"""Target lock state machine on top of ByteTrack.

    IDLE ─start─► LOCKED ─not detected─► COAST (Kalman prediction, up to coast_s)
                    ▲                      │  same track re-matched, or a NEW track of the
                    └──── re-acquired ◄────┤  same class appears near the prediction
                                           └─ coast_s ran out ─► LOST ─(lost_hold_s)─► IDLE

Commands (start/stop/next/prev/select) come from MAVLink or the dev window through
the same methods, and each returns True/False, which becomes the MAVLink COMMAND_ACK.
"""

import math
import time

import numpy as np

IDLE, LOCKED, COAST, LOST = "IDLE", "LOCKED", "COAST", "LOST"


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class TargetLock:
    def __init__(self, frame_size, hfov_deg, vfov_deg=None, coast_s=1.5, lost_hold_s=2.0):
        self.w, self.h = frame_size
        self.hfov = math.radians(hfov_deg)
        # Without a given VFOV, derive it for square pixels and no sensor crop.
        self.vfov = math.radians(vfov_deg) if vfov_deg else 2 * math.atan(math.tan(self.hfov / 2) * self.h / self.w)
        self.coast_s, self.lost_hold_s = coast_s, lost_hold_s
        self.state = IDLE
        self.track_id = None
        self.cls = None
        self.box = None  # current target box (frame pixels); predicted while in COAST
        self.state_since = time.monotonic()
        self.lost_frame = 0

    # ---- commands -------------------------------------------------------
    def _lock(self, track):
        self.track_id, self.cls, self.box = track.track_id, track.cls, track.xyxy
        self._set(LOCKED)
        return True

    def start_point(self, x, y, tracks):
        """Lock the track whose box contains (x, y) in pixels. If none does, lock the nearest centre within 10% of the frame width."""
        inside = [t for t in tracks if t.xyxy[0] <= x <= t.xyxy[2] and t.xyxy[1] <= y <= t.xyxy[3]]
        if inside:
            return self._lock(min(inside, key=lambda t: np.prod(t.xyxy[2:] - t.xyxy[:2])))  # smallest wins
        near = self._nearest(tracks, x, y, 0.1 * self.w)
        return self._lock(near) if near else False

    def start_rect(self, rect, tracks):
        from .bytetrack import iou_matrix
        if not tracks:
            return False
        ious = iou_matrix(np.array([rect], dtype=float), np.array([t.xyxy for t in tracks]))[0]
        if ious.max() > 0.1:
            return self._lock(tracks[int(ious.argmax())])
        return self.start_point(*_center(rect), tracks)

    def start_auto(self, tracks):
        """Lock the detection nearest the image centre (the crosshair)."""
        near = self._nearest(tracks, self.w / 2, self.h / 2, float("inf"))
        return self._lock(near) if near else False

    def select_id(self, track_id, tracks):
        for t in tracks:
            if t.track_id == track_id:
                return self._lock(t)
        return False

    def cycle(self, step, tracks):
        """Next/prev target, ordered left to right as the operator sees them."""
        if not tracks:
            return False
        order = sorted(tracks, key=lambda t: _center(t.xyxy)[0])
        ids = [t.track_id for t in order]
        if self.track_id in ids:
            return self._lock(order[(ids.index(self.track_id) + step) % len(order)])
        if self.box is None:  # nothing locked yet: start from the crosshair
            return self.start_auto(tracks)
        # Locked target isn't visible (COAST/LOST): step to the first track to its right or left.
        ref = _center(self.box)[0]
        xs = [_center(t.xyxy)[0] for t in order]
        if step > 0:
            i = next((k for k, x in enumerate(xs) if x > ref), 0)
        else:
            i = next((k for k in reversed(range(len(xs))) if xs[k] < ref), len(xs) - 1)
        return self._lock(order[i])

    def stop(self):
        self.track_id = self.box = self.cls = None
        self._set(IDLE)
        return True

    # ---- per frame ------------------------------------------------------
    def update(self, tracker, frame_id):
        if self.state == IDLE:
            return
        now = time.monotonic()
        if self.state == LOST:
            if now - self.state_since > self.lost_hold_s:
                self.stop()
            return
        active = [t for t in tracker.tracked if t.activated]
        hit = next((t for t in active if t.track_id == self.track_id), None)
        if hit:
            self.box, self.cls = hit.xyxy, hit.cls
            if self.state != LOCKED:
                self._set(LOCKED)
            return
        if self.state == LOCKED:
            self._set(COAST)
            self.lost_frame = frame_id
        lost = next((t for t in tracker.lost if t.track_id == self.track_id), None)
        if lost is not None:
            self.box = lost.xyxy
        # Re-acquire: ByteTrack sometimes starts a new ID for the same object after a
        # miss. Only accept tracks BORN after the lock was lost, so an unrelated object
        # that was already being tracked nearby can't take over the lock.
        cx, cy = _center(self.box)
        gate = max(1.5 * math.hypot(self.box[2] - self.box[0], self.box[3] - self.box[1]), 0.05 * self.w)
        fresh = [t for t in active if t.start_frame >= self.lost_frame and t.cls == self.cls]
        near = self._nearest(fresh, cx, cy, gate)
        if near:
            self._lock(near)
        elif now - self.state_since > self.coast_s:
            self._set(LOST)

    def _set(self, state):
        self.state, self.state_since = state, time.monotonic()

    @staticmethod
    def _nearest(tracks, x, y, max_dist):
        best, best_d = None, max_dist
        for t in tracks:
            cx, cy = _center(t.xyxy)
            d = math.hypot(cx - x, cy - y)
            if d <= best_d:
                best, best_d = t, d
        return best

    # ---- outputs --------------------------------------------------------
    def angle_error(self):
        """(azimuth, elevation) in degrees from the boresight to the target centre. Positive = right / up."""
        if self.box is None:
            return None
        cx, cy = _center(self.box)
        nx = (cx - self.w / 2) / (self.w / 2)
        ny = (self.h / 2 - cy) / (self.h / 2)
        return (math.degrees(math.atan(nx * math.tan(self.hfov / 2))),
                math.degrees(math.atan(ny * math.tan(self.vfov / 2))))

    def normalized_box(self):
        if self.box is None:
            return None
        x1, y1, x2, y2 = self.box
        return (max(0.0, x1 / self.w), max(0.0, y1 / self.h), min(1.0, x2 / self.w), min(1.0, y2 / self.h))
