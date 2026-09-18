"""Target lock state machine, in two independent modes.

    AI takip     - the lock follows a ByteTrack track_id (YOLO detections).
    Sahne takip  - the lock follows a patch of the scene through CSRT (see scene_track.py),
                   with no detector involved.

Both share one state machine, so the MAVLink status output doesn't care which is driving:

    IDLE ─start─► LOCKED ─not detected─► COAST (Kalman prediction, up to coast_s)
                    ▲                      │  same track re-matched, or a NEW track of the
                    └──── re-acquired ◄────┤  same class appears near the prediction
                                           └─ coast_s ran out ─► LOST ─(lost_hold_s)─► IDLE

AI mode separates SELECTING a target from ENGAGING it: `candidates` holds the five
detections nearest the crosshair (ordered left to right, as the operator sees them),
select_step/select_id move the highlight, and only engage() actually takes the lock. The
operator's "İLERİ / GERİ / TAKİP BAŞLAT" buttons map straight onto those.

Commands come from MAVLink or the dev window through the same methods, and each returns
True/False, which becomes the MAVLink COMMAND_ACK.
"""

import math
import time

import numpy as np

from .scene_track import SceneTracker

IDLE, LOCKED, COAST, LOST = "IDLE", "LOCKED", "COAST", "LOST"
AI, SCENE = "ai", "scene"
MAX_CANDIDATES = 5


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class TargetLock:
    def __init__(self, frame_size, hfov_deg, vfov_deg=None, coast_s=1.5, lost_hold_s=2.0, gate="M"):
        self.w, self.h = frame_size
        self.hfov = math.radians(hfov_deg)
        # Without a given VFOV, derive it for square pixels and no sensor crop.
        self.vfov = math.radians(vfov_deg) if vfov_deg else 2 * math.atan(math.tan(self.hfov / 2) * self.h / self.w)
        self.coast_s, self.lost_hold_s = coast_s, lost_hold_s
        self.state = IDLE
        self.mode = AI
        self.track_id = None
        self.cls = None
        self.box = None  # current target box (frame pixels); predicted while in COAST
        self.state_since = time.monotonic()
        self.lost_frame = 0
        self.scene = SceneTracker(frame_size, gate)
        self.candidates = []  # the five tracks nearest the crosshair, left to right
        self.sel_id = None  # highlighted candidate; engage() is what locks it

    # ---- selection (AI mode) --------------------------------------------
    def refresh_candidates(self, tracks):
        """Recompute the candidate set. Called once per frame, before commands run, so
        İLERİ/GERİ and TAKİP BAŞLAT act on exactly what the operator is looking at."""
        nearest = sorted(tracks, key=lambda t: math.dist(_center(t.xyxy), (self.w / 2, self.h / 2)))
        self.candidates = sorted(nearest[:MAX_CANDIDATES], key=lambda t: _center(t.xyxy)[0])
        ids = [t.track_id for t in self.candidates]
        if self.sel_id not in ids:
            # Selection fell out of the set (target left, or nothing selected yet): fall back
            # to the one nearest the crosshair rather than leaving a dangling highlight.
            self.sel_id = nearest[0].track_id if nearest else None

    @property
    def selected(self):
        return next((t for t in self.candidates if t.track_id == self.sel_id), None)

    def select_step(self, step):
        """Seçili hedef İLERİ / GERİ. Moves the highlight only; the lock is untouched."""
        ids = [t.track_id for t in self.candidates]
        if not ids:
            return False
        i = ids.index(self.sel_id) if self.sel_id in ids else 0
        self.sel_id = ids[(i + step) % len(ids)]
        return True

    def select_id(self, track_id):
        """Highlight a candidate by track ID. Only the current five are selectable."""
        if track_id not in [t.track_id for t in self.candidates]:
            return False
        self.sel_id = track_id
        return True

    def engage(self):
        """Seçili hedef takip BAŞLAT."""
        target = self.selected
        return self._lock(target) if target else False

    # ---- commands -------------------------------------------------------
    def _lock(self, track):
        self.scene.stop()  # taking an AI target ends any scene track
        self.mode = AI
        self.track_id, self.cls, self.box = track.track_id, track.cls, track.xyxy
        self.sel_id = track.track_id
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
        """Lock the detection nearest the image centre (the crosshair), in one step."""
        near = self._nearest(tracks, self.w / 2, self.h / 2, float("inf"))
        return self._lock(near) if near else False

    def start_scene(self, frame):
        """Sabit sahne/obje takip BAŞLAT: lock whatever is inside the gate. `frame` must be
        the clean frame, before any symbology is drawn on it."""
        if not self.scene.start(frame):
            return False
        self.mode = SCENE
        self.track_id, self.cls = None, None
        self.box = self.scene.gate_box()
        self._set(LOCKED)
        return True

    def stop(self):
        self.scene.stop()
        self.mode = AI
        self.track_id = self.box = self.cls = None
        self._set(IDLE)
        return True

    # ---- per frame ------------------------------------------------------
    def update(self, tracker, frame_id, frame=None):
        if self.state == IDLE:
            return
        now = time.monotonic()
        if self.state == LOST:
            if now - self.state_since > self.lost_hold_s:
                self.stop()
            return
        if self.mode == SCENE:
            self._update_scene(frame, now)
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

    def _update_scene(self, frame, now):
        ok, box = self.scene.update(frame)
        if ok:
            self.box = box
            if self.state != LOCKED:
                self._set(LOCKED)
        elif self.state == LOCKED:
            # CSRT has no Kalman prediction to coast on: hold the last box and give the
            # patch coast_s to come back (a brief occlusion or a motion blur burst).
            self._set(COAST)
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
