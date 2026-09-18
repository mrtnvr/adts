"""A small ByteTrack in numpy + scipy (arXiv:2110.06864, same thresholds as
Ultralytics' bytetrack.yaml).

Written here instead of reusing ultralytics.trackers because importing Ultralytics
pulls in torch. On the Pi, torch would add hundreds of MB and seconds of startup for
nothing, since detection runs on the Hailo. It also gives the target lock direct
access to lost tracks and their Kalman predictions, which track.py had to dig out of
model.predictor.trackers[0].
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h), as in ByteTrack/DeepSORT."""

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.w_pos, self.w_vel = 1 / 20, 1 / 160

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-2, self.w_pos * h,
               self.w_vel * h, self.w_vel * h, 1e-5, self.w_vel * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        return mean + K @ (z - self.H @ mean), cov - K @ S @ K.T


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def linear_assignment(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr, mc = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [r for r in range(cost.shape[0]) if r not in mr], [c for c in range(cost.shape[1]) if c not in mc]


class Track:
    _next_id = 1

    def __init__(self, xyxy, score, cls):
        self.xyxy_det = np.asarray(xyxy, dtype=np.float64)
        self.score, self.cls = float(score), int(cls)
        self.mean = self.cov = None
        self.state, self.activated = NEW, False
        self.track_id = 0
        self.frame_id = self.start_frame = 0

    @property
    def xyxy(self):
        if self.mean is None:
            return self.xyxy_det.copy()
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def _assign_id(self):
        # IDs are handed out on confirmation, not on creation. Otherwise every
        # one-frame noise detection uses up an ID and the operator sees IDs in the
        # hundreds after a minute, which is bad for "select ID" commands.
        self.track_id = Track._next_id
        Track._next_id += 1

    def activate(self, kf, frame_id):
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(self.xyxy_det))
        self.state = TRACKED
        self.activated = frame_id == 1  # like ByteTrack: confirmed at once only on the first frame
        if self.activated:
            self._assign_id()
        self.frame_id = self.start_frame = frame_id

    def update(self, kf, det, frame_id):
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(det.xyxy_det))
        self.score, self.cls = det.score, det.cls
        if not self.activated:
            self._assign_id()
        self.state, self.activated, self.frame_id = TRACKED, True, frame_id


class ByteTracker:
    def __init__(self, high=0.25, low=0.1, new_track=0.25, match=0.8, buffer_frames=30):
        self.high, self.low, self.new_track, self.match = high, low, new_track, match
        self.buffer_frames = buffer_frames
        self.kf = KalmanXYAH()
        self.frame_id = 0
        self.tracked, self.lost = [], []

    @staticmethod
    def _dist(tracks, dets, fuse_score):
        iou = iou_matrix(np.array([t.xyxy for t in tracks]).reshape(-1, 4),
                         np.array([d.xyxy_det for d in dets]).reshape(-1, 4))
        if fuse_score and len(dets):
            iou = iou * np.array([d.score for d in dets])[None, :]
        return 1 - iou

    def update(self, det):
        """det: detector.Detections. Returns the confirmed, currently matched tracks."""
        self.frame_id += 1
        fid = self.frame_id
        all_dets = [Track(b, s, c) for b, s, c in zip(det.xyxy, det.conf, det.cls)]
        high = [d for d in all_dets if d.score >= self.high]
        low = [d for d in all_dets if self.low <= d.score < self.high]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool:
            if t.state != TRACKED:
                t.mean[7] = 0  # ByteTrack zeroes a lost track's height velocity
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        activated, refound = [], []
        # 1st pass: high-score detections against every track, lost ones included.
        m, u_t, u_d = linear_assignment(self._dist(pool, high, True), self.match)
        for ti, di in m:
            t = pool[ti]
            (activated if t.state == TRACKED else refound).append(t)
            t.update(self.kf, high[di], fid)
        # 2nd pass: low-score detections against the still-unmatched tracked (not lost) ones.
        remain = [pool[i] for i in u_t if pool[i].state == TRACKED]
        m, u_t2, _ = linear_assignment(self._dist(remain, low, False), 0.5)
        for ti, di in m:
            remain[ti].update(self.kf, low[di], fid)
            activated.append(remain[ti])
        newly_lost = []
        for i in u_t2:
            t = remain[i]
            t.state = LOST
            newly_lost.append(t)
        # Unconfirmed (1-frame-old) tracks get one chance to match the leftover high detections.
        left = [high[i] for i in u_d]
        m, u_unc, u_d = linear_assignment(self._dist(unconfirmed, left, True), 0.7)
        for ti, di in m:
            unconfirmed[ti].update(self.kf, left[di], fid)
            activated.append(unconfirmed[ti])
        for i in u_unc:
            unconfirmed[i].state = REMOVED
        for i in u_d:
            d = left[i]
            if d.score >= self.new_track:
                d.activate(self.kf, fid)
                activated.append(d)

        still_lost = [t for t in self.lost if t.state == LOST and t not in refound]
        still_lost += newly_lost
        self.lost = [t for t in still_lost if fid - t.frame_id <= self.buffer_frames]
        self.tracked = [t for t in activated + refound if t.state == TRACKED]
        return [t for t in self.tracked if t.activated]

    def predict_only(self):
        """Step every track's Kalman filter without a detection (for skipped detection frames)."""
        for t in self.tracked + self.lost:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        return [t for t in self.tracked if t.activated]
