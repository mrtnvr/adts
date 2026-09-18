"""Synthetic checks for ByteTracker + TargetLock state transitions. Runs anywhere (numpy/scipy only):
    python3 -m pytest tests/ -q
"""

import time

import numpy as np

from adts.bytetrack import ByteTracker
from adts.detector import Detections
from adts.target import COAST, IDLE, LOCKED, LOST, TargetLock

W, H = 1280, 720


def dets(*boxes, conf=0.9, cls=0):
    if not boxes:
        return Detections.empty()
    return Detections(np.array(boxes, np.float32), np.full(len(boxes), conf, np.float32), np.full(len(boxes), cls, np.int32))


def moving(x, y=300, s=40):
    return (x, y, x + s, y + s)


def test_ids_stable_while_moving():
    trk = ByteTracker()
    ids = set()
    for i in range(30):
        out = trk.update(dets(moving(100 + 5 * i), moving(800 - 5 * i, 500)))
        if i > 1:
            ids |= {t.track_id for t in out}
    assert len(ids) == 2


def test_track_survives_short_gap_with_same_id():
    trk = ByteTracker()
    for i in range(10):
        trk.update(dets(moving(100 + 5 * i)))
    tid = trk.tracked[0].track_id
    for i in range(10, 15):  # 5-frame occlusion
        trk.update(dets())
    assert any(t.track_id == tid for t in trk.lost)
    out = trk.update(dets(moving(100 + 5 * 15)))  # comes back where the Kalman filter predicted
    assert [t.track_id for t in out] == [tid]


def test_lock_coast_relock():
    trk, lock = ByteTracker(), TargetLock((W, H), 66, coast_s=0.5)
    for i in range(5):
        tracks = trk.update(dets(moving(600 + 2 * i, 340)))
        lock.update(trk, trk.frame_id)
    assert lock.start_auto(tracks) and lock.state == LOCKED
    tid = lock.track_id
    trk.update(dets())
    lock.update(trk, trk.frame_id)
    assert lock.state == COAST and lock.box is not None
    tracks = trk.update(dets(moving(600 + 2 * 6, 340)))
    lock.update(trk, trk.frame_id)
    assert lock.state == LOCKED and lock.track_id == tid


def test_lock_lost_then_idle():
    trk, lock = ByteTracker(), TargetLock((W, H), 66, coast_s=0.1, lost_hold_s=0.1)
    for i in range(3):
        tracks = trk.update(dets(moving(600, 340)))
    lock.start_auto(tracks)
    for _ in range(3):
        trk.update(dets())
        lock.update(trk, trk.frame_id)
        time.sleep(0.06)
    assert lock.state == LOST
    time.sleep(0.12)
    lock.update(trk, trk.frame_id)
    assert lock.state == IDLE


def test_reacquire_only_new_tracks_of_same_class():
    trk, lock = ByteTracker(), TargetLock((W, H), 66, coast_s=5)
    for _ in range(3):
        tracks = trk.update(dets(moving(600, 340), moving(700, 340)))
    lock.select_id(tracks[0].track_id, tracks)
    old_neighbour = next(t.track_id for t in tracks if t.track_id != lock.track_id)
    locked_id = lock.track_id
    # the target jumps far (as if the detector re-found it offset): ByteTrack can't
    # match it by IoU, so it starts a new track. The neighbour stays put.
    for _ in range(3):
        trk.update(dets(moving(640, 380), moving(700, 340)))
        lock.update(trk, trk.frame_id)
    assert lock.state == LOCKED
    assert lock.track_id not in (locked_id, old_neighbour)


def test_cycle_left_to_right_and_point():
    trk, lock = ByteTracker(), TargetLock((W, H), 66)
    for _ in range(3):
        tracks = trk.update(dets(moving(100), moving(600), moving(1100)))
    by_x = sorted(tracks, key=lambda t: t.xyxy[0])
    assert lock.start_point(620, 320, tracks) and lock.track_id == by_x[1].track_id
    lock.cycle(1, tracks)
    assert lock.track_id == by_x[2].track_id
    lock.cycle(1, tracks)
    assert lock.track_id == by_x[0].track_id  # wraps
    lock.cycle(-1, tracks)
    assert lock.track_id == by_x[2].track_id
    assert not lock.start_point(5, 5, tracks)  # empty spot -> refused


def test_angle_error_signs():
    lock = TargetLock((W, H), 66)
    lock.state, lock.box = LOCKED, np.array([W - 20, 0, W, 20.0])  # top-right corner
    az, el = lock.angle_error()
    assert 32 < az < 33.1 and el > 0
