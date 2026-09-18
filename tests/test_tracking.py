"""Synthetic checks for ByteTracker + TargetLock state transitions, both tracking modes.
Runs anywhere (numpy/scipy/opencv, no camera and no accelerator):
    python3 -m pytest tests/ -q
"""

import time

import cv2
import numpy as np

from adts.bytetrack import ByteTracker
from adts.detector import Detections
from adts.target import AI, COAST, IDLE, LOCKED, LOST, SCENE, TargetLock

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
    lock.refresh_candidates(tracks)
    assert lock.select_id(tracks[0].track_id) and lock.engage()
    old_neighbour = next(t.track_id for t in tracks if t.track_id != lock.track_id)
    locked_id = lock.track_id
    # the target jumps far (as if the detector re-found it offset): ByteTrack can't
    # match it by IoU, so it starts a new track. The neighbour stays put.
    for _ in range(3):
        trk.update(dets(moving(640, 380), moving(700, 340)))
        lock.update(trk, trk.frame_id)
    assert lock.state == LOCKED
    assert lock.track_id not in (locked_id, old_neighbour)


def test_selection_moves_left_to_right_without_locking():
    trk, lock = ByteTracker(), TargetLock((W, H), 66)
    for _ in range(3):
        tracks = trk.update(dets(moving(100), moving(600), moving(1100)))
    by_x = sorted(tracks, key=lambda t: t.xyxy[0])
    lock.refresh_candidates(tracks)
    assert lock.sel_id == by_x[1].track_id  # nearest the crosshair to begin with
    assert lock.select_step(1) and lock.sel_id == by_x[2].track_id
    assert lock.select_step(1) and lock.sel_id == by_x[0].track_id  # wraps
    assert lock.select_step(-1) and lock.sel_id == by_x[2].track_id
    assert lock.state == IDLE  # selecting alone never engages
    assert lock.engage() and lock.state == LOCKED and lock.track_id == by_x[2].track_id


def test_candidates_are_the_five_nearest_the_crosshair():
    trk, lock = ByteTracker(), TargetLock((W, H), 66)
    xs = [40, 200, 380, 560, 620, 700, 900, 1200]  # 8 targets, only 5 may be selectable
    for _ in range(3):
        tracks = trk.update(dets(*[moving(x) for x in xs]))
    lock.refresh_candidates(tracks)
    assert len(lock.candidates) == 5
    cand_x = [t.xyxy[0] for t in lock.candidates]
    assert cand_x == sorted(cand_x)  # left to right, as the operator sees them
    far = min(tracks, key=lambda t: t.xyxy[0])  # x=40, nowhere near the centre
    assert far.track_id not in [t.track_id for t in lock.candidates]
    assert not lock.select_id(far.track_id)  # only the five are selectable


def test_engage_without_detections_is_refused():
    trk, lock = ByteTracker(), TargetLock((W, H), 66)
    lock.refresh_candidates(trk.update(dets()))
    assert not lock.engage() and lock.state == IDLE


def test_point_lock_still_works():
    trk, lock = ByteTracker(), TargetLock((W, H), 66)
    for _ in range(3):
        tracks = trk.update(dets(moving(100), moving(600), moving(1100)))
    by_x = sorted(tracks, key=lambda t: t.xyxy[0])
    assert lock.start_point(620, 320, tracks) and lock.track_id == by_x[1].track_id
    assert not lock.start_point(5, 5, tracks)  # empty spot -> refused


def test_scene_lock_follows_a_translating_scene():
    # Blurred noise, not raw noise: CSRT needs structure at a scale bigger than one pixel,
    # and so does any real scene.
    noise = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 6)
    trk, lock = ByteTracker(), TargetLock((W, H), 66)
    assert lock.start_scene(base) and lock.state == LOCKED and lock.mode == SCENE
    x0 = lock.box[0]
    for shift in range(4, 33, 4):  # the whole scene slides left, 4 px per frame
        lock.update(trk, 0, np.roll(base, -shift, axis=1))
    assert lock.state == LOCKED
    assert lock.box[0] < x0 - 20  # the gate contents were followed, not the gate position
    assert lock.track_id is None  # a scene patch has no ByteTrack identity
    lock.stop()
    assert lock.state == IDLE and lock.mode == AI


def test_scene_lock_coasts_then_gives_up():
    class NeverFinds:
        """Stands in for CSRT losing the patch, so the coast/lost timers are what is tested."""
        gate = "M"

        def gate_box(self):
            return (0.0, 0.0, 100.0, 100.0)

        def start(self, frame):
            return True

        def update(self, frame):
            return False, None

        def stop(self):
            pass

    trk, lock = ByteTracker(), TargetLock((W, H), 66, coast_s=0.1, lost_hold_s=0.1)
    lock.scene = NeverFinds()
    assert lock.start_scene(np.zeros((H, W, 3), np.uint8))
    lock.update(trk, 0, None)
    assert lock.state == COAST
    time.sleep(0.12)
    lock.update(trk, 0, None)
    assert lock.state == LOST
    time.sleep(0.12)
    lock.update(trk, 0, None)
    assert lock.state == IDLE


def test_angle_error_signs():
    lock = TargetLock((W, H), 66)
    lock.state, lock.box = LOCKED, np.array([W - 20, 0, W, 20.0])  # top-right corner
    az, el = lock.angle_error()
    assert 32 < az < 33.1 and el > 0


def test_gate_change_starts_a_three_second_preview():
    from adts.scene_track import PREVIEW_S, SceneTracker

    st = SceneTracker((W, H))
    assert not st.previewing  # nothing changed yet
    assert st.set_gate("L") and st.previewing
    assert st.preview_until - time.monotonic() <= PREVIEW_S + 0.01
    st.preview_until = time.monotonic() - 0.01  # fast-forward past the window
    assert not st.previewing
    assert st.step_gate(1) and st.previewing  # step_gate goes through set_gate too


def test_bad_gate_size_is_refused_without_starting_a_preview():
    from adts.scene_track import SceneTracker

    st = SceneTracker((W, H))
    assert not st.set_gate("XL")
    assert not st.previewing
