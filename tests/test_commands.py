"""Checks for the operator command path: MAVLink sub-code -> (kind, arg) -> effect.

The decode half is a pure function, so the wire protocol is tested without opening a link:
    python3 -m pytest tests/ -q
"""

import numpy as np
from pymavlink import mavutil

from adts.bytetrack import ByteTracker
from adts.commands import Controller
from adts.mavlink_io import decode_user_command
from adts.classes import WALDO_NAMES
from adts.render import COLOR_NAMES, OverlayConfig
from adts.target import AI, IDLE, LOCKED, SCENE, TargetLock

M = mavutil.mavlink
W, H = 1280, 720


class FakeTrack:
    """Enough of a ByteTrack track for selection and locking."""

    def __init__(self, track_id, x, cls=1):
        self.track_id, self.cls = track_id, cls
        self.xyxy = np.array([float(x), 340.0, float(x) + 40, 380.0])


def setup(ai_on=True):
    lock = TargetLock((W, H), 66)
    tracker = ByteTracker()
    ctl = Controller(lock, tracker, OverlayConfig(), WALDO_NAMES, ai_on=ai_on)
    return ctl, lock, tracker


def frame():
    return np.zeros((H, W, 3), np.uint8)


def test_decode_matches_the_documented_table():
    table = {
        (M.MAV_CMD_USER_1, 1, 0): ("cand", 1),
        (M.MAV_CMD_USER_1, -1, 0): ("cand", -1),
        (M.MAV_CMD_USER_1, 0, 7): ("select", 7),
        (M.MAV_CMD_USER_1, 2, 0): ("auto", None),
        (M.MAV_CMD_USER_1, 3, 0): ("engage", None),
        (M.MAV_CMD_USER_1, 4, 0): ("stop", None),
        (M.MAV_CMD_USER_1, 5, -1): ("ai", -1),
        (M.MAV_CMD_USER_2, 1, 0): ("scene_start", None),
        (M.MAV_CMD_USER_2, 0, 0): ("scene_stop", None),
        (M.MAV_CMD_USER_2, 2, 2): ("gate", 2),
        (M.MAV_CMD_USER_3, 1, 0): ("overlay", 0),
        (M.MAV_CMD_USER_3, 2, 1): ("reticle", 1),
        (M.MAV_CMD_USER_3, 3, 1): ("lang", 1),
        (M.MAV_CMD_USER_3, 4, 4): ("color", 4),
    }
    for (cmd, p1, p2), expected in table.items():
        assert decode_user_command(cmd, float(p1), float(p2)) == expected, (cmd, p1, p2)


def test_decode_refuses_unknown_subcommands():
    assert decode_user_command(M.MAV_CMD_USER_1, 9, 0) is None
    assert decode_user_command(M.MAV_CMD_USER_2, 7, 0) is None
    assert decode_user_command(M.MAV_CMD_USER_3, 0, 0) is None
    assert decode_user_command(M.MAV_CMD_CAMERA_STOP_TRACKING, 0, 0) is None


def test_overlay_and_reticle_flags():
    ctl, _, _ = setup()
    assert ctl.run("overlay", 0, []) and ctl.overlay.on is False
    assert ctl.run("overlay", -1, []) and ctl.overlay.on is True
    assert ctl.run("reticle", 0, []) and ctl.overlay.reticle is False
    assert not ctl.run("overlay", 7, [])  # out of range -> FAILED ack, nothing changed
    assert ctl.overlay.on is True


def test_colour_gate_and_language():
    ctl, lock, _ = setup()
    assert ctl.run("color", 2, []) and COLOR_NAMES[ctl.overlay.color_idx] == "red"
    assert ctl.run("color", -1, []) and COLOR_NAMES[ctl.overlay.color_idx] == "white"
    assert not ctl.run("color", 99, [])
    assert ctl.run("gate", 2, []) and lock.scene.gate == "L"
    assert ctl.run("gate", -1, []) and lock.scene.gate == "S"  # wraps round
    assert ctl.names[1] == "Insan"
    assert ctl.run("lang", 1, []) and ctl.names[1] == "Person"
    assert ctl.run("lang", -1, []) and ctl.names[1] == "Insan"


def test_selection_then_engage():
    ctl, lock, _ = setup()
    tracks = [FakeTrack(1, 100), FakeTrack(2, 600), FakeTrack(3, 1100)]
    lock.refresh_candidates(tracks)
    assert ctl.run("cand", 1, tracks) and lock.sel_id == 3
    assert lock.state == IDLE  # still only a highlight
    assert ctl.run("engage", None, tracks) and lock.state == LOCKED and lock.track_id == 3
    assert ctl.run("stop", None, tracks) and lock.state == IDLE


def test_ai_off_drops_the_ai_lock_and_its_tracks():
    ctl, lock, tracker = setup()
    tracks = [FakeTrack(1, 600)]
    lock.refresh_candidates(tracks)
    assert ctl.run("engage", None, tracks) and lock.state == LOCKED
    tracker.tracked = list(tracks)
    assert ctl.run("ai", 0, tracks)
    assert ctl.ai_on is False and lock.state == IDLE and tracker.tracked == []
    assert lock.candidates == [] and lock.sel_id is None
    assert ctl.run("ai", -1, tracks) and ctl.ai_on is True


def test_ai_off_leaves_a_scene_track_running():
    ctl, lock, _ = setup()
    assert ctl.run("scene_start", None, [], frame()) and lock.mode == SCENE
    assert ctl.run("ai", 0, [])
    assert lock.mode == SCENE and lock.state == LOCKED  # independent of the detector


def test_scene_cancel_only_cancels_a_scene_track():
    ctl, lock, _ = setup()
    tracks = [FakeTrack(1, 600)]
    lock.refresh_candidates(tracks)
    ctl.run("engage", None, tracks)
    assert not ctl.run("scene_stop", None, tracks)  # an AI lock is not the scene page's to drop
    assert lock.state == LOCKED and lock.mode == AI
    assert ctl.run("scene_start", None, tracks, frame()) and lock.mode == SCENE
    assert ctl.run("scene_stop", None, tracks) and lock.state == IDLE


def test_unknown_command_is_refused():
    ctl, _, _ = setup()
    assert not ctl.run("nope", None, [])


def test_a_malformed_command_does_not_kill_command_reception():
    """MAVLink uses NaN for "leave unchanged", so a GCS really can send one. It must cost the
    offending packet and nothing else - a dead rx thread would mean no commands for the rest
    of the flight."""
    from adts.mavlink_io import MavlinkIO

    io = MavlinkIO("udpin:127.0.0.1:14611", frame_size=(W, H))
    try:
        gcs = mavutil.mavlink_connection("udpout:127.0.0.1:14611", source_system=1, source_component=1)
        nan = float("nan")
        gcs.mav.command_long_send(1, M.MAV_COMP_ID_CAMERA, M.MAV_CMD_USER_1, 0, nan, 0, 0, 0, 0, 0, 0)
        gcs.mav.command_long_send(1, M.MAV_COMP_ID_CAMERA, M.MAV_CMD_USER_1, 0, 3, 0, 0, 0, 0, 0, 0)
        cmd = io.commands.get(timeout=10)
        assert (cmd.kind, cmd.arg) == ("engage", None)
    finally:
        io.close()
