"""Checks for tools/gcs_link.py, the shared connection used by gcs_sim.py and gcs_gui.py:
every command method sends what adts/mavlink_io.py expects, and ACK/track/rect replies
come back through on_event. A real loopback UDP link end to end, no serial hardware
needed, no tkinter needed (that part is only exercised by hand - see gcs_gui.py's
docstring for the manual check):
    python3 -m pytest tests/ -q
"""

import queue
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from gcs_link import GcsLink  # noqa: E402

from adts.mavlink_io import MavlinkIO  # noqa: E402

W, H = 1280, 720
PORT = 14625


def linked(port=PORT):
    """A MavlinkIO (the ADTS side) and a GcsLink (the GCS side) talking over loopback UDP,
    with the GcsLink's events collected on a queue instead of a callback doing real work."""
    io = MavlinkIO(f"udpin:127.0.0.1:{port}", frame_size=(W, H))
    events = queue.Queue()
    gcs = GcsLink(f"udpout:127.0.0.1:{port}", on_event=lambda k, d: events.put((k, d)))
    return io, gcs, events


def test_commands_reach_the_tracker_and_ack_comes_back():
    io, gcs, events = linked()
    try:
        gcs.engage()
        cmd = io.commands.get(timeout=10)
        assert (cmd.kind, cmd.arg) == ("engage", None)
        cmd.done(True)
        kind, data = events.get(timeout=10)
        assert kind == "ack" and data[1] == "MAV_RESULT_ACCEPTED"
    finally:
        io.close()
        gcs.close()


def test_every_command_method_sends_the_right_subcode():
    io, gcs, _events = linked(PORT + 1)
    try:
        table = [
            (gcs.cand_next, ("cand", 1)), (gcs.cand_prev, ("cand", -1)),
            (lambda: gcs.select(3), ("select", 3)), (gcs.auto, ("auto", None)),
            (gcs.cancel, ("stop", None)),
            (gcs.scene_start, ("scene_start", None)), (gcs.scene_stop, ("scene_stop", None)),
            (lambda: gcs.gate(2), ("gate", 2)),
            (lambda: gcs.overlay(0), ("overlay", 0)), (lambda: gcs.reticle(1), ("reticle", 1)),
            (lambda: gcs.lang(1), ("lang", 1)), (lambda: gcs.color(3), ("color", 3)),
            (gcs.stop, ("stop", None)), (gcs.info, None),
        ]
        for send, expected in table:
            send()
            if expected is None:  # info -> CAMERA_INFORMATION, not a queued command
                continue
            cmd = io.commands.get(timeout=10)
            assert (cmd.kind, cmd.arg) == expected
            cmd.done(True)
    finally:
        io.close()
        gcs.close()


def test_track_state_and_rect_events_are_parsed():
    io, gcs, events = linked(PORT + 2)
    try:
        io.set_status("LOCKED", (0.1, 0.2, 0.3, 0.4), (5.0, -2.5))
        deadline = time.time() + 5
        seen_track = seen_rect = False
        while time.time() < deadline and not (seen_track and seen_rect):
            kind, data = events.get(timeout=5)
            if kind == "track":
                assert data[0] == "LOCKED" and data[1:] == pytest.approx((5.0, -2.5), abs=1e-4)
                seen_track = True
            elif kind == "rect":
                assert data == pytest.approx((0.1, 0.2, 0.3, 0.4), abs=1e-5)
                seen_rect = True
        assert seen_track and seen_rect
    finally:
        io.close()
        gcs.close()


def test_point_and_rect_and_camera_info():
    io, gcs, events = linked(PORT + 3)
    try:
        gcs.point(0.25, 0.5)
        cmd = io.commands.get(timeout=10)
        assert cmd.kind == "point" and cmd.arg == pytest.approx((0.25 * W, 0.5 * H), abs=1e-2)
        cmd.done(False)
        kind, data = events.get(timeout=10)
        assert kind == "ack" and data[1] == "MAV_RESULT_FAILED"

        gcs.rect(0.1, 0.1, 0.5, 0.5)
        cmd = io.commands.get(timeout=10)
        assert cmd.kind == "rect" and cmd.arg == pytest.approx((0.1 * W, 0.1 * H, 0.5 * W, 0.5 * H), abs=1e-2)
        cmd.done(True)
        events.get(timeout=10)  # that rect's ack

        gcs.info()
        kind, data = events.get(timeout=10)
        assert kind == "camera_info"
        assert data[0] == "ADTS" and data[1:3] == (W, H)
    finally:
        io.close()
        gcs.close()
