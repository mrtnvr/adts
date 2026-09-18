"""Smoke checks for the symbology. A KeyError or a None class name in here would only show
up on the drone, where nobody can attach a debugger, so every overlay state gets drawn once.
"""

import numpy as np

from adts.classes import WALDO_NAMES, display_names
from adts.render import COLOR_NAMES, OverlayConfig, draw
from adts.target import COAST, LOCKED, LOST, TargetLock

W, H = 640, 360
STATS = {"fps": 30.0, "det_ms": 12.0, "temp_c": 55.0, "mav": True, "rec": True, "ai": True}


class FakeTrack:
    def __init__(self, track_id, x, cls=1):
        self.track_id, self.cls = track_id, cls
        self.xyxy = np.array([float(x), 150.0, float(x) + 30, 190.0])


def render(lock, overlay, tracks=(), stats=None, lang="tr"):
    img = np.zeros((H, W, 3), np.uint8)
    draw(img, list(tracks), lock, display_names(WALDO_NAMES, lang), {**STATS, **(stats or {})}, overlay)
    return img


def test_every_lock_state_and_colour_draws():
    lock = TargetLock((W, H), 66)
    tracks = [FakeTrack(1, 100), FakeTrack(2, 300), FakeTrack(3, 500)]
    lock.refresh_candidates(tracks)
    for idx in range(len(COLOR_NAMES)):
        overlay = OverlayConfig(color_idx=idx)
        assert render(lock, overlay, tracks).any()  # IDLE: gate + candidates
        lock.engage()
        for state in (LOCKED, COAST, LOST):
            lock.state = state
            assert render(lock, overlay, tracks).any()
        lock.stop()
        lock.refresh_candidates(tracks)


def test_scene_target_without_a_class_draws():
    lock = TargetLock((W, H), 66)
    assert lock.start_scene(np.zeros((H, W, 3), np.uint8))
    assert render(lock, OverlayConfig()).any()  # lock.cls is None in scene mode


def test_detection_off_and_both_languages_draw():
    lock = TargetLock((W, H), 66)
    for lang in ("tr", "en"):
        assert render(lock, OverlayConfig(lang=lang), stats={"ai": False}, lang=lang).any()


def test_overlay_off_leaves_the_frame_untouched():
    lock, tracks = TargetLock((W, H), 66), [FakeTrack(1, 300)]
    lock.refresh_candidates(tracks)
    lock.engage()
    img = np.full((H, W, 3), 77, np.uint8)
    before = img.copy()
    draw(img, tracks, lock, WALDO_NAMES, STATS, OverlayConfig(on=False))
    assert np.array_equal(img, before)


def test_gate_preview_draws_even_while_locked():
    lock = TargetLock((W, H), 66)
    assert lock.start_scene(np.zeros((H, W, 3), np.uint8))
    lock.scene.set_gate("L")  # freshly changed: previewing() is now True
    assert render(lock, OverlayConfig()).any()


def test_outline_pass_never_shifts_the_fill_out_of_registration():
    """Regression for the old thick+2-vs-thick outline: Hershey glyphs advance wider at a
    thicker stroke, so drawing the outline pass thicker than the fill pass drifted the two
    out of registration on long strings (e.g. the AZ/EL readout), printing as a ghosted
    double image instead of a clean outline."""
    from adts.render import _text

    img = np.full((60, 400, 3), 90, np.uint8)
    _text(img, "AZ +3.8  EL -3.4", (12, 40), 0.55, (0, 255, 0), 1)
    green = np.array([0, 255, 0])
    fill_cols = np.where((img == green).all(axis=2).any(axis=0))[0]
    # A drifted outline leaves stray green pixels beyond the last real glyph column (the
    # "ghost" tail of a mismatched second pass); a clean render's rightmost green pixel sits
    # at the actual last stroke, with no black outline pixels beyond it.
    black = np.array([0, 0, 0])
    black_cols = np.where((img == black).all(axis=2).any(axis=0))[0]
    assert black_cols.max() <= fill_cols.max() + 2  # outline hugs the fill, not past it
