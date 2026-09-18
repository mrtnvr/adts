"""One place where every operator command is executed, whatever carried it: MAVLink over
UART to the flight controller, MAVLink over UDP from a GCS, or the dev window's keys and
mouse. adts/mavlink_io.py only decodes the wire format into a (kind, arg) pair; what a
command MEANS lives here.

Every handler returns True/False and that bool becomes the COMMAND_ACK the GCS sees, so a
refused command (nothing to lock onto, out-of-range parameter) shows up at the other end
instead of silently doing nothing.
"""

from .classes import display_names
from .render import COLORS
from .scene_track import GATE_SIZES
from .target import AI, IDLE, SCENE

LANGS = ("tr", "en")


def resolve_flag(value, current):
    """p2 convention for on/off settings: 0 = off, 1 = on, -1 = toggle. None = bad value."""
    if value < 0:
        return not current
    return bool(value) if value in (0, 1) else None


def resolve_index(value, current, n):
    """p2 convention for multi-value settings: an index, or -1 for the next value round."""
    if value < 0:
        return (current + 1) % n
    return value if 0 <= value < n else None


class Controller:
    def __init__(self, lock, tracker, overlay, class_names, ai_on=True):
        self.lock = lock
        self.tracker = tracker
        self.overlay = overlay
        self.class_names = list(class_names)  # canonical English names, straight from the detector
        self.names = display_names(self.class_names, overlay.lang)
        self.ai_on = ai_on

    def run(self, kind, arg, tracks, frame=None):
        """`frame` must be the CLEAN frame (before draw()), because starting a scene track
        hands it to CSRT."""
        lock = self.lock
        if kind == "point":
            return lock.start_point(arg[0], arg[1], tracks)
        if kind == "rect":
            return lock.start_rect(arg, tracks)
        if kind == "auto":
            return lock.start_auto(tracks)
        if kind == "cand":
            return lock.select_step(arg)
        if kind == "select":
            return lock.select_id(arg)
        if kind == "engage":
            return lock.engage()
        if kind == "stop":
            return lock.stop()
        if kind == "ai":
            return self._set_ai(arg)
        if kind == "scene_start":
            return lock.start_scene(frame)
        if kind == "scene_stop":
            # Scene cancel only cancels a scene track; an AI lock is cancelled by "stop", so
            # a stray press on the wrong page can't drop the target the operator is on.
            return lock.stop() if lock.mode == SCENE and lock.state != IDLE else False
        if kind == "gate":
            idx = resolve_index(arg, GATE_SIZES.index(lock.scene.gate), len(GATE_SIZES))
            return idx is not None and lock.scene.set_gate(GATE_SIZES[idx])
        if kind == "overlay":
            return self._set_flag("on", arg)
        if kind == "reticle":
            return self._set_flag("reticle", arg)
        if kind == "lang":
            return self._set_lang(arg)
        if kind == "color":
            idx = resolve_index(arg, self.overlay.color_idx, len(COLORS))
            if idx is None:
                return False
            self.overlay.color_idx = idx
            return True
        return False

    def _set_flag(self, field, value):
        want = resolve_flag(value, getattr(self.overlay, field))
        if want is None:
            return False
        setattr(self.overlay, field, want)
        return True

    def _set_lang(self, value):
        idx = resolve_index(value, LANGS.index(self.overlay.lang), len(LANGS))
        if idx is None:
            return False
        self.overlay.lang = LANGS[idx]
        self.names = display_names(self.class_names, self.overlay.lang)
        return True

    def _set_ai(self, value):
        want = resolve_flag(value, self.ai_on)
        if want is None:
            return False
        if want != self.ai_on:
            self.ai_on = want
            if not want:
                # Detection is about to stop: drop an AI lock and the stale tracks behind it,
                # so nothing keeps drifting on screen. A scene track is unaffected - that is
                # the whole point of having it.
                if self.lock.mode == AI and self.lock.state != IDLE:
                    self.lock.stop()
                self.tracker.reset()
                self.lock.candidates, self.lock.sel_id = [], None
        return True
