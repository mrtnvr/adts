"""Symbology overlay burned into the output video (HDMI / UDP / recording).

Everything drawn here answers to the operator's overlay commands (see commands.py): the
whole overlay switches off for a clean image, the reticle switches off on its own, and the
label language and symbology colour change in flight.

Every label is ASCII, drawn with plain cv2 (see WALDO_NAMES_TR in classes.py for why).
Colours are BGR.
"""

import time
from dataclasses import dataclass

import cv2

from .classes import ui_text
from .target import AI, COAST, IDLE, LOCKED, LOST, SCENE

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
DIM_COLOR = (130, 130, 130)
# COAST and LOST keep fixed warning colours whatever the operator picked: "the lock is
# slipping" must not be something a colour preference can hide.
COAST_COLOR, LOST_COLOR = (0, 220, 255), (0, 0, 255)

COLOR_NAMES = ("green", "blue", "red", "white", "black")
COLORS = ((0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0))


@dataclass
class OverlayConfig:
    """Runtime overlay state. Every field is settable over MAVLink in flight."""
    on: bool = True
    reticle: bool = True
    lang: str = "tr"
    color_idx: int = 0

    @property
    def color(self):
        return COLORS[self.color_idx]


def _contrast(color):
    """Outline/fill partner for a colour, so black symbology stays readable on dark video."""
    return WHITE if color == BLACK else BLACK


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, _contrast(color), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _crosshair(img, color):
    h, w = img.shape[:2]
    cx, cy, gap, arm = w // 2, h // 2, max(6, w // 160), max(20, w // 40)
    for (x1, y1, x2, y2) in ((cx - gap - arm, cy, cx - gap, cy), (cx + gap, cy, cx + gap + arm, cy),
                             (cx, cy - gap - arm, cx, cy - gap), (cx, cy + gap, cx, cy + gap + arm)):
        cv2.line(img, (x1, y1), (x2, y2), _contrast(color), 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def _brackets(img, box, color, thick=2):
    x1, y1, x2, y2 = (int(v) for v in box)
    lx, ly = max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * lx, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * ly), color, thick, cv2.LINE_AA)


def _box_label(img, box, label, color, thick=1):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.35, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 3, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 3), FONT, 0.35, _contrast(color), 1, cv2.LINE_AA)


def draw(img, tracks, lock, names, stats, overlay):
    """tracks: confirmed ByteTrack tracks. lock: TargetLock. names: display names by class index.
    stats: dict with fps, det_ms, temp_c, mav (bool/None), rec (bool), ai (bool).
    overlay: OverlayConfig."""
    if not overlay.on:
        return img  # temiz goruntu: nothing burned in, recording included
    txt = ui_text(overlay.lang)
    h, w = img.shape[:2]
    base = overlay.color
    locked = lock.state != IDLE
    ai_locked = locked and lock.mode == AI
    cand_ids = [t.track_id for t in lock.candidates]

    for t in tracks:
        if ai_locked and t.track_id == lock.track_id:
            continue  # drawn below as the target
        cls_name = names[t.cls]
        if t.track_id in cand_ids:
            n = cand_ids.index(t.track_id) + 1
            chosen = t.track_id == lock.sel_id
            _box_label(img, t.xyxy, f"{n} {t.track_id} {cls_name}", base, 2 if chosen else 1)
        else:
            _box_label(img, t.xyxy, f"{t.track_id} {cls_name}", DIM_COLOR if locked else base)

    if overlay.reticle:
        _crosshair(img, base)

    if lock.state == IDLE:
        # Nothing engaged: show the scene-track gate so the operator can aim it.
        _brackets(img, lock.scene.gate_box(), base, 1)
        gx1, gy1, _, gy2 = lock.scene.gate_box()
        _text(img, f"{txt['gate']} {lock.scene.gate}", (int(gx1), int(gy2) + 18), 0.45, base)

    color = {LOCKED: base, COAST: COAST_COLOR, LOST: LOST_COLOR, IDLE: base}[lock.state]
    if locked and lock.box is not None:
        _brackets(img, lock.box, color, 2 if lock.state == LOCKED else 1)
        x1, y1, x2, y2 = lock.box
        tcx, tcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.line(img, (w // 2, h // 2), (tcx, tcy), color, 1, cv2.LINE_AA)
        what = txt["scene"] if lock.mode == SCENE else (names[lock.cls] if lock.cls is not None else "")
        _text(img, f"{txt['tgt']} {lock.track_id if lock.mode == AI else ''} {what}".replace("  ", " "),
              (int(x1), max(14, int(y1) - 6)), 0.45, color)

    state_label = txt[lock.state]
    if lock.state != IDLE:
        state_label += f" {lock.track_id}" if lock.mode == AI else f" {txt['scene']}"
    y = 28
    _text(img, state_label, (12, y), 0.8, color, 2)
    y += 26
    _text(img, txt["scene_mode"] if lock.mode == SCENE else txt["ai_mode"], (12, y), 0.5, base)
    y += 24
    if stats.get("ai", True):
        _text(img, f"{txt['det']} {len(tracks)}", (12, y), 0.5, base)
    else:
        _text(img, f"{txt['ai']} {txt['off']}", (12, y), 0.5, LOST_COLOR)
    y += 24
    sel = lock.selected
    if sel is not None and not (ai_locked and sel.track_id == lock.track_id):
        # Which target "takip baslat" would engage, and the ID that "select ID" expects.
        _text(img, f"{txt['sel']} {sel.track_id} {names[sel.cls]}", (12, y), 0.5, base)
        y += 24
    err = lock.angle_error() if locked else None
    if err:  # fixed left column, so it never collides with target labels near the frame edge
        _text(img, f"AZ {err[0]:+5.1f}  EL {err[1]:+5.1f}", (12, y), 0.55, color)

    right = [f"{stats.get('fps', 0):4.1f} FPS  {stats.get('det_ms', 0):4.0f} ms"]
    if stats.get("temp_c") is not None:
        right.append(f"CPU {stats['temp_c']:.0f}C")
    mav = stats.get("mav")
    if mav is not None:
        right.append("MAV OK" if mav else "MAV --")
    for i, s in enumerate(right):
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.5, 1)
        warn = s == "MAV --" or (s.startswith("CPU") and stats["temp_c"] >= 80)
        _text(img, s, (w - tw - 12, 24 + i * 22), 0.5, LOST_COLOR if warn else base)
    if stats.get("rec"):
        cv2.circle(img, (w - 20, h - 20), 7, LOST_COLOR, -1)
    _text(img, time.strftime("%H:%M:%S"), (12, h - 14), 0.45, base)
    return img
