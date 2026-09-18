"""Symbology overlay burned into the output video (HDMI / UDP / recording).

Every label is ASCII, drawn with plain cv2 (see WALDO_NAMES_TR in classes.py for why).
Colours are BGR.
"""

import time

import cv2

from .target import COAST, IDLE, LOCKED, LOST

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
DET_COLOR = (255, 180, 0)
DIM_COLOR = (130, 130, 130)
STATE_COLOR = {LOCKED: (0, 255, 0), COAST: (0, 220, 255), LOST: (0, 0, 255), IDLE: (255, 255, 255)}


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _crosshair(img):
    h, w = img.shape[:2]
    cx, cy, gap, arm = w // 2, h // 2, max(6, w // 160), max(20, w // 40)
    for (x1, y1, x2, y2) in ((cx - gap - arm, cy, cx - gap, cy), (cx + gap, cy, cx + gap + arm, cy),
                             (cx, cy - gap - arm, cx, cy - gap), (cx, cy + gap, cx, cy + gap + arm)):
        cv2.line(img, (x1, y1), (x2, y2), BLACK, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), WHITE, 1, cv2.LINE_AA)


def _brackets(img, box, color, thick=2):
    x1, y1, x2, y2 = (int(v) for v in box)
    lx, ly = max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * lx, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * ly), color, thick, cv2.LINE_AA)


def draw(img, tracks, lock, names, stats):
    """tracks: confirmed ByteTrack tracks. lock: TargetLock. names: display names by class index.
    stats: dict with fps, det_ms, temp_c, mav (bool/None), rec (bool)."""
    h, w = img.shape[:2]
    locked = lock.state != IDLE
    for t in tracks:
        if locked and t.track_id == lock.track_id:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        color = DIM_COLOR if locked else DET_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f"{t.track_id} {names[t.cls]}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.35, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 3, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3), FONT, 0.35, BLACK, 1, cv2.LINE_AA)

    _crosshair(img)

    color = STATE_COLOR[lock.state]
    if locked and lock.box is not None:
        _brackets(img, lock.box, color, 2 if lock.state == LOCKED else 1)
        x1, y1, x2, y2 = lock.box
        tcx, tcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.line(img, (w // 2, h // 2), (tcx, tcy), color, 1, cv2.LINE_AA)
        _text(img, f"TGT {lock.track_id} {names[lock.cls] if lock.cls is not None else ''}",
              (int(x1), max(14, int(y1) - 6)), 0.45, color)

    state_label = lock.state if lock.state == IDLE else f"{lock.state} {lock.track_id}"
    _text(img, state_label, (12, 28), 0.8, color, 2)
    _text(img, f"DET {len(tracks)}", (12, 54), 0.5)
    err = lock.angle_error() if locked else None
    if err:  # fixed spot top-left, so it never collides with target labels near the frame edge
        _text(img, f"AZ {err[0]:+5.1f}  EL {err[1]:+5.1f}", (12, 80), 0.55, color)

    right = [f"{stats.get('fps', 0):4.1f} FPS  {stats.get('det_ms', 0):4.0f} ms"]
    if stats.get("temp_c") is not None:
        right.append(f"CPU {stats['temp_c']:.0f}C")
    mav = stats.get("mav")
    if mav is not None:
        right.append("MAV OK" if mav else "MAV --")
    for i, s in enumerate(right):
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.5, 1)
        warn = s == "MAV --" or (s.startswith("CPU") and stats["temp_c"] >= 80)
        _text(img, s, (w - tw - 12, 24 + i * 22), 0.5, (0, 0, 255) if warn else WHITE)
    if stats.get("rec"):
        cv2.circle(img, (w - 20, h - 20), 7, (0, 0, 255), -1)
    _text(img, time.strftime("%H:%M:%S"), (12, h - 14), 0.45)
    return img
