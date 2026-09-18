#!/usr/bin/env python3
"""Run YOLO + ByteTrack/BotSORT multi-object tracking on a video/stream, live by default.

Defaults to --device gpu --weights waldo-l-p2 --show: this box is actually a
Jetson Orin NX (not the Pi 5 the project originally targeted), and WALDO30
is a fast, general-purpose overhead detector that's a good live
vehicle/infrastructure tracker (see WALDO_WEIGHTS comment below) - but it has
weak Person recall on close-range rescue scenes, so pass --weights fast (or
11l/11x/26x) for ARGUS's own, human-detection-tuned checkpoints; pass
--device cpu for the original NCNN Raspberry Pi 5 preview path.

Source can be a video file, an RTSP/HTTP stream URL, a webcam index ("0"), a directory
of image frames, or a directory of video files (mp4/mkv/avi/mov/webm/... - each is run
through the pipeline in turn, e.g. --source videos/ to batch-test a folder of clips).
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torchvision
from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.nms import TorchNMS

# This machine's torchvision build (0.18.0a0) has broken/mismatched compiled
# ops for its torch build (2.5.0a0), so torchvision.ops.nms raises at call
# time. Ultralytics ships its own pure-PyTorch NMS that "matches torchvision
# behavior exactly" as a fallback for environments without torchvision ops -
# use it directly instead of the broken compiled op.
torchvision.ops.nms = TorchNMS.nms

# PyTorch defaults to one CPU thread per core (8 here) for its own ops
# (preprocess/postprocess tensor work, NMS, etc). For this pipeline's many
# small, short-lived calls that's a net loss, not a win: thread pool
# spin-up/sync overhead dominates over the actual parallel work available.
# Benchmarked on a real run (GPU inference itself is unaffected - this only
# governs PyTorch's CPU-side ops): 8 threads -> 58ms/frame (17 FPS), 2 threads
# -> 39ms/frame (26 FPS). This must be set before any model is loaded.
torch.set_num_threads(2)

# torch.set_num_threads alone did NOT cap this process's real CPU usage - `ps`
# still showed ~550% CPU (5.5 cores) with it set, because OpenCV has its own,
# separate thread pool for decode/resize/plot/imshow that torch doesn't touch.
# Capping both together (measured with a real cv2 window, not headless):
# 66ms/frame (15 FPS) vs the ~100ms/frame (10-11 FPS) seen live with only
# torch capped - the two pools were both free to compete for all 8 cores.
cv2.setNumThreads(2)

ROOT = Path(__file__).parent

# NCNN (not raw .pt) is the default backend: ~4x faster than PyTorch at the
# same imgsz on this CPU (benchmarked: 1280px 4.73s/frame -> 1.23s/frame).
# RoblabWhGe only ships 3 ARGUS-tuned sizes (11l/11x/26x) - "fast" is the
# same 11l weights re-exported at a rectangular 288x480 (matches the 16:9
# source instead of padding a square to it), the fastest option found in
# testing (0.14s/frame) with no accuracy tradeoff versus square 480
# (0.90s/frame) - square 480 letterboxes ~44% of the tensor with dead
# padding since the source isn't square. int8 (480 square) was tried and
# rejected: ncnn2int8 only quantizes plain Convolution layers, leaving
# YOLO11's Attention/C2PSA/DFL blocks in fp32, so the resulting fp32<->int8
# requant/dequant overhead made it SLOWER (541-611ms) than fp16. NCNN
# exports are FIXED-SHAPE - requesting a mismatched imgsz hangs rather than
# erroring, so each entry carries the imgsz (int = square, tuple = (h, w))
# it was exported at and main() enforces it.
WEIGHTS = {
    "fast": {"path": ROOT / "weights" / "argus_yolo11l_480_ncnn_model", "imgsz": (288, 480)},  # default
    "11l": {"path": ROOT / "weights" / "argus_yolo11l_640_ncnn_model", "imgsz": 640},
    "11x": {"path": ROOT / "weights" / "argus_yolo11x_1280_ncnn_model", "imgsz": 1280},
    "26x": {"path": ROOT / "weights" / "argus_yolo26x_1280_ncnn_model", "imgsz": 1280},  # most accurate, heaviest
}

# GPU path (this box is actually a Jetson Orin NX, not a Pi 5). Benchmarked
# on a real frame: raw .pt 11l @1280 FP16 on the GPU (120ms/frame, 14
# detections) already beats CPU NCNN "fast" 288x480 (142ms/frame, 5
# detections) while running at full native resolution instead of a
# downscaled crop. Picks up the TensorRT .engine automatically once
# export_trt.py finishes building it; falls back to the plain .pt until
# then. No GPU equivalent of "fast" - that's an NCNN-only imgsz-shrink trick
# the GPU doesn't need.
GPU_WEIGHTS = {
    "11l": ROOT / "weights" / "argus_yolo11l_1280",
    "11x": ROOT / "weights" / "argus_yolo11x_1280",
    "26x": ROOT / "weights" / "argus_yolo26x_1280",
}
GPU_IMGSZ = 1280


def resolve_gpu_weights(weights_key):
    if weights_key == "fast":
        weights_key = "11l"
    base = GPU_WEIGHTS[weights_key]
    engine, pt = base.with_suffix(".engine"), base.with_suffix(".pt")
    return (engine if engine.exists() else pt), weights_key


# WALDO30 (github.com/stephansturges/WALDO) - general-purpose overhead
# detector, NOT ARGUS-tuned: ~2x faster than ARGUS 11l (171-224ms vs 387ms
# for waldo-n/-n-p2) with fast, confident LightVehicle/Truck/Bus detection,
# but weak Person recall on close-range rescue scenes (0 detections at
# conf=0.25 in testing; best candidate was only 0.18 even at conf=0.05) -
# WALDO was trained on general overhead imagery, not fine-tuned on
# close-range rescue scenes the way ARGUS's own human class was. Use for
# vehicle/infrastructure awareness, not as a human-detection replacement for
# ARGUS. GPU-only - only .pt checkpoints were pulled, no NCNN export made.
WALDO_WEIGHTS = {
    "waldo-n": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n_640x640.pt", "imgsz": 640},
    "waldo-n-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n-p2_640x640.pt", "imgsz": 640},
    "waldo-l-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l-p2_1024x1024.pt", "imgsz": 1024},
    "waldo-m": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8m_640x640.pt", "imgsz": 640},
    "waldo-l": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l_640x640.pt", "imgsz": 640},  # no P2 head, unlike waldo-l-p2
}

# Display-name translations for WALDO's 12 classes, keyed by --lang code. "en" is WALDO's own
# English names (empty = passthrough, see the WALDO_NAMES.get(name, name) fallback where this
# is used). Person/LightVehicle/Bike/Digger have no exact one-word equivalent in several of
# these languages (they cover a broader real-world category) - each is a judgment call, not a
# literal dictionary translation; the rest are unambiguous direct translations.
#
# fr/de/es/pt/tr are ASCII-transliterated on purpose (e.g. Turkish ç->c, ı->i, İ->I, ş->s, ğ->g,
# ü->u, ö->o; German follows its own standard ae/oe/ue/ss fallback instead of just dropping the
# umlaut): any non-ASCII character in a label forces ultralytics' Annotator (--no-show path) or
# this file's own draw_boxes (--show path, see its `font` argument) onto the PIL/Unicode-font
# text path, measured at ~16ms/frame here vs ~2ms for the plain cv2.putText path used when every
# label is pure ASCII - a bigger cost than the tracker itself. ru/uk/ar/zh have no reasonable
# ASCII form at all (Cyrillic/Arabic/Chinese script) and always pay that cost when selected.
WALDO_NAMES = {
    "en": {},
    "tr": {
        "LightVehicle": "Otomobil",
        "Person": "Insan",
        "Building": "Bina",
        "UPole": "Elektrik Diregi",
        "Boat": "Tekne",
        "Bike": "Bisiklet/Motosiklet",
        "Container": "Konteyner",
        "Truck": "Kamyon",
        "Gastank": "Gaz Tanki",
        "Digger": "Is Makinesi",
        "SolarPanels": "Gunes Paneli",
        "Bus": "Otobus",
    },
    "fr": {
        "LightVehicle": "Vehicule leger",
        "Person": "Personne",
        "Building": "Batiment",
        "UPole": "Poteau electrique",
        "Boat": "Bateau",
        "Bike": "Velo/Moto",
        "Container": "Conteneur",
        "Truck": "Camion",
        "Gastank": "Reservoir de gaz",
        "Digger": "Excavatrice",
        "SolarPanels": "Panneaux solaires",
        "Bus": "Autobus",
    },
    "de": {
        "LightVehicle": "Leichtfahrzeug",
        "Person": "Person",
        "Building": "Gebaeude",
        "UPole": "Strommast",
        "Boat": "Boot",
        "Bike": "Fahrrad/Motorrad",
        "Container": "Container",
        "Truck": "LKW",
        "Gastank": "Gastank",
        "Digger": "Bagger",
        "SolarPanels": "Solarmodule",
        "Bus": "Bus",
    },
    "es": {
        "LightVehicle": "Vehiculo ligero",
        "Person": "Persona",
        "Building": "Edificio",
        "UPole": "Poste electrico",
        "Boat": "Barco",
        "Bike": "Bici/Moto",
        "Container": "Contenedor",
        "Truck": "Camion",
        "Gastank": "Tanque de gas",
        "Digger": "Excavadora",
        "SolarPanels": "Paneles solares",
        "Bus": "Autobus",
    },
    "pt": {
        "LightVehicle": "Veiculo leve",
        "Person": "Pessoa",
        "Building": "Edificio",
        "UPole": "Poste eletrico",
        "Boat": "Barco",
        "Bike": "Bicicleta/Moto",
        "Container": "Contentor",
        "Truck": "Caminhao",
        "Gastank": "Tanque de gas",
        "Digger": "Escavadora",
        "SolarPanels": "Paineis solares",
        "Bus": "Onibus",
    },
    "ru": {
        "LightVehicle": "Автомобиль",
        "Person": "Человек",
        "Building": "Здание",
        "UPole": "Электростолб",
        "Boat": "Лодка",
        "Bike": "Велосипед/Мотоцикл",
        "Container": "Контейнер",
        "Truck": "Грузовик",
        "Gastank": "Газовый бак",
        "Digger": "Экскаватор",
        "SolarPanels": "Солнечные панели",
        "Bus": "Автобус",
    },
    "uk": {
        "LightVehicle": "Автомобіль",
        "Person": "Людина",
        "Building": "Будівля",
        "UPole": "Електростовп",
        "Boat": "Човен",
        "Bike": "Велосипед/Мотоцикл",
        "Container": "Контейнер",
        "Truck": "Вантажівка",
        "Gastank": "Газовий бак",
        "Digger": "Екскаватор",
        "SolarPanels": "Сонячні панелі",
        "Bus": "Автобус",
    },
    "ar": {
        "LightVehicle": "سيارة",
        "Person": "شخص",
        "Building": "مبنى",
        "UPole": "عمود كهرباء",
        "Boat": "قارب",
        "Bike": "دراجة",
        "Container": "حاوية",
        "Truck": "شاحنة",
        "Gastank": "خزان غاز",
        "Digger": "حفارة",
        "SolarPanels": "ألواح شمسية",
        "Bus": "حافلة",
    },
    "zh": {
        "LightVehicle": "轻型车辆",
        "Person": "人",
        "Building": "建筑物",
        "UPole": "电线杆",
        "Boat": "船",
        "Bike": "自行车/摩托车",
        "Container": "集装箱",
        "Truck": "卡车",
        "Gastank": "燃气罐",
        "Digger": "挖掘机",
        "SolarPanels": "太阳能板",
        "Bus": "公交车",
    },
}


BOX_COLOR = (255, 180, 0)  # BGR, detect-mode box color
DIM_COLOR = (110, 110, 110)  # non-selected boxes once a target is picked (click-to-highlight)
TARGET_COLOR = (0, 255, 0)  # selected track
ROI_COLOR = (0, 200, 255)


class TrackSelection:
    """Click-to-highlight state: an earlier version of this project used a classical CV
    tracker (OpenCV CSRT) that swapped in per-target once clicked, turning detection off
    entirely. YOLO+ByteTrack keep running on every object regardless of selection - this
    only changes which track_id gets drawn prominently. ByteTrack's own track_id already
    gives every object a stable identity across frames, so "locking onto" one is just
    filtering the draw, not swapping in a separate CV tracker."""

    def __init__(self):
        self.selected_id = None
        self.boxes = []  # this frame's boxes in FULL-FRAME coords, kept current by the main loop for on_mouse to hit-test

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, track_id, cls_name, conf in self.boxes:
                # track_id is None in --no-track mode (no stable identity to select) -
                # skip it, since a "selected" None would be indistinguishable from the
                # cleared (no selection) state below.
                if track_id is not None and x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = track_id
                    break
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_id = None


def draw_roi_rect(frame, roi_box):
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)
    cv2.putText(frame, "ROI", (x1 + 4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ROI_COLOR, 1, cv2.LINE_AA)


def boxes_from_results(r):
    """Normalize a real detection Results object into plain (x1,y1,x2,y2,track_id,cls_name,conf) tuples."""
    out = []
    for box in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        cls_name = r.names[int(box.cls[0])]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        out.append((x1, y1, x2, y2, track_id, cls_name, conf))
    return out


def boxes_from_tracks(stracks, names):
    """Same shape as boxes_from_results, but read off the tracker's own Kalman-predicted
    state (STrack.xyxy) after multi_predict() - i.e. no new detection this frame at all."""
    out = []
    for st in stracks:
        x1, y1, x2, y2 = (int(v) for v in st.xyxy)
        out.append((x1, y1, x2, y2, st.track_id, names[int(st.cls)], float(st.score)))
    return out


def resolve_video_sources(source):
    """A single source (file, webcam index, stream URL, image dir) is returned unchanged -
    ultralytics' own loader (--no-show path) or cv2.VideoCapture (--show path) handle those
    directly. A directory is instead expanded here to every video file it directly contains
    (sorted, non-recursive), so --source can point at a folder of test clips and --show will
    play them one after another instead of failing to open the directory as a single video."""
    p = Path(source)
    if not p.is_dir():
        return [source]
    videos = sorted(str(f) for f in p.iterdir() if f.suffix.lower().lstrip(".") in VID_FORMATS)
    if not videos:
        raise SystemExit(f"No video files ({', '.join(sorted(VID_FORMATS))}) found in {p}")
    return videos


def merge_lost_tracks(boxes_list, tracker, names):
    # ByteTrack's track_buffer (30 frames by default) keeps a track that
    # missed a match alive in tracker.lost_stracks rather than deleting it
    # immediately - specifically to survive a brief, spurious miss. But
    # ultralytics' own r.boxes only ever returns CURRENTLY matched tracks, so
    # a track sitting in that not-yet-expired window still visually blinks
    # off for those frames even though ByteTrack itself hasn't given up on
    # it. Show it anyway, at its last Kalman position - on real detection
    # frames too, not just skipped ones, since this flicker isn't a
    # --detect-stride artifact, it's WALDO's own per-frame confidence noise
    # occasionally dropping a track for a single cycle (confirmed empirically:
    # a track can vanish between two consecutive REAL detections).
    existing_ids = {b[4] for b in boxes_list}
    out = list(boxes_list)
    for st in tracker.lost_stracks:
        if st.track_id not in existing_ids:
            x1, y1, x2, y2 = (int(v) for v in st.xyxy)
            out.append((x1, y1, x2, y2, st.track_id, names[int(st.cls)], float(st.score)))
    return out


def smooth_boxes(boxes_list, smoothed_state, alpha):
    # Even a real (matched) detection frame has visible left-right jitter -
    # ByteTrack's own Kalman update trusts each new measurement fairly
    # quickly, which is good for association robustness but not for a
    # visually still box on a near-stationary object. This is a second,
    # simple EMA on top, keyed by track_id (which ByteTrack never reuses, so
    # no stale-state risk): alpha=1 is unsmoothed/raw, lower alpha = smoother
    # but laggier. Untracked boxes (track_id is None, i.e. --no-track) pass
    # through unsmoothed - there's no stable key to smooth them against.
    if alpha >= 1.0:
        return boxes_list
    out = []
    for x1, y1, x2, y2, track_id, cls_name, conf in boxes_list:
        if track_id is not None:
            prev = smoothed_state.get(track_id)
            if prev is not None:
                x1 = alpha * x1 + (1 - alpha) * prev[0]
                y1 = alpha * y1 + (1 - alpha) * prev[1]
                x2 = alpha * x2 + (1 - alpha) * prev[2]
                y2 = alpha * y2 + (1 - alpha) * prev[3]
            smoothed_state[track_id] = (x1, y1, x2, y2)
        out.append((int(x1), int(y1), int(x2), int(y2), track_id, cls_name, conf))
    return out


def get_unicode_font(size=13):
    """PIL font with broad script coverage (Latin/Cyrillic/CJK; Arabic still needs shape_text()
    first, see there) for a --lang whose WALDO_NAMES aren't ASCII (ru/uk/ar/zh) - cv2.putText's
    built-in Hershey fonts have no non-ASCII glyphs at all, unlike the Latin-with-diacritics
    languages (fr/de/es/pt/tr), which are ASCII-transliterated in WALDO_NAMES instead so they
    can stay on the fast cv2 text path. Cached to disk by ultralytics' own check_font() (which
    this also reuses to avoid bundling a font file) after the first download."""
    from PIL import ImageFont
    from ultralytics.utils.checks import check_font

    return ImageFont.truetype(str(check_font("Arial.Unicode.ttf")), size)


_arabic_shaper = None  # None = not checked yet, False = unavailable, else the (reshape, get_display) pair


def shape_text(text, lang):
    """Arabic letters join to their neighbors (unlike Latin/Cyrillic/CJK) - drawing the raw
    codepoints gives disconnected, isolated-form letters instead of the connected script a
    reader expects. arabic_reshaper + python-bidi fix that (join letters, then reverse to
    visual/display order, since PIL's ImageDraw does no bidi of its own); fall back to the raw
    string, with one warning, if they're not installed - nothing else here needs them.

    The import is attempted only once (cached in _arabic_shaper, not retried per call): a
    failed `import` isn't cached in sys.modules the way a successful one is, so retrying it
    every frame re-walks sys.path every time - measured at ~1.3ms/frame extra here, on top of
    the PIL text path's own cost, for nothing (the answer never changes at runtime)."""
    if lang != "ar":
        return text
    global _arabic_shaper
    if _arabic_shaper is None:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display

            _arabic_shaper = (arabic_reshaper.reshape, get_display)
        except ImportError:
            print("--lang ar: arabic_reshaper/python-bidi not installed - Arabic labels will render as "
                  "disconnected letters. pip install arabic-reshaper python-bidi for correct shaping.")
            _arabic_shaper = False
    if _arabic_shaper is False:
        return text
    reshape, get_display = _arabic_shaper
    return get_display(reshape(text))


_label_tile_cache = {}


def get_label_tile(font, lang, text):
    """Pre-renders `text` (a WALDO class name - at most 12 distinct strings per language, see
    WALDO_NAMES) once onto a small transparent RGBA tile and caches it by (lang, text), reused
    every later frame via Image.paste(tile, pos, tile) (the tile as its own alpha mask) instead
    of re-rasterizing through FreeType each time draw_boxes needs it.

    This matters because PIL's ImageDraw.text() turned out to have a large, mostly CONTENT-
    independent per-call cost here (~0.2-0.35ms regardless of whether the string is "id:3 " or
    a full Chinese class name - confirmed by benchmarking font sizes 10-28px, which barely
    moved it either) - so caching only pays off for the part of a label that's actually
    repeated across calls. track_id and confidence change on nearly every call and aren't
    cached; the class name is the same string every time a given class is drawn, and pasting a
    cached tile measured ~40x faster than re-rendering it (~0.01ms vs ~0.35ms)."""
    key = (lang, text)
    tile = _label_tile_cache.get(key)
    if tile is None:
        from PIL import Image, ImageDraw

        shaped = shape_text(text, lang)
        tw, th = font.getbbox(shaped)[2:4]
        tile = Image.new("RGBA", (max(tw, 1) + 2, max(th, 1) + 2), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((0, 0), shaped, font=font, fill=(0, 0, 0, 255))
        _label_tile_cache[key] = tile
    return tile


def draw_text_pil(frame, items, font):
    """Draws each item (see draw_boxes) onto `frame` in place, through a small PIL crop around
    just that label's own few hundred pixels - not a numpy<->PIL round trip on the WHOLE frame
    per label (or even once for all labels): that cost is dominated by the conversion itself,
    which scales with total frame size regardless of how much of the frame is actually text -
    typically >99% of it, for a handful of small labels. Measured on a 1024x576 frame, 5
    labels: ~8.4ms/frame whole-frame vs ~0.6ms/frame cropped.

    The id/confidence parts are always ASCII and rendered fresh (see get_label_tile for why
    that's not much cheaper by itself); the class name is a cached tile, just pasted in place.

    Also skips the BGR<->RGB channel swap PIL normally wants: nothing here reorders the array's
    underlying byte order (`Image.fromarray(crop, "RGB")` just lies about what the 3 channels
    are called), and text is always drawn in plain black (see draw_boxes/get_label_tile), which
    is identical either way - fewer full-size conversions."""
    import numpy as np
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    for x, y, tw, th, prefix, prefix_w, tile, suffix in items:
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        # A little padding: PIL can draw a few px outside its own reported bbox (descenders,
        # antialiasing) - better to waste a few pixels of crop than clip them.
        x1, y1 = min(x + tw + 2, w), min(y + th + 4, h)
        if x1 <= x0 or y1 <= y0:
            continue
        # .copy(): a frame slice is a strided VIEW, not contiguous unless it spans full rows -
        # PIL's fromarray needs a contiguous buffer, and this copy is the crop-sized one we want.
        crop = Image.fromarray(frame[y0:y1, x0:x1].copy(), mode="RGB")
        draw = ImageDraw.Draw(crop)
        cur_x = x - x0
        ty = y - y0
        if prefix:
            draw.text((cur_x, ty), prefix, font=font, fill=(0, 0, 0))
            cur_x += prefix_w
        crop.paste(tile, (cur_x, ty), tile)
        cur_x += tile.size[0]
        if suffix:
            draw.text((cur_x, ty), suffix, font=font, fill=(0, 0, 0))
        frame[y0:y1, x0:x1] = np.asarray(crop)


def draw_boxes(frame, boxes_list, offset=(0, 0), selected_id=None, font=None, lang="en"):
    # Manual drawing (not r.plot()) so the exact same code path handles both a
    # real Results object's boxes (via boxes_from_results) and the tracker's
    # own predicted-only state on a skipped detection frame (via
    # boxes_from_tracks) - r.plot() only ever works on a genuine Results tied
    # to an actual inference call, which skipped frames don't have. Plain
    # cv2.putText when font is None (labels are ASCII - see WALDO_NAMES) is
    # cheaper than r.plot()'s PIL path too. offset shifts from a --roi crop's
    # origin back to full-frame coordinates; (0, 0) when --roi is off.
    #
    # selected_id: click-to-highlight target (see TrackSelection) - when set,
    # every other box is dimmed instead of hidden, so the rest of the scene
    # (and the tracker/detector still running on it) stays visible.
    #
    # font: a PIL ImageFont (see get_unicode_font), for a --lang whose names aren't ASCII. Box
    # outlines and label backgrounds are still drawn with cv2 either way; only the label TEXT
    # is deferred to one draw_text_pil() call for the whole frame at the end (using a cached
    # tile for the class-name part, see get_label_tile), instead of a numpy<->PIL round trip
    # and a fresh render of the whole string per box.
    ox, oy = offset
    pending_labels = []
    for x1, y1, x2, y2, track_id, cls_name, conf in boxes_list:
        x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
        is_target = selected_id is not None and track_id == selected_id
        color = TARGET_COLOR if is_target else (BOX_COLOR if selected_id is None else DIM_COLOR)
        thickness = 2 if is_target else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if font is None:
            label = f"id:{track_id} {cls_name} {conf:.2f}" if track_id is not None else f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
            cv2.rectangle(frame, (x1, y1 - th - 3), (x1 + tw + 2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            prefix = f"id:{track_id} " if track_id is not None else ""
            suffix = f" {conf:.2f}"
            tile = get_label_tile(font, lang, cls_name)
            prefix_w, prefix_h = font.getbbox(prefix)[2:4] if prefix else (0, 0)
            suffix_w, suffix_h = font.getbbox(suffix)[2:4]
            tile_w, tile_h = tile.size
            tw = prefix_w + tile_w + suffix_w
            th = max(prefix_h, tile_h, suffix_h)
            cv2.rectangle(frame, (x1, y1 - th - 3), (x1 + tw + 2, y1), color, -1)
            pending_labels.append((x1 + 1, y1 - th - 3, tw, th, prefix, prefix_w, tile, suffix))
    if font is not None and pending_labels:
        draw_text_pil(frame, pending_labels, font)


def draw_stats(frame, model_name, dt, fps):
    # Top-right, outlined text (black stroke + colored fill) for readability over any
    # frame content. dt is the full per-frame elapsed time (inference + tracker +
    # plot/display), in ms, not just the model's own inference time - "the image
    # actually reaching the screen took this long", matching what fps is computed from.
    lines = [model_name, f"{dt * 1000:.0f} ms | {fps:.1f} FPS"]
    for i, text in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = frame.shape[1] - tw - 10
        y = 20 + i * (th + 10)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def limit_ncnn_threads(model, imgsz, n_threads):
    """Cap NCNN's thread pool so this dev box previews Pi 5-realistic speed (4 cores, not this box's 8).

    Must be set BEFORE the network is loaded, not after: NCNN packs some conv
    paths (gemm/winograd) for whatever thread count was active at load time,
    and setting net.opt.num_threads afterwards leaves those stuck - confirmed
    by NCNN's own warning ("will use load-time value 8") when tried that way.
    ultralytics has no hook for this, so we monkeypatch ncnn.Net.__init__ to
    apply it right at construction, then trigger the (otherwise lazy) load
    with a dummy predict while the patch is active.
    """
    import numpy as np
    import ncnn as pyncnn

    orig_init = pyncnn.Net.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.opt.num_threads = n_threads

    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    pyncnn.Net.__init__ = patched_init
    try:
        model.predict(np.zeros((h, w, 3), dtype="uint8"), imgsz=imgsz, device="cpu", verbose=False)
    finally:
        pyncnn.Net.__init__ = orig_init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        choices=(*WEIGHTS.keys(), *WALDO_WEIGHTS.keys()),
        default="waldo-l-p2",
        help="which checkpoint to use (default: waldo-l-p2 - see WALDO_WEIGHTS comment above), or an ARGUS checkpoint "
             "(fast/11l/11x/26x - ARGUS-tuned, better human recall on rescue scenes, see WEIGHTS/GPU_WEIGHTS above).",
    )
    parser.add_argument(
        "--lang",
        choices=sorted(WALDO_NAMES.keys()),
        default="tr",
        help="label language for WALDO's class names (see WALDO_NAMES above); no effect on ARGUS weights, which keep "
             "their own English class names (human/vehicle). en/fr/de/es/pt/tr stay ASCII-transliterated for the fast "
             "cv2 text path; ru/uk/ar/zh use real Cyrillic/Arabic/Chinese script, which forces the slower PIL "
             "text-rendering path (cv2.putText's built-in fonts are ASCII-only) - expect a bigger FPS hit from those "
             "than from the language choice itself.",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "assets" / "mosaic_val.jpg"),
        help="video file, stream URL, webcam index, or image/dir to track on. A directory of "
             "VIDEOS (mp4/mkv/avi/mov/webm/...) runs each one in turn, e.g. --source videos/ "
             "to batch-test a folder of clips",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="must match the chosen weights' export size; omit to use it automatically")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--exclude-classes", default="UPole",
                         help="comma-separated class names to drop before tracking (matched against the loaded model's own names, so this "
                              "is a no-op for ARGUS weights, which don't have a UPole class); empty string = track every class")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True,
                         help="default: on (ByteTrack/BotSORT IDs, via model.track()). --no-track calls plain model.predict() instead, "
                              "skipping the tracker entirely - useful to check whether it's actually costing FPS. In practice it usually "
                              "isn't: ByteTrack's Kalman+IoU matching is sub-millisecond, versus the 150-450ms YOLO forward pass itself "
                              "(see WALDO_WEIGHTS/exclude-classes comments) - if FPS barely moves with --no-track, the bottleneck is the "
                              "model/imgsz, not the tracker, and waldo-n(-p2) or a smaller imgsz is the real lever.")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="ultralytics tracker config (bytetrack.yaml or botsort.yaml)",
    )
    parser.add_argument("--pi-cores", type=int, default=4,
                         help="cap NCNN threads to this many, to preview Raspberry Pi 5 (4 cores) speed instead of this dev machine's full core count; 0 = use all cores")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu",
                         help="gpu (default) = run on this box's Jetson Orin GPU (uses the TensorRT .engine if exported, else the plain .pt). "
                              "cpu = NCNN Pi-5-preview path; ignores --weights=fast and --pi-cores when gpu")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                         help="open a live window (default: on - pass --no-show to instead save every frame to outputs/tracks/, "
                              "impractical for a long video). Draws ByteTrack/BotSORT IDs, which persist a track through a few missed "
                              "detections instead of the raw per-frame flicker you get from detect-only mode at --detect-stride 1")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="cap display playback rate to this many FPS when --show is set, so a recorded video plays at realistic speed "
                              "instead of fast-forwarding (default: 30, matches this project's source videos); 0 = uncapped")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                         help="with --show, also write the exact annotated frames being displayed (boxes, IDs, model name/ms/FPS overlay) "
                              "to an mp4 in outputs/tracks/ (one file per input video, named after it); no effect with --no-show, which "
                              "already saves every frame as a jpg")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False,
                         help="print a per-stage ms breakdown every frame (decode+other/preprocess/inference/postprocess from "
                              "ultralytics' own r.speed, plus plot/copy/draw/imshow measured here) to find which stage is actually slow, "
                              "instead of guessing from the single combined ms/FPS number on screen")
    parser.add_argument("--roi", type=float, default=0.0,
                         help="crop each frame to this fraction of width/height, centered, before detecting/tracking/displaying (e.g. "
                              "0.5 = center half); 0 = off (full frame, default), otherwise must be in (0, 1]. Fewer source pixels to "
                              "decode/resize is a genuine speed win here (unlike the NCNN weights above, which are FIXED-SHAPE exports that "
                              "resize the crop back up to the same imgsz regardless) - the same real-world object also fills more of the "
                              "model's input after cropping, which helps small-object recall")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True,
                         help="FP16 inference on GPU (default: on; no effect with --device cpu, PyTorch CPU FP16 isn't a speed win). "
                              "Benchmarked on waldo-m: 56.9ms FP32 -> 40.4ms FP16, ~29%% faster, for free (no accuracy tradeoff noted)")
    parser.add_argument("--detect-stride", type=int, default=1,
                         help="run YOLO once every this many frames (default: 1, every frame; requires --track). On skipped frames, "
                              "no inference runs at all - instead ByteTrack/BotSORT's own Kalman filter is stepped directly "
                              "(tracker.multi_predict()) and its predicted positions are drawn, same physics-based extrapolation the "
                              "tracker already does internally for a frame where a track's detection went briefly unmatched - this "
                              "just calls that step explicitly instead of running a real (and skipped) detection first.")
    parser.add_argument("--smooth", type=float, default=1.0,
                         help="EMA smoothing factor per track ID for the drawn box position, on top of ByteTrack/BotSORT's own Kalman "
                              "update (which still lets real per-frame detection noise show as visible left-right jitter on a near-"
                              "still object) - default 1.0 = raw/unsmoothed; lower (e.g. 0.4) = smoother but laggier")
    args = parser.parse_args()

    if args.detect_stride > 1 and not args.track:
        raise SystemExit("--detect-stride > 1 needs a tracker to fill the skipped frames; pass --track (the default) or drop --detect-stride.")

    if not 0.0 <= args.roi <= 1.0:
        # >1.0 would make rx/ry negative below, and raw_frame[ry:ry+rh, rx:rx+rw] with a
        # negative start silently wraps around (numpy slicing), cropping the wrong region
        # instead of erroring.
        raise SystemExit(f"--roi must be between 0 and 1.0 (0 = off, full frame); got {args.roi}.")

    if args.weights in WALDO_WEIGHTS:
        if args.device != "gpu":
            raise SystemExit(f"--weights {args.weights} is GPU-only (no NCNN export was made for WALDO); pass --device gpu.")
        choice = WALDO_WEIGHTS[args.weights]
        if args.imgsz is not None and args.imgsz != choice["imgsz"]:
            raise SystemExit(f"--weights {args.weights} is native imgsz={choice['imgsz']}; pass that or omit --imgsz.")
        args.imgsz = choice["imgsz"]
        weights_path = choice["path"]
        print(f"--device gpu: using {weights_path.name} (WALDO, non-ARGUS-tuned - see WALDO_WEIGHTS comment)")
    elif args.device == "gpu":
        weights_path, resolved_key = resolve_gpu_weights(args.weights)
        if resolved_key != args.weights:
            print(f"--device gpu: '{args.weights}' has no GPU checkpoint (NCNN-only trick); using '{resolved_key}' @1280 instead.")
        if args.imgsz is not None and args.imgsz != GPU_IMGSZ:
            raise SystemExit(f"--device gpu checkpoints are all native imgsz={GPU_IMGSZ}; pass --imgsz {GPU_IMGSZ} or omit --imgsz.")
        args.imgsz = GPU_IMGSZ
        print(f"--device gpu: using {weights_path.name}")
    else:
        choice = WEIGHTS[args.weights]
        if args.imgsz is None:
            args.imgsz = choice["imgsz"]
        elif args.imgsz != choice["imgsz"]:
            raise SystemExit(
                f"--weights {args.weights} is an NCNN export fixed at imgsz={choice['imgsz']}; "
                f"a mismatched --imgsz {args.imgsz} won't error, it will just hang. "
                f"Pass --imgsz {choice['imgsz']} or omit --imgsz."
            )
        weights_path = choice["path"]

    model = YOLO(str(weights_path))
    if args.device == "cpu" and args.pi_cores > 0:
        limit_ncnn_threads(model, args.imgsz, args.pi_cores)

    # Drop excluded classes (default: UPole - not a search-and-rescue target)
    # BEFORE tracking, not just at display time: fewer surviving boxes means
    # less NMS/tracker (Kalman + IoU matching) work per frame. This does NOT
    # touch the YOLO backbone forward pass itself (still the dominant cost,
    # 150-450ms here on WALDO l-p2/Orin NX) - it only trims the cheap
    # postprocessing tail, so expect a small, not dramatic, FPS gain.
    exclude = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    class_ids = [i for i, name in model.names.items() if name not in exclude] if exclude else None

    # Translate WALDO's English class names to --lang for display - AFTER
    # exclude_classes above, which matches against the model's original
    # (English) names, so --exclude-classes UPole keeps working regardless of --lang.
    label_font = None
    if args.weights in WALDO_WEIGHTS:
        lang_table = WALDO_NAMES[args.lang]
        # model.names has no setter (YOLO wraps a torch nn.Module) - the
        # mutable dict actually lives on the underlying DetectionModel.
        model.model.names = {i: lang_table.get(name, name) for i, name in model.names.items()}
        if not all(name.isascii() for name in lang_table.values()):
            label_font = get_unicode_font()
    elif args.lang != "tr":
        print(f"--lang {args.lang}: ignored, only applies to WALDO weights (ARGUS checkpoints keep their own English class names).")

    if not args.show:
        half = args.half and args.device == "gpu"
        results = (
            model.track(
                source=args.source, imgsz=args.imgsz, conf=args.conf, device=0 if args.device == "gpu" else "cpu",
                tracker=args.tracker, classes=class_ids, half=half, persist=True, stream=True,
            ) if args.track else
            model.predict(
                source=args.source, imgsz=args.imgsz, conf=args.conf, device=0 if args.device == "gpu" else "cpu",
                classes=class_ids, half=half, stream=True,
            )
        )

    if args.show:
        # Manual cv2.VideoCapture instead of handing `source` straight to
        # ultralytics (as the --no-show path above still does): --roi needs
        # to crop each raw frame before it ever reaches the model, which
        # isn't something model.track(source=...)'s internal loader exposes.
        # This also gives 'f' (skip 60 frames) a capture object of our own to
        # seek, instead of reaching into model.predictor.dataset.cap. A
        # directory --source is expanded to its video files here and each is
        # played in turn, reusing the same model/window across the batch.
        sources = resolve_video_sources(args.source)
        batch = len(sources) > 1

        mode_label = args.tracker if args.track else "no-track"
        controls = "click=highlight, right-click=clear, f=skip 60 frames"
        if batch:
            controls += ", n=next video"
        controls += ", q=quit"
        window_name = f"ARGUS-YOLO track ({args.weights}, {mode_label}) - {controls}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        selection = TrackSelection()
        cv2.setMouseCallback(window_name, selection.on_mouse)
        predict_kwargs = dict(imgsz=args.imgsz, conf=args.conf, device=0 if args.device == "gpu" else "cpu",
                               classes=class_ids, half=(args.half and args.device == "gpu"), verbose=False)
        record_dir = ROOT / "outputs" / "tracks"

        quit_all = False
        for video_idx, video_source in enumerate(sources):
            cap_source = int(video_source) if video_source.isdigit() else video_source
            cap = cv2.VideoCapture(cap_source)
            if not cap.isOpened():
                if batch:
                    print(f"Could not open {video_source} - skipping.")
                    continue
                raise SystemExit(f"Could not open source: {video_source}")

            if batch:
                print(f"[{video_idx + 1}/{len(sources)}] {video_source}")
            if args.track and model.predictor is not None:
                # Fresh tracker per video: with persist=True below, ByteTrack/BotSORT
                # would otherwise carry track IDs and Kalman state over from the
                # previous (unrelated) clip and briefly "coast" into this one.
                model.predictor.trackers[0].reset()

            ema_fps = 0.0
            frame_count = 0  # frame 0 always detects, so model.predictor exists before any --detect-stride skip needs it
            smoothed_state = {}  # track_id -> last EMA-smoothed (x1,y1,x2,y2), for smooth_boxes()
            selection.selected_id = None
            writer = None
            if args.record:
                record_dir.mkdir(parents=True, exist_ok=True)
                record_path = record_dir / f"{Path(video_source).stem}_{args.weights}.mp4"
                # writer is opened lazily below, once the first annotated frame's
                # actual pixel size is known, rather than guessed up front

            while True:
                t_iter0 = time.time()

                t0 = time.time()
                ok, raw_frame = cap.read()
                decode_ms = (time.time() - t0) * 1000
                if not ok:
                    print(f"End of video: {video_source}")
                    break

                detect_input = raw_frame
                roi_box = None
                if args.roi > 0:
                    # Crop BEFORE detection, not after: fewer source pixels means a
                    # genuinely cheaper decode->resize step (unlike the NCNN weights above,
                    # which are fixed-shape and resize the crop back up regardless), and the
                    # same real-world object now fills more of the model's input - the
                    # small-object recall benefit that comes from "zooming in". raw_frame
                    # itself is left untouched (full frame) so it can still be shown with the
                    # ROI marked.
                    h, w = raw_frame.shape[:2]
                    rw, rh = int(w * args.roi), int(h * args.roi)
                    rx, ry = (w - rw) // 2, (h - rh) // 2
                    roi_box = (rx, ry, rx + rw, ry + rh)
                    detect_input = raw_frame[ry:ry + rh, rx:rx + rw]

                t0 = time.time()
                sp = None
                if frame_count % args.detect_stride == 0:
                    if args.track:
                        r = model.track(detect_input, tracker=args.tracker, persist=True, **predict_kwargs)[0]
                    else:
                        r = model.predict(detect_input, **predict_kwargs)[0]
                    boxes_list = boxes_from_results(r)
                    sp = r.speed  # ultralytics' own breakdown, ms - only exists for a real inference call
                else:
                    # Skipped detection frame: step ByteTrack/BotSORT's own Kalman
                    # filter directly instead of running (and discarding) a real
                    # detection - see --detect-stride help for the rationale.
                    # model.predictor only exists once a real track() call above
                    # has run at least once, which frame_count==0 always is.
                    tracker = model.predictor.trackers[0]
                    tracker.multi_predict(tracker.tracked_stracks)
                    boxes_list = boxes_from_tracks(tracker.tracked_stracks, model.names)
                frame_count += 1
                if args.track:
                    boxes_list = merge_lost_tracks(boxes_list, model.predictor.trackers[0], model.names)
                boxes_list = smooth_boxes(boxes_list, smoothed_state, args.smooth)
                detect_ms = (time.time() - t0) * 1000

                # r.plot() only ever works on a genuine Results tied to a real
                # inference call, which a skipped frame doesn't have - draw_boxes
                # handles both cases uniformly instead (see its docstring).
                t0 = time.time()
                frame = raw_frame.copy()
                offset = roi_box[:2] if roi_box is not None else (0, 0)
                # click-to-highlight needs FULL-FRAME coords to hit-test against
                # (on_mouse gets screen/window coords) - boxes_list itself stays
                # in ROI-crop-relative coords until draw_boxes applies offset.
                ox, oy = offset
                selection.boxes = [(x1 + ox, y1 + oy, x2 + ox, y2 + oy, tid, cls_name, conf)
                                    for x1, y1, x2, y2, tid, cls_name, conf in boxes_list]
                if roi_box is not None:
                    draw_roi_rect(frame, roi_box)
                draw_boxes(frame, boxes_list, offset, selected_id=selection.selected_id, font=label_font, lang=args.lang)
                plot_ms = (time.time() - t0) * 1000
                copy_ms = 0.0

                sleep_ms = 0.0
                if args.fps > 0:
                    remaining = (1.0 / args.fps) - (time.time() - t_iter0)
                    if remaining > 0:
                        time.sleep(remaining)
                        sleep_ms = remaining * 1000

                dt = time.time() - t_iter0
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                ema_fps = inst_fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * inst_fps

                t0 = time.time()
                draw_stats(frame, args.weights, dt, ema_fps)
                draw_ms = (time.time() - t0) * 1000
                t0 = time.time()
                cv2.imshow(window_name, frame)
                progress = f" | {Path(video_source).name} ({video_idx + 1}/{len(sources)})" if batch else ""
                cv2.setWindowTitle(window_name, f"{window_name}{progress} | {ema_fps:.1f} FPS (uncapped)")
                imshow_ms = (time.time() - t0) * 1000
                if args.record:
                    if writer is None:
                        h, w = frame.shape[:2]
                        write_fps = args.fps if args.fps > 0 else 30.0
                        writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), write_fps, (w, h))
                        print(f"--record: writing {w}x{h} @ {write_fps:.0f}fps -> {record_path}")
                    writer.write(frame)  # exactly what's on screen, overlay included

                if args.profile:
                    if sp is not None:
                        pre_ms, inf_ms, post_ms = sp.get("preprocess", 0.0), sp.get("inference", 0.0), sp.get("postprocess", 0.0)
                        other_ms = detect_ms - (pre_ms + inf_ms + post_ms)  # tracker update + python/ultralytics call overhead
                        stage = f"other={other_ms:5.1f}ms  preprocess={pre_ms:5.1f}ms  inference={inf_ms:5.1f}ms  postprocess={post_ms:5.1f}ms"
                    else:
                        stage = f"[skip-detect] multi_predict={detect_ms:5.1f}ms"
                    print(f"decode={decode_ms:5.1f}ms  {stage}  plot={plot_ms:4.1f}ms  copy={copy_ms:4.1f}ms  draw={draw_ms:4.1f}ms  "
                          f"imshow={imshow_ms:4.1f}ms  sleep={sleep_ms:4.1f}ms  |  TOTAL={dt*1000:6.1f}ms ({1/dt:4.1f} FPS)")

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC - quits the whole batch, not just this video
                    quit_all = True
                    break
                elif batch and key == ord("n"):
                    print("Skipping to next video.")
                    break
                elif key == ord("f"):
                    # Track IDs aren't reset across the jump; ByteTrack will just drop
                    # the now-unmatched old tracks and spawn fresh ones within a few
                    # frames, same as it does after any ordinary missed detection.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 60)
                    frame_count = 0  # force a real detection next frame - old tracks' Kalman state is stale across the cut

            cap.release()
            if writer is not None:
                writer.release()
                print(f"--record: saved -> {record_path}")
            if quit_all:
                break
        cv2.destroyAllWindows()
    else:
        out_dir = ROOT / "outputs" / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        counts = {}  # per-source running index, so a folder --source names/numbers frames per-video instead of one global count
        for r in results:
            stem = Path(r.path).stem
            i = counts.get(stem, 0)
            counts[stem] = i + 1
            ids = r.boxes.id
            ids = ids.int().tolist() if ids is not None else []
            out_path = out_dir / f"{stem}_{i}.jpg"
            r.save(filename=str(out_path))
            print(f"[{stem} {i}] {len(ids)} tracks, ids={ids} -> {out_path}")


if __name__ == "__main__":
    main()
