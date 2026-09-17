#!/usr/bin/env python3
"""Run YOLO + ByteTrack/BotSORT multi-object tracking on a video/stream, live by default.

Defaults to --device gpu --weights waldo-l-p2 --show: this box is actually a
Jetson Orin NX (not the Pi 5 the project originally targeted), and WALDO30
is a fast, general-purpose overhead detector that's a good live
vehicle/infrastructure tracker (see WALDO_WEIGHTS comment below) - but it has
weak Person recall on close-range rescue scenes, so pass --weights fast (or
11l/11x/26x) for ARGUS's own, human-detection-tuned checkpoints; pass
--device cpu for the original NCNN Raspberry Pi 5 preview path.

Source can be a video file, an RTSP/HTTP stream URL, a webcam index ("0"),
or a directory of image frames.
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torchvision
from ultralytics import YOLO
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

# Turkish display names for WALDO's 12 classes - user-chosen for the 4 with a
# real translation judgment call (Person/LightVehicle/Bike/Digger cover a
# broader real-world category than any single Turkish word, see git history
# for the tradeoffs discussed); the rest are unambiguous direct translations.
# ASCII-transliterated (ç->c, ı->i, İ->I, ş->s, ğ->g, ü->u, ö->o) on purpose:
# any non-ASCII character here forces ultralytics' Annotator onto its PIL/
# Unicode-font text path, measured at ~16ms/frame here vs ~2ms for the plain
# cv2.putText path used when every label is pure ASCII - a bigger cost than
# the tracker itself. Words stay Turkish, just spelled without diacritics.
WALDO_NAMES_TR = {
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


def draw_boxes(frame, boxes_list, offset=(0, 0), selected_id=None):
    # Manual drawing (not r.plot()) so the exact same code path handles both a
    # real Results object's boxes (via boxes_from_results) and the tracker's
    # own predicted-only state on a skipped detection frame (via
    # boxes_from_tracks) - r.plot() only ever works on a genuine Results tied
    # to an actual inference call, which skipped frames don't have. Plain
    # cv2.putText (labels are already ASCII, see WALDO_NAMES_TR) - cheaper
    # than r.plot()'s PIL path too. offset shifts from a --roi crop's origin
    # back to full-frame coordinates; (0, 0) when --roi is off.
    #
    # selected_id: click-to-highlight target (see TrackSelection) - when set,
    # every other box is dimmed instead of hidden, so the rest of the scene
    # (and the tracker/detector still running on it) stays visible.
    ox, oy = offset
    for x1, y1, x2, y2, track_id, cls_name, conf in boxes_list:
        x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
        is_target = selected_id is not None and track_id == selected_id
        color = TARGET_COLOR if is_target else (BOX_COLOR if selected_id is None else DIM_COLOR)
        thickness = 2 if is_target else 1
        label = f"id:{track_id} {cls_name} {conf:.2f}" if track_id is not None else f"{cls_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
        cv2.rectangle(frame, (x1, y1 - th - 3), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)


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
        "--source",
        default=str(ROOT / "assets" / "mosaic_val.jpg"),
        help="video file, stream URL, webcam index, or image/dir to track on",
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
                              "to an mp4 in outputs/tracks/; no effect with --no-show, which already saves every frame as a jpg")
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

    # Translate WALDO's English class names to Turkish for display - AFTER
    # exclude_classes above, which matches against the model's original
    # (English) names, so --exclude-classes UPole keeps working unchanged.
    if args.weights in WALDO_WEIGHTS:
        # model.names has no setter (YOLO wraps a torch nn.Module) - the
        # mutable dict actually lives on the underlying DetectionModel.
        model.model.names = {i: WALDO_NAMES_TR.get(name, name) for i, name in model.names.items()}

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
        # seek, instead of reaching into model.predictor.dataset.cap.
        cap_source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open source: {args.source}")

        mode_label = args.tracker if args.track else "no-track"
        window_name = f"ARGUS-YOLO track ({args.weights}, {mode_label}) - click=highlight, right-click=clear, f=skip 60 frames, q=quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        selection = TrackSelection()
        cv2.setMouseCallback(window_name, selection.on_mouse)
        ema_fps = 0.0
        frame_count = 0  # frame 0 always detects, so model.predictor exists before any --detect-stride skip needs it
        smoothed_state = {}  # track_id -> last EMA-smoothed (x1,y1,x2,y2), for smooth_boxes()
        writer = None
        if args.record:
            record_dir = ROOT / "outputs" / "tracks"
            record_dir.mkdir(parents=True, exist_ok=True)
            record_path = record_dir / f"{Path(args.source).stem}_{args.weights}.mp4"
            # writer is opened lazily below, once the first annotated frame's
            # actual pixel size is known, rather than guessed up front
        predict_kwargs = dict(imgsz=args.imgsz, conf=args.conf, device=0 if args.device == "gpu" else "cpu",
                               classes=class_ids, half=(args.half and args.device == "gpu"), verbose=False)
        while True:
            t_iter0 = time.time()

            t0 = time.time()
            ok, raw_frame = cap.read()
            decode_ms = (time.time() - t0) * 1000
            if not ok:
                print("End of video.")
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
            draw_boxes(frame, boxes_list, offset, selected_id=selection.selected_id)
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
            cv2.setWindowTitle(window_name, f"{window_name} | {ema_fps:.1f} FPS (uncapped)")
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
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("f"):
                # Track IDs aren't reset across the jump; ByteTrack will just drop
                # the now-unmatched old tracks and spawn fresh ones within a few
                # frames, same as it does after any ordinary missed detection.
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 60)
                frame_count = 0  # force a real detection next frame - old tracks' Kalman state is stale across the cut
        cap.release()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print(f"--record: saved -> {record_path}")
    else:
        out_dir = ROOT / "outputs" / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(results):
            ids = r.boxes.id
            ids = ids.int().tolist() if ids is not None else []
            out_path = out_dir / f"{Path(args.source).stem}_{i}.jpg"
            r.save(filename=str(out_path))
            print(f"[{i}] {len(ids)} tracks, ids={ids} -> {out_path}")


if __name__ == "__main__":
    main()
