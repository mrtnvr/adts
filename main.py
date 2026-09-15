#!/usr/bin/env python3
"""ARGUS-YOLO interactive click-to-track viewer.

Two modes:
  - DETECT: YOLO runs every frame, drawing all detected boxes. This is the
    only mode where YOLO runs.
  - TRACK: triggered by left-clicking a detected box. YOLO is switched off
    entirely; an OpenCV CSRT tracker (classical CV, no neural net) follows
    just that box every frame instead - much cheaper than re-running YOLO.
    Right-click (or losing the target) returns to DETECT mode.

In DETECT mode, YOLO now runs in a background thread instead of blocking the
video loop: the main loop reads/displays frames as fast as decode allows and
always hands the detector its newest frame (never a queue - if the detector
is still busy, newer frames simply replace the pending one). Display FPS and
detector FPS are drawn separately top-right so it's clear the video itself is
no longer capped at YOLO's inference speed, even though each individual
detection still takes as long as it takes. Expect detector FPS to jump much
further once you click into TRACK mode, since YOLO stops running entirely.

ROI (region of interest): a fixed, centered box covering 2/3 of the frame in
each dimension is active from the start - YOLO only runs on that crop, not
the full frame. 'c' clears it (back to full frame), 'r' restores it. Note
this is NOT a speed optimization here - the NCNN exports are FIXED-SHAPE, so
the crop gets resized back up to the same imgsz the model always runs at,
meaning roughly the same compute either way. What it actually buys you:
small/distant objects fill more of the model's fixed input after cropping
(effectively zooming in), which helps recall on small targets, and
background outside the ROI can't generate false positives.

Detection stride (--detect-stride, default 4): YOLO runs once every Nth video
frame (real time, anchored to the source's own FPS - see DetectorThread). The
frames in between don't freeze - each box's position is EXTRAPOLATED forward
from the velocity observed between the last two actual detections (matched by
class + IoU). This is extrapolation, not interpolation: true interpolation
would need the *next* detection's result to blend towards, which doesn't
exist yet in a live/causal video loop (a webcam has no future frames to look
ahead into) - so this projects the recent trend forward instead, same
principle a Kalman-filter tracker uses. A freshly-detected box (age 0) is
always shown exactly as detected; extrapolation only affects the frames
before the next real one. Unlike merely skipping submissions (measured to
have zero effect on actual YOLO CPU time - the detector thread already
self-throttles to hardware's max pace regardless of submission rate), this
idles the detector thread itself between runs, so it genuinely frees CPU.

CPU-only by design: the initial deployment target for this project is a
Raspberry Pi 5, which has no CUDA GPU.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import torchvision
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS

# This machine's torchvision build (0.18.0a0) has broken/mismatched compiled
# ops for its torch build (2.5.0a0), so torchvision.ops.nms raises at call
# time. Ultralytics ships its own pure-PyTorch NMS that "matches torchvision
# behavior exactly" as a fallback for environments without torchvision ops -
# use it directly instead of the broken compiled op.
torchvision.ops.nms = TorchNMS.nms

ROOT = Path(__file__).parent

# NCNN (not raw .pt) is the default backend: on this CPU it runs ~4x faster
# than PyTorch at the same imgsz (benchmarked: 1280px 4.73s/frame -> 1.23s/frame).
# RoblabWhGe only ships 3 ARGUS-tuned sizes (11l/11x/26x) - there is no smaller
# fine-tuned checkpoint, so "a smaller model" here means the same 11l weights
# re-exported differently. Tested on a real frame from videos/ankara_drone_clip20s.mp4:
#   640px square: 1.12s/frame, 8 detections   (old default)
#   480px square: 0.90s/frame, 10 detections
#   320px square: 0.19s/frame, only 2-3 detections - drops most small objects, rejected
#   int8 (480 square, calibrated on 337 real frames): BROKEN - best confidence
#          0.12 vs the normal ~0.5-0.9, because ncnn2int8 only quantizes plain
#          Convolution/ConvolutionDepthwise layers, leaving YOLO11's Attention/
#          C2PSA/DFL blocks in fp32 - the resulting fp32<->int8 boundary
#          requant/dequant overhead even made it SLOWER (541-611ms) than fp16.
#          Rejected; would need QAT (needs training data, which we don't have)
#          to fix properly, not just PTQ calibration.
#   288x480 rectangular (matches the 1280x720 source's 16:9 aspect - square
#          480 letterboxes ~44% of the tensor with dead padding since the
#          source is 16:9, not square): 0.14s/frame, same 10 detections -
#          FASTEST option found, no accuracy tradeoff versus square 480.
#          ("fast" - now the default.) FP16 on top of this rect export gave
#          no further gain (within measurement noise on this box), so fp32.
# NCNN exports are FIXED-SHAPE: requesting an imgsz other than the one below
# hangs (confirmed) rather than erroring, so each entry carries its own
# required imgsz (int = square, tuple = (h, w)) and main() enforces it.
WEIGHTS = {
    "fast": {"path": ROOT / "weights" / "argus_yolo11l_480_ncnn_model", "imgsz": (288, 480)},  # default
    "11l": {"path": ROOT / "weights" / "argus_yolo11l_640_ncnn_model", "imgsz": 640},
    "11x": {"path": ROOT / "weights" / "argus_yolo11x_1280_ncnn_model", "imgsz": 1280},
    "26x": {"path": ROOT / "weights" / "argus_yolo26x_1280_ncnn_model", "imgsz": 1280},  # most accurate, heaviest
}

# GPU path (this box is actually a Jetson Orin NX, not a Pi 5 - `nvidia-smi`/
# torch.cuda both confirm a working CUDA device). Benchmarked on a real frame
# from videos/ankara_drone_clip20s.mp4: raw .pt 11l @1280 FP16 on the GPU
# (120ms/frame, 14 detections) already beats CPU NCNN "fast" 288x480 (142ms/
# frame, 5 detections) while running at full native resolution instead of a
# downscaled crop. TensorRT (.engine) is faster still - export_trt.py builds
# an FP16 engine once; this picks it up automatically the moment it exists,
# no code change needed, and falls back to the plain .pt in the meantime.
# There's no GPU equivalent of "fast": that key is an NCNN-only trick to claw
# back CPU speed by shrinking imgsz, which the GPU doesn't need.
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


# WALDO30 (github.com/stephansturges/WALDO) - a general-purpose overhead-imagery
# detector (YOLOv8, MIT-ish license, civilian use only), NOT ARGUS-tuned.
# 12 classes incl. LightVehicle/Truck/Bus/Person/Building/UPole/Container/etc,
# used as-is here - draw_detect_boxes() reads labels from model.names, so the
# richer taxonomy just works, no code change needed there.
# GPU-only: only .pt checkpoints were pulled, no NCNN export was made/tested.
# Benchmarked on assets/mosaic_val.jpg (a close-range rescue scene, RTX-class
# accuracy claims don't apply - this is an Orin NX): waldo-n and waldo-n-p2
# are ~2x faster than argus 11l (171-224ms vs 387ms) with strong, confident
# LightVehicle/Truck/Bus detections (conf 0.4-0.9+) - a genuinely good fast
# vehicle pre-filter. BUT Person recall on this domain is weak: 0 detections
# at the standard conf=0.25 across all 3 WALDO sizes on an image with real
# people in it; even at conf=0.05 the best candidate was only 0.18 - WALDO
# was trained on general "30ft-to-satellite" overhead imagery, not fine-tuned
# on close-range rescue scenes the way ARGUS's own human class was. Use WALDO
# for fast vehicle/infrastructure awareness; keep ARGUS for human detection,
# the actual search-and-rescue mission.
WALDO_WEIGHTS = {
    "waldo-n": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n_640x640.pt", "imgsz": 640},
    "waldo-n-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n-p2_640x640.pt", "imgsz": 640},
    "waldo-l-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l-p2_1024x1024.pt", "imgsz": 1024},
    "waldo-m": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8m_640x640.pt", "imgsz": 640},
    "waldo-l": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l_640x640.pt", "imgsz": 640},  # no P2 head, unlike waldo-l-p2
}

WINDOW_NAME = "ARGUS-YOLO - click=track, right-click=clear, c=disable ROI, r=enable ROI, f=skip 60 frames, q=quit"

BOX_COLOR = (255, 180, 0)  # detect-mode boxes (BGR)
TARGET_COLOR = (0, 255, 0)  # tracked target
ROI_COLOR = (0, 200, 255)
ROI_FRACTION = 2 / 3  # fixed ROI covers this fraction of the frame in each dimension, centered


def compute_fixed_roi(frame_shape):
    h, w = frame_shape[:2]
    rw, rh = int(w * ROI_FRACTION), int(h * ROI_FRACTION)
    x1, y1 = (w - rw) // 2, (h - rh) // 2
    return (x1, y1, x1 + rw, y1 + rh)


def make_cv_tracker():
    # CSRT: OpenCV's classical (non-NN) single-object tracker - accurate and
    # real-time on CPU, needs opencv-contrib-python (cv2.legacy namespace).
    return cv2.legacy.TrackerCSRT_create()


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


class DetectorThread:
    """Runs YOLO in the background so the video loop never blocks on it.

    submit() always overwrites the pending frame rather than queuing - if the
    detector is still busy with the previous one, older frames are simply
    dropped and it moves straight to the newest. This is what actually makes
    YOLO "faster" here: not the model itself, but decoupling it from display
    so the video no longer waits on every single inference.

    min_interval is a genuine compute throttle, unlike submission-side frame
    skipping: measured empirically, gating how often submit() is called does
    NOT reduce actual YOLO CPU time, because this thread already only ever
    processes the newest available frame and drops the rest - it runs at
    hardware's natural max pace (~7 FPS here) regardless of submission rate
    (confirmed: 16 detections in ~2.5s at both stride=1 and stride=4). To
    genuinely free CPU for other work, the thread itself must idle - so after
    each completed inference, if less than min_interval elapsed, it sleeps
    out the difference before picking up the next frame.
    """

    def __init__(self, model, imgsz, conf, names, min_interval=0.0, device="cpu"):
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.names = names
        self.min_interval = min_interval
        self.device = device
        self._lock = threading.Lock()
        self._pending_frame = None
        self._new_frame = threading.Event()
        self.boxes = []  # most recent result: (x1, y1, x2, y2, cls_name, conf)
        self.version = 0  # bumped every time self.boxes is replaced, so callers can detect a fresh result
        self.fps = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._pending_frame = frame
        self._new_frame.set()

    def stop(self):
        self._running = False
        self._new_frame.set()
        self._thread.join(timeout=2)

    def _worker(self):
        ema = 0.0
        while self._running:
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()
            with self._lock:
                frame = self._pending_frame
            if frame is None:
                continue

            t0 = time.time()
            results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
            r = results[0]
            boxes = []
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                cls_name = self.names[int(b.cls[0])]
                conf = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, cls_name, conf))
            dt = time.time() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            ema = inst if ema == 0.0 else 0.9 * ema + 0.1 * inst

            self.boxes = boxes
            self.fps = ema
            self.version += 1

            if self.min_interval > 0:
                remaining = self.min_interval - dt
                if remaining > 0:
                    time.sleep(remaining)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter)


class BoxExtrapolator:
    """Projects box positions forward from the velocity between the last two detections.

    Not interpolation: interpolation would blend towards the *next* detection,
    which doesn't exist yet in a live video loop. This extrapolates the recent
    trend instead (same principle a Kalman-filter tracker uses), so it works
    identically for a recorded file and a live webcam.
    """

    IOU_MATCH_THRESHOLD = 0.2
    MAX_T = 1.5  # cap how far past the normal stride we'll keep projecting if a detection is late

    def __init__(self):
        self.prev = []  # detection before last: (x1, y1, x2, y2, cls_name, conf)
        self.curr = []  # most recent detection
        self.age = 0  # display frames since curr was captured

    def update(self, new_boxes):
        self.prev = self.curr
        self.curr = new_boxes
        self.age = 0

    def tick(self):
        self.age += 1

    def get_boxes(self, stride):
        if not self.curr:
            return []
        t = min(self.age / stride, self.MAX_T)
        if not self.prev or t == 0:
            return self.curr

        used_prev = set()
        out = []
        for x1, y1, x2, y2, cls_name, conf in self.curr:
            best_iou, best_j = 0.0, -1
            for j, (px1, py1, px2, py2, pcls, _pconf) in enumerate(self.prev):
                if pcls != cls_name or j in used_prev:
                    continue
                iou = box_iou((x1, y1, x2, y2), (px1, py1, px2, py2))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou > self.IOU_MATCH_THRESHOLD:
                used_prev.add(best_j)
                px1, py1, px2, py2 = self.prev[best_j][:4]
                ex1 = int(x1 + (x1 - px1) * t)
                ey1 = int(y1 + (y1 - py1) * t)
                ex2 = int(x2 + (x2 - px2) * t)
                ey2 = int(y2 + (y2 - py2) * t)
                out.append((ex1, ey1, ex2, ey2, cls_name, conf))
            else:
                out.append((x1, y1, x2, y2, cls_name, conf))  # new object - no prior velocity to project
        return out


class App:
    """Owns interaction state: which mode we're in, the active CV tracker, and the ROI."""

    def __init__(self):
        self.mode = "detect"  # or "track"
        self.boxes = []  # detect-mode boxes, in FULL-FRAME coords: (x1, y1, x2, y2, cls_name, conf)
        self.cv_tracker = None
        self.target_label = None
        self.current_frame = None  # kept up to date by the main loop for on_mouse to use
        self.roi = None  # (x1, y1, x2, y2) in full-frame coords, or None = whole frame
        self.fixed_roi = None  # the computed 2/3-frame rectangle, set once frame size is known; 'r'/'c' toggle self.roi between this and None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == "detect":
            for x1, y1, x2, y2, cls_name, conf in self.boxes:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    tracker = make_cv_tracker()
                    tracker.init(self.current_frame, (x1, y1, x2 - x1, y2 - y1))
                    self.cv_tracker = tracker
                    self.target_label = f"{cls_name} {conf:.2f}"
                    self.mode = "track"
                    break
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mode = "detect"
            self.cv_tracker = None
            self.target_label = None


def draw_fps(frame, real_fps):
    # Single number, the actual measured loop rate (frame read -> detect/extrapolate
    # -> draw -> show) - what's really reaching the screen, not a per-stage
    # breakdown someone has to interpret.
    text = f"{real_fps:.1f} FPS"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = frame.shape[1] - tw - 15
    cv2.putText(frame, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_mode(frame, mode):
    text = "YOLO: DETECTING" if mode == "detect" else "YOLO: OFF (CSRT tracking)"
    color = (255, 180, 0) if mode == "detect" else (0, 255, 0)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)


def draw_detect_boxes(frame, boxes):
    for x1, y1, x2, y2, cls_name, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_roi(frame, roi):
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)
        cv2.putText(frame, "ROI", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2, cv2.LINE_AA)


def draw_target_box(frame, bbox, label):
    x, y, w, h = (int(v) for v in bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), TARGET_COLOR, 3)
    text = f"TARGET {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), TARGET_COLOR, -1)
    cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="video file path, RTSP/HTTP URL, or webcam index (e.g. '0')")
    parser.add_argument("--weights", choices=(*WEIGHTS.keys(), *WALDO_WEIGHTS.keys()), default="fast",
                         help="ARGUS-YOLO checkpoint (default: fast, 288x480 rectangular), or a waldo-* checkpoint - a general-purpose, "
                              "non-ARGUS-tuned overhead detector (12 classes incl. LightVehicle/Truck/Bus/Person/Building/...); "
                              "fast vehicle detection but weak Person recall on this domain (see WALDO_WEIGHTS comment above). GPU-only.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                         help="cpu = NCNN Pi-5-preview path (default, unchanged; not available for waldo-* weights). gpu = run on this box's "
                              "Jetson Orin GPU (ARGUS: native 1280px, uses the TensorRT .engine if exported else the plain .pt; "
                              "waldo-*: that checkpoint's own native imgsz); ignores --pi-cores, and --weights=fast maps to ARGUS 11l@1280")
    parser.add_argument("--imgsz", type=int, default=None, help="must match the chosen weights' export size (see --weights); omit to use it automatically")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pi-cores", type=int, default=4,
                         help="cap NCNN threads to this many, to preview Raspberry Pi 5 (4 cores) speed instead of this dev machine's full core count; 0 = use all cores")
    parser.add_argument("--detect-stride", type=int, default=4,
                         help="run YOLO once every this many video frames (real-time, anchored to the source's own FPS, not the display loop's) - genuinely throttles CPU by idling the detector thread between runs, not just skipping submissions. Boxes in between are extrapolated from the last two detections' velocity, not frozen (1 = detect every frame, no throttle/extrapolation)")
    parser.add_argument("--fps", type=float, default=0.0,
                         help="cap display playback rate to this many FPS, so a recorded video plays at realistic speed instead of fast-forwarding through it as fast as decode+draw allow; 0 = uncapped (default)")
    args = parser.parse_args()

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

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.video}")

    # detect-stride is expressed in video frames, not display loop iterations
    # (display can run uncapped/arbitrarily fast) - so the real-time interval
    # it implies is anchored to the source's own frame rate, independent of
    # --fps. Falls back to 30 if the source doesn't report one (some webcams).
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_interval = args.detect_stride / native_fps

    model = YOLO(str(weights_path))
    names = model.names
    if args.device == "cpu" and args.pi_cores > 0:
        limit_ncnn_threads(model, args.imgsz, args.pi_cores)
    detector_device = 0 if args.device == "gpu" else "cpu"
    detector = DetectorThread(model, args.imgsz, args.conf, names, min_interval=min_interval, device=detector_device)

    app = App()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, app.on_mouse)

    extrapolator = BoxExtrapolator()
    last_detector_version = -1
    frame_count = 0

    ema_fps = 0.0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            app.current_frame = frame
            if app.fixed_roi is None:
                app.fixed_roi = compute_fixed_roi(frame.shape)
                app.roi = app.fixed_roi  # active from the first frame

            if app.mode == "detect":
                if frame_count % args.detect_stride == 0:
                    if app.roi is not None:
                        rx1, ry1, rx2, ry2 = app.roi
                        detector.submit(frame[ry1:ry2, rx1:rx2])  # crop only - see module docstring on why this isn't a speed win
                    else:
                        detector.submit(frame)  # non-blocking: just replaces whatever the thread hasn't picked up yet

                if detector.version != last_detector_version:
                    last_detector_version = detector.version
                    if app.roi is not None:
                        rx1, ry1, rx2, ry2 = app.roi
                        fresh = [(x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1, cls_name, conf)
                                 for x1, y1, x2, y2, cls_name, conf in detector.boxes]
                    else:
                        fresh = detector.boxes
                    extrapolator.update(fresh)
                else:
                    extrapolator.tick()
                app.boxes = extrapolator.get_boxes(args.detect_stride)
                draw_detect_boxes(frame, app.boxes)
            else:
                ok_t, bbox = app.cv_tracker.update(frame)
                if ok_t:
                    draw_target_box(frame, bbox, app.target_label)
                else:
                    cv2.putText(frame, "Target lost - back to detection", (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    app.mode = "detect"
                    app.cv_tracker = None
                    app.target_label = None

            draw_mode(frame, app.mode)
            draw_roi(frame, app.roi)

            if args.fps > 0:
                remaining = (1.0 / args.fps) - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

            dt = time.time() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            ema_fps = inst_fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * inst_fps
            draw_fps(frame, ema_fps)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("r"):
                app.roi = app.fixed_roi
            elif key == ord("c"):
                app.roi = None
            elif key == ord("f"):
                # 60 frames is a discontinuous jump - old boxes/tracker no longer
                # correspond to anything real, so drop them rather than let
                # extrapolation or CSRT carry stale state across the cut.
                new_pos = cap.get(cv2.CAP_PROP_POS_FRAMES) + 60
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                extrapolator = BoxExtrapolator()
                last_detector_version = detector.version
                app.boxes = []
                app.mode = "detect"
                app.cv_tracker = None
                app.target_label = None
            frame_count += 1
    finally:
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
