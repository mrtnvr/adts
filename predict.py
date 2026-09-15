#!/usr/bin/env python3
"""Run ARGUS-YOLO inference on a sample image, CPU-only.

CPU-only is intentional: the initial deployment target for this project is a
Raspberry Pi 5, which has no CUDA GPU. Running CPU-only here on the dev
machine (even though it has a Jetson GPU available) keeps behavior
representative of the target hardware.
"""

import argparse
from pathlib import Path

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

# NCNN (not raw .pt) is the default backend: ~4x faster than PyTorch at the
# same imgsz on this CPU. RoblabWhGe only ships 3 ARGUS-tuned sizes
# (11l/11x/26x) - "fast" is the same 11l weights re-exported at a rectangular
# 288x480 (matches the 16:9 source instead of padding a square to it - see
# main.py for the full size/format comparison, including why int8 was tried
# and rejected). NCNN exports are FIXED-SHAPE - requesting a mismatched
# imgsz hangs rather than erroring, so each entry carries the imgsz (int =
# square, tuple = (h, w)) it was exported at and main() enforces it.
WEIGHTS = {
    "fast": {"path": ROOT / "weights" / "argus_yolo11l_480_ncnn_model", "imgsz": (288, 480)},  # default
    "11l": {"path": ROOT / "weights" / "argus_yolo11l_640_ncnn_model", "imgsz": 640},
    "11x": {"path": ROOT / "weights" / "argus_yolo11x_1280_ncnn_model", "imgsz": 1280},
    "26x": {"path": ROOT / "weights" / "argus_yolo26x_1280_ncnn_model", "imgsz": 1280},  # most accurate, heaviest
}

# GPU path - see main.py for the full rationale (this box is actually a
# Jetson Orin NX, not a Pi 5). Picks up the TensorRT .engine automatically
# once export_trt.py finishes building it; falls back to the plain .pt until
# then. No GPU equivalent of "fast" - that's an NCNN-only imgsz-shrink trick.
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


# WALDO30 (github.com/stephansturges/WALDO) - see main.py for the full
# rationale/benchmarks. General-purpose overhead detector, NOT ARGUS-tuned:
# fast, confident LightVehicle/Truck/Bus detection but weak Person recall on
# close-range rescue scenes (0 detections at conf=0.25 in testing; best
# candidate was only 0.18 even at conf=0.05). Use for vehicle/infrastructure
# awareness, not as a human-detection replacement for ARGUS. GPU-only - only
# .pt checkpoints were pulled, no NCNN export made.
WALDO_WEIGHTS = {
    "waldo-n": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n_640x640.pt", "imgsz": 640},
    "waldo-n-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n-p2_640x640.pt", "imgsz": 640},
    "waldo-l-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l-p2_1024x1024.pt", "imgsz": 1024},
    "waldo-m": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8m_640x640.pt", "imgsz": 640},
    "waldo-l": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l_640x640.pt", "imgsz": 640},  # no P2 head, unlike waldo-l-p2
}


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
        default="fast",
        help="which checkpoint to use (default: fast, ARGUS 288x480 rectangular), or a waldo-* checkpoint - "
             "general-purpose, non-ARGUS-tuned (see WALDO_WEIGHTS comment above). GPU-only.",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "assets" / "mosaic_val.jpg"),
        help="image (or dir/video) to run detection on",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                         help="cpu = NCNN Pi-5-preview path (default, unchanged). gpu = run on this box's Jetson Orin GPU at native 1280px "
                              "(uses the TensorRT .engine if exported, else the plain .pt); ignores --weights=fast and --pi-cores")
    parser.add_argument("--imgsz", type=int, default=None, help="must match the chosen weights' export size; omit to use it automatically")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pi-cores", type=int, default=4,
                         help="cap NCNN threads to this many, to preview Raspberry Pi 5 (4 cores) speed instead of this dev machine's full core count; 0 = use all cores")
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

    model = YOLO(str(weights_path))
    if args.device == "cpu" and args.pi_cores > 0:
        limit_ncnn_threads(model, args.imgsz, args.pi_cores)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=0 if args.device == "gpu" else "cpu",
    )

    out_dir = ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        n_dets = len(r.boxes)
        out_path = out_dir / f"{Path(args.source).stem}_{i}.jpg"
        r.save(filename=str(out_path))
        print(f"[{i}] {n_dets} detections -> {out_path}")


if __name__ == "__main__":
    main()
