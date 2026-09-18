#!/usr/bin/env python3
"""Run WALDO detection on a sample image, GPU-only (WALDO only ships .pt checkpoints - no
NCNN export exists for a CPU-only preview)."""

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

# See track.py's WALDO_WEIGHTS comment for the accuracy/speed tradeoffs between checkpoints.
WALDO_WEIGHTS = {
    "waldo-n": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n_640x640.pt", "imgsz": 640},
    "waldo-n-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8n-p2_640x640.pt", "imgsz": 640},
    "waldo-l-p2": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l-p2_1024x1024.pt", "imgsz": 1024},
    "waldo-m": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8m_640x640.pt", "imgsz": 640},
    "waldo-l": {"path": ROOT / "weights" / "waldo" / "WALDO30_yolov8l_640x640.pt", "imgsz": 640},  # no P2 head, unlike waldo-l-p2
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", choices=sorted(WALDO_WEIGHTS.keys()), default="waldo-l-p2",
                         help="which WALDO checkpoint to use (default: waldo-l-p2)")
    parser.add_argument("--source", required=True, help="image (or dir/video) to run detection on")
    parser.add_argument("--imgsz", type=int, default=None, help="must match the chosen weights' export size; omit to use it automatically")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    choice = WALDO_WEIGHTS[args.weights]
    if args.imgsz is not None and args.imgsz != choice["imgsz"]:
        raise SystemExit(f"--weights {args.weights} is native imgsz={choice['imgsz']}; pass that or omit --imgsz.")
    args.imgsz = choice["imgsz"]
    weights_path = choice["path"]

    model = YOLO(str(weights_path))
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, device=0)

    out_dir = ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        n_dets = len(r.boxes)
        out_path = out_dir / f"{Path(args.source).stem}_{i}.jpg"
        r.save(filename=str(out_path))
        print(f"[{i}] {n_dets} detections -> {out_path}")


if __name__ == "__main__":
    main()
