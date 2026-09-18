#!/usr/bin/env python3
"""Check a compiled .hef against the float model, using the same frames.

Step 1 (Jetson, has Ultralytics): write reference detections from the ONNX/.pt
    python3 tools/hef_check.py make-ref --model models/waldo_yolov8m_640.onnx --out models/ref_m.npz
Step 2 (Pi, has hailo_platform): run the .hef through adts.detector.HailoDetector on the same frames
    python3 tools/hef_check.py check --model models/waldo_yolov8m_640.hef --ref models/ref_m.npz

This checks, together:
  - HailoDetector's NMS output parsing and box coordinate mapping (not testable without the chip);
    boxes landing in the wrong place show up as near-zero matches
  - the INT8 quantisation loss: expect roughly 80-95% matches, fewer on small Person boxes
Annotated side-by-side images go to --vis-dir, to look at by eye.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adts.bytetrack import iou_matrix  # noqa: E402
from adts.classes import WALDO_NAMES  # noqa: E402
from adts.detector import make_detector  # noqa: E402

CONF = 0.25


def frames(n):
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "calib").glob("*.jpg"))
    if not files:
        raise SystemExit("no calib/*.jpg: run export_hailo.py first (and copy calib/ to the Pi)")
    for f in files[:: max(1, len(files) // n)][:n]:
        yield f.name, cv2.imread(str(f))


def draw(img, xyxy, cls, color):
    for b, c in zip(xyxy.astype(int), cls):
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), color, 1)
        cv2.putText(img, WALDO_NAMES[c], (b[0], b[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("make-ref", "check"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default="models/ref.npz")
    ap.add_argument("--out", default=None, help="make-ref output (default: --ref)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--vis-dir", default="outputs/hef_check")
    args = ap.parse_args()

    # Frames are already 640x640 letterboxed, so imgsz=640 makes letterboxing a no-op on both sides.
    det = make_detector(args.model, imgsz=640, conf=CONF)
    if args.mode == "make-ref":
        data = {}
        for name, img in frames(args.n):
            d = det(img)
            data[name + ":xyxy"], data[name + ":cls"] = d.xyxy, d.cls
        out = args.out or args.ref
        np.savez_compressed(out, **data)
        print(f"wrote reference detections for {len(data) // 2} frames -> {out}")
        return

    ref = np.load(args.ref)
    vis = Path(args.vis_dir)
    vis.mkdir(parents=True, exist_ok=True)
    matched = total = extra = 0
    per_cls = {}
    for name, img in frames(args.n):
        if name + ":xyxy" not in ref:
            continue
        rx, rc = ref[name + ":xyxy"], ref[name + ":cls"]
        d = det(img)
        iou = iou_matrix(rx, d.xyxy) if len(rx) and len(d) else np.zeros((len(rx), len(d)))
        hit = ((iou > 0.5) & (rc[:, None] == d.cls[None, :])).any(axis=1) if len(d) else np.zeros(len(rx), bool)
        matched += int(hit.sum())
        total += len(rx)
        extra += int(len(d) - hit.sum()) if len(d) > hit.sum() else 0
        for c, h in zip(rc, hit):
            s = per_cls.setdefault(WALDO_NAMES[c], [0, 0])
            s[0] += int(h); s[1] += 1
        canvas = np.hstack([img.copy(), img.copy()])
        draw(canvas[:, :640], rx, rc, (0, 255, 0))
        draw(canvas[:, 640:], d.xyxy, d.cls, (0, 200, 255))
        cv2.imwrite(str(vis / name), canvas)
    print(f"reference boxes found by the .hef: {matched}/{total} ({100 * matched / max(total, 1):.0f}%), extra boxes: {extra}")
    for c, (h, n) in sorted(per_cls.items(), key=lambda kv: -kv[1][1]):
        print(f"  {c:14s} {h:4d}/{n:<4d} {100 * h / n:5.1f}%")
    print(f"side-by-side images (left = float reference, right = .hef) -> {vis}")
    if total and matched / total < 0.3:
        print("VERY LOW match: most likely the NMS output parsing/box order in HailoDetector._parse_nms, not quantisation.")


if __name__ == "__main__":
    main()
