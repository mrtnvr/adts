#!/usr/bin/env python3
"""Get WALDO30 ready for the Hailo-8L. Runs here (Jetson). The compile step itself runs on an x86 PC.

For each chosen WALDO checkpoint:
  1. export an ONNX at 640, fixed shape, opset 11 (what Hailo's YOLOv8 flow expects)
  2. check the ONNX gives the same detections as the .pt on real frames. If it
     doesn't, the export is broken, and there's no point spending an hour compiling it.
Then, once for all models:
  3. write calibration frames from videos/ into calib/: 640x640 letterboxed, exactly
     as adts.detector.letterbox feeds the chip at runtime. The Dataflow Compiler uses
     them to pick the INT8 ranges, so they have to look like real flight footage.
  4. print the hailomz compile commands to run on the x86 PC.

    python3 export_hailo.py                       # n + m (the recommended pair)
    python3 export_hailo.py --models n m l n-p2   # more candidates
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torchvision
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS

from adts.bytetrack import iou_matrix
from adts.classes import WALDO_NAMES
from adts.detector import letterbox

torchvision.ops.nms = TorchNMS.nms  # the Jetson's broken torchvision ops, see track.py

ROOT = Path(__file__).parent
WALDO = {
    "n": "WALDO30_yolov8n_640x640",
    "n-p2": "WALDO30_yolov8n-p2_640x640",
    "m": "WALDO30_yolov8m_640x640",
    "l": "WALDO30_yolov8l_640x640",
}
IMGSZ = 640


def sample_frames(videos, n_total):
    """Evenly spaced frames across every video, weighted by length, so no single clip dominates the calibration set."""
    counts = {}
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        counts[v] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
    total = sum(counts.values())
    for v, n in counts.items():
        k = max(1, round(n_total * n / total)) if total else 0
        cap = cv2.VideoCapture(str(v))
        for idx in np.linspace(0, max(n - 1, 0), k).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                yield v, int(idx), frame
        cap.release()


def compare(pt, onnx_path, frames):
    """Match ONNX detections to .pt detections (conf >= 0.25) by class and IoU > 0.9."""
    m_pt, m_onnx = YOLO(str(pt)), YOLO(str(onnx_path), task="detect")
    kw = dict(imgsz=IMGSZ, conf=0.25, device="cpu", verbose=False)
    matched = total = 0
    for f in frames:
        a = m_pt.predict(f, **kw)[0].boxes
        b = m_onnx.predict(f, **kw)[0].boxes
        total += len(a)
        if not len(a) or not len(b):
            continue
        iou = iou_matrix(a.xyxy.numpy(), b.xyxy.numpy())
        same_cls = a.cls.numpy()[:, None] == b.cls.numpy()[None, :]
        matched += int(((iou > 0.9) & same_cls).any(axis=1).sum())
    return matched, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["n", "m"], choices=list(WALDO))
    ap.add_argument("--calib", type=int, default=1024, help="number of calibration frames (Hailo recommends ~1024)")
    ap.add_argument("--calib-dir", default=str(ROOT / "calib"))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted((ROOT / "videos").glob("*.mp4"))
    # Pre-letterbox to a 640 square: on a raw frame, .pt predict uses minimal
    # rectangular padding (e.g. 640x384) while the fixed-shape ONNX always gets
    # 640x640. The inputs then differ slightly, and about 10% of borderline
    # detections flip, which looks like a broken export when it isn't.
    check_frames = [letterbox(f, IMGSZ)[0] for _, _, f in sample_frames(videos, 12)]

    for key in args.models:
        pt = ROOT / "weights" / "waldo" / f"{WALDO[key]}.pt"
        model = YOLO(str(pt))
        names = [model.names[i] for i in sorted(model.names)]
        if names != WALDO_NAMES:
            raise SystemExit(f"{pt.name}: class list {names} != adts.classes.WALDO_NAMES. Update classes.py first.")
        print(f"[{key}] exporting {pt.name} -> ONNX {IMGSZ}x{IMGSZ} opset 11")
        exported = Path(model.export(format="onnx", imgsz=IMGSZ, opset=11, simplify=True, dynamic=False, batch=1, device="cpu"))
        target = out_dir / f"waldo_yolov8{key.replace('-', '')}_{IMGSZ}.onnx"
        exported.replace(target)
        matched, total = compare(pt, target, check_frames)
        verdict = "OK" if total == 0 or matched / total > 0.95 else "MISMATCH, don't compile this"
        print(f"[{key}] ONNX vs .pt: {matched}/{total} detections match -> {verdict}   ({target})")

    calib_dir = Path(args.calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for v, idx, frame in sample_frames(videos, args.calib):
        # Resize to the tracker's processing size first (1280x720 by default), then
        # letterbox, exactly as the runtime does.
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        img, _, _ = letterbox(frame, IMGSZ)
        cv2.imwrite(str(calib_dir / f"{v.stem}_{idx:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"\nwrote {n} calibration frames -> {calib_dir}")

    print("\n=== On the x86 Ubuntu PC (Hailo AI SW Suite / Dataflow Compiler + Model Zoo installed) ===")
    print("Copy models/*.onnx and calib/ over, then for each model:")
    for key in args.models:
        if key == "n-p2":
            print("  # n-p2 has a 4th (stride-4) detection head: the stock yolov8 config doesn't fit it.\n"
                  "  # It needs a custom YAML with 4 end-node pairs. See docs/DEPLOY_PI.md.")
            continue
        arch = f"yolov8{key}"
        print(f"  hailomz compile {arch} --ckpt waldo_{arch}_{IMGSZ}.onnx --hw-arch hailo8l "
              f"--calib-path calib/ --classes {len(WALDO_NAMES)} --performance")
        print(f"  mv {arch}.hef waldo_{arch}_{IMGSZ}.hef")
    print("Copy the .hef files to the Pi under ~/adts/models/.")


if __name__ == "__main__":
    main()
