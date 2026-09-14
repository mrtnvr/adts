#!/usr/bin/env python3
"""Run ARGUS-YOLO + ByteTrack multi-object tracking on a video/stream, CPU-only.

CPU-only is intentional: the initial deployment target for this project is a
Raspberry Pi 5, which has no CUDA GPU. Running CPU-only here on the dev
machine (even though it has a Jetson GPU available) keeps behavior
representative of the target hardware.

Source can be a video file, an RTSP/HTTP stream URL, a webcam index ("0"),
or a directory of image frames.
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
        choices=WEIGHTS.keys(),
        default="fast",
        help="which ARGUS-YOLO checkpoint to use (default: fast, 288x480 rectangular)",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "assets" / "mosaic_val.jpg"),
        help="video file, stream URL, webcam index, or image/dir to track on",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="must match the chosen weights' export size; omit to use it automatically")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="ultralytics tracker config (bytetrack.yaml or botsort.yaml)",
    )
    parser.add_argument("--pi-cores", type=int, default=4,
                         help="cap NCNN threads to this many, to preview Raspberry Pi 5 (4 cores) speed instead of this dev machine's full core count; 0 = use all cores")
    args = parser.parse_args()

    choice = WEIGHTS[args.weights]
    if args.imgsz is None:
        args.imgsz = choice["imgsz"]
    elif args.imgsz != choice["imgsz"]:
        raise SystemExit(
            f"--weights {args.weights} is an NCNN export fixed at imgsz={choice['imgsz']}; "
            f"a mismatched --imgsz {args.imgsz} won't error, it will just hang. "
            f"Pass --imgsz {choice['imgsz']} or omit --imgsz."
        )

    model = YOLO(str(choice["path"]))
    if args.pi_cores > 0:
        limit_ncnn_threads(model, args.imgsz, args.pi_cores)
    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device="cpu",
        tracker=args.tracker,
        persist=True,
        stream=True,
    )

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
