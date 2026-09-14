#!/usr/bin/env python3
"""Export all ARGUS-YOLO weights to NCNN format for Raspberry Pi 5 deployment.

Raw PyTorch inference is too slow on a Pi 5 CPU; NCNN is Ultralytics'
recommended format for usable FPS on Raspberry Pi. Run once here, then copy
the resulting `*_ncnn_model/` directories onto the Pi 5.
"""

from pathlib import Path

import torchvision
from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS

# See predict.py: this machine's torchvision build has broken compiled ops,
# so route NMS (used by export's internal validation pass) through
# Ultralytics' pure-PyTorch fallback instead.
torchvision.ops.nms = TorchNMS.nms

ROOT = Path(__file__).parent

WEIGHTS = [
    ROOT / "weights" / "argus_yolo11l_1280.pt",
    ROOT / "weights" / "argus_yolo11x_1280.pt",
    ROOT / "weights" / "argus_yolo26x_1280.pt",
]


def main():
    for weights_path in WEIGHTS:
        print(f"Exporting {weights_path.name} -> NCNN ...")
        model = YOLO(str(weights_path))
        exported = model.export(format="ncnn", imgsz=1280, device="cpu")
        print(f"  done: {exported}")


if __name__ == "__main__":
    main()
