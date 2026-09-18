"""Detector backends that return plain numpy detections in frame pixel coordinates.

- HailoDetector: a compiled .hef on the Hailo-8L (the Pi deployment path). The YOLOv8
  NMS runs on the chip, so the host only letterboxes the input and unpacks the result.
- UltralyticsDetector: .pt / .engine / .onnx through Ultralytics. This is the
  development path on the Jetson, and the way to check an exported ONNX before it goes
  to the Hailo compiler.

Both letterbox the same way (centred, pad value 114, like Ultralytics), so the
calibration frames export_hailo.py writes match what the chip sees at runtime.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .classes import WALDO_NAMES


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4) float32, frame pixels
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int32

    @staticmethod
    def empty():
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))

    def __len__(self):
        return len(self.conf)

    def filter(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])


def letterbox(img, size):
    """Resize keeping aspect ratio, pad to size x size. Returns (padded, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = resized
    return out, r, (px, py)


class _ClassFilter:
    def __init__(self, names, exclude, conf):
        self.names = list(names)
        self.conf = conf
        excluded = {n.strip() for n in exclude if n.strip()}
        self.keep = np.array([n not in excluded for n in self.names], dtype=bool)

    def apply(self, det):
        if not len(det):
            return det
        return det.filter((det.conf >= self.conf) & self.keep[det.cls])


class HailoDetector:
    def __init__(self, hef_path, conf=0.1, exclude=(), names=WALDO_NAMES):
        # Import here, not at module level: hailo_platform only exists on the Pi (from
        # apt's hailo-all). The Jetson dev path never touches it.
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        self._model = self._vdevice.create_infer_model(str(hef_path))
        self._model.set_batch_size(1)
        self._model.input().set_format_type(FormatType.UINT8)
        self._model.output().set_format_type(FormatType.FLOAT32)
        self._configured_ctx = self._model.configure()
        self._configured = self._configured_ctx.__enter__()
        self._bindings = self._configured.create_bindings()
        self._out = np.empty(self._model.output().shape, dtype=np.float32)
        self._bindings.output().set_buffer(self._out)
        in_shape = self._model.input().shape  # (H, W, C) for a Hailo-compiled YOLOv8
        if in_shape[0] != in_shape[1]:
            raise ValueError(f"expected a square model input, got {in_shape}")
        self.imgsz = int(in_shape[0])
        self.filter = _ClassFilter(names, exclude, conf)
        self.names = self.filter.names

    def __call__(self, frame_bgr):
        img, r, (px, py) = letterbox(frame_bgr, self.imgsz)
        rgb = np.ascontiguousarray(img[:, :, ::-1])  # the HEF expects RGB, the camera gives BGR
        self._bindings.input().set_buffer(rgb)
        self._configured.run([self._bindings], 10000)
        boxes, scores, classes = self._parse_nms(self._bindings.output().get_buffer())
        if not len(scores):
            return Detections.empty()
        # The chip's NMS gives [ymin, xmin, ymax, xmax], normalised to the letterboxed input.
        s = self.imgsz
        xyxy = np.stack([boxes[:, 1] * s, boxes[:, 0] * s, boxes[:, 3] * s, boxes[:, 2] * s], axis=1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
        h, w = frame_bgr.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        return self.filter.apply(Detections(xyxy.astype(np.float32), scores.astype(np.float32), classes.astype(np.int32)))

    def _parse_nms(self, out):
        """Unpack HailoRT's by-class NMS output.

        The Python bindings normally return a list with one (n_i, 5) array per class.
        This also accepts the raw flat layout: for each class, a count followed by
        max_boxes rows of [ymin, xmin, ymax, xmax, score].
        UNVERIFIED on real hardware: this came from the docs and Hailo's examples. Check
        it against a known frame on the Pi (see docs/DEPLOY_PI.md, step 4).
        """
        boxes, scores, classes = [], [], []
        if isinstance(out, (list, tuple)):
            per_class = out[0] if len(out) == 1 and isinstance(out[0], (list, tuple)) else out
            for c, dets in enumerate(per_class):
                dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
                if len(dets):
                    boxes.append(dets[:, :4]); scores.append(dets[:, 4]); classes.append(np.full(len(dets), c))
        else:
            flat = np.asarray(out, dtype=np.float32).ravel()
            n_cls = len(self.names)
            max_boxes = (len(flat) // n_cls - 1) // 5
            for c in range(n_cls):
                base = c * (1 + 5 * max_boxes)
                n = int(flat[base])
                if n:
                    dets = flat[base + 1: base + 1 + 5 * n].reshape(n, 5)
                    boxes.append(dets[:, :4]); scores.append(dets[:, 4]); classes.append(np.full(n, c))
        if not scores:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)

    def close(self):
        self._configured_ctx.__exit__(None, None, None)
        self._vdevice.release()


class UltralyticsDetector:
    def __init__(self, weights, imgsz=640, conf=0.1, exclude=(), device=None, half=True):
        import torchvision
        from ultralytics import YOLO
        from ultralytics.utils.nms import TorchNMS

        # The Jetson's torchvision build has broken compiled ops (see track.py).
        torchvision.ops.nms = TorchNMS.nms
        weights = Path(weights)
        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() and weights.suffix != ".onnx" else "cpu"
        self.model = YOLO(str(weights), task="detect")
        self.imgsz = imgsz
        self.kwargs = dict(imgsz=imgsz, conf=conf, device=device, half=bool(half and device != "cpu"), verbose=False)
        names = self.model.names
        self.filter = _ClassFilter([names[i] for i in sorted(names)], exclude, conf)
        self.names = self.filter.names

    def __call__(self, frame_bgr):
        b = self.model.predict(frame_bgr, **self.kwargs)[0].boxes
        return self.filter.apply(Detections(
            b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(np.int32)))

    def close(self):
        pass


def make_detector(model_path, imgsz=640, conf=0.1, exclude=()):
    if Path(model_path).suffix == ".hef":
        return HailoDetector(model_path, conf=conf, exclude=exclude)
    return UltralyticsDetector(model_path, imgsz=imgsz, conf=conf, exclude=exclude)
