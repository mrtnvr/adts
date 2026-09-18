"""HailoDetector's NMS parsing and letterbox mapping, fed synthetic chip output (no Hailo needed)."""

import numpy as np

from adts.classes import WALDO_NAMES
from adts.detector import HailoDetector, _ClassFilter


class _FakeBindings:
    def __init__(self, out):
        self.out = out

    def input(self):
        return self

    def output(self):
        return self

    def set_buffer(self, buf):
        self.inp = buf

    def get_buffer(self):
        return self.out


class _FakeConfigured:
    def run(self, bindings, timeout):
        pass


def _detector(out):
    d = HailoDetector.__new__(HailoDetector)
    d.imgsz = 640
    d.filter = _ClassFilter(WALDO_NAMES, ["UPole"], 0.1)
    d.names = d.filter.names
    d._bindings, d._configured = _FakeBindings(out), _FakeConfigured()
    return d


# 1280x720 frame -> letterbox scale 0.5, 640x360 content, pad_y = 140
FRAME = np.zeros((720, 1280, 3), np.uint8)
# a Person at frame box (400, 300)-(480, 420) -> input (200, 290)-(240, 350) -> normalised [ymin, xmin, ymax, xmax]
PERSON = [290 / 640, 200 / 640, 350 / 640, 240 / 640, 0.8]


def _per_class(**rows):
    return [np.array(rows.get(n, np.zeros((0, 5))), np.float32).reshape(-1, 5) for n in WALDO_NAMES]


def test_list_layout_maps_back_to_frame_pixels():
    d = _detector(_per_class(Person=[PERSON]))(FRAME)
    assert len(d) == 1 and WALDO_NAMES[d.cls[0]] == "Person"
    np.testing.assert_allclose(d.xyxy[0], [400, 300, 480, 420], atol=0.5)


def test_batch_wrapped_list_layout():
    d = _detector([_per_class(Person=[PERSON])])(FRAME)
    assert len(d) == 1


def test_flat_layout_and_class_filter():
    max_boxes = 3
    flat = np.zeros(len(WALDO_NAMES) * (1 + 5 * max_boxes), np.float32)
    for c, row in ((1, PERSON), (3, PERSON)):  # Person + UPole (excluded)
        base = c * (1 + 5 * max_boxes)
        flat[base] = 1
        flat[base + 1: base + 6] = row
    d = _detector(flat)(FRAME)
    assert [WALDO_NAMES[c] for c in d.cls] == ["Person"]
    np.testing.assert_allclose(d.xyxy[0], [400, 300, 480, 420], atol=0.5)


def test_empty():
    assert len(_detector(_per_class())(FRAME)) == 0
