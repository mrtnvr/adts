"""Sabit sahne/obje takibi: a classical CSRT tracker locked onto whatever sits inside the
centre gate, with no detector involved.

The operator aims the gate (S/M/L) at a patch of the scene and starts tracking; from then on
OpenCV follows that patch by appearance alone. This is the mode to use when the target isn't
one of WALDO's 12 classes, or when AI detection is switched off entirely.

An earlier version of the project (track.py) used CSRT as THE tracker and turned detection
off per target. Here it is a second, independent mode next to YOLO + ByteTrack: which one
holds the lock is TargetLock.mode.
"""

import cv2

GATE_SIZES = ("S", "M", "L")
# Gate side as a fraction of frame WIDTH (square gate, so S/M/L mean the same angular size
# regardless of aspect ratio).
GATE_FRACTION = {"S": 0.08, "M": 0.15, "L": 0.25}


def _new_csrt():
    """CSRT moved from opencv-contrib's legacy API into the main namespace in OpenCV 4.5.1.
    The Pi runs apt's python3-opencv and the Jetson opencv-contrib-python, so accept both."""
    for factory in (getattr(cv2, "TrackerCSRT_create", None),
                    getattr(getattr(cv2, "TrackerCSRT", None), "create", None),
                    getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)):
        if factory is not None:
            return factory()
    raise RuntimeError("this OpenCV build has no CSRT tracker; needs OpenCV >= 4.5.1 or opencv-contrib")


class SceneTracker:
    def __init__(self, frame_size, gate="M"):
        self.w, self.h = frame_size
        self.gate = gate if gate in GATE_FRACTION else "M"
        self._tracker = None

    @property
    def active(self):
        return self._tracker is not None

    def set_gate(self, gate):
        """Gate size only decides what a future start() grabs, so changing it mid-track is
        harmless and leaves the current lock alone."""
        if gate not in GATE_FRACTION:
            return False
        self.gate = gate
        return True

    def step_gate(self, step=1):
        return self.set_gate(GATE_SIZES[(GATE_SIZES.index(self.gate) + step) % len(GATE_SIZES)])

    def gate_box(self):
        side = GATE_FRACTION[self.gate] * self.w
        cx, cy = self.w / 2, self.h / 2
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    def start(self, frame):
        """Lock the gate contents. `frame` must be the CLEAN frame: started on a rendered one,
        CSRT would happily track our own brackets."""
        if frame is None:
            return False
        x1, y1, x2, y2 = (int(v) for v in self.gate_box())
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return False
        self._tracker = _new_csrt()
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        return True

    def update(self, frame):
        """-> (ok, xyxy). ok False once CSRT loses confidence; the caller's coast/lost timers
        take it from there."""
        if self._tracker is None or frame is None:
            return False, None
        ok, box = self._tracker.update(frame)
        if not ok:
            return False, None
        x, y, w, h = box
        return True, (float(x), float(y), float(x + w), float(y + h))

    def stop(self):
        self._tracker = None
