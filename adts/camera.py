"""Frame sources. Each runs its own capture thread and hands out only the newest frame.

If detection falls behind the camera, old frames are dropped, not queued. On a drone,
latency matters more than processing every frame.
"""

import threading
import time

import cv2


class _LatestFrame:
    def __init__(self):
        self._cond = threading.Condition()
        self._frame, self._seq, self._t = None, 0, 0.0
        self.running = True
        self.error = None

    def _publish(self, frame, t):
        with self._cond:
            self._frame, self._t = frame, t
            self._seq += 1
            self._cond.notify_all()

    def read(self, last_seq=0, timeout=2.0):
        """Wait for a frame newer than last_seq. Returns (frame, t, seq), or (None, 0, last_seq) at end/timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or not self.running, timeout)
            if self._seq <= last_seq:
                return None, 0.0, last_seq
            return self._frame, self._t, self._seq

    def _finish(self, error=None):
        self.error = error
        with self._cond:
            self.running = False
            self._cond.notify_all()


class PiCameraSource(_LatestFrame):
    """CSI camera through picamera2/libcamera. The Pi's hardware ISP does debayering
    and scaling, so frames arrive already at `size` with no CPU cost.
    "RGB888" in picamera2 means BGR byte order, which is what OpenCV expects."""

    def __init__(self, size=(1280, 720), fps=30, camera_num=0):
        super().__init__()
        from picamera2 import Picamera2  # only on the Pi (apt: python3-picamera2)

        self.size = size
        self.cam = Picamera2(camera_num)
        cfg = self.cam.create_video_configuration(
            main={"size": size, "format": "RGB888"}, buffer_count=4,
            controls={"FrameRate": fps})
        self.cam.configure(cfg)
        self.cam.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="picam")
        self._thread.start()

    def _run(self):
        try:
            while self.running:
                self._publish(self.cam.capture_array("main"), time.monotonic())
        except Exception as e:  # camera unplugged / libcamera error
            self._finish(e)

    def close(self):
        self.running = False
        self._thread.join(timeout=2)
        self.cam.stop()
        self.cam.close()


class VideoSource(_LatestFrame):
    """Video file, RTSP URL or webcam index, resized to `size`.

    With realtime=True (the default) a file plays at its own frame rate and drops frames
    the pipeline can't keep up with, like a live camera would. realtime=False hands out
    every frame for offline evaluation.
    """

    def __init__(self, source, size=(1280, 720), realtime=True, loop=False):
        super().__init__()
        self.size, self.realtime, self.loop = size, realtime, loop
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._consumed = threading.Event()
        self._consumed.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="video")
        self._thread.start()

    def read(self, last_seq=0, timeout=2.0):
        out = super().read(last_seq, timeout)
        self._consumed.set()
        return out

    def _run(self):
        t_next = time.monotonic()
        while self.running:
            if not self.realtime:
                self._consumed.wait()
                self._consumed.clear()
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    continue
                break
            if (frame.shape[1], frame.shape[0]) != tuple(self.size):
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            if self.realtime:
                t_next += 1.0 / self.fps
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.monotonic()
            self._publish(frame, time.monotonic())
        self._finish()

    def close(self):
        self.running = False
        self._consumed.set()
        self._thread.join(timeout=2)
        self.cap.release()
