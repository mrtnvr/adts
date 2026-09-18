"""Output sinks for the labelled video.

GstSink starts one gst-launch-1.0 subprocess and pipes raw BGR frames to its stdin.
This works with any OpenCV build (the pip wheels have no GStreamer), and the same
command line runs on the Pi and the Jetson:
    HDMI  -> kmssink. Needs no desktop session: boot to console, or the desktop holds the display.
    UDP   -> x264enc (software: the Pi 5 has no H.264 encoder) -> RTP -> udpsink
    file  -> the same encoder output -> MPEG-TS. Unlike MP4, a TS file stays playable after a
             sudden power cut, and power cuts are normal on a drone.
Frames go through a one-slot queue on a writer thread, so a slow encoder drops frames
instead of stalling the tracker.
"""

import queue
import shlex
import subprocess
import threading

import cv2


class GstSink:
    def __init__(self, size, fps=30, hdmi=False, udp=None, record=None, bitrate_kbps=3000):
        w, h = size
        self.frame_bytes = w * h * 3
        # do-timestamp + framerate=0/1 (variable rate): stamp each frame with the time it
        # arrives. A fixed rate would make recordings play too fast whenever the tracker
        # runs below `fps`. rawvideoparse re-frames the stream, since a pipe read can return
        # part of a frame.
        src = (f"fdsrc fd=0 do-timestamp=true blocksize={self.frame_bytes} ! "
               f"rawvideoparse use-sink-caps=false width={w} height={h} format=bgr framerate=0/1 ! "
               f"queue leaky=downstream max-size-buffers=2 ! videoconvert ! tee name=raw")
        branches = []
        if hdmi:
            branches.append("raw. ! queue leaky=downstream max-size-buffers=1 ! kmssink sync=false")
        if udp or record:
            enc = (f"raw. ! queue leaky=downstream max-size-buffers=2 ! video/x-raw,format=I420 ! "
                   f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} "
                   f"key-int-max={int(fps)} threads=2 ! video/x-h264,profile=baseline ! h264parse config-interval=1 ! tee name=h264")
            branches.append(enc)
            if udp:
                host, port = udp.rsplit(":", 1)
                branches.append(f"h264. ! queue ! rtph264pay pt=96 config-interval=1 ! udpsink host={host} port={port} sync=false async=false")
            if record:
                branches.append(f"h264. ! queue ! mpegtsmux ! filesink location={shlex.quote(str(record))} sync=false async=false")
        if not branches:
            raise ValueError("GstSink needs at least one of hdmi/udp/record")
        self.pipeline = " ".join([src] + branches)
        self.proc = subprocess.Popen(["gst-launch-1.0", "-q", "-e"] + shlex.split(self.pipeline),
                                     stdin=subprocess.PIPE, bufsize=0)
        self._q = queue.Queue(maxsize=1)
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="gst-out")
        self._thread.start()

    def submit(self, frame):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(frame)

    def _run(self):
        while True:
            frame = self._q.get()
            if frame is None:
                break
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError):
                print("video_out: gst-launch exited; pipeline was:\n  " + self.pipeline)
                break

    def close(self):
        try:
            self._q.get_nowait()  # drop any pending frame so the stop marker always fits
        except queue.Empty:
            pass
        self._q.put(None)
        self._thread.join(timeout=2)
        try:
            self.proc.stdin.close()  # EOF -> with -e, gst sends EOS so the TS file is finalised
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# Every operator command has a key here, so the whole MAVLink command set can be exercised
# on the dev machine without a GCS. Enter arrives as 13 or 10 depending on the platform.
KEYS = {
    ord("s"): ("auto", None), ord("x"): ("stop", None),
    ord("n"): ("cand", 1), ord("p"): ("cand", -1),
    13: ("engage", None), 10: ("engage", None),
    ord("g"): ("scene_start", None), ord("b"): ("scene_stop", None), ord("k"): ("gate", -1),
    ord("d"): ("ai", -1), ord("o"): ("overlay", -1), ord("r"): ("reticle", -1),
    ord("l"): ("lang", -1), ord("c"): ("color", -1),
}
KEY_HELP = ("click=track, n/p=sel, Enter=engage, s=auto, x=stop, g/b=scene on/off, k=gate, "
            "d=AI, o/r=overlay/reticle, l=lang, c=color, q=quit")


class WindowSink:
    """Dev-only cv2 window. Keys and the mouse become the same commands MAVLink sends:
    left-click = track point, right-click = stop, and KEYS above for the rest."""

    def __init__(self, title, on_command):
        self.title, self.on_command = title, on_command
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, self._mouse)
        self.quit = False

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_command("point", (x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.on_command("stop", None)

    def submit(self, frame):
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.quit = True
        elif key in KEYS:
            self.on_command(*KEYS[key])

    def close(self):
        cv2.destroyWindow(self.title)
