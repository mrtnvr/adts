"""ADTS onboard tracker main loop.

Pi 5 + Hailo-8L on the drone:
    python3 -m adts --source picam --model models/waldo_yolov8m_640.hef \\
        --mav /dev/ttyAMA0 --hdmi --record /home/pi/rec

Dev on the Jetson (video file, cv2 window with keys/mouse, MAVLink over UDP):
    python3 -m adts --source videos/ankara_drone_test.mp4 \\
        --model weights/waldo/WALDO30_yolov8m_640x640.pt --window --mav udpin:0.0.0.0:14555

Threads: camera capture (latest frame only), MAVLink rx/tx, video writer. The main
loop runs command -> detect -> track -> lock -> render -> output.
"""

import argparse
import queue
import shutil
import time
from pathlib import Path

from .bytetrack import ByteTracker
from .classes import display_names
from .detector import make_detector
from .render import draw
from .target import TargetLock


def parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def cpu_temp():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="picam", help="'picam' (CSI camera), a video file/URL, or a webcam index")
    ap.add_argument("--camera-num", type=int, default=0, help="CSI port for --source picam (Pi 5 has two)")
    ap.add_argument("--size", type=parse_size, default=(1280, 720), help="processing + output resolution, WxH")
    ap.add_argument("--fps", type=int, default=30, help="camera frame rate (picam) and output stream rate")
    ap.add_argument("--model", required=True, help=".hef (Hailo) or .pt/.engine/.onnx (Ultralytics, dev)")
    ap.add_argument("--imgsz", type=int, default=640, help="model input for Ultralytics models (a .hef has its own)")
    ap.add_argument("--conf", type=float, default=0.1, help="detector floor; ByteTrack uses 0.1-0.25 as its low-score band")
    ap.add_argument("--exclude-classes", default="UPole", help="comma-separated WALDO class names to drop")
    ap.add_argument("--lang", choices=("tr", "en"), default="tr", help="class label language on the video")
    ap.add_argument("--hfov", type=float, default=66.0,
                    help="camera horizontal FOV, degrees (Camera Module 3 = 66, CM3 Wide = 102); used for the angle error")
    ap.add_argument("--vfov", type=float, default=None, help="vertical FOV; derived from --hfov and aspect if omitted")
    ap.add_argument("--coast", type=float, default=1.5, help="seconds to predict a lost target before declaring LOST")
    ap.add_argument("--detect-stride", type=int, default=1,
                    help="run the detector every Nth frame, Kalman-predict in between (1 = every frame)")
    ap.add_argument("--mav", default=None, help="MAVLink connection: /dev/ttyAMA0, udpin:0.0.0.0:14555, tcp:host:port ...")
    ap.add_argument("--mav-baud", type=int, default=921600)
    ap.add_argument("--sysid", type=int, default=1, help="vehicle MAVLink system ID (the camera joins it as compid 100)")
    ap.add_argument("--hdmi", action="store_true", help="labelled video to HDMI (kmssink; console boot, no desktop)")
    ap.add_argument("--udp", default=None, metavar="HOST:PORT", help="labelled video as H.264 RTP to HOST:PORT")
    ap.add_argument("--record", default=None, metavar="DIR", help="record labelled video (.ts) into DIR")
    ap.add_argument("--window", action="store_true", help="dev: show a cv2 window with key/mouse control")
    ap.add_argument("--no-realtime", action="store_true", help="file sources: process every frame instead of dropping")
    ap.add_argument("--loop", action="store_true", help="file sources: loop forever")
    ap.add_argument("--profile", action="store_true", help="print per-stage timings every second")
    args = ap.parse_args()

    detector = make_detector(args.model, imgsz=args.imgsz, conf=args.conf, exclude=args.exclude_classes.split(","))
    names = display_names(detector.names, args.lang)

    if args.source == "picam":
        from .camera import PiCameraSource
        cam = PiCameraSource(args.size, args.fps, args.camera_num)
    else:
        from .camera import VideoSource
        cam = VideoSource(args.source, args.size, realtime=not args.no_realtime, loop=args.loop)

    tracker = ByteTracker(buffer_frames=max(30, args.fps))
    lock = TargetLock(args.size, args.hfov, args.vfov, coast_s=args.coast)

    local_cmds = queue.Queue()
    mav = None
    if args.mav:
        from .mavlink_io import MavlinkIO
        mav = MavlinkIO(args.mav, args.mav_baud, args.sysid, frame_size=args.size)
        print(f"MAVLink: {args.mav} as sysid {args.sysid} compid 100")

    sinks = []
    rec_path = None
    if args.record:
        Path(args.record).mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(args.record).free / 1e9
        if free_gb < 2:
            # A full SD card mid-flight can break logging and the OS. Refuse to start recording instead.
            print(f"--record: only {free_gb:.1f} GB free in {args.record}, recording DISABLED")
        else:
            rec_path = Path(args.record) / time.strftime("adts_%Y%m%d_%H%M%S.ts")
    if args.hdmi or args.udp or rec_path:
        from .video_out import GstSink
        sinks.append(GstSink(args.size, args.fps, hdmi=args.hdmi, udp=args.udp, record=rec_path))
        print(f"video out: hdmi={args.hdmi} udp={args.udp} record={rec_path}")
    window = None
    if args.window:
        from .video_out import WindowSink
        window = WindowSink("ADTS - click=track, s=auto, n/p=next/prev, x=stop, q=quit",
                            lambda kind, arg: local_cmds.put((kind, arg, None)))
        sinks.append(window)

    def run_command(kind, arg, tracks):
        if kind == "point":
            return lock.start_point(arg[0], arg[1], tracks)
        if kind == "rect":
            return lock.start_rect(arg, tracks)
        if kind == "auto":
            return lock.start_auto(tracks)
        if kind == "cycle":
            return lock.cycle(arg, tracks)
        if kind == "select":
            return lock.select_id(arg, tracks)
        if kind == "stop":
            return lock.stop()
        return False

    seq, frame_idx = 0, 0
    tracks = []
    ema_fps, det_ms = 0.0, 0.0
    prof = {"det": 0.0, "trk": 0.0, "draw": 0.0, "n": 0}
    t_prev = t_report = time.monotonic()
    temp, t_temp = cpu_temp(), 0.0
    try:
        while not (window and window.quit):
            frame, t_cap, seq = cam.read(seq)
            if frame is None:
                if not cam.running:
                    if cam.error:
                        print(f"camera error: {cam.error}")
                    break
                continue
            frame = frame.copy()  # the camera thread may reuse its buffer; drawing needs our own

            t0 = time.monotonic()
            if frame_idx % args.detect_stride == 0:
                dets = detector(frame)
                t1 = time.monotonic()
                tracks = tracker.update(dets)
                det_ms = (t1 - t0) * 1000
            else:
                t1 = t0
                tracks = tracker.predict_only()
            t2 = time.monotonic()
            frame_idx += 1

            # Commands run after tracking so they act on the tracks the operator sees now.
            while True:
                try:
                    kind, arg, mc = local_cmds.get_nowait()
                except queue.Empty:
                    if mav is None:
                        break
                    try:
                        mc = mav.commands.get_nowait()
                    except queue.Empty:
                        break
                    kind, arg = mc.kind, mc.arg
                ok = run_command(kind, arg, tracks)
                print(f"command {kind} {arg if arg is not None else ''} -> {'OK' if ok else 'FAILED'} ({lock.state} {lock.track_id})")
                if mc is not None:
                    mc.done(ok)

            lock.update(tracker, tracker.frame_id)
            if mav is not None:
                mav.set_status(lock.state, lock.normalized_box(), lock.angle_error())

            now = time.monotonic()
            dt, t_prev = now - t_prev, now
            ema_fps = (1 / dt if ema_fps == 0 else 0.9 * ema_fps + 0.1 / dt) if dt > 0 else ema_fps
            if now - t_temp > 2:
                temp, t_temp = cpu_temp(), now
            draw(frame, tracks, lock, names, {
                "fps": ema_fps, "det_ms": det_ms, "temp_c": temp,
                "mav": mav.connected if mav else None, "rec": rec_path is not None})
            t3 = time.monotonic()
            for s in sinks:
                s.submit(frame)

            prof["det"] += t1 - t0; prof["trk"] += t2 - t1; prof["draw"] += t3 - t2; prof["n"] += 1
            if args.profile and now - t_report >= 1.0:
                n = prof["n"]
                print(f"{ema_fps:5.1f} FPS | detect {prof['det'] / n * 1000:5.1f} ms  track {prof['trk'] / n * 1000:4.1f} ms  "
                      f"draw {prof['draw'] / n * 1000:4.1f} ms | latency(capture->out) {(time.monotonic() - t_cap) * 1000:5.1f} ms | "
                      f"{len(tracks)} tracks | {lock.state} {lock.track_id or ''}")
                prof = {"det": 0.0, "trk": 0.0, "draw": 0.0, "n": 0}
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        for s in sinks:
            s.close()
        cam.close()
        if mav:
            mav.close()
        detector.close()


if __name__ == "__main__":
    main()
