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
from .commands import Controller
from .detector import make_detector
from .render import COLOR_NAMES, OverlayConfig, draw
from .scene_track import GATE_SIZES
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
    ap.add_argument("--lang", choices=("tr", "en"), default="en", help="starting overlay language (changeable in flight)")
    ap.add_argument("--overlay-color", choices=COLOR_NAMES, default="green",
                    help="starting symbology colour (changeable in flight)")
    ap.add_argument("--gate", choices=[g.lower() for g in GATE_SIZES], default="m",
                    help="starting scene-track gate size (changeable in flight)")
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

    if args.source == "picam":
        from .camera import PiCameraSource
        cam = PiCameraSource(args.size, args.fps, args.camera_num)
    else:
        from .camera import VideoSource
        cam = VideoSource(args.source, args.size, realtime=not args.no_realtime, loop=args.loop)

    tracker = ByteTracker(buffer_frames=max(30, args.fps))
    lock = TargetLock(args.size, args.hfov, args.vfov, coast_s=args.coast, gate=args.gate.upper())
    overlay = OverlayConfig(lang=args.lang, color_idx=COLOR_NAMES.index(args.overlay_color))
    ctl = Controller(lock, tracker, overlay, detector.names)

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
        from .video_out import WindowSink, KEY_HELP
        window = WindowSink(f"ADTS - {KEY_HELP}", lambda kind, arg: local_cmds.put((kind, arg, None)))
        sinks.append(window)

    seq, frame_idx = 0, 0
    tracks = []
    ema_fps = 0.0
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
            if not ctl.ai_on:
                # Detection switched off: the Hailo (or the GPU) stays idle, and nothing is
                # drawn but the scene track, if one is running.
                t1, tracks = t0, []
            elif frame_idx % args.detect_stride == 0:
                dets = detector(frame)
                t1 = time.monotonic()
                tracks = tracker.update(dets)
            else:
                t1 = t0
                tracks = tracker.predict_only()
            t2 = time.monotonic()
            frame_idx += 1
            lock.refresh_candidates(tracks)

            # Commands run after tracking so they act on the tracks the operator sees now, and
            # BEFORE draw() so a scene track starts on a clean frame, not on our own symbology.
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
                ok = ctl.run(kind, arg, tracks, frame)
                print(f"command {kind} {arg if arg is not None else ''} -> {'OK' if ok else 'FAILED'} ({lock.state} {lock.track_id})")
                if mc is not None:
                    mc.done(ok)

            lock.update(tracker, tracker.frame_id, frame)
            if mav is not None:
                mav.set_status(lock.state, lock.normalized_box(), lock.angle_error())

            now = time.monotonic()
            dt, t_prev = now - t_prev, now
            ema_fps = (1 / dt if ema_fps == 0 else 0.9 * ema_fps + 0.1 / dt) if dt > 0 else ema_fps
            if now - t_temp > 2:
                temp, t_temp = cpu_temp(), now
            draw(frame, tracks, lock, ctl.names, {
                "fps": ema_fps, "temp_c": temp, "ai": ctl.ai_on,
                "mav": mav.connected if mav else None, "rec": rec_path is not None}, overlay)
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
