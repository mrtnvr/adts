#!/usr/bin/env python3
"""Stand-in for a GCS or flight controller, to test the tracker's MAVLink interface
without ArduPilot. Sends heartbeats and tracking commands, and prints the ACKs and
tracking status that come back. See gcs_link.py for the connection/command plumbing this
shares with tools/gcs_gui.py, and tools/gcs_gui.py itself for a point-and-click version of
the same thing.

    python3 -m adts ... --mav udpin:0.0.0.0:14555        # tracker listens
    python3 tools/gcs_sim.py udpout:127.0.0.1:14555       # this connects

Interactive commands:
    yapay zeka takip : next | prev | engage | select ID | auto | cancel | ai on|off|toggle
    sabit sahne takip: scene | scene stop | gate s|m|l|next
    overlay          : overlay on|off|toggle | reticle on|off|toggle | lang tr|en|next
                       color green|blue|red|white|black|next
    dogrudan         : point X Y | rect X1 Y1 X2 Y2 (normalised 0..1) | stop | info | quit

next/prev only MOVE the highlight between the five detections nearest the crosshair;
engage is what actually starts tracking the highlighted one.

Or run a script:  --script "auto; wait 3; next; wait 2; stop"
"""

import argparse
import sys
import time

from gcs_link import COLORS, GATES, LANGS, ONOFF, GcsLink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="udpout:host:port, tcp:host:port, or a serial device (e.g. /dev/ttyUSB0, COM3)")
    ap.add_argument("--sysid", type=int, default=1, help="tracker's system ID")
    ap.add_argument("--baud", type=int, default=921600, help="serial baud rate; ignored for udp/tcp")
    ap.add_argument("--script", default=None)
    ap.add_argument("--quiet-status", action="store_true", help="don't print TRK_ERR updates")
    args = ap.parse_args()

    stats = {"status": 0, "ack": 0}
    last_print = [0.0]

    def on_event(kind, data):
        if kind == "ack":
            stats["ack"] += 1
            print(f"  ACK cmd={data[0]} result={data[1]}")
        elif kind == "camera_info":
            vendor, w, h, flags = data
            print(f"  CAMERA_INFORMATION {vendor} {w}x{h} flags={flags}")
        elif kind == "track":
            stats["status"] += 1
            if not args.quiet_status and time.time() - last_print[0] > 0.5:
                last_print[0] = time.time()
                state, az, el = data
                print(f"  TRK {state:6s} az={az:+6.2f} el={el:+6.2f}")
        elif kind == "rect" and data is not None and not args.quiet_status:
            if time.time() - last_print[0] > 0.45:
                x1, y1, x2, y2 = data
                print(f"  RECT ({x1:.3f},{y1:.3f})-({x2:.3f},{y2:.3f})")
        elif kind == "error":
            print(f"  link error: {data}")

    link = GcsLink(args.url, sysid=args.sysid, baud=args.baud, on_event=on_event)

    def run(line):
        parts = line.split()
        if not parts:
            return True
        op, rest = parts[0], parts[1:]
        print(f"> {line}")

        def nums():
            return [float(x) for x in rest]

        def pick(table):
            key = rest[0].lower() if rest else "next"
            if key not in table:
                print(f"  bad argument: {rest[0]}")
                return None
            return table[key]

        if op == "auto":
            link.auto()
        elif op == "point":
            link.point(*nums())
        elif op == "rect":
            link.rect(*nums()[:4])
        elif op == "next":
            link.cand_next()
        elif op == "prev":
            link.cand_prev()
        elif op == "select":
            link.select(nums()[0])
        elif op == "engage":
            link.engage()
        elif op == "cancel":
            link.cancel()
        elif op == "ai":
            v = pick(ONOFF)
            if v is not None:
                link.ai(v)
        elif op == "scene":
            (link.scene_stop if rest and rest[0] == "stop" else link.scene_start)()
        elif op == "gate":
            v = pick(GATES)
            if v is not None:
                link.gate(v)
        elif op in ("overlay", "reticle"):
            v = pick(ONOFF)
            if v is not None:
                (link.overlay if op == "overlay" else link.reticle)(v)
        elif op == "lang":
            v = pick(LANGS)
            if v is not None:
                link.lang(v)
        elif op == "color":
            v = pick(COLORS)
            if v is not None:
                link.color(v)
        elif op == "stop":
            link.stop()
        elif op == "info":
            link.info()
        elif op == "wait":
            time.sleep(nums()[0])
        elif op == "quit":
            return False
        else:
            print("  unknown command")
        return True

    time.sleep(1.5)  # let the link register with the tracker
    if args.script:
        for line in args.script.split(";"):
            run(line.strip())
        time.sleep(1)
        print(f"summary: {stats['ack']} ACKs, {stats['status']} TRK_ERR status messages")
        return
    for line in sys.stdin:
        if not run(line.strip()):
            break


if __name__ == "__main__":
    main()
