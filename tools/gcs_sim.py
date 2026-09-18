#!/usr/bin/env python3
"""Stand-in for a GCS or flight controller, to test the tracker's MAVLink interface
without ArduPilot. Sends heartbeats and tracking commands, and prints the ACKs and
tracking status that come back.

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
import math
import os
import sys
import threading
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

M = mavutil.mavlink
CAM = M.MAV_COMP_ID_CAMERA

ONOFF = {"off": 0, "on": 1, "toggle": -1}
GATES = {"s": 0, "m": 1, "l": 2, "next": -1}
LANGS = {"tr": 0, "en": 1, "next": -1}
COLORS = {"green": 0, "blue": 1, "red": 2, "white": 3, "black": 4, "next": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--sysid", type=int, default=1, help="tracker's system ID")
    ap.add_argument("--script", default=None)
    ap.add_argument("--quiet-status", action="store_true", help="don't print TRK_ERR updates")
    args = ap.parse_args()

    # Pretend to be the autopilot of the same vehicle (compid 1).
    conn = mavutil.mavlink_connection(args.url, source_system=args.sysid, source_component=1)
    stats = {"status": 0, "ack": 0}

    def rx():
        last_print = 0
        while True:
            msg = conn.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "COMMAND_ACK":
                stats["ack"] += 1
                print(f"  ACK cmd={msg.command} result={mavutil.mavlink.enums['MAV_RESULT'][msg.result].name}")
            elif t == "CAMERA_INFORMATION":
                vendor = bytes(msg.vendor_name).rstrip(b"\0").decode()
                print(f"  CAMERA_INFORMATION {vendor} {msg.resolution_h}x{msg.resolution_v} flags={msg.flags}")
            elif t == "DEBUG_VECT" and msg.name.startswith("TRK_ERR"):
                stats["status"] += 1
                if not args.quiet_status and time.time() - last_print > 0.5:
                    last_print = time.time()
                    state = ["IDLE", "LOCKED", "COAST", "LOST"][int(msg.z)]
                    print(f"  TRK {state:6s} az={msg.x:+6.2f} el={msg.y:+6.2f}")
            elif t == "CAMERA_TRACKING_IMAGE_STATUS" and msg.tracking_status == 1 and not args.quiet_status:
                if not math.isnan(msg.rec_top_x) and time.time() - last_print > 0.45:
                    print(f"  RECT ({msg.rec_top_x:.3f},{msg.rec_top_y:.3f})-({msg.rec_bottom_x:.3f},{msg.rec_bottom_y:.3f})")

    def hb():
        while True:
            conn.mav.heartbeat_send(M.MAV_TYPE_QUADROTOR, M.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, M.MAV_STATE_ACTIVE)
            time.sleep(1)

    threading.Thread(target=rx, daemon=True).start()
    threading.Thread(target=hb, daemon=True).start()

    def cmd(c, *p):
        p = list(p) + [0] * (7 - len(p))
        conn.mav.command_long_send(args.sysid, CAM, c, 0, *p)

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
            cmd(M.MAV_CMD_USER_1, 2)
        elif op == "point":
            a = nums()
            cmd(M.MAV_CMD_CAMERA_TRACK_POINT, a[0], a[1], 0.02)
        elif op == "rect":
            cmd(M.MAV_CMD_CAMERA_TRACK_RECTANGLE, *nums()[:4])
        elif op == "next":
            cmd(M.MAV_CMD_USER_1, 1)
        elif op == "prev":
            cmd(M.MAV_CMD_USER_1, -1)
        elif op == "select":
            cmd(M.MAV_CMD_USER_1, 0, nums()[0])
        elif op == "engage":
            cmd(M.MAV_CMD_USER_1, 3)
        elif op == "cancel":
            cmd(M.MAV_CMD_USER_1, 4)
        elif op == "ai":
            v = pick(ONOFF)
            if v is not None:
                cmd(M.MAV_CMD_USER_1, 5, v)
        elif op == "scene":
            cmd(M.MAV_CMD_USER_2, 0 if rest and rest[0] == "stop" else 1)
        elif op == "gate":
            v = pick(GATES)
            if v is not None:
                cmd(M.MAV_CMD_USER_2, 2, v)
        elif op in ("overlay", "reticle"):
            v = pick(ONOFF)
            if v is not None:
                cmd(M.MAV_CMD_USER_3, 1 if op == "overlay" else 2, v)
        elif op == "lang":
            v = pick(LANGS)
            if v is not None:
                cmd(M.MAV_CMD_USER_3, 3, v)
        elif op == "color":
            v = pick(COLORS)
            if v is not None:
                cmd(M.MAV_CMD_USER_3, 4, v)
        elif op == "stop":
            cmd(M.MAV_CMD_CAMERA_STOP_TRACKING)
        elif op == "info":
            cmd(M.MAV_CMD_REQUEST_MESSAGE, M.MAVLINK_MSG_ID_CAMERA_INFORMATION)
        elif op == "wait":
            time.sleep(nums()[0])
        elif op == "quit":
            return False
        else:
            print("  unknown command")
        return True

    time.sleep(1.5)  # let the udpout link register with the tracker
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
