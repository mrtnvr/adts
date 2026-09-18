"""Shared MAVLink link for the GCS test stand-ins in this folder (gcs_sim.py, gcs_gui.py):
opens a connection - a serial device (a UART-TTL USB adapter wired to the flight
controller's telemetry port, or straight into the Pi's UART for a benchtop test) or a
udp/tcp URL for dev-machine testing without hardware - sends heartbeats, and turns each
operator action into the COMMAND_LONG ADTS expects. See adts/mavlink_io.py for what these
commands mean on the receiving end; this module only builds and sends them.

Pretends to be the vehicle's autopilot (component 1) on `sysid`, the same role ArduPilot
plays for real, so ADTS's compid-100 camera component accepts its commands.
"""

import math
import os
import threading
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

M = mavutil.mavlink
CAM = M.MAV_COMP_ID_CAMERA
STATE_NAMES = ("IDLE", "LOCKED", "COAST", "LOST")

# p2 conventions from adts/mavlink_io.py: an on/off setting takes one of ONOFF's values, a
# multi-value setting one of its own table's values - -1 means "next" in both.
ONOFF = {"off": 0, "on": 1, "toggle": -1}
GATES = {"s": 0, "m": 1, "l": 2, "next": -1}
LANGS = {"tr": 0, "en": 1, "next": -1}
COLORS = {"green": 0, "blue": 1, "red": 2, "white": 3, "black": 4, "next": -1}


class GcsLink:
    """Connects, sends 1 Hz heartbeats, and reports incoming messages through `on_event`
    (called from the rx thread - callers marshal onto their own thread/loop if needed):

        on_event("ack",     (command_id, "ACCEPTED" | "FAILED" | ...))
        on_event("track",   (state_name, azimuth_deg, elevation_deg))
        on_event("rect",    (x1, y1, x2, y2) | None)   # normalised 0..1, None = no target
        on_event("camera_info", (vendor, width, height, flags))
        on_event("error",   message)
    """

    def __init__(self, url, sysid=1, baud=921600, on_event=None):
        self.on_event = on_event or (lambda kind, data: None)
        self.sysid = sysid
        self.conn = mavutil.mavlink_connection(url, baud=baud, source_system=sysid, source_component=1)
        self.running = True
        self._tx_lock = threading.Lock()
        for fn in (self._rx, self._hb):
            threading.Thread(target=fn, daemon=True).start()

    def close(self):
        self.running = False
        self.conn.close()

    # ---- receive ----------------------------------------------------------
    def _rx(self):
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                self.on_event("error", str(e))
                time.sleep(0.5)
                continue
            if msg is None:
                continue
            t = msg.get_type()
            if t == "COMMAND_ACK":
                self.on_event("ack", (msg.command, M.enums["MAV_RESULT"][msg.result].name))
            elif t == "CAMERA_INFORMATION":
                vendor = bytes(msg.vendor_name).rstrip(b"\0").decode(errors="replace")
                self.on_event("camera_info", (vendor, msg.resolution_h, msg.resolution_v, msg.flags))
            elif t == "DEBUG_VECT" and msg.name.startswith("TRK_ERR"):
                self.on_event("track", (STATE_NAMES[int(msg.z)], msg.x, msg.y))
            elif t == "CAMERA_TRACKING_IMAGE_STATUS":
                active = msg.tracking_status == 1 and not math.isnan(msg.rec_top_x)
                self.on_event("rect", (msg.rec_top_x, msg.rec_top_y, msg.rec_bottom_x, msg.rec_bottom_y)
                             if active else None)

    def _hb(self):
        while self.running:
            self._send(lambda: self.conn.mav.heartbeat_send(
                M.MAV_TYPE_QUADROTOR, M.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, M.MAV_STATE_ACTIVE))
            time.sleep(1)

    # ---- transmit -----------------------------------------------------------
    def _send(self, fn):
        with self._tx_lock:
            try:
                fn()
            except Exception as e:
                self.on_event("error", str(e))

    def _cmd(self, c, *p):
        p = list(p) + [0] * (7 - len(p))
        self._send(lambda: self.conn.mav.command_long_send(self.sysid, CAM, c, 0, *p))

    # ---- direct camera tracking (standard MAVLink) ---------------------------
    def point(self, x, y):
        self._cmd(M.MAV_CMD_CAMERA_TRACK_POINT, x, y, 0.02)

    def rect(self, x1, y1, x2, y2):
        self._cmd(M.MAV_CMD_CAMERA_TRACK_RECTANGLE, x1, y1, x2, y2)

    def stop(self):
        self._cmd(M.MAV_CMD_CAMERA_STOP_TRACKING)

    def info(self):
        self._cmd(M.MAV_CMD_REQUEST_MESSAGE, M.MAVLINK_MSG_ID_CAMERA_INFORMATION)

    # ---- yapay zeka takip: MAV_CMD_USER_1 ------------------------------------
    def cand_next(self):
        self._cmd(M.MAV_CMD_USER_1, 1)

    def cand_prev(self):
        self._cmd(M.MAV_CMD_USER_1, -1)

    def select(self, track_id):
        self._cmd(M.MAV_CMD_USER_1, 0, track_id)

    def auto(self):
        self._cmd(M.MAV_CMD_USER_1, 2)

    def engage(self):
        self._cmd(M.MAV_CMD_USER_1, 3)

    def cancel(self):
        self._cmd(M.MAV_CMD_USER_1, 4)

    def ai(self, value):
        self._cmd(M.MAV_CMD_USER_1, 5, value)

    # ---- sabit sahne/obje takip: MAV_CMD_USER_2 ------------------------------
    def scene_start(self):
        self._cmd(M.MAV_CMD_USER_2, 1)

    def scene_stop(self):
        self._cmd(M.MAV_CMD_USER_2, 0)

    def gate(self, value):
        self._cmd(M.MAV_CMD_USER_2, 2, value)

    # ---- overlay: MAV_CMD_USER_3 ----------------------------------------------
    def overlay(self, value):
        self._cmd(M.MAV_CMD_USER_3, 1, value)

    def reticle(self, value):
        self._cmd(M.MAV_CMD_USER_3, 2, value)

    def lang(self, value):
        self._cmd(M.MAV_CMD_USER_3, 3, value)

    def color(self, value):
        self._cmd(M.MAV_CMD_USER_3, 4, value)
