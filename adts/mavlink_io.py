"""MAVLink link to the flight controller (ArduPilot) or a GCS.

The Pi shows up as a MAVLink camera component (compid 100, MAV_COMP_ID_CAMERA) on the
vehicle's system ID, and uses the standard camera tracking protocol plus three
ADTS-specific MAV_CMD_USER_* commands, because standard MAVLink has no command for
selecting a target, switching detection off or configuring symbology.

Two parameter conventions run through the USER commands: for an on/off setting, p2 is
0 = off, 1 = on, -1 = toggle; for a multi-value setting, p2 is the index, or -1 for the
next value round (so a GCS with one button per function still reaches every value).

  in:  MAV_CMD_CAMERA_TRACK_POINT      (2004) p1,p2 = x,y normalised 0..1   -> lock the detection there
       MAV_CMD_CAMERA_TRACK_RECTANGLE  (2005) p1..p4 = x1,y1,x2,y2 0..1     -> lock the best-overlapping detection
       MAV_CMD_CAMERA_STOP_TRACKING    (2010)                               -> stop
       MAV_CMD_USER_1 (31010) yapay zeka takip:
                              p1 = +1 / -1        secili hedef ILERI / GERI (moves the highlight only)
                              p1 = 0, p2 = ID     highlight that track ID
                              p1 = 2              highlight+lock the detection nearest the crosshair
                              p1 = 3              secili hedef takip BASLAT
                              p1 = 4              takip IPTAL
                              p1 = 5, p2          yapay zeka tespit ACIK/KAPALI
       MAV_CMD_USER_2 (31011) sabit sahne/obje takip:
                              p1 = 1              takip BASLAT (lock the gate contents, no detector)
                              p1 = 0              takip IPTAL
                              p1 = 2, p2          kapi boyutu 0=S 1=M 2=L
       MAV_CMD_USER_3 (31012) overlay:
                              p1 = 1, p2          overlay ACIK/KAPALI
                              p1 = 2, p2          reticle ACIK/KAPALI
                              p1 = 3, p2          dil 0=tr 1=en
                              p1 = 4, p2          renk 0=yesil 1=mavi 2=kirmizi 3=beyaz 4=siyah
       MAV_CMD_REQUEST_MESSAGE (512) for CAMERA_INFORMATION (259)
  out: HEARTBEAT (1 Hz, MAV_TYPE_CAMERA)
       CAMERA_TRACKING_IMAGE_STATUS (status_hz): target rectangle, normalised
       DEBUG_VECT name "TRK_ERR" (status_hz): x = azimuth deg, y = elevation deg, z = state (0 idle, 1 locked, 2 coast, 3 lost)
       COMMAND_ACK for every command, with the real result (ACCEPTED / FAILED)

The status output is the same whichever tracking mode holds the lock, so the autopilot
side never needs to know whether a detection or a scene patch is being followed.

This only REPORTS where the target is. It never commands the gimbal or the vehicle:
what ArduPilot does with the data (Lua script, gimbal, GUIDED) is set up on the
autopilot side and enabled separately.
"""

import os
import queue
import threading
import time

# The tracking messages are MAVLink 2 only. pymavlink picks v1 unless this is set before import.
os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

M = mavutil.mavlink
STATE_CODE = {"IDLE": 0, "LOCKED": 1, "COAST": 2, "LOST": 3}
TRACK_CMDS = {M.MAV_CMD_CAMERA_TRACK_POINT, M.MAV_CMD_CAMERA_TRACK_RECTANGLE,
              M.MAV_CMD_CAMERA_STOP_TRACKING, M.MAV_CMD_USER_1, M.MAV_CMD_USER_2, M.MAV_CMD_USER_3}
OVERLAY_SUBS = {1: "overlay", 2: "reticle", 3: "lang", 4: "color"}


def decode_user_command(cmd, p1, p2):
    """Map a MAV_CMD_USER_* command onto the internal (kind, arg) pair commands.py runs.
    Returns None for an unknown sub-command, which becomes a FAILED ack. Kept free of
    any connection state so the wire protocol can be tested without a link."""
    p1, p2 = int(p1), int(p2)
    if cmd == M.MAV_CMD_USER_1:
        if p1 in (1, -1):
            return "cand", p1
        return {0: ("select", p2), 2: ("auto", None), 3: ("engage", None),
                4: ("stop", None), 5: ("ai", p2)}.get(p1)
    if cmd == M.MAV_CMD_USER_2:
        return {1: ("scene_start", None), 0: ("scene_stop", None), 2: ("gate", p2)}.get(p1)
    if cmd == M.MAV_CMD_USER_3:
        sub = OVERLAY_SUBS.get(p1)
        return (sub, p2) if sub else None
    return None


class MavCommand:
    def __init__(self, kind, arg, ack):
        self.kind, self.arg, self._ack = kind, arg, ack

    def done(self, ok):
        self._ack(ok)


class MavlinkIO:
    def __init__(self, url, baud=921600, sysid=1, compid=M.MAV_COMP_ID_CAMERA, status_hz=20,
                 frame_size=(1280, 720)):
        self.sysid, self.compid = sysid, compid
        self.conn = mavutil.mavlink_connection(url, baud=baud, source_system=sysid, source_component=compid,
                                               autoreconnect=True)
        self.commands = queue.Queue()
        self.frame_size = frame_size
        self.status_period = 1.0 / status_hz
        self._status = ("IDLE", None, None)
        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self.last_peer_heartbeat = 0.0
        self.running = True
        self.t0 = time.monotonic()
        for fn, name in ((self._rx, "mav-rx"), (self._tx, "mav-tx")):
            threading.Thread(target=fn, daemon=True, name=name).start()

    @property
    def connected(self):
        return time.monotonic() - self.last_peer_heartbeat < 3.0

    def set_status(self, state, norm_box, angle_err):
        with self._lock:
            self._status = (state, norm_box, angle_err)

    # ---- receive --------------------------------------------------------
    def _rx(self):
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.5)
            except Exception as e:  # serial glitch: keep the thread alive
                print(f"mavlink rx error: {e}")
                time.sleep(0.5)
                continue
            if msg is None:
                continue
            t = msg.get_type()
            if t == "HEARTBEAT" and msg.get_srcComponent() != self.compid:
                self.last_peer_heartbeat = time.monotonic()
            elif t == "COMMAND_LONG" and msg.target_system in (0, self.sysid) and msg.target_component in (0, self.compid):
                try:
                    self._on_command(msg)
                except Exception as e:
                    # A malformed command must never take command reception down with it for
                    # the rest of the flight. NaN params are the likely one: MAVLink uses NaN
                    # for "leave unchanged", and int(nan) raises.
                    print(f"mavlink: ignored bad command {msg.command}: {e}")

    def _on_command(self, msg):
        cmd = msg.command
        src = (msg.get_srcSystem(), msg.get_srcComponent())

        def ack(ok, cmd=cmd, src=src):
            result = M.MAV_RESULT_ACCEPTED if ok else M.MAV_RESULT_FAILED
            self._send(lambda: self.conn.mav.command_ack_send(cmd, result, 0, 0, src[0], src[1]))

        if cmd == M.MAV_CMD_REQUEST_MESSAGE and int(msg.param1) == M.MAVLINK_MSG_ID_CAMERA_INFORMATION:
            self._send_camera_information()
            ack(True)
            return
        if cmd not in TRACK_CMDS:
            if msg.target_component == self.compid:  # addressed to us specifically: say we don't support it
                self._send(lambda: self.conn.mav.command_ack_send(cmd, M.MAV_RESULT_UNSUPPORTED, 0, 0, src[0], src[1]))
            return
        w, h = self.frame_size
        if cmd == M.MAV_CMD_CAMERA_TRACK_POINT:
            c = ("point", (msg.param1 * w, msg.param2 * h))
        elif cmd == M.MAV_CMD_CAMERA_TRACK_RECTANGLE:
            c = ("rect", (msg.param1 * w, msg.param2 * h, msg.param3 * w, msg.param4 * h))
        elif cmd == M.MAV_CMD_CAMERA_STOP_TRACKING:
            c = ("stop", None)
        else:
            c = decode_user_command(cmd, msg.param1, msg.param2)
            if c is None:  # a USER command we know, with a sub-command we don't
                ack(False)
                return
        self.commands.put(MavCommand(c[0], c[1], ack))

    # ---- transmit -------------------------------------------------------
    def _send(self, fn):
        with self._tx_lock:
            try:
                fn()
            except Exception as e:
                print(f"mavlink tx error: {e}")

    def _send_camera_information(self):
        w, h = self.frame_size
        flags = M.CAMERA_CAP_FLAGS_HAS_TRACKING_POINT | M.CAMERA_CAP_FLAGS_HAS_TRACKING_RECTANGLE
        self._send(lambda: self.conn.mav.camera_information_send(
            self._boot_ms(), list(b"ADTS".ljust(32, b"\0")), list(b"WALDO-Hailo".ljust(32, b"\0")), 0,
            0, 0, 0, w, h, 0, flags, 0, b""))

    def _boot_ms(self):
        return int((time.monotonic() - self.t0) * 1000) & 0xFFFFFFFF

    def _tx(self):
        next_hb = 0.0
        while self.running:
            now = time.monotonic()
            if now >= next_hb:
                next_hb = now + 1.0
                self._send(lambda: self.conn.mav.heartbeat_send(M.MAV_TYPE_CAMERA, M.MAV_AUTOPILOT_INVALID, 0, 0, M.MAV_STATE_ACTIVE))
            with self._lock:
                state, box, err = self._status
            active = state in ("LOCKED", "COAST")
            nan = float("nan")
            x1, y1, x2, y2 = box if (box and active) else (nan, nan, nan, nan)
            self._send(lambda: self.conn.mav.camera_tracking_image_status_send(
                M.CAMERA_TRACKING_STATUS_FLAGS_ACTIVE if active else
                (M.CAMERA_TRACKING_STATUS_FLAGS_ERROR if state == "LOST" else M.CAMERA_TRACKING_STATUS_FLAGS_IDLE),
                M.CAMERA_TRACKING_MODE_RECTANGLE if active else M.CAMERA_TRACKING_MODE_NONE,
                M.CAMERA_TRACKING_TARGET_DATA_IN_STATUS if active else 0,  # 0 = no target data (no enum entry)
                (x1 + x2) / 2 if active else nan, (y1 + y2) / 2 if active else nan, nan, x1, y1, x2, y2))
            az, el = err if (err and active) else (0.0, 0.0)
            self._send(lambda: self.conn.mav.debug_vect_send(b"TRK_ERR", self._boot_ms() * 1000, az, el, float(STATE_CODE[state])))
            time.sleep(self.status_period)

    def close(self):
        self.running = False
        self.conn.close()
