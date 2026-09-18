# Deploying ADTS on Raspberry Pi 5 + Hailo-8L

The target: an onboard tracker on the drone. A CSI camera feeds WALDO30, which runs on
the Hailo-8L (13 TOPS). ByteTrack and the target lock run on the Pi CPU. Commands arrive
and results go out as MAVLink over UART to ArduPilot. Labelled video goes out on HDMI,
and optionally as H.264 over UDP and/or a recording.

```
CSI cam ─► picamera2 (ISP) ─► letterbox 640 ─► Hailo-8L WALDO (INT8, NMS on-chip)
                                                    │
FC TELEM2 ◄─ MAVLink/UART ─► commands ─► ByteTrack ─► TargetLock ─► symbology ─► HDMI / UDP / .ts
```

Code layout (`adts/`):

| module | role |
|---|---|
| `camera.py` | `PiCameraSource` (CSI) / `VideoSource` (files, dev). Newest frame only, never queues |
| `detector.py` | `HailoDetector` (.hef) / `UltralyticsDetector` (.pt/.onnx/.engine, dev) |
| `bytetrack.py` | ByteTrack in numpy + scipy (no torch on the Pi) |
| `target.py` | IDLE → LOCKED → COAST → LOST state machine in two modes (AI / scene); candidate selection; angle error from the camera FOV |
| `scene_track.py` | sabit sahne/obje takip: OpenCV CSRT locked on the centre gate (S/M/L), no detector |
| `commands.py` | where every operator command is executed, whatever carried it (MAVLink or dev keys) |
| `render.py` | symbology: crosshair, gate, candidates, target brackets, state, AZ/EL, FPS, temperature, MAV link. On/off, language and colour switch in flight |
| `video_out.py` | `GstSink` (HDMI kmssink / x264 RTP / MPEG-TS recording), `WindowSink` (dev) |
| `mavlink_io.py` | MAVLink camera component: tracking commands in, tracking status + angle error out |

---

## 1. Hardware

- Pi 5 (8 GB), **Active Cooler**, Hailo-8L (AI Kit M.2 or AI HAT+ 13T), CSI camera on CAM0.
  The Pi 5 uses the narrow 22-pin camera connector, so use a 22-to-15-pin cable for older cameras.
- **Power:** 5 V, **5 A** BEC. The Pi, Hailo and camera together draw about 10–12 W, with peaks above that.
  An undersized BEC causes brownouts and SD card corruption. On a BEC without USB-PD, set
  `PSU_MAX_CURRENT=5000` with `sudo rpi-eeprom-config --edit`.
- **UART to the flight controller** (3.3 V logic on both sides; don't connect the 5 V pins):

  | Pi 5 header | FC TELEM2 |
  |---|---|
  | pin 8  GPIO14 TXD | RX |
  | pin 10 GPIO15 RXD | TX |
  | pin 6  GND | GND |

- **HDMI** (micro-HDMI port 0) to the video transmitter's air unit.
- Mount the camera on vibration dampers. Leave airflow over the cooler inside the enclosure.

## 2. OS setup (once)

Raspberry Pi OS Bookworm **64-bit Lite**. Clone this repo into `~/adts`, then:

```bash
sudo deploy/pi_setup.sh     # packages, PCIe Gen3, UART on / serial console off, console boot, venv, service
sudo reboot
```

Safe to re-run — each step checks what's already done and skips it. Useful flags: `--dry-run`
to preview without changing anything, `-y`/`--yes` to skip confirmations (unattended runs),
`--user <name>` if the account to install for isn't auto-detected. `deploy/pi_setup.sh --help`
lists them all. It also handles a few things by hand: OS/board/architecture mismatches, no
network, `hailo-all` not (yet) available from apt, an existing `.venv` or systemd unit, missing
`raspi-config`, and more — each prints what happened and what to do about it.

Checks after the reboot:

```bash
deploy/pi_setup.sh --verify         # runs the checks below and reports pass/fail (no sudo needed)
```

or by hand:

```bash
hailortcli fw-control identify      # must print the device, and "HAILO8L" as the architecture
hailortcli --version                # HailoRT version: the .hef must come from a matching DFC/AI SW Suite release
rpicam-hello --list-cameras         # camera detected
ls -l /dev/ttyAMA0                  # UART present
vcgencmd get_throttled              # 0x0 = no undervoltage or throttling (run again after a long test)
```

If the HDMI video link doesn't report its display modes (EDID), force a mode by adding
this to the single line in `/boot/firmware/cmdline.txt`:
`video=HDMI-A-1:1280x720@60D`

## 3. Compile WALDO for the Hailo (on an x86 PC)

The Hailo Dataflow Compiler only runs on **x86-64 Ubuntu** (22.04), not on the Pi or the
Jetson. Install the *Hailo AI Software Suite* (Docker is simplest) from the Hailo
Developer Zone (free account). Use the release that matches the Pi's `hailortcli --version`.

On the Jetson:

```bash
python3 export_hailo.py            # models/waldo_yolov8{n,m}_640.onnx + calib/ (1024 frames) + prints the commands
```

The export script checks each ONNX against the `.pt` (it must print `OK`). The
calibration frames are letterboxed exactly the way the runtime feeds the chip.
**Replace or add to `calib/` with frames from your own camera, flown at your
operating altitudes, once you have some.** INT8 accuracy depends on it.

On the x86 PC:

```bash
hailomz compile yolov8m --ckpt waldo_yolov8m_640.onnx --hw-arch hailo8l --calib-path calib/ --classes 12 --performance
hailomz compile yolov8n --ckpt waldo_yolov8n_640.onnx --hw-arch hailo8l --calib-path calib/ --classes 12 --performance
```

- `--hw-arch hailo8l` is required. A Hailo-8 build won't load on the 8L.
- `--performance` searches longer for a faster compile (it can take hours). Leave it out for a quick first build.
- The YOLOv8 NMS config in the Model Zoo sets the score threshold at ~0.2. For ByteTrack,
  a lower threshold (0.1) is slightly better, because it uses low-score boxes to keep
  tracks alive. It's set in the model's NMS config JSON (`nms_scores_th`). Optional.
- **n-p2** (extra stride-4 head, better on small objects) doesn't fit the stock `yolov8` config.
  It needs a custom YAML with 4 end-node pairs. Only worth it if n/m miss small people.

Copy the `.hef` files to the Pi at `~/adts/models/`.

## 4. Check the .hef on the Pi (don't skip)

`HailoDetector`'s NMS parsing was written from the HailoRT docs and unit-tested on
synthetic output. It hasn't run on a real chip yet. `models/ref_{n,m}.npz` hold the
float model's detections for 40 of the calibration frames:

```bash
cd ~/adts && source .venv/bin/activate
python3 tools/hef_check.py check --model models/waldo_yolov8m_640.hef --ref models/ref_m.npz
```

- About 80–95% of reference boxes found: fine. That's the normal INT8 loss. Look at the
  per-class line for **Person** in particular.
- Under 30%: the parsing or box order is wrong (the tool says so). Open the side-by-side
  images in `outputs/hef_check/` and fix `HailoDetector._parse_nms`.

Raw chip speed, for reference: `hailortcli run models/waldo_yolov8m_640.hef`

## 5. First run by hand

```bash
python3 -m adts --source picam --model models/waldo_yolov8m_640.hef --hfov 66 \
    --mav /dev/ttyAMA0 --hdmi --record ~/rec --profile
```

`--profile` prints FPS, detect/track/draw time and capture-to-output latency every
second. The target is **≥ 25 FPS** end to end. If you're below that:
1. Try `waldo_yolov8n_640.hef` (much lighter).
2. `--detect-stride 2` runs detection every other frame. The Kalman filter predicts in
   between, so symbology still updates every frame.
3. `--size 960x540` makes resize/draw/encode cheaper (the model input stays 640).

`--hfov` must match your lens: Camera Module 3 = 66°, CM3 Wide = 102°, HQ camera = depends on the lens.
It only affects the reported AZ/EL angles, not the detection.

## 6. ArduPilot setup

| Param | Value | Why |
|---|---|---|
| `SERIAL2_PROTOCOL` | 2 | MAVLink2 on TELEM2 |
| `SERIAL2_BAUD` | 921 | 921600 baud (matches `--mav-baud`) |
| `SYSID_THISMAV` (`MAV_SYSID` on newer firmware) | 1 | must equal the tracker's `--sysid` |

ArduPilot routes MAVLink between its ports. After it has seen the tracker's heartbeat
(component 100) on TELEM2, commands a GCS sends over the telemetry radio to
system 1 / component 100 get forwarded to the Pi. The tracker's status gets forwarded
back the same way.

**Commands the tracker accepts** (COMMAND_LONG to sysid 1, compid 100):

| Command | Params | Action |
|---|---|---|
| `MAV_CMD_CAMERA_TRACK_POINT` (2004) | p1=x, p2=y (0..1) | lock the detection at/near that point |
| `MAV_CMD_CAMERA_TRACK_RECTANGLE` (2005) | p1..p4 = x1,y1,x2,y2 (0..1) | lock the detection that best overlaps the rectangle |
| `MAV_CMD_CAMERA_STOP_TRACKING` (2010) | – | stop |
| `MAV_CMD_REQUEST_MESSAGE` (512) | p1=259 | replies with CAMERA_INFORMATION (advertises tracking support) |

**Yapay zeka takip** — `MAV_CMD_USER_1` (31010). Selecting and engaging are separate steps:
`±1` only moves the highlight between the five detections nearest the crosshair, and `p1=3`
is what starts tracking the highlighted one.

| Params | Action |
|---|---|
| p1=+1 / -1 | secili hedef ILERI / GERI (highlight only, left-to-right order) |
| p1=0, p2=ID | highlight that track ID (only the current five are selectable) |
| p1=2 | auto: highlight and lock the detection nearest the crosshair, in one step |
| p1=3 | secili hedef takip BASLAT |
| p1=4 | takip IPTAL |
| p1=5, p2 | yapay zeka tespit: 0 kapali, 1 acik, -1 degistir. Off leaves the Hailo idle and drops any AI lock; a scene track keeps running |

**Sabit sahne/obje takip** — `MAV_CMD_USER_2` (31011). CSRT on the gate contents, so the
target doesn't have to be one of WALDO's 12 classes, and it works with detection switched off.

| Params | Action |
|---|---|
| p1=1 | takip BASLAT: lock whatever is inside the gate |
| p1=0 | takip IPTAL (refused with FAILED if no scene track is running) |
| p1=2, p2 | kapi boyutu: 0=S, 1=M, 2=L, -1 = next. The gate is drawn at the centre whenever nothing is being tracked; a size change flashes it as a jagged (tirtikli) box for 3s, on top of whatever else is on screen, then it settles back to the thin steady-state outline |

**Overlay** — `MAV_CMD_USER_3` (31012). All four take effect on the next frame, on HDMI,
UDP and the recording alike.

| Params | Action |
|---|---|
| p1=1, p2 | overlay: 0 kapali (clean image), 1 acik, -1 degistir |
| p1=2, p2 | reticle: 0 kapali, 1 acik, -1 degistir |
| p1=3, p2 | dil: 0=tr, 1=en, -1 degistir |
| p1=4, p2 | renk: 0=yesil, 1=mavi, 2=kirmizi, 3=beyaz, 4=siyah, -1 = next |

Two parameter conventions run through all three: an on/off setting takes `0` / `1` / `-1`
(toggle), and a multi-value setting takes an index or `-1` for the next value round, so a GCS
with one button per function can still reach every value. COAST and LOST keep their amber and
red whatever colour is picked — a slipping lock must not be something a colour preference can
hide.

Each command gets a `COMMAND_ACK`: `ACCEPTED`, or `FAILED` (e.g. nothing detected at that point).

**What the tracker sends** (20 Hz):
- `CAMERA_TRACKING_IMAGE_STATUS`: status (idle/active/error=lost), target rectangle normalised 0..1
- `DEBUG_VECT` named `TRK_ERR`: x = azimuth error °, y = elevation error ° (positive = right/up), z = state (0 idle, 1 locked, 2 coast, 3 lost).
  It shows in Mission Planner's status tab and in the dataflash log, and a Lua script can read it.

The tracker only **reports**. Turning AZ/EL into gimbal or vehicle motion is done on the
ArduPilot side (Lua script or companion logic). That's a separate step, and it gets
flight-tested last.

## 7. Autostart and hardening

```bash
sudo nano /etc/default/adts          # ADTS_ARGS: model, hfov, record dir...
sudo systemctl enable --now adts
journalctl -u adts -f                # live log
```

- The service restarts on any crash (camera or Hailo error, etc.) after 2 s. The CPU governor is pinned to `performance`.
- Recording: a new `.ts` file per start. TS stays playable after a power cut. Recording is
  refused if less than 2 GB is free. Record to a USB SSD if possible.
- Before flight: enable a read-only root filesystem (`sudo raspi-config` → Performance →
  Overlay FS), with the recording directory on a separate partition or USB drive, so
  power cuts can't corrupt the OS.
- Soak test: 1 hour inside the closed drone enclosure with `--profile`. Pass = FPS holds,
  `vcgencmd get_throttled` = `0x0`, CPU temperature below 80 °C.

## Testing without the drone (Jetson / PC)

```bash
# tracker on a video, MAVLink over UDP, window with keyboard/mouse control
python3 -m adts --source videos/ankara_drone_test.mp4 \
    --model weights/waldo/WALDO30_yolov8m_640x640.pt --mav udpin:0.0.0.0:14555 --window
# fake GCS, interactive: next|prev|engage|select ID|auto|cancel|ai on|off, scene|scene stop|
#   gate s|m|l, overlay|reticle on|off|toggle, lang tr|en, color green|blue|red|white|black,
#   point X Y|rect X1 Y1 X2 Y2|stop|info
python3 tools/gcs_sim.py udpout:127.0.0.1:14555
# ...or scripted, for a quick pass over the whole command set:
python3 tools/gcs_sim.py udpout:127.0.0.1:14555 --script \
    "next; wait 1; engage; wait 2; cancel; scene; wait 2; gate l; scene stop; ai off; wait 1; ai on"
# unit tests (tracker, both lock modes, command decoding, symbology, Hailo output parsing)
python3 -m pytest tests/ -q
```

`--window` gives every command a key, so the whole set can be driven without a GCS:
click = track point, `n`/`p` = secili hedef ileri/geri, `Enter` = takip baslat, `s` = auto,
`x` = takip iptal, `g`/`b` = sahne takip baslat/iptal, `k` = kapi boyutu, `d` = yapay zeka
tespit, `o`/`r` = overlay/reticle, `l` = dil, `c` = renk, `q` = cikis.

Against ArduPilot SITL instead of the fake GCS, point `--mav` at a SITL output port
(e.g. `udpin:0.0.0.0:14555` with `--out=udp:127.0.0.1:14555` on `sim_vehicle.py`).

## Testing over a real UART-TTL USB adapter, no drone

For a benchtop test where ADTS runs on the Pi/Jetson and the "GCS" is your own PC talking
over an actual USB-serial adapter instead of a network link: wire the adapter's TX/RX/GND
to the board's UART header (crossed: adapter TX → board RX), then on the ADTS side use the
serial device instead of a udpin URL:

```bash
python3 -m adts --source videos/ankara_drone_test.mp4 \
    --model weights/waldo/WALDO30_yolov8m_640x640.pt --mav /dev/ttyUSB0 --window
```

On the PC, either drive it from the terminal with `tools/gcs_sim.py` (same commands as
above, just a serial device instead of a udpout URL: `python3 tools/gcs_sim.py
/dev/ttyUSB0` on Linux/macOS, `python3 tools/gcs_sim.py COM3` on Windows), or use the
point-and-click panel:

```bash
pip install pyserial   # only needed for the port dropdown; the field also takes typed input
python3 tools/gcs_gui.py
```

Pick the adapter's port (refresh with the ⟳ button) and the FC's baud rate, Connect, and
every button sends the same MAV_CMD_USER_* command tools/gcs_sim.py does - one button per
row in the MAVLink command table above, plus a small canvas for TRACK_POINT (click) and
TRACK_RECTANGLE (drag). The log at the bottom shows every ACK and the live LOCKED/COAST/
LOST state with its AZ/EL readout. `tools/gcs_link.py` is the connection code both tools
share; needs tkinter (bundled with python.org's Windows/macOS installers, `sudo apt
install python3-tk` on Linux).
