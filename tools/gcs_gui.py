#!/usr/bin/env python3
"""Point-and-click stand-in for a GCS, to test ADTS over a real UART-TTL USB adapter (or a
udp/tcp link) without a drone. Every button sends the same MAVLink command tools/gcs_sim.py
does - see gcs_link.py for the shared connection/command code and adts/mavlink_io.py for
what each command means on the receiving end.

    python3 -m adts --source videos/x.mp4 --model weights/waldo/WALDO30_yolov8m_640x640.pt \\
        --mav /dev/ttyUSB0 --window                 # tracker listens on the adapter
    python3 tools/gcs_gui.py                        # pick the same port and Connect

Needs tkinter (ships with python.org's Windows/macOS installers; on Linux install your
distro's package, e.g. `sudo apt install python3-tk`) and pyserial (`pip install pyserial`)
for the port list - without pyserial the port field still takes typed input.
"""

import queue
import sys
import time
import tkinter as tk
from tkinter import ttk

from gcs_link import COLORS, GATES, GcsLink

try:
    from serial.tools.list_ports import comports
except ImportError:
    comports = None

DEFAULT_PORT = "COM3" if sys.platform == "win32" else "/dev/ttyUSB0"
BAUDS = (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600)
STATE_COLOR = {"IDLE": "#888888", "LOCKED": "#2ecc71", "COAST": "#f1c40f", "LOST": "#e74c3c"}


class GcsGui:
    def __init__(self, root):
        self.root = root
        root.title("ADTS GCS test panel")
        root.geometry("760x640")
        root.minsize(760, 500)
        self.link = None
        # The link's rx thread calls into on_event from a background thread; tkinter is
        # not thread-safe, so events are queued here and drained on the Tk main loop
        # instead of touching widgets directly from that thread.
        self.events = queue.Queue()
        self.selected_id = tk.StringVar(value="")

        self._build_connection_row()
        self._build_status_row()
        self._build_command_panel()
        self._build_log()
        self._set_controls_enabled(False)  # nothing to send before Connect

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._drain_events)

    # ---- connection -------------------------------------------------------
    def _build_connection_row(self):
        row = ttk.Frame(self.root, padding=8)
        row.pack(fill="x")
        ttk.Label(row, text="Port:").pack(side="left")
        ports = [p.device for p in comports()] if comports else []
        self.port = ttk.Combobox(row, values=ports, width=16)
        self.port.set(ports[0] if ports else DEFAULT_PORT)
        self.port.pack(side="left", padx=4)
        if comports:
            ttk.Button(row, text="↻", width=2, command=self._refresh_ports).pack(side="left")

        ttk.Label(row, text="Baud:").pack(side="left", padx=(10, 0))
        self.baud = ttk.Combobox(row, values=BAUDS, width=8)
        self.baud.set(921600)
        self.baud.pack(side="left", padx=4)

        self.connect_btn = ttk.Button(row, text="Connect", command=self._toggle_connect)
        self.connect_btn.pack(side="left", padx=(10, 0))
        self.conn_dot = tk.Canvas(row, width=14, height=14, highlightthickness=0)
        self.conn_dot.pack(side="left", padx=6)
        self._set_connected(False)

    def _refresh_ports(self):
        self.port["values"] = [p.device for p in comports()]

    def _set_connected(self, ok):
        self.conn_dot.delete("all")
        self.conn_dot.create_oval(2, 2, 12, 12, fill="#2ecc71" if ok else "#555555", outline="")

    def _toggle_connect(self):
        if self.link is None:
            url = self.port.get().strip()
            try:
                baud = int(self.baud.get())
            except ValueError:
                baud = 921600
            try:
                self.link = GcsLink(url, baud=baud, on_event=lambda k, d: self.events.put((k, d)))
            except Exception as e:
                self._log(f"connect failed: {e}")
                return
            self._log(f"connecting to {url} @ {baud}...")
            self.connect_btn.config(text="Disconnect")
            self._set_connected(True)
            self._set_controls_enabled(True)
        else:
            self.link.close()
            self.link = None
            self._log("disconnected")
            self.connect_btn.config(text="Connect")
            self._set_connected(False)
            self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for w in self._command_widgets:
            w.config(state=state)

    # ---- status -------------------------------------------------------------
    def _build_status_row(self):
        row = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        row.pack(fill="x")
        self.state_label = tk.Label(row, text="---", font=("", 16, "bold"), fg=STATE_COLOR["IDLE"])
        self.state_label.pack(side="left")
        self.az_el_label = ttk.Label(row, text="")
        self.az_el_label.pack(side="left", padx=12)

    # ---- commands -----------------------------------------------------------
    def _build_command_panel(self):
        self._command_widgets = []
        panel = ttk.Frame(self.root, padding=8)
        panel.pack(fill="x")

        ai = ttk.LabelFrame(panel, text="Yapay zeka takip", padding=6)
        ai.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._button_row(ai, [("< Geri", self._on("cand_prev")), ("Ileri >", self._on("cand_next"))])
        self._button_row(ai, [("Takip BASLAT", self._on("engage")), ("IPTAL", self._on("cancel"))])
        self._button_row(ai, [("AUTO (nearest)", self._on("auto"))])
        idrow = ttk.Frame(ai)
        idrow.pack(fill="x", pady=2)
        ttk.Label(idrow, text="Select ID:").pack(side="left")
        e = ttk.Entry(idrow, textvariable=self.selected_id, width=6)
        e.pack(side="left", padx=4)
        self._command_widgets.append(e)
        self._button_row(idrow, [("Select", self._select_id)], side="left")
        self._button_row(ai, [("AI ON", self._on("ai", 1)), ("AI OFF", self._on("ai", 0)),
                              ("AI TOGGLE", self._on("ai", -1))])

        scene = ttk.LabelFrame(panel, text="Sabit sahne/obje takip", padding=6)
        scene.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self._button_row(scene, [("Takip BASLAT", self._on("scene_start")), ("IPTAL", self._on("scene_stop"))])
        self._button_row(scene, [(f"Gate {g.upper()}", self._on("gate", v)) for g, v in GATES.items() if g != "next"])
        self._button_row(scene, [("Gate next", self._on("gate", -1))])

        overlay = ttk.LabelFrame(panel, text="Overlay", padding=6)
        overlay.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._button_row(overlay, [("Overlay ON", self._on("overlay", 1)), ("OFF", self._on("overlay", 0)),
                                   ("toggle", self._on("overlay", -1))])
        self._button_row(overlay, [("Reticle ON", self._on("reticle", 1)), ("OFF", self._on("reticle", 0)),
                                   ("toggle", self._on("reticle", -1))])
        self._button_row(overlay, [("TR", self._on("lang", 0)), ("EN", self._on("lang", 1)),
                                   ("next", self._on("lang", -1))])
        self._button_row(overlay, [(c.capitalize(), self._on("color", v)) for c, v in COLORS.items() if c != "next"])

        direct = ttk.LabelFrame(panel, text="Dogrudan", padding=6)
        direct.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(direct, text="Click = TRACK_POINT, drag = TRACK_RECTANGLE\n(normalised 0..1 over this square)",
                 justify="left", wraplength=220).pack(anchor="w")
        self.canvas = tk.Canvas(direct, width=160, height=120, bg="#222222", highlightthickness=1,
                                highlightbackground="#555555")
        self.canvas.pack(pady=4)
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self._command_widgets.append(self.canvas)
        self._drag_start = None
        self._button_row(direct, [("STOP", self._on("stop")), ("INFO", self._on("info"))])

        for c in range(2):
            panel.grid_columnconfigure(c, weight=1)

    def _button_row(self, parent, items, side="top"):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2, side=side if side != "top" else "top")
        for text, cmd in items:
            b = ttk.Button(row, text=text, command=cmd)
            b.pack(side="left", padx=2, fill="x", expand=True)
            self._command_widgets.append(b)

    def _on(self, method, *args):
        def call():
            if self.link is None:
                return
            try:
                getattr(self.link, method)(*args)
                self._log(f"-> {method}{args if args else ''}")
            except Exception as e:
                self._log(f"send failed: {e}")
        return call

    def _select_id(self):
        try:
            tid = float(self.selected_id.get())
        except ValueError:
            self._log("select: not a number")
            return
        self._on("select", tid)()

    def _canvas_press(self, ev):
        self._drag_start = (ev.x, ev.y)
        self.canvas.delete("drag")

    def _canvas_drag(self, ev):
        if self._drag_start:
            self.canvas.delete("drag")
            self.canvas.create_rectangle(*self._drag_start, ev.x, ev.y, outline="#2ecc71", tags="drag")

    def _canvas_release(self, ev):
        if self._drag_start is None or self.link is None:
            return
        x0, y0 = self._drag_start
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if abs(ev.x - x0) < 4 and abs(ev.y - y0) < 4:
            self.link.point(x0 / w, y0 / h)
            self._log(f"-> point({x0 / w:.2f}, {y0 / h:.2f})")
        else:
            x1, x2 = sorted((x0 / w, ev.x / w))
            y1, y2 = sorted((y0 / h, ev.y / h))
            self.link.rect(x1, y1, x2, y2)
            self._log(f"-> rect({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
        self._drag_start = None

    # ---- log / incoming events -----------------------------------------------
    def _build_log(self):
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="both", expand=True)
        self.log = tk.Text(frame, height=8, state="disabled", bg="#111111", fg="#dddddd", font=("Courier", 9))
        sb = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _log(self, line):
        self.log.config(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {line}\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _drain_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                self._handle_event(kind, data)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_events)

    def _handle_event(self, kind, data):
        if kind == "ack":
            self._log(f"ACK cmd={data[0]} {data[1]}")
        elif kind == "camera_info":
            vendor, w, h, flags = data
            self._log(f"CAMERA_INFORMATION {vendor} {w}x{h} flags={flags}")
        elif kind == "track":
            state, az, el = data
            self.state_label.config(text=state, fg=STATE_COLOR.get(state, "#dddddd"))
            self.az_el_label.config(text=f"AZ {az:+5.1f}  EL {el:+5.1f}" if state != "IDLE" else "")
        elif kind == "error":
            self._log(f"link error: {data}")

    def _on_close(self):
        if self.link is not None:
            self.link.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GcsGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
