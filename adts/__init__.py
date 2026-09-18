"""ADTS onboard tracker: WALDO detection + ByteTrack + target lock, built for a
Raspberry Pi 5 with a Hailo-8L (13 TOPS) accelerator, a CSI camera, MAVLink over
UART to ArduPilot and HDMI/UDP video out. Development runs on the Jetson via the
Ultralytics detector backend and video files.

Run with: python3 -m adts --help
"""
