"""MAVLink2 must be selected before pymavlink.mavutil is imported anywhere in the process:
the dialect it binds is process-wide and sticky (set once at that first import, unaffected
by the env var afterwards), not per test module. adts/mavlink_io.py and tools/gcs_link.py
already guard their own import this way, but test_commands.py imports pymavlink.mavutil
directly for MAV_CMD_* constants - if pytest happens to import that file before either of
those, the whole session is stuck on MAVLink1, and v2-only sends (CAMERA_TRACKING_IMAGE_STATUS,
COMMAND_ACK's target system/component fields) start failing in whatever test runs next.
Setting it here, before pytest imports any test module, makes the outcome independent of
collection order.
"""
import os

os.environ.setdefault("MAVLINK20", "1")
