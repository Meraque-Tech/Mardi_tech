"""ROS 2 node for the ESP32 JSON GNSS receiver."""

import json
import math
import queue
import threading
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool, String

from serial import Serial, SerialException
from serial.tools import list_ports


FIX_STATE = {
    0: "NO_FIX",
    1: "DEAD_RECKONING",
    2: "2D",
    3: "3D",
    4: "GNSS_DR",
    5: "TIME_ONLY",
}
CORR_AGE = [
    "N/A", "0-1s", "1-2s", "2-5s", "5-10s", "10-15s", "15-20s",
    "20-30s", "30-45s", "45-60s", "60-90s", "90-120s", ">120s",
]
KNOWN_USB_IDS = [
    (0x10C4, 0xEA60),
    (0x1A86, 0x7523),
    (0x0403, 0x6001),
    (0x303A, 0x1001),
]
RTK_VALID_STATES = {"RTK_FIXED", "RTK_FLOAT"}


def rtk_state(pvt: Dict[str, Any]) -> str:
    """Keep the RTK classification used by base_receiver.py."""
    if pvt["fix"] < 2:
        return "NO_FIX"
    if pvt["carr"] == 2:
        return "RTK_FIXED"
    if pvt["carr"] == 1:
        return "RTK_FLOAT"
    if pvt["diff"]:
        return "DGNSS"
    return "3D_GNSS"


def rtk_status_from_state(state: Any) -> bool:
    """Return whether the receiver currently has an RTK solution."""
    try:
        return state in RTK_VALID_STATES
    except TypeError:
        return False


def find_esp32_port() -> Optional[str]:
    for port_info in list_ports.comports():
        if (port_info.vid, port_info.pid) in KNOWN_USB_IDS:
            return port_info.device
    return None


def enrich_pvt(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Add exactly the derived fields produced by the original script."""
    msg = dict(msg)
    msg["rtkState"] = rtk_state(msg)
    msg["fixLabel"] = FIX_STATE.get(msg["fix"], "UNKNOWN")
    corr_age = msg["corrAge"]
    msg["corrAgeLabel"] = (
        CORR_AGE[corr_age] if 0 <= corr_age < len(CORR_AGE) else "?"
    )
    msg["timestamp"] = time.monotonic()
    return msg


class RoverGnssReader:
    """Blocking serial reader kept separate from ROS publishing."""

    def __init__(self, port: str, baud: int, reconnect_interval: float):
        self.fixed_port = port
        self.baud = baud
        self.reconnect_interval = reconnect_interval
        self.serial_conn: Optional[Serial] = None

    def connect(self, stop_event: threading.Event) -> bool:
        while not stop_event.is_set() and self.serial_conn is None:
            port = self.fixed_port or find_esp32_port()
            if port is None:
                stop_event.wait(self.reconnect_interval)
                continue
            try:
                self.serial_conn = Serial(port, self.baud, timeout=1)
                return True
            except SerialException:
                stop_event.wait(self.reconnect_interval)
        return self.serial_conn is not None

    def disconnect(self) -> None:
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except SerialException:
                pass
            self.serial_conn = None

    def stream(self, stop_event: threading.Event):
        while not stop_event.is_set():
            if self.serial_conn is None and not self.connect(stop_event):
                break
            try:
                line = self.serial_conn.readline().decode("ascii", errors="ignore").strip()
            except (SerialException, OSError, AttributeError):
                self.disconnect()
                continue

            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "pvt":
                try:
                    yield enrich_pvt(msg)
                except (KeyError, TypeError, ValueError):
                    # Invalid PVT records are ignored, matching the original
                    # stream's behavior for malformed JSON while keeping the
                    # ROS node alive.
                    continue
            else:
                yield msg

    def close(self) -> None:
        self.disconnect()


class RoverGnssNode(Node):
    """Publish standard position data and the complete enriched PVT JSON."""

    def __init__(self) -> None:
        super().__init__("base_receiver")
        self.declare_parameter("port", "")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("frame_id", "gps")
        self.declare_parameter("stale_timeout", 3.0)
        self.declare_parameter("reconnect_interval", 2.0)
        self.declare_parameter("fix_topic", "/receiver/fix")
        self.declare_parameter("pvt_topic", "/gnss/pvt")
        self.declare_parameter("rtk_status_topic", "/gnss/rtk_status")

        port = self.get_parameter("port").value
        baud = int(self.get_parameter("baud").value)
        reconnect_interval = float(self.get_parameter("reconnect_interval").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.stale_timeout = float(self.get_parameter("stale_timeout").value)

        self.fix_pub = self.create_publisher(
            NavSatFix, str(self.get_parameter("fix_topic").value), 10
        )
        self.pvt_pub = self.create_publisher(
            String, str(self.get_parameter("pvt_topic").value), 10
        )
        self.rtk_status_pub = self.create_publisher(
            Bool, str(self.get_parameter("rtk_status_topic").value), 10
        )

        self._messages: queue.Queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._reader = RoverGnssReader(str(port), baud, reconnect_interval)
        self._last_pvt_monotonic: Optional[float] = None
        self._reader_thread = threading.Thread(target=self._read_serial, daemon=True)
        self._reader_thread.start()
        self._timer = self.create_timer(0.02, self._drain_messages)
        self._stale_logged = False
        self._publish_rtk_status(False)

        self.get_logger().info(f"Publishing NavSatFix on {self.fix_pub.topic_name}")
        self.get_logger().info(f"Publishing PVT JSON on {self.pvt_pub.topic_name}")
        self.get_logger().info(
            f"Publishing RTK status on {self.rtk_status_pub.topic_name}"
        )

    def _read_serial(self) -> None:
        for msg in self._reader.stream(self._stop_event):
            if self._stop_event.is_set():
                break
            try:
                self._messages.put_nowait(msg)
            except queue.Full:
                try:
                    self._messages.get_nowait()
                    self._messages.put_nowait(msg)
                except queue.Empty:
                    pass

    def _drain_messages(self) -> None:
        while True:
            try:
                msg = self._messages.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") == "pvt":
                self._publish_pvt(msg)
            elif msg.get("msg"):
                self.get_logger().info(f"[receiver] {msg['msg']}")

        if (
            self._last_pvt_monotonic is not None
            and time.monotonic() - self._last_pvt_monotonic > self.stale_timeout
            and not self._stale_logged
        ):
            self.get_logger().warning("GNSS PVT data is stale")
            self._publish_rtk_status(False)
            self._stale_logged = True

    def _publish_pvt(self, pvt: Dict[str, Any]) -> None:
        now = self.get_clock().now().to_msg()
        fix = NavSatFix()
        fix.header.stamp = now
        fix.header.frame_id = self.frame_id
        fix.latitude = float(pvt["lat"])
        fix.longitude = float(pvt["lon"])
        fix.altitude = float(pvt.get("alt", float("nan")))

        fix.status.status = (
            NavSatStatus.STATUS_NO_FIX
            if int(pvt["fix"]) < 2
            else NavSatStatus.STATUS_FIX
        )
        fix.status.service = NavSatStatus.SERVICE_GPS

        hacc = pvt.get("hacc")
        vacc = pvt.get("vacc")
        if self._valid_accuracy(hacc) and self._valid_accuracy(vacc):
            fix.position_covariance[0] = float(hacc) ** 2
            fix.position_covariance[4] = float(hacc) ** 2
            fix.position_covariance[8] = float(vacc) ** 2
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        else:
            # Do not fabricate vertical accuracy when the receiver does not
            # provide it. The complete hacc value remains available in /gnss/pvt.
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.fix_pub.publish(fix)
        pvt_msg = String()
        pvt_msg.data = json.dumps(pvt, separators=(",", ":"), allow_nan=False)
        self.pvt_pub.publish(pvt_msg)
        self._publish_rtk_status(rtk_status_from_state(pvt.get("rtkState")))

        self._last_pvt_monotonic = float(pvt["timestamp"])
        self._stale_logged = False

    def _publish_rtk_status(self, value: bool) -> None:
        status_msg = Bool()
        status_msg.data = value
        self.rtk_status_pub.publish(status_msg)

    @staticmethod
    def _valid_accuracy(value: Any) -> bool:
        try:
            return math.isfinite(float(value)) and float(value) >= 0.0
        except (TypeError, ValueError):
            return False

    def destroy_node(self):
        self._stop_event.set()
        self._reader.close()
        self._reader_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverGnssNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
