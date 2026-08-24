"""ROS 2 node for the ESP32 JSON GNSS receiver."""

import json
import math
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

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
POSITION_FIX_STATES = {2, 3, 4}
RTK_VALID_STATES = {"RTK_FIXED", "RTK_FLOAT"}


def has_position_fix(fix: Any) -> bool:
    """Return whether a receiver fix type contains a usable position."""
    try:
        return fix in POSITION_FIX_STATES
    except TypeError:
        return False


def navsat_status_for_fix(fix: Any) -> int:
    """Map the receiver fix type to the ROS NavSatStatus convention."""
    if has_position_fix(fix):
        return NavSatStatus.STATUS_FIX
    return NavSatStatus.STATUS_NO_FIX


def rtk_state(pvt: Dict[str, Any]) -> str:
    """Keep the RTK classification used by base_receiver.py."""
    if not has_position_fix(pvt["fix"]):
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


def _integer_field(pvt: Dict[str, Any], name: str) -> int:
    value = pvt[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_float_field(pvt: Dict[str, Any], name: str) -> float:
    value = pvt[name]
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_pvt(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a PVT record before it reaches ROS publishers."""
    if not isinstance(msg, dict):
        raise ValueError("PVT record must be a JSON object")

    normalized = dict(msg)
    fix = _integer_field(normalized, "fix")
    if fix not in FIX_STATE:
        raise ValueError("fix is outside the supported receiver fix types")

    carr = _integer_field(normalized, "carr")
    if carr not in (0, 1, 2):
        raise ValueError("carr must be 0, 1, or 2")

    corr_age = _integer_field(normalized, "corrAge")
    if corr_age < 0:
        raise ValueError("corrAge must be non-negative")

    diff = normalized["diff"]
    if isinstance(diff, bool):
        normalized["diff"] = diff
    elif isinstance(diff, int) and diff in (0, 1):
        normalized["diff"] = bool(diff)
    else:
        raise ValueError("diff must be a boolean or 0/1")

    latitude = _finite_float_field(normalized, "lat")
    longitude = _finite_float_field(normalized, "lon")
    altitude = _finite_float_field(normalized, "alt")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("lat must be between -90 and 90 degrees")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("lon must be between -180 and 180 degrees")

    normalized["fix"] = fix
    normalized["carr"] = carr
    normalized["corrAge"] = corr_age
    normalized["lat"] = latitude
    normalized["lon"] = longitude
    normalized["alt"] = altitude

    for accuracy_name in ("hacc", "vacc"):
        if normalized.get(accuracy_name) is None:
            continue
        accuracy = _finite_float_field(normalized, accuracy_name)
        if accuracy < 0.0:
            raise ValueError(f"{accuracy_name} must be non-negative")
        normalized[accuracy_name] = accuracy

    return normalized


def _reject_non_finite_json_constant(value: str):
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def find_esp32_port() -> Optional[str]:
    for port_info in list_ports.comports():
        if (port_info.vid, port_info.pid) in KNOWN_USB_IDS:
            return port_info.device
    return None


def enrich_pvt(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Add exactly the derived fields produced by the original script."""
    msg = normalize_pvt(msg)
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

    def __init__(
        self,
        port: str,
        baud: int,
        reconnect_interval: float,
        stale_reconnect_timeout: float,
        event_callback: Optional[Callable[[str, str], None]] = None,
        serial_factory: Callable[..., Serial] = Serial,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.fixed_port = port
        self.baud = baud
        self.reconnect_interval = reconnect_interval
        self.stale_reconnect_timeout = stale_reconnect_timeout
        self.event_callback = event_callback
        self.serial_factory = serial_factory
        self.monotonic = monotonic
        self.serial_conn: Optional[Serial] = None
        self.connected_port: Optional[str] = None
        self.connected_monotonic: Optional[float] = None
        self.last_valid_pvt_monotonic: Optional[float] = None
        self.outage_started_monotonic: Optional[float] = None
        self.reconnect_count = 0

    def _emit(self, level: str, message: str) -> None:
        if self.event_callback is not None:
            self.event_callback(level, message)

    def _begin_outage(self, now: Optional[float] = None) -> None:
        if self.outage_started_monotonic is None:
            self.outage_started_monotonic = (
                self.monotonic() if now is None else now
            )

    def connect(self, stop_event: threading.Event) -> bool:
        while not stop_event.is_set() and self.serial_conn is None:
            port = self.fixed_port or find_esp32_port()
            if port is None:
                self._emit(
                    "warning",
                    "GNSS serial device not found; retrying in "
                    f"{self.reconnect_interval:.1f} seconds",
                )
                stop_event.wait(self.reconnect_interval)
                continue
            try:
                connection = self.serial_factory(port, self.baud, timeout=1)
                try:
                    connection.reset_input_buffer()
                except (SerialException, OSError, AttributeError) as error:
                    self._emit(
                        "warning",
                        "GNSS serial input buffer reset failed on "
                        f"{port}: {error}",
                    )
                self.serial_conn = connection
                self.connected_port = port
                self.connected_monotonic = self.monotonic()
                self.last_valid_pvt_monotonic = None
                self._emit("info", f"GNSS serial connected on {port}")
                return True
            except (SerialException, OSError) as error:
                self._emit(
                    "warning",
                    f"GNSS serial open failed on {port}: {error}; retrying in "
                    f"{self.reconnect_interval:.1f} seconds",
                )
                stop_event.wait(self.reconnect_interval)
        return self.serial_conn is not None

    def disconnect(self) -> None:
        connection = self.serial_conn
        self.serial_conn = None
        self.connected_port = None
        self.connected_monotonic = None
        self.last_valid_pvt_monotonic = None
        if connection is not None:
            try:
                connection.close()
            except (SerialException, OSError):
                pass

    def _stream_is_stale(self, now: float) -> bool:
        reference = self.last_valid_pvt_monotonic
        if reference is None:
            reference = self.connected_monotonic
        return (
            reference is not None
            and now - reference > self.stale_reconnect_timeout
        )

    def _recover_stale_connection(
        self,
        stop_event: threading.Event,
        now: float,
    ) -> None:
        reference = self.last_valid_pvt_monotonic
        if reference is None:
            reference = self.connected_monotonic
        stale_age = 0.0 if reference is None else max(0.0, now - reference)
        port = (
            self.connected_port
            or self.fixed_port
            or "auto-discovered device"
        )
        self._begin_outage(now)
        self.reconnect_count += 1
        self._emit(
            "warning",
            f"GNSS PVT stream stale for {stale_age:.1f} seconds on {port}; "
            f"closing serial connection and reconnecting "
            f"(attempt {self.reconnect_count})",
        )
        self.disconnect()
        stop_event.wait(self.reconnect_interval)

    def _record_valid_pvt(self, now: float) -> None:
        self.last_valid_pvt_monotonic = now
        if self.outage_started_monotonic is not None:
            outage_duration = max(0.0, now - self.outage_started_monotonic)
            self._emit(
                "info",
                "GNSS PVT stream recovered after "
                f"{outage_duration:.1f} seconds",
            )
            self.outage_started_monotonic = None
            self.reconnect_count = 0

    def stream(self, stop_event: threading.Event):
        while not stop_event.is_set():
            if self.serial_conn is None and not self.connect(stop_event):
                break
            try:
                raw_line = self.serial_conn.readline()
                line = raw_line.decode("ascii", errors="ignore").strip()
            except (SerialException, OSError, AttributeError) as error:
                self._begin_outage()
                self.reconnect_count += 1
                self._emit(
                    "warning",
                    f"GNSS serial read failed: {error}; closing connection "
                    "and "
                    f"reconnecting (attempt {self.reconnect_count})",
                )
                self.disconnect()
                stop_event.wait(self.reconnect_interval)
                continue

            parsed_msg = None
            if line.startswith("{"):
                try:
                    candidate = json.loads(
                        line,
                        parse_constant=_reject_non_finite_json_constant,
                    )
                except (json.JSONDecodeError, ValueError):
                    candidate = None
                if isinstance(candidate, dict):
                    if candidate.get("type") == "pvt":
                        try:
                            parsed_msg = enrich_pvt(candidate)
                            self._record_valid_pvt(self.monotonic())
                        except (KeyError, TypeError, ValueError):
                            # Invalid PVT records are ignored. If they persist,
                            # the valid-PVT watchdog below reopens the serial
                            # device instead of leaving the stream wedged.
                            parsed_msg = None
                    else:
                        parsed_msg = candidate

            if parsed_msg is not None:
                yield parsed_msg

            now = self.monotonic()
            if self._stream_is_stale(now):
                self._recover_stale_connection(stop_event, now)

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
        self.declare_parameter("stale_reconnect_timeout", 5.0)
        self.declare_parameter("fix_topic", "/receiver/fix")
        self.declare_parameter("pvt_topic", "/gnss/pvt")
        self.declare_parameter("rtk_status_topic", "/gnss/rtk_status")

        port = self.get_parameter("port").value
        baud = int(self.get_parameter("baud").value)
        reconnect_interval = float(
            self.get_parameter("reconnect_interval").value
        )
        stale_reconnect_timeout = float(
            self.get_parameter("stale_reconnect_timeout").value
        )
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.stale_timeout = float(self.get_parameter("stale_timeout").value)

        if reconnect_interval <= 0.0:
            raise ValueError("reconnect_interval must be greater than zero")
        if stale_reconnect_timeout <= 0.0:
            raise ValueError(
                "stale_reconnect_timeout must be greater than zero"
            )
        if stale_reconnect_timeout <= self.stale_timeout:
            self.get_logger().warning(
                "stale_reconnect_timeout should be greater than stale_timeout "
                "so GNSS is marked stale before the serial connection is "
                "reopened"
            )

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
        self._reader = RoverGnssReader(
            str(port),
            baud,
            reconnect_interval,
            stale_reconnect_timeout,
            event_callback=self._log_reader_event,
        )
        self._last_pvt_monotonic: Optional[float] = None
        self._reader_thread = threading.Thread(
            target=self._read_serial,
            daemon=True,
        )
        self._reader_thread.start()
        self._timer = self.create_timer(0.02, self._drain_messages)
        self._stale_logged = False
        self._publish_rtk_status(False)

        self.get_logger().info(
            f"Publishing NavSatFix on {self.fix_pub.topic_name}"
        )
        self.get_logger().info(
            f"Publishing PVT JSON on {self.pvt_pub.topic_name}"
        )
        self.get_logger().info(
            f"Publishing RTK status on {self.rtk_status_pub.topic_name}"
        )

    def _log_reader_event(self, level: str, message: str) -> None:
        logger = self.get_logger()
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

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
                try:
                    self._publish_pvt(msg)
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    OverflowError,
                ) as error:
                    self.get_logger().warning(
                        f"Dropping invalid GNSS PVT record: {error}"
                    )
            elif msg.get("msg"):
                self.get_logger().info(f"[receiver] {msg['msg']}")

        if (
            self._last_pvt_monotonic is not None
            and time.monotonic() - self._last_pvt_monotonic
            > self.stale_timeout
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

        fix.status.status = navsat_status_for_fix(pvt["fix"])
        fix.status.service = NavSatStatus.SERVICE_GPS

        hacc = pvt.get("hacc")
        vacc = pvt.get("vacc")
        if self._valid_accuracy(hacc) and self._valid_accuracy(vacc):
            fix.position_covariance[0] = float(hacc) ** 2
            fix.position_covariance[4] = float(hacc) ** 2
            fix.position_covariance[8] = float(vacc) ** 2
            fix.position_covariance_type = (
                NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            )
        else:
            # Do not fabricate vertical accuracy when the receiver does not
            # provide it. The complete hacc value remains available in
            # /gnss/pvt.
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
