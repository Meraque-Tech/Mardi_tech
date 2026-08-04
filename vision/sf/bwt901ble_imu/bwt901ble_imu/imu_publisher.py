"""Publish WitMotion BWT901BLE5.0 samples as sensor_msgs/Imu messages."""

import asyncio
import math
import queue
import subprocess
import threading
from typing import Dict, Iterable, Optional, Tuple

from bleak import BleakScanner
import device_model
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from tf2_ros import TransformBroadcaster


GRAVITY_METERS_PER_SECOND_SQUARED = 9.80665
DEGREES_TO_RADIANS = math.pi / 180.0

# RELIABLE so both imu_complementary_filter (rclcpp default QoS) and
# imu_filter_madgwick (BEST_EFFORT) can subscribe; a RELIABLE writer is
# compatible with BEST_EFFORT readers, but not the other way around.
IMU_PUBLISHER_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)
SDK_MAGNETIC_DIVISOR = 120.0
MILLIGAUSS_TO_TESLA = 1.0e-7


def _normalized_quaternion(
    quaternion: Iterable[float],
) -> Optional[Tuple[float, float, float, float]]:
    """Return a normalized (w, x, y, z) quaternion, if it is valid."""
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None

    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude < 1.0e-9:
        return None
    return tuple(value / magnitude for value in values)


def _euler_degrees_to_quaternion(
    roll_degrees: float,
    pitch_degrees: float,
    yaw_degrees: float,
) -> Tuple[float, float, float, float]:
    """Convert SDK roll, pitch and yaw in degrees to (w, x, y, z)."""
    roll = float(roll_degrees) * DEGREES_TO_RADIANS
    pitch = float(pitch_degrees) * DEGREES_TO_RADIANS
    yaw = float(yaw_degrees) * DEGREES_TO_RADIANS

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def build_imu_message(
    sample: Dict[str, float],
    stamp,
    frame_id: str,
    orientation_covariance: Iterable[float],
    angular_velocity_covariance: Iterable[float],
    linear_acceleration_covariance: Iterable[float],
) -> Imu:
    """Convert one DeviceModel data dictionary into a ROS Imu message."""
    message = Imu()
    message.header.stamp = stamp
    message.header.frame_id = frame_id

    # DeviceModel reports acceleration in g and angular rate in degrees/second.
    message.linear_acceleration.x = (
        float(sample['AccX']) * GRAVITY_METERS_PER_SECOND_SQUARED
    )
    message.linear_acceleration.y = (
        float(sample['AccY']) * GRAVITY_METERS_PER_SECOND_SQUARED
    )
    message.linear_acceleration.z = (
        float(sample['AccZ']) * GRAVITY_METERS_PER_SECOND_SQUARED
    )
    message.angular_velocity.x = float(sample['AsX']) * DEGREES_TO_RADIANS
    message.angular_velocity.y = float(sample['AsY']) * DEGREES_TO_RADIANS
    message.angular_velocity.z = float(sample['AsZ']) * DEGREES_TO_RADIANS

    quaternion = None
    if all(key in sample for key in ('Q0', 'Q1', 'Q2', 'Q3')):
        quaternion = _normalized_quaternion(
            (sample['Q0'], sample['Q1'], sample['Q2'], sample['Q3'])
        )
    if quaternion is None:
        quaternion = _euler_degrees_to_quaternion(
            sample['AngX'], sample['AngY'], sample['AngZ']
        )

    message.orientation.w = quaternion[0]
    message.orientation.x = quaternion[1]
    message.orientation.y = quaternion[2]
    message.orientation.z = quaternion[3]
    message.orientation_covariance = list(orientation_covariance)
    message.angular_velocity_covariance = list(angular_velocity_covariance)
    message.linear_acceleration_covariance = list(
        linear_acceleration_covariance
    )
    return message


def build_rotation_transform(
    imu_message: Imu,
    parent_frame: str,
) -> TransformStamped:
    """Build a zero-translation TF using an Imu message's orientation."""
    transform = TransformStamped()
    transform.header.stamp = imu_message.header.stamp
    transform.header.frame_id = parent_frame
    transform.child_frame_id = imu_message.header.frame_id
    transform.transform.translation.x = 0.0
    transform.transform.translation.y = 0.0
    transform.transform.translation.z = 0.0
    transform.transform.rotation.x = imu_message.orientation.x
    transform.transform.rotation.y = imu_message.orientation.y
    transform.transform.rotation.z = imu_message.orientation.z
    transform.transform.rotation.w = imu_message.orientation.w
    return transform


def build_magnetic_field_message(
    sample: Dict[str, float],
    stamp,
    frame_id: str,
    covariance: Iterable[float],
) -> MagneticField:
    """Undo the SDK's /120 scaling and convert milligauss to tesla."""
    message = MagneticField()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    scale = SDK_MAGNETIC_DIVISOR * MILLIGAUSS_TO_TESLA
    message.magnetic_field.x = float(sample['HX']) * scale
    message.magnetic_field.y = float(sample['HY']) * scale
    message.magnetic_field.z = float(sample['HZ']) * scale
    message.magnetic_field_covariance = list(covariance)
    return message


class Bwt901BleImuNode(Node):
    """Connect to a BWT901 over BLE and publish decoded IMU frames."""

    def __init__(self) -> None:
        super().__init__('bwt901ble_imu_publisher')

        self._restart_bluetooth_service()

        self.declare_parameter('device_address', 'F7:75:A3:1D:9C:AB')
        self.declare_parameter('device_name_filter', 'WT')
        self.declare_parameter('scan_timeout', 20.0)
        self.declare_parameter('reconnect_delay', 5.0)
        self.declare_parameter('topic', 'imu/data')
        self.declare_parameter('mag_topic', 'imu/mag')
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('tf_parent_frame', 'imu_reference')
        self.declare_parameter('orientation_covariance', [0.0] * 9)
        self.declare_parameter('angular_velocity_covariance', [0.0] * 9)
        self.declare_parameter('linear_acceleration_covariance', [0.0] * 9)
        self.declare_parameter('magnetic_field_covariance', [0.0] * 9)

        self._device_address = str(
            self.get_parameter('device_address').value
        ).strip()
        self._device_name_filter = str(
            self.get_parameter('device_name_filter').value
        )
        self._scan_timeout = float(self.get_parameter('scan_timeout').value)
        self._reconnect_delay = float(
            self.get_parameter('reconnect_delay').value
        )
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._publish_tf = bool(self.get_parameter('publish_tf').value)
        self._tf_parent_frame = str(
            self.get_parameter('tf_parent_frame').value
        )
        if self._publish_tf and self._tf_parent_frame == self._frame_id:
            raise ValueError('tf_parent_frame and frame_id must be different')
        self._orientation_covariance = self._read_covariance_parameter(
            'orientation_covariance'
        )
        self._angular_velocity_covariance = self._read_covariance_parameter(
            'angular_velocity_covariance'
        )
        self._linear_acceleration_covariance = (
            self._read_covariance_parameter('linear_acceleration_covariance')
        )
        self._magnetic_field_covariance = self._read_covariance_parameter(
            'magnetic_field_covariance'
        )

        topic = str(self.get_parameter('topic').value)
        mag_topic = str(self.get_parameter('mag_topic').value)
        self._publisher = self.create_publisher(
            Imu, topic, IMU_PUBLISHER_QOS
        )
        self._mag_publisher = self.create_publisher(
            MagneticField, mag_topic, IMU_PUBLISHER_QOS
        )
        self._tf_broadcaster = (
            TransformBroadcaster(self) if self._publish_tf else None
        )
        self._stop_event = threading.Event()
        self._device_lock = threading.Lock()
        self._device = None
        self._dropped_samples = 0
        self._samples = queue.Queue(maxsize=512)
        self._publish_guard = self.create_guard_condition(
            self._publish_queued_samples
        )

        self._ble_thread = threading.Thread(
            target=self._run_ble_worker,
            name='bwt901_ble_worker',
            daemon=True,
        )
        self._ble_thread.start()

        selection = self._device_address or (
            f'first device containing "{self._device_name_filter}"'
        )
        self.get_logger().info(
            f'Publishing {selection} on {topic} with frame_id={self._frame_id}'
        )
        self.get_logger().info(f'Publishing magnetic field on {mag_topic}')
        if self._publish_tf:
            self.get_logger().info(
                f'Publishing rotation TF {self._tf_parent_frame} -> '
                f'{self._frame_id}'
            )

    def _restart_bluetooth_service(self) -> None:
        """Power-cycle the BLE adapter once at startup via org.bluez.

        BlueZ can leave stale connection state behind after a prior
        run's disconnect, which then makes the next BLE connect hang
        or fail even though the sensor itself is fine (the WitMotion
        phone app reconnects immediately because it never goes
        through BlueZ). Toggling the adapter off/on over D-Bus clears
        that state before the first connection attempt. This runs as
        an unprivileged D-Bus call (via bluetoothctl) rather than
        restarting bluetoothd itself, since the container has no
        root/sudo and a read-only rootfs; it needs the host's D-Bus
        socket mounted read-write (see docker-compose.yaml).
        """
        try:
            self.get_logger().info('Power-cycling the Bluetooth adapter')
            subprocess.run(
                ['bluetoothctl', 'power', 'off'],
                capture_output=True, timeout=15, check=True,
            )
            subprocess.run(
                ['bluetoothctl', 'power', 'on'],
                capture_output=True, timeout=15, check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.get_logger().warning(
                f'Could not power-cycle the Bluetooth adapter: {error}'
            )

    def _read_covariance_parameter(self, name: str) -> Tuple[float, ...]:
        parameter_values = self.get_parameter(name).value
        values = tuple(float(value) for value in parameter_values)
        if len(values) != 9:
            raise ValueError(f'{name} must contain exactly 9 values')
        return values

    def _device_updated(self, model: device_model.DeviceModel) -> None:
        if self._stop_event.is_set():
            return

        sample = dict(model.deviceData)
        try:
            self._samples.put_nowait(sample)
        except queue.Full:
            try:
                self._samples.get_nowait()
                self._samples.put_nowait(sample)
                self._dropped_samples += 1
            except (queue.Empty, queue.Full):
                return

        try:
            self._publish_guard.trigger()
        except RuntimeError:
            # The ROS context may already be shutting down.
            pass

    def _publish_queued_samples(self) -> None:
        while True:
            try:
                sample = self._samples.get_nowait()
            except queue.Empty:
                return

            self._publish_sample(sample)

    def _publish_sample(self, sample: Dict[str, float]) -> None:
        stamp = self.get_clock().now().to_msg()
        try:
            message = build_imu_message(
                sample=sample,
                stamp=stamp,
                frame_id=self._frame_id,
                orientation_covariance=self._orientation_covariance,
                angular_velocity_covariance=(
                    self._angular_velocity_covariance
                ),
                linear_acceleration_covariance=(
                    self._linear_acceleration_covariance
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().warning(
                f'Ignoring incomplete BWT901 sample: {error}',
                throttle_duration_sec=5.0,
            )
            return

        self._publisher.publish(message)
        if all(key in sample for key in ('HX', 'HY', 'HZ')):
            magnetic_message = build_magnetic_field_message(
                sample=sample,
                stamp=stamp,
                frame_id=self._frame_id,
                covariance=self._magnetic_field_covariance,
            )
            self._mag_publisher.publish(magnetic_message)
        if self._tf_broadcaster is not None:
            transform = build_rotation_transform(
                message,
                self._tf_parent_frame,
            )
            self._tf_broadcaster.sendTransform(transform)

    def _run_ble_worker(self) -> None:
        try:
            asyncio.run(self._connection_loop())
        # Keep BLE exceptions out of the ROS executor thread.
        except Exception as error:
            if not self._stop_event.is_set():
                self.get_logger().error(
                    f'BLE worker stopped: {type(error).__name__}: {error}'
                )

    async def _connection_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ble_device = await self._find_device()
                if ble_device is None:
                    self.get_logger().warning(
                        'BWT901 BLE device was not found; scanning again',
                        throttle_duration_sec=10.0,
                    )
                else:
                    identity = f'{ble_device.name} ({ble_device.address})'
                    self.get_logger().info(
                        f'Connecting to {identity}'
                    )
                    model = device_model.DeviceModel(
                        'BWT901BLE5.0',
                        ble_device,
                        self._device_updated,
                        on_connected=lambda: self.get_logger().info(
                            f'Connected to {identity}; streaming IMU data'
                        ),
                    )
                    with self._device_lock:
                        self._device = model
                    await model.openDevice()
                    if not self._stop_event.is_set():
                        self.get_logger().warning('BWT901 disconnected')
            except Exception as error:
                if not self._stop_event.is_set():
                    self.get_logger().error(
                        'BWT901 connection failed: '
                        f'{type(error).__name__}: {error}'
                    )
            finally:
                with self._device_lock:
                    self._device = None

            await self._wait_or_stop(self._reconnect_delay)

    async def _find_device(self):
        if self._device_address:
            self.get_logger().info(
                f'Scanning for BWT901 at {self._device_address}'
            )
            return await BleakScanner.find_device_by_address(
                self._device_address,
                timeout=self._scan_timeout,
            )

        self.get_logger().info(
            'Scanning for a BLE device whose name contains '
            f'"{self._device_name_filter}"'
        )
        devices = await BleakScanner.discover(timeout=self._scan_timeout)
        for candidate in devices:
            if (
                candidate.name is not None
                and self._device_name_filter in candidate.name
            ):
                return candidate
        return None

    async def _wait_or_stop(self, duration: float) -> None:
        loop = asyncio.get_running_loop()
        end_time = loop.time() + max(0.0, duration)
        while not self._stop_event.is_set() and loop.time() < end_time:
            await asyncio.sleep(min(0.2, end_time - loop.time()))

    def stop(self) -> None:
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        with self._device_lock:
            model = self._device
        if model is not None:
            model.closeDevice()
        if self._ble_thread.is_alive():
            self._ble_thread.join(timeout=3.0)

        if self._dropped_samples:
            self.get_logger().warning(
                f'Dropped {self._dropped_samples} queued IMU samples'
            )

    def destroy_node(self):
        self.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Bwt901BleImuNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
