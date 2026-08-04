import math
import unittest

from builtin_interfaces.msg import Time

from bwt901ble_imu.imu_publisher import build_imu_message
from bwt901ble_imu.imu_publisher import build_magnetic_field_message
from bwt901ble_imu.imu_publisher import build_rotation_transform


ZERO_COVARIANCE = [0.0] * 9


class TestImuConversion(unittest.TestCase):

    def test_sdk_units_are_converted_to_ros_si_units(self):
        message = build_imu_message(
            sample={
                'AccX': 1.0,
                'AccY': -0.5,
                'AccZ': 0.0,
                'AsX': 180.0,
                'AsY': -90.0,
                'AsZ': 0.0,
                'AngX': 0.0,
                'AngY': 0.0,
                'AngZ': 90.0,
            },
            stamp=Time(sec=12, nanosec=34),
            frame_id='imu_link',
            orientation_covariance=ZERO_COVARIANCE,
            angular_velocity_covariance=ZERO_COVARIANCE,
            linear_acceleration_covariance=ZERO_COVARIANCE,
        )

        self.assertTrue(
            math.isclose(message.linear_acceleration.x, 9.80665)
        )
        self.assertTrue(
            math.isclose(message.linear_acceleration.y, -4.903325)
        )
        self.assertTrue(math.isclose(message.angular_velocity.x, math.pi))
        self.assertTrue(
            math.isclose(message.angular_velocity.y, -math.pi / 2.0)
        )
        self.assertTrue(
            math.isclose(message.orientation.w, math.sqrt(0.5))
        )
        self.assertTrue(
            math.isclose(message.orientation.z, math.sqrt(0.5))
        )
        self.assertEqual(message.header.frame_id, 'imu_link')
        self.assertEqual(message.header.stamp.sec, 12)

    def test_device_quaternion_is_preferred_and_normalized(self):
        message = build_imu_message(
            sample={
                'AccX': 0.0,
                'AccY': 0.0,
                'AccZ': 1.0,
                'AsX': 0.0,
                'AsY': 0.0,
                'AsZ': 0.0,
                'AngX': 0.0,
                'AngY': 0.0,
                'AngZ': 0.0,
                'Q0': 2.0,
                'Q1': 0.0,
                'Q2': 0.0,
                'Q3': 2.0,
            },
            stamp=Time(),
            frame_id='imu_link',
            orientation_covariance=ZERO_COVARIANCE,
            angular_velocity_covariance=ZERO_COVARIANCE,
            linear_acceleration_covariance=ZERO_COVARIANCE,
        )

        self.assertTrue(
            math.isclose(message.orientation.w, math.sqrt(0.5))
        )
        self.assertTrue(
            math.isclose(message.orientation.z, math.sqrt(0.5))
        )

    def test_rotation_transform_matches_imu_header_and_orientation(self):
        message = build_imu_message(
            sample={
                'AccX': 0.0,
                'AccY': 0.0,
                'AccZ': 1.0,
                'AsX': 0.0,
                'AsY': 0.0,
                'AsZ': 0.0,
                'AngX': 0.0,
                'AngY': 0.0,
                'AngZ': 90.0,
            },
            stamp=Time(sec=12, nanosec=34),
            frame_id='imu_link',
            orientation_covariance=ZERO_COVARIANCE,
            angular_velocity_covariance=ZERO_COVARIANCE,
            linear_acceleration_covariance=ZERO_COVARIANCE,
        )

        transform = build_rotation_transform(message, 'imu_reference')

        self.assertEqual(transform.header.frame_id, 'imu_reference')
        self.assertEqual(transform.child_frame_id, 'imu_link')
        self.assertEqual(transform.header.stamp.sec, 12)
        self.assertEqual(transform.transform.translation.x, 0.0)
        self.assertEqual(
            transform.transform.rotation,
            message.orientation,
        )

    def test_magnetic_field_is_converted_to_tesla(self):
        message = build_magnetic_field_message(
            sample={
                'HX': 3.0,
                'HY': -4.0,
                'HZ': 5.0,
            },
            stamp=Time(sec=12, nanosec=34),
            frame_id='imu_link',
            covariance=ZERO_COVARIANCE,
        )

        self.assertTrue(math.isclose(message.magnetic_field.x, 36.0e-6))
        self.assertTrue(math.isclose(message.magnetic_field.y, -48.0e-6))
        self.assertTrue(math.isclose(message.magnetic_field.z, 60.0e-6))
        self.assertEqual(message.header.frame_id, 'imu_link')
        self.assertEqual(message.header.stamp.sec, 12)
