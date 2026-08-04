import device_model
import unittest


def _little_endian_int16(value):
    unsigned = value & 0xFFFF
    return [unsigned & 0xFF, unsigned >> 8]


class TestDeviceModel(unittest.TestCase):

    def test_combined_imu_frame_is_decoded(self):
        callbacks = []
        model = device_model.DeviceModel('test', None, callbacks.append)

        frame = [0x55, 0x61]
        for raw_value in (
            2048,    # AccX = 1 g
            -4096,   # AccY = -2 g
            1024,    # AccZ = 0.5 g
            16384,   # AsX = 1000 degrees/s
            -8192,   # AsY = -500 degrees/s
            0,       # AsZ = 0 degrees/s
            16384,   # AngX = 90 degrees
            -8192,   # AngY = -45 degrees
            0,       # AngZ = 0 degrees
        ):
            frame.extend(_little_endian_int16(raw_value))

        model.processData(frame)

        self.assertEqual(callbacks, [model])
        self.assertEqual(model.get('AccX'), 1.0)
        self.assertEqual(model.get('AccY'), -2.0)
        self.assertEqual(model.get('AccZ'), 0.5)
        self.assertEqual(model.get('AsX'), 1000.0)
        self.assertEqual(model.get('AsY'), -500.0)
        self.assertEqual(model.get('AngX'), 90.0)
        self.assertEqual(model.get('AngY'), -45.0)

    def test_quaternion_register_response_is_decoded(self):
        model = device_model.DeviceModel('test', None, lambda unused: None)
        frame = [0x55, 0x71, 0x51, 0x00]
        for raw_value in (16384, 0, 0, -16384):
            frame.extend(_little_endian_int16(raw_value))
        frame.extend([0] * (20 - len(frame)))

        model.processData(frame)

        self.assertEqual(model.get('Q0'), 0.5)
        self.assertEqual(model.get('Q1'), 0.0)
        self.assertEqual(model.get('Q2'), 0.0)
        self.assertEqual(model.get('Q3'), -0.5)
