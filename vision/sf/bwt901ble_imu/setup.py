from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'bwt901ble_imu'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    py_modules=['device_model'],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py')),
        ),
    ],
    install_requires=['setuptools', 'bleak>=0.20'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='dj',
    maintainer_email='debanikroy92@gmail.com',
    description=(
        'ROS 2 sensor_msgs/Imu publisher for the WitMotion BWT901BLE5.0.'
    ),
    license='Meraque Proprietary License',
    entry_points={
        'console_scripts': [
            'imu_publisher = bwt901ble_imu.imu_publisher:main',
        ],
    },
)
