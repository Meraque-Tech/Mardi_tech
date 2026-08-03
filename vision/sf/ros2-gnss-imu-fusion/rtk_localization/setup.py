from glob import glob
import os

from setuptools import find_packages, setup


package_name = "rtk_localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=["setuptools", "pymap3d"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "gnss_fix_enu_odom = rtk_localization.gnss_odom_node:main",
            # Preserve the executable name used by the original package.
            "gnss_pvt_enu_odom = rtk_localization.gnss_odom_node:main",
        ],
    },
)
