from setuptools import find_packages, setup


package_name = "base_receiver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            ["config/receiver.yaml"],
        ),
        ("share/" + package_name + "/launch", ["launch/receiver.launch.py"]),
    ],
    install_requires=["setuptools", "pyserial", "pymap3d"],
    tests_require=["pytest"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "base_receiver = base_receiver.receiver_node:main",
            "gnss_enu = base_receiver.enu_node:main",
            "gnss_motion_classifier = base_receiver.motion_node:main",
        ],
    },
)
