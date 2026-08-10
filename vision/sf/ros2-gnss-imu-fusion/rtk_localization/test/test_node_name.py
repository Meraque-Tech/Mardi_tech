import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_default_launch_and_config_use_rtk_localization_node_name():
    node_source = (
        PACKAGE_ROOT / 'rtk_localization' / 'gnss_odom_node.py'
    ).read_text(encoding='utf-8')
    assert 'super().__init__("rtk_localization")' in node_source

    launch_tree = ast.parse(
        (PACKAGE_ROOT / 'launch' / 'gnss_odom.launch.py').read_text(
            encoding='utf-8'
        )
    )
    odometry_node = next(
        node
        for node in ast.walk(launch_tree)
        if isinstance(node, ast.Call)
        and any(
            keyword.arg == 'package'
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 'rtk_localization'
            for keyword in node.keywords
        )
    )
    node_name = next(
        keyword.value.value
        for keyword in odometry_node.keywords
        if keyword.arg == 'name'
    )
    assert node_name == 'rtk_localization'

    config_lines = (
        PACKAGE_ROOT / 'config' / 'gnss_odom.yaml'
    ).read_text(encoding='utf-8').splitlines()
    first_setting = next(line for line in config_lines if line.strip())
    assert first_setting == 'rtk_localization:'
