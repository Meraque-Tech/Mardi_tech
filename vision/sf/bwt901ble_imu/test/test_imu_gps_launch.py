import ast
from pathlib import Path


def _keyword_string(call, keyword_name):
    keyword = next(
        item for item in call.keywords if item.arg == keyword_name
    )
    assert isinstance(keyword.value, ast.Constant)
    return keyword.value.value


def test_imu_gps_launch_requires_rtk_localization():
    launch_path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'imu_gps_raw.launch.py'
    )
    tree = ast.parse(launch_path.read_text(encoding='utf-8'))

    localization_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == 'localization_node'
            for target in node.targets
        )
    )
    localization_call = localization_assignment.value
    assert isinstance(localization_call, ast.Call)
    assert _keyword_string(localization_call, 'package') == 'rtk_localization'
    assert _keyword_string(localization_call, 'executable') == 'gnss_fix_enu_odom'
    assert _keyword_string(localization_call, 'name') == 'rtk_localization'
    assert all(
        keyword.arg != 'condition' for keyword in localization_call.keywords
    )

    launch_description = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'LaunchDescription'
    )
    actions = launch_description.args[0]
    assert isinstance(actions, ast.List)
    action_names = {
        item.id for item in actions.elts if isinstance(item, ast.Name)
    }
    assert 'localization_node' in action_names
    assert 'shutdown_on_localization_exit' in action_names

    shutdown_target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == 'target_action'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'localization_node'
    )
    assert shutdown_target.value.id == 'localization_node'
