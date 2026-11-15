"""Local startup receipt: clients reject a pre-V3.1 or outdated planner process."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

class PlannerRestartRequired(ValueError):
    """A live planner loaded a different source version; waiting cannot fix it."""

def fingerprint():
    return hashlib.sha256(b''.join(p.read_bytes() for p in (
        ROOT/'dual_v2_2_upper_planner_v3_1.py',ROOT/'scripts/curobo_planner_v3_1.py',
        ROOT/'scripts/servo_cartesian_policy_v3_1.py',ROOT/'scripts/v3_1_feedback.py',
        ROOT/'scripts/v3_1_cpu_fk.py',ROOT/'scripts/v3_1_panel_path.py',
        ROOT/'scripts/v3_1_panel_path_bridge.cpp',ROOT/'scripts/v3_1_execution.py',
        ROOT/'scripts/v3_1_motion_bridge.py',
        ROOT/'scripts/book_lower_pose.py',
        ROOT/'scripts/baseline_withdraw_path.py',
        ROOT/'scripts/baseline_approach_path.py',ROOT/'scripts/baseline_approach_planner.py',
        ROOT/'scripts/baseline_ready_pose.py',
        ROOT/'scripts/baseline_plan_speed.py', ROOT/'scripts/planner_lifecycle.py',
        ROOT/'scripts/book_post_lift_pose.py',
        ROOT/'scripts/book_planning_budget.py', ROOT/'scripts/book_planning_diagnostics.py',
        ROOT/'scripts/v3_1_direct_servo.py',ROOT/'scripts/v3_1_direct_servo.json',
        ROOT.parent/'cuarm_rt_control/cuarm_robotics/src/interpolation.cpp',
        ROOT.parent/'cuarm_rt_control/cuarm_robotics/src/interpolation.h',Path(__file__)))).hexdigest()

def process_start(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]

def receipt_path(port):return Path(f'/tmp/cuarm-v31-{os.getuid()}-{int(port)}.json')

def publish_ready(port):
    path=receipt_path(port);tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),start=process_start(os.getpid()),fingerprint=fingerprint(),port=int(port))))
    tmp.replace(path)

def require_ready(port=10176):
    try:
        receipt=json.loads(receipt_path(port).read_text())
        if receipt['port']!=int(port) or receipt['start']!=process_start(int(receipt['pid'])):
            raise ValueError('stale planner identity')
        if receipt['fingerprint']!=fingerprint():
            raise PlannerRestartRequired(
                f"Planner PID={receipt['pid']}仍在运行，但代码已修改；必须正常重启Planner加载新代码。"
                '仅修改抓取参数不需要重启；本次不是网络断流，也不要启动第二份Planner。')
    except PlannerRestartRequired:
        raise
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise ValueError('请先启动独立规划器 dual_v2_2_upper_planner_v3_1.py（端口 10176）；V3.1 不连接原规划器。') from error


def check_panel_port_available(endpoint):
    import socket
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
        try:probe.bind(endpoint)
        except OSError as error:
            raise RuntimeError(f'Panel UDP {endpoint} is in use. Stop the previous planner before starting V3.1; do not run both against one robot.') from error
