"""V3.4 through a local V3.1 planner connected to the 155 cockpit.

Both head RGB-D and wrist RGB are read from 155 over authenticated SSH.
The original camera calibration checks remain active; no local USB fallback.
This launcher never starts a planner, cockpit, RT, or enables robot control.
"""
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
PLANNER_ROOT = ROOT.parent/'dual_v2_2_control/cuarm_upper_planner'
PLANNER_ENTRY = PLANNER_ROOT/'dual_v2_2_upper_planner_v3_1.py'
DEFAULT_ROBOT_HOST = '10.30.3.155'
REMOTE_ADAPTER = PLANNER_ROOT/'scripts/v3_1_seldon.py'
REMOTE_LINK_REPORT = Path(f'/tmp/cuarm-v31-seldon-link-{os.getuid()}-30241.json')


def robot_address(value):
    address = ipaddress.IPv4Address(value)
    if address.is_unspecified or address.is_loopback or address.is_multicast or str(address) == '255.255.255.255':
        raise ValueError('Remote robot must have a concrete LAN IPv4 address')
    return str(address)


def require_remote_adapter(pid, host):
    try:
        report = json.loads(REMOTE_LINK_REPORT.read_text())
        if PLANNER_ENTRY.name == 'dual_v2_2_upper_planner_board.py':
            from v3_1_panel_process import verify_publisher
            verify_publisher(report, pid)
        elif type(report['pid']) is not int or report['pid'] != pid:
            raise ValueError('诊断记录不属于当前 Planner')
        if report['remote'] != [host, 30240]:
            raise ValueError('诊断记录的机器人地址不匹配')
        stamp = report['updated_monotonic']
        if type(stamp) not in (int, float) or not 0 <= time.monotonic()-stamp <= 5.:
            raise ValueError('远程链路诊断记录已过期')
        if report['error'] is not None:
            raise ValueError('远程链路报告故障：'+str(report['error']))
        if report['adapter_sha256'] != hashlib.sha256(REMOTE_ADAPTER.read_bytes()).hexdigest():
            raise ValueError('运行中的 Planner 仍加载旧版远程适配器')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError('远程适配器未就绪，请重启本机 V3.1 Planner（无需重启 155）。原因：'+str(error)) from error


def require_remote_planner(host):
    host = robot_address(host)
    scripts = str(PLANNER_ROOT/'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import v3_1_readiness as readiness
    readiness.require_ready(10176)
    try:
        receipt = json.loads(readiness.receipt_path(10176).read_text())
        pid = int(receipt['pid'])
        if pid <= 0 or receipt['start'] != readiness.process_start(pid):
            raise ValueError('Planner process changed')
        process = Path('/proc')/str(pid)
        argv = (process/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        cwd = (process/'cwd').resolve(strict=True)
        entries = [index for index, value in enumerate(argv)
                   if Path(value).name == PLANNER_ENTRY.name and
                   (Path(value) if Path(value).is_absolute() else cwd/value).resolve() == PLANNER_ENTRY.resolve()]
        if len(entries) != 1:
            raise ValueError('Unexpected planner entry')
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument('--robot-host', nargs='?', const=DEFAULT_ROBOT_HOST)
        parser.add_argument('--planner-bind', default='0.0.0.0')
        settings, unknown = parser.parse_known_args(argv[entries[0]+1:])
        if unknown or settings.robot_host is None or robot_address(settings.robot_host) != host:
            raise ValueError('Planner is not connected to the requested remote robot')
        environment = dict(item.split('=', 1) for item in
                           (process/'environ').read_bytes().decode().split('\0') if '=' in item)
        expected = {'CUARM_V31_TARGET_LOCAL_IP': '127.0.0.1',
                    'CUARM_V31_TARGET_REMOTE_IP': '127.0.0.1',
                    'CUARM_V31_TARGET_LOCAL_PORT': '10176',
                    'CUARM_V31_TARGET_REMOTE_PORT': '10175'}
        if any(environment.get(key, value) != value for key, value in expected.items()):
            raise ValueError('Planner Agent endpoints differ from 127.0.0.1:10175/10176')
        require_remote_adapter(pid, host)
        readiness.require_ready(10176)
        if receipt['start'] != readiness.process_start(pid) or receipt != json.loads(readiness.receipt_path(10176).read_text()):
            raise ValueError('Planner changed during readiness check')
    except (OSError, KeyError, TypeError, ValueError, SystemExit) as error:
        raise ValueError(
            f'请先启动 {PLANNER_ENTRY.name} --robot-host {host}，等待 Curobo is ready；'
            f'本入口拒绝本机路由、其他机器人或不匹配的 Agent 端口。原因：{error}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--robot-host', default=DEFAULT_ROBOT_HOST,
                        help='Expected robot address; must match the already-running V3.1 planner')
    parser.add_argument('--ssh-user', default='erazer')
    parser.add_argument('--ssh-control', type=Path, help='Existing SSH ControlMaster socket (optional with key authentication)')
    parser.add_argument('--remote-python', default='/home/erazer/Seldon/keyboard_cockpit/.venv/bin/python')
    parser.add_argument('--head-port', type=int, default=8792, help='155 loopback head-rgbd-v1 service; must publish native RGB-D')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check-config', action='store_true', help='Offline V3.4 configuration/calibration check')
    checks.add_argument('--check-planner-only', action='store_true', help='Read local readiness and process metadata only')
    checks.add_argument('--check-cameras', action='store_true', help='Capture head/wrist on 155 only; no planner or robot connection')
    args, remaining = parser.parse_known_args(argv)
    try:
        host = robot_address(args.robot_host)
    except ValueError as error:
        parser.error(str(error))
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_-]*', args.ssh_user) or not Path(args.remote_python).is_absolute():
        parser.error('Use a valid SSH username and absolute remote Python path')
    if not 1024 <= args.head_port <= 65535:
        parser.error('head-port must be 1024..65535')
    if (args.check_planner_only or args.check_cameras) and remaining:
        parser.error('Unknown planner check arguments: '+' '.join(remaining))
    if any(value.split('=', 1)[0].startswith('--remote-rgbd') or value.startswith('--remote-camera-') for value in remaining):
        parser.error('Use this entry SSH/head-port options; legacy remote-rgbd overrides are not supported')
    settings = dict(host=host, ssh_user=args.ssh_user, ssh_control=args.ssh_control,
                    remote_python=args.remote_python, head_port=args.head_port)
    print(f'V3.4 remote: Agent 127.0.0.1:10175 → local V3.1 :10176 → {host}:30240', flush=True)
    print(f'头部 RGB-D 与腕部 RGB 均来自 {host}；保留标定校验，不回退到本机相机。', flush=True)
    if not args.check_config and not args.check_cameras:
        require_remote_planner(host)
    if args.check_planner_only:
        print('远程 Planner 启动目标与就绪记录检查通过；未连接相机或发送机器人命令。')
        return
    if not args.check_config:
        from cart_book_grasp_v3_4_remote_camera import check_sources, remote_cameras
        check_sources(settings, include_wrist=args.check_cameras)
        if args.check_cameras:
            return
    from cuart_book_grasp_v3_4 import main as original_main
    if args.check_config:
        return original_main(['--check-config']+remaining)
    with remote_cameras(settings):
        return original_main(remaining)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(f'V3.4 remote: {error}', file=sys.stderr)
        raise SystemExit(2)
