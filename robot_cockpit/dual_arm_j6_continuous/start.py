"""Prepare the 155 dual-gripper services and local CuRobo from one command."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import socket
import stat
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parent
V = ROOT.parents[1]
RUNTIME = ROOT / 'runtime'
SOCKET = ROOT.parent / 'dual_arm_control/runtime/ssh.sock'
PLANNER = ROOT / 'planner.py'
CUROBO = '/mnt/ssd/robotarm/env/curobo/bin/python'
ACTIVE_RELEASE = RUNTIME / 'active_release.json'
REMOTE = (json.loads(ACTIVE_RELEASE.read_text())['remote'] if ACTIVE_RELEASE.is_file()
          else '/mnt/data/erazer/book_grasp/dual_arm_control_runtime/candidate_20261111')
REMOTE_PYTHON = '/home/erazer/Seldon/keyboard_cockpit/.venv/bin/python'
ROBOT_IP = '10.30.3.155'
HOST = 'erazer@' + ROBOT_IP
PC_IP = '10.30.3.100'
GUI = 'http://127.0.0.1:8811'
TARGET_PORT = 10176
FEEDBACK_PORT = 30241
READY = Path(f'/tmp/cuarm-dual-arm-j6-continuous-{os.getuid()}-{TARGET_PORT}.json')
LINK = Path(f'/tmp/cuarm-v31-seldon-link-{os.getuid()}-{FEEDBACK_PORT}.json')


def ssh_check():
    env = dict(os.environ, LC_ALL='C')
    return subprocess.run(['ssh', '-S', str(SOCKET), '-O', 'check', HOST],
                          capture_output=True, text=True, env=env, timeout=5)


def ensure_tunnel():
    result = ssh_check()
    if result.returncode == 0:
        return
    # Only remove our own socket when OpenSSH confirms there is no listener.
    if SOCKET.exists():
        old = SOCKET.lstat()
        if (not stat.S_ISSOCK(old.st_mode) or old.st_uid != os.getuid()
                or 'Connection refused' not in result.stderr):
            raise RuntimeError('无法复用现有 SSH 转发，请检查 ' + str(SOCKET))
        SOCKET.unlink()
    print('连接155，必要时按终端提示输入SSH密码。', flush=True)
    subprocess.run(['ssh', '-M', '-S', str(SOCKET), '-fNT', '-o', 'ConnectTimeout=8',
                    '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=10',
                    '-o', 'ServerAliveCountMax=3', '-L', '127.0.0.1:8811:127.0.0.1:8810',
                    HOST], check=True, timeout=120)
    if ssh_check().returncode:
        raise RuntimeError('155的 SSH 转发未就绪')


def remote(argv, *, use_master=True, timeout=90):
    command = ['ssh', '-T', '-o', 'ConnectTimeout=8']
    if use_master:
        command.extend(['-S', str(SOCKET), '-o', 'BatchMode=yes'])
    result = subprocess.run(command + [HOST, shlex.join(argv)], capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        # The remote launcher sometimes prints a usable URL, including its token.
        output = (result.stderr + '\n' + result.stdout).splitlines()
        raise RuntimeError('\n'.join(line for line in output if '#token=' not in line).strip()
                           or f'155启动步骤失败（退出码{result.returncode}）')
    return result.stdout


def publish_launcher(*, use_master=True):
    """Publish source as an immutable launcher; never edit active release files."""
    source = (ROOT/'start_155.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    destination = '/mnt/data/erazer/book_grasp/dual_arm_control_runtime/launcher_releases/'+digest+'/start_155.py'
    installer = '''import hashlib, pathlib, sys
data=sys.stdin.buffer.read()
path=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError('Launcher source checksum mismatch')
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    if path.read_bytes()!=data: raise RuntimeError('Immutable launcher differs')
else:
    with path.open('xb') as f: f.write(data)
if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise RuntimeError('Installed launcher checksum mismatch')
'''
    command = ['ssh', '-T', '-o', 'ConnectTimeout=8']
    if use_master:
        command += ['-S', str(SOCKET), '-o', 'BatchMode=yes']
    command += [HOST, shlex.join([REMOTE_PYTHON, '-c', installer, destination, digest])]
    subprocess.run(command, input=source, check=True, timeout=20,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    record = dict(source=str(ROOT/'start_155.py'), sha256=digest, remote=destination,
                  profile_root=REMOTE, startup_reference='accept-current',
                  hardware_started_by_publish=False)
    temporary = RUNTIME/'launcher_release.tmp'
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(RUNTIME/'launcher_release.json')
    return destination


def start_remote(*args, use_master=True):
    launcher = publish_launcher(use_master=use_master)
    result = remote(['env', 'DUAL_GRIPPER_PROFILE_ROOT='+REMOTE,
                     REMOTE_PYTHON, launcher, *args], use_master=use_master)
    for line in result.splitlines():
        if '#token=' not in line:
            print(line, flush=True)


def precheck():
    if not Path(CUROBO).is_file() or not PLANNER.is_file():
        raise RuntimeError('找不到本机CuRobo或双夹爪Planner入口')
    route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', ROBOT_IP], text=True))
    if not route or route[0].get('prefsrc') != PC_IP:
        raise RuntimeError(f'到155的路由源地址应为{PC_IP}，请检查有线网卡')


def existing_planner():
    pids = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if os.fsencode(PLANNER) in argv[1:]:
            pids.append(int(path.parent.name))
    if len(pids) > 1:
        raise RuntimeError('发现重复双夹爪Planner，请正常停止重复实例')
    return pids[0] if pids else None


def check_ports():
    for address in [('127.0.0.1', TARGET_PORT), (PC_IP, FEEDBACK_PORT)]:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.bind(address)
            except OSError as error:
                raise RuntimeError(f'UDP {address[0]}:{address[1]} 已占用，无法启动双夹爪Planner') from error


def launch_planner():
    check_ports()
    env = dict(os.environ, BOOK_PACKET_TRACE_DIR=str(RUNTIME / 'packet_trace'),
               CUARM_V31_TARGET_LOCAL_IP='127.0.0.1', CUARM_V31_TARGET_REMOTE_IP='127.0.0.1',
               CUARM_V31_TARGET_LOCAL_PORT=str(TARGET_PORT), CUARM_V31_TARGET_REMOTE_PORT='10175')
    with (RUNTIME / 'planner.log').open('ab', buffering=0) as log:
        log.write(f'\n双夹爪启动 unix_s={time.time():.6f}\n'.encode())
        return subprocess.Popen([CUROBO, '-u', str(PLANNER)], cwd=V, env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def wait_planner(pid, process):
    shared = V / 'dual_v2_2_control/cuarm_upper_planner/scripts'
    adapters = ROOT / 'planner_scripts'
    sys.path[:0] = [str(adapters), str(shared)]
    from v3_1_panel_process import verify_publisher
    from v3_1_readiness import fingerprint, process_start

    start = process_start(pid)
    expected_code = fingerprint()
    adapter_sha = hashlib.sha256((adapters / 'v3_1_seldon.py').read_bytes()).hexdigest()
    deadline = time.monotonic() + 100
    previous = None
    reason = None
    while time.monotonic() < deadline:
        if (process is not None and process.poll() is not None) or not Path(f'/proc/{pid}').exists():
            raise RuntimeError('双夹爪Planner已退出，请查看 ' + str(RUNTIME / 'planner.log'))
        try:
            receipt = json.loads(READY.read_text())
            if (receipt['pid'], receipt['start'], receipt['port']) != (pid, start, TARGET_PORT):
                raise ValueError('等待当前Planner初始化')
            if receipt['fingerprint'] != expected_code:
                raise RuntimeError(f'Planner PID={pid} 的代码版本已变化，请正常重启该Planner')
            report = json.loads(LINK.read_text())
            verify_publisher(report, pid)
            if (report['error'] is not None or report['remote'] != [ROBOT_IP, 30240]
                    or report['adapter_sha256'] != adapter_sha
                    or not 0 <= time.monotonic() - report['updated_monotonic'] <= 5):
                raise ValueError('155反馈尚未恢复，或通信地址和版本不匹配')
            current = (report['publisher_pid'], report['updated_monotonic'], report['received'])
            if previous is not None and current[0] == previous[0] and current[1] > previous[1] and current[2] > previous[2]:
                return report
            if previous is None or current[0] != previous[0] or current[1] > previous[1]:
                previous = current
            detail = 'CuRobo已初始化，等待155反馈计数增长'
        except (OSError, ValueError, KeyError, TypeError) as error:
            previous = None
            detail = str(error)
        if detail != reason:
            print('等待Planner：' + detail, flush=True)
            reason = detail
        time.sleep(.4)
    raise RuntimeError('Planner未收到持续反馈：' + str(reason) + '。日志：' + str(RUNTIME / 'planner.log'))


def gui_status():
    url = json.loads(remote(['cat', REMOTE + '/runtime/web.json'], timeout=10))['url']
    parsed = urlsplit(url)
    if (parsed.scheme, parsed.netloc, parsed.path) != ('http', '127.0.0.1:8810', '/'):
        raise RuntimeError('双夹爪工作台地址不匹配')
    token = parse_qs(parsed.fragment)['token'][0]
    opener = build_opener(ProxyHandler({}))
    with opener.open(GUI + '/', timeout=4) as response:
        if '双臂工作台' not in response.read(100_000).decode():
            raise RuntimeError('正式工作台未发布，请检查155的GUI版本')
    request = Request(GUI + '/api/state', headers={'X-Arm-Token': token})
    with opener.open(request, timeout=4) as response:
        state = json.load(response)
    if (state.get('rt_assembly') != 'dual_gripper' or state.get('gripper_installed') != [True, True]
            or state.get('simulation') or not state.get('rt_running') or not state.get('fresh')
            or not 0 <= state.get('age_ms', -1) <= 250):
        raise RuntimeError('工作台未取得新双夹爪配置的实时RT反馈')
    return GUI + "/#" + parsed.fragment, state


def restart_idle_planner(pid):
    """Normal shutdown of this profile only; never revoke an active owner."""
    if pid is None:
        return
    _, state = gui_status()
    if state.get('active') or (state.get('preparation') or {}).get('active') or state['planner'].get('enabled'):
        raise RuntimeError('当前仍有活动控制；先在原程序或驾驶舱正常停止外部Planner控制，再用 --restart-planner。RT保持运行。')
    if existing_planner() != pid:
        raise RuntimeError('Planner进程发生变化，取消重启')
    print(f'正常退出旧Planner PID={pid}；不关闭155 RT。', flush=True)
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic()+12.
    while existing_planner() == pid:
        if time.monotonic() >= deadline:
            raise RuntimeError('旧Planner未正常退出；未强杀、未启动第二份')
        time.sleep(.1)



def launch_optional_preview_cameras():
    """Best effort background image-service startup; never gate RT/Planner on head RGB-D."""
    log_path = RUNTIME / 'optional_preview_cameras.log'
    try:
        with log_path.open('ab') as log:
            subprocess.Popen(
                ['ssh', '-T', '-S', str(SOCKET), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                 HOST, shlex.join(['timeout', '30s', REMOTE_PYTHON,
                                  '/home/erazer/deepseek_camera_tool/ensure_ready.py'])],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print('头部/右腕预览服务后台准备，不等待其就绪；任务测量只使用左腕D405。日志：'+str(log_path), flush=True)
    except OSError as error:
        print('可选预览服务未启动（不影响任务）：'+str(error), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--restart-planner', action='store_true', help='控制空闲时正常退出旧CuRobo并加载新版，不关闭155 RT')
    mode.add_argument('--rt-only', action='store_true', help='只准备双臂RT和工作台')
    mode.add_argument('--check', action='store_true', help='只检查文件和155运行配置')
    args = parser.parse_args()
    if not args.rt_only:
        precheck()
    if args.check:
        start_remote('--check', use_master=ssh_check().returncode == 0)
        return

    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / 'startup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('另一个双夹爪启动程序正在运行') from error
        pid = None if args.rt_only else existing_planner()
        ensure_tunnel()
        if not args.rt_only and pid is None:
            check_ports()
        if not args.rt_only:
            launch_optional_preview_cameras()
        print('准备155双臂RT和本机工作台。', flush=True)
        start_remote('--rt-only' if args.rt_only else '--planner-host', *(() if args.rt_only else (PC_IP,)))
        url, state = gui_status()
        if args.restart_planner:
            restart_idle_planner(pid)
            pid = existing_planner()
        if not args.rt_only:
            process = None
            if pid is None:
                if state.get('active') or state.get('preparation', {}).get('active') or state['planner'].get('enabled'):
                    raise RuntimeError('已有活动控制会话，当前无法启动新的Planner')
                process = launch_planner()
                pid = process.pid
                print(f'已启动本机CuRobo PID={pid}，日志 {RUNTIME / "planner.log"}', flush=True)
            else:
                print(f'复用本机CuRobo PID={pid}。', flush=True)
            report = wait_planner(pid, process)
            from j6_only import require_capability
            require_capability(pid)
            url, state = gui_status()
            planner = state['planner']
            if (not planner['connected'] or planner['ip'] != PC_IP or planner['port'] != FEEDBACK_PORT
                    or planner['local_port'] != 30240):
                raise RuntimeError('155侧Planner反馈地址不匹配')
            print(f'Planner已连接，155→本机反馈持续到达，通信进程 PID={report["publisher_pid"]}。', flush=True)
        print(f'服务准备完成。双臂RT反馈 {state["age_ms"]} ms。', flush=True)
        print('本机页面：' + url, flush=True)
        opener = shutil.which('xdg-open')
        if opener and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            subprocess.Popen([opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('双臂启动未完成：' + (str(error) or type(error).__name__), file=sys.stderr)
        raise SystemExit(1)
