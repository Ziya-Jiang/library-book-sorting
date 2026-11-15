"""Run on155: ensure camera-only dependencies before opening a display subscription."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

HERE = Path(__file__).resolve().parent
COCKPIT = Path('/home/erazer/Seldon/keyboard_cockpit')
OPENER = build_opener(ProxyHandler({}))


def health(port, path):
    try:
        with OPENER.open(f'http://127.0.0.1:{port}{path}', timeout=1) as response:
            value = json.load(response)
        if port == 8792:
            return value if 'head' in value and 'wrist' in value else None
        return value if value.get('runtime') == '155' and value.get('fps_per_channel', 0) > 0 else None
    except (OSError, ValueError):
        return None


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), .3):
            return True
    except OSError:
        return False


def wait_health(port, path, timeout):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        result = health(port, path)
        if result is not None:
            return result
        time.sleep(.25)
    raise RuntimeError(f'155端口{port}{path}未就绪；请查看相应服务日志。')


def ensure_ready():
    # Serialize this tool's preparation without touching the robot control lock/session.
    lock_path = Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}'))/'vision-tool-start.lock'
    with lock_path.open('a') as lock:
        deadline = time.monotonic()+35
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError('另一Visualization Tool仍在准备相机，请稍后重试。')
                time.sleep(.2)
        if health(8792, '/status') is None:
            if listening(8792):
                wait_health(8792, '/status', 10)
            else:
                print('155：启动共享头部/右腕图像源（8792）。', flush=True)
                runtime = COCKPIT/'runtime'
                runtime.mkdir(exist_ok=True)
                logfile = runtime/'vision_tool_head.log'
                with logfile.open('ab') as output:
                    subprocess.Popen([str(COCKPIT/'.venv/bin/python'), '-m', 'seldon_cockpit.head_service'],
                                     cwd=COCKPIT, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                try:
                    wait_health(8792, '/status', 25)
                except RuntimeError as error:
                    raise RuntimeError(f'{error} 相机日志：{logfile}') from error
        else:
            print('155：共享图像源8792已就绪。', flush=True)
        if health(8796, '/health') is None:
            if listening(8796):
                wait_health(8796, '/health', 5)
            else:
                print('155：启动Vision Tool（8796）。', flush=True)
                r = subprocess.run(['systemctl', '--user', 'start', 'deepseek-camera-tool.service'],
                                   capture_output=True, text=True, timeout=15)
                if r.returncode:
                    raise RuntimeError('Vision Tool启动失败：'+r.stderr[-1000:])
                wait_health(8796, '/health', 12)
        print('155：Vision Tool8796已就绪；仅启动图像服务。', flush=True)


if __name__ == '__main__':
    try:
        ensure_ready()
    except Exception as error:
        print(f'图像服务准备失败：{error}', file=sys.stderr)
        sys.exit(1)
