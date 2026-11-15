"""One managed Rerun viewer shared by sequential right-arm trial runs."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

DIRECTORY = Path('/tmp') / f'V-right-trial-viewer-{os.getuid()}'


def identity(pid, executable, port):
    try:
        root = Path('/proc') / str(int(pid))
        args = (root/'cmdline').read_bytes().split(b'\0')
        if not args or Path(os.fsdecode(args[0])).resolve() != executable.resolve():
            return None
        slot = args.index(b'--port')
        if args[slot+1] != str(port).encode():
            return None
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # Linux process start ticks: excludes PID reuse.
    except (OSError, ValueError, IndexError):
        return None


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.15):
            return True
    except OSError:
        return False


def ensure_viewer(executable):
    executable = Path(executable)
    DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
    registry = DIRECTORY/'viewer.json'
    with (DIRECTORY/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            record = json.loads(registry.read_text())
            valid = (type(record['port']) is int and 0 < record['port'] < 65536
                     and record.get('start') is not None
                     and identity(record['pid'], executable, record['port']) == record['start'])
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = dict(os.environ, RERUN_APP_ONLY='true')
            with (DIRECTORY/'viewer.log').open('wb') as log:
                process = subprocess.Popen([str(executable), '--bind', '127.0.0.1', '--port', str(port),
                    '--memory-limit', '512MiB', '--hide-welcome-screen', '--newest-first'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
            stamp = identity(process.pid, executable, port)
            # Popen can return before the child has exec'd the viewer binary.
            deadline = time.monotonic()+3.
            while stamp is None and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
                stamp = identity(process.pid, executable, port)
            if stamp is None:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3.)
                raise RuntimeError('Rerun窗口进程未启动；查看 '+str(DIRECTORY/'viewer.log'))
            record = dict(pid=process.pid, port=port, start=stamp)
            registry.write_text(json.dumps(record))
        deadline = time.monotonic()+8.
        while not listening(record['port']):
            if identity(record['pid'], executable, record['port']) != record['start']:
                raise RuntimeError('Rerun窗口已退出；查看 '+str(DIRECTORY/'viewer.log'))
            if time.monotonic() >= deadline:
                raise RuntimeError('Rerun窗口尚未就绪；保留进程，不另开第二个窗口')
            time.sleep(.05)
        print(('复用' if valid else '打开')+'右臂试验Rerun单窗口。', flush=True)
        return record['port']
