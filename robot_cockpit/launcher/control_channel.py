"""A private SSH TCP connection for control, never the image/preview master."""
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import tempfile

HOST = 'erazer@10.30.3.155'


def command(control, port, *, remote_port=8780):
    return ['ssh', '-M', '-S', str(control), '-fNT',
            '-o', 'ControlMaster=yes', '-o', 'ControlPersist=no',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=2',
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=no',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'Compression=no',
            '-o', 'NumberOfPasswordPrompts=1',
            '-L', f'127.0.0.1:{port}:127.0.0.1:{remote_port}', HOST]


@contextmanager
def dedicated_control(*, remote_port=8780):
    # A fresh private ControlPath forces a separate TCP transport even when
    # SSH config or the cockpit already has a shared master. No credential or
    # remote authorized_keys change is needed; normal SSH auth may prompt.
    with tempfile.TemporaryDirectory(prefix='book-control-') as folder:
        control = Path(folder)/'ssh.sock'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        print('建立独立控制连接，与图像/网页预览分流；SSH需要认证时请按提示输入。', flush=True)
        try:
            subprocess.run(command(control, port, remote_port=remote_port), check=True)
            subprocess.run(['ssh', '-S', str(control), '-O', 'check', HOST],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3)
            yield f'ws://127.0.0.1:{port}/ws'
        finally:
            # Only this context's randomly named master; never terminate the
            # shared cockpit/camera SSH or the robot services it carries.
            if control.exists():
                try:
                    subprocess.run(['ssh', '-S', str(control), '-O', 'exit', HOST],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=3, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    print('独立控制SSH尚未确认退出；未操作共享连接。', flush=True)
