"""Private launcher/agent startup handshake; standalone entries remain unchanged."""
from contextlib import contextmanager
import json
import os
import socket
import time

ENV_FD = 'BOOK_GRASP_STARTUP_FD'
_gate = None


class StartupAborted(SystemExit):
    """Do not let the normal per-grasp retry handler retry a closed launcher."""


class StartupGate:
    def __init__(self, fd, allow_wait=False):
        self.channel = socket.socket(fileno=int(fd))
        self.channel.set_inheritable(False)
        if self.channel.family != socket.AF_UNIX or self.channel.type != socket.SOCK_STREAM:
            self.channel.close()
            raise RuntimeError('无效的启动器就绪通道')
        self.channel.settimeout(.1)
        self.released = False
        self.allow_wait = allow_wait

    def ready(self, cancelled=lambda: False, timeout=15.):
        if self.released:
            return
        if cancelled():
            raise RuntimeError('初始化期间收到停止/暂停，未进入抓取任务')
        self.channel.sendall(json.dumps(dict(stage='models_ready',pid=os.getpid())).encode()+b'\n')
        print('模型与相机初始化完成；等待启动器确认控制使能，尚未发送任务。',flush=True)
        deadline = time.monotonic()+timeout
        reply = bytearray()
        while time.monotonic()<deadline:
            if cancelled():
                raise RuntimeError('等待使能期间收到停止/暂停，未进入抓取任务')
            try:
                chunk = self.channel.recv(32)
            except socket.timeout:
                continue
            if not chunk:
                raise RuntimeError('启动器已断开；未进入抓取任务')
            reply.extend(chunk)
            while b'\n' in reply:
                line, _, rest = reply.partition(b'\n')
                reply = bytearray(rest)
                if line == b'WAIT' and self.allow_wait:
                    deadline = time.monotonic()+timeout
                    continue
                if line != b'GO' or reply:
                    raise RuntimeError('启动器未允许执行抓取任务')
                if cancelled():
                    raise RuntimeError('收到停止/暂停，未进入抓取任务')
                self.released = True
                self.channel.close()
                return
            if len(reply)>16:
                raise RuntimeError('启动器放行消息无效')
        raise RuntimeError('等待启动器使能超时；未进入抓取任务')

    def close(self):
        self.channel.close()


@contextmanager
def installed():
    global _gate
    fd = os.environ.pop(ENV_FD,None)
    if fd is None:
        yield
        return
    if _gate is not None:
        raise RuntimeError('重复安装启动器就绪通道')
    gate = StartupGate(fd, allow_wait=os.environ.get('BOOK_GRASP_MANUAL_START') == '1')
    _gate = gate
    try:
        yield
    finally:
        _gate = None
        gate.close()


def wait_for_launcher(cancelled=lambda: False):
    if _gate is not None:
        try:
            _gate.ready(cancelled)
        except (OSError,RuntimeError) as error:
            raise StartupAborted(str(error)) from error
