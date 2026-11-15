"""Local Planner lifetime only: never stops robot RT or a live peer Planner."""
import ctypes
import fcntl
import json
import os
import platform
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        if path.stat().st_uid != os.getuid(): return None
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return fields[19], int(fields[1])
    except (OSError, IndexError): return None


def pidfd_syscall(number, *args):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('Unsupported pidfd platform; no process was signalled')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def open_pidfd(pid):
    return pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def signal_pidfd(fd, sig):
    return pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def arm_parent_death(parent_pid):
    # Kernel enforcement also works if the child is stuck in Python cleanup.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot arm publisher parent-death signal')
    if os.getppid() != parent_pid:
        raise SystemExit('Planner parent already exited; publisher will not open UDP')


def reap_orphan_publishers(worker=None, timeout=2.):
    worker = Path(worker or Path(__file__).with_name('v3_1_panel_worker.py')).resolve()
    for entry in Path('/proc').glob('[0-9]*'):
        pid = int(entry.name); before = identity(pid)
        if before is None: continue
        try:
            argv = (entry/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if not Path(argv[0]).name.startswith('python'): continue
            args = argv[1:]
            while args and args[0] in ('-u', '-B'): args = args[1:]
            if len(args) != 5 or args[1] != '--fd' or args[3] != '--parent-pid': continue
            if (entry/'cwd'/args[0]).resolve() != worker: continue
            claimed = int(args[4]); int(args[2])
            # A live claimed parent or an attached child is never reclaimed.
            if before[1] == claimed or identity(claimed) is not None: continue
            fd = open_pidfd(pid)
        except (OSError, ValueError, IndexError): continue
        try:
            if identity(pid) != before or identity(claimed) is not None: continue
            signal_pidfd(fd, signal.SIGTERM)
            deadline = time.monotonic()+timeout
            while identity(pid) == before and time.monotonic() < deadline: time.sleep(.05)
            if identity(pid) == before:
                signal_pidfd(fd, signal.SIGKILL)
                deadline = time.monotonic()+timeout
                while identity(pid) == before and time.monotonic() < deadline: time.sleep(.05)
            if identity(pid) == before:
                raise RuntimeError(f'Orphan publisher {pid} has not exited; do not start another Planner')
            print(f'已清理旧Planner孤立UDP子进程 PID={pid}；未触及RT或活动Planner。', flush=True)
        except ProcessLookupError: pass
        finally: os.close(fd)


class PlannerLifetime:
    def __init__(self, port=10176, directory='/tmp'):
        self.directory = Path(directory); self.port = port
        self.lock = None; self.handlers = {}; self.stopping = False
        self.start = identity(os.getpid())[0]

    def __enter__(self):
        self.lock = (self.directory/f'cuarm-v31-{os.getuid()}-{self.port}.lock').open('a+')
        try: fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close(); self.lock = None
            raise RuntimeError('已有Planner或正在退出清理；未启动第二份，请等待旧进程退出。')
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.handlers[sig] = signal.signal(sig, self.stop)
        return self

    def stop(self, *_):
        if not self.stopping:
            self.stopping = True
            raise KeyboardInterrupt

    def __exit__(self, *_):
        receipt = self.directory/f'cuarm-v31-{os.getuid()}-{self.port}.json'
        try:
            value = json.loads(receipt.read_text())
            if value.get('pid') == os.getpid() and value.get('start') == self.start:
                receipt.unlink()
        except (OSError, ValueError): pass
        for sig, handler in self.handlers.items(): signal.signal(sig, handler)
        if self.lock is not None: self.lock.close()
        # Keep the lock inode: unlinking allows two independent flock owners.
