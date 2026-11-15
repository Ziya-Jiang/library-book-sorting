"""Bounded RT/backend output on the data volume; no control-thread disk I/O."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import threading

ROOT = Path('/mnt/data/erazer/book_grasp/dual_arm_control_runtime/logs')
OUTPUT_BYTES = 16 * 1024 * 1024
OUTPUT_BACKUPS = 4


def log_path(name):
    if name not in ('rt.log', 'backend.log'):
        raise ValueError('Unknown RT console log')
    if os.stat('/mnt/data').st_dev == os.stat('/').st_dev:
        raise RuntimeError('RT logging requires mounted /mnt/data; no system-disk fallback')
    directory = ROOT / 'console'
    directory.mkdir(parents=True, exist_ok=True)
    if directory.stat().st_dev != os.stat('/mnt/data').st_dev:
        raise RuntimeError('RT log directory is outside the data volume')
    return directory / name


class BoundedOutput:
    def __init__(self, path, max_bytes=OUTPUT_BYTES, backups=OUTPUT_BACKUPS):
        if max_bytes < 1 or backups < 1:
            raise ValueError('Invalid output retention')
        self.path, self.max_bytes, self.backups = Path(path), max_bytes, backups
        self.file = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = os.open(str(self.path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for i in range(backups + 1):
                self._regular(self._path(i))
            self._open()
            if self.size >= max_bytes:
                self._rotate()
        except BaseException:
            self.close()
            raise

    def _path(self, index):
        return self.path if not index else Path(str(self.path) + '.' + str(index))

    @staticmethod
    def _regular(path):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode):
            raise ValueError('Refusing non-regular log path: ' + str(path))
        return True

    def _open(self):
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        self.file = os.fdopen(fd, 'ab', buffering=0)
        self.size = os.fstat(fd).st_size

    def _rotate(self):
        self.file.close()
        self.file = None
        for i in range(self.backups + 1):
            self._regular(self._path(i))
        self._path(self.backups).unlink(missing_ok=True)
        for i in range(self.backups - 1, -1, -1):
            if self._regular(self._path(i)):
                self._path(i).replace(self._path(i + 1))
        self._open()

    def write(self, data):
        view = memoryview(data)
        while view:
            if self.size >= self.max_bytes:
                self._rotate()
            length = min(len(view), self.max_bytes - self.size)
            written = self.file.write(view[:length])
            if not written:
                raise OSError('No progress writing RT output')
            self.size += written
            view = view[written:]

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None


class OutputCapture:
    """Own the RT pipe while its backend lives; a disk error never stops draining."""
    def __init__(self, path, **limits):
        self.writer = BoundedOutput(path, **limits)
        self.read_fd, self.write_fd = os.pipe()
        self._tail, self.error = b'', None
        self.thread = threading.Thread(target=self._drain, name='rt-output', daemon=True)
        self.thread.start()

    def fileno(self):
        return self.write_fd

    def _drain(self):
        try:
            with os.fdopen(self.read_fd, 'rb', buffering=0) as stream:
                while True:
                    block = stream.read(65536)
                    if not block:
                        break
                    self._tail = (self._tail + block)[-3000:]
                    if self.error is None:
                        try:
                            self.writer.write(block)
                        except Exception as error:
                            self.error = 'RT console disk logging disabled: ' + str(error)
                            self.writer.close()
        finally:
            self.writer.close()

    def tail(self):
        text = self._tail.decode(errors='replace')
        return text if self.error is None else text + '\n[' + self.error + ']'

    def close(self):
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None
        self.thread.join(timeout=2)


@contextmanager
def detached_output(path):
    """A separate sink survives the short-lived startup command without wrapping RT."""
    read_fd, write_fd = os.pipe()
    ready_read, ready_write = os.pipe()
    process = None
    try:
        process = subprocess.Popen([sys.executable, '-m', 'backend.rt_logs', '--sink', str(path),
                                    '--ready-fd', str(ready_write)],
                                   stdin=read_fd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   pass_fds=(ready_write,), cwd=Path(__file__).resolve().parents[1],
                                   start_new_session=True)
        os.close(read_fd); read_fd = None
        os.close(ready_write); ready_write = None
        readable, _, _ = select.select([ready_read], [], [], 5)
        if not readable or os.read(ready_read, 1) != b'1':
            raise RuntimeError('Backend log sink could not initialize; backend not started')
        yield write_fd
    finally:
        for fd in (read_fd, write_fd, ready_read, ready_write):
            if fd is not None:
                os.close(fd)
        if process is not None:
            try:
                process.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                # Keep a reference and reap it if the launcher remains alive.
                # A short-lived launcher may exit while the backend still writes.
                threading.Thread(target=process.wait, name='backend-log-reaper', daemon=True).start()


def _sink(path, ready_fd):
    writer = BoundedOutput(path)
    os.write(ready_fd, b'1'); os.close(ready_fd)
    failed = False
    try:
        while True:
            block = sys.stdin.buffer.read1(65536)
            if not block:
                break
            if not failed:
                try:
                    writer.write(block)
                except Exception:
                    failed = True
                    writer.close()  # Keep draining; never SIGPIPE the backend for a disk error.
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sink', type=Path, required=True)
    parser.add_argument('--ready-fd', type=int, required=True)
    args = parser.parse_args()
    _sink(args.sink, args.ready_fd)
