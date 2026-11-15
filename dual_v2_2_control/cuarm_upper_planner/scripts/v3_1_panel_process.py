"""Private PC IPC for the board Planner's independent UDP publisher.

The child owns both the socket and the original transport interlocks. It never
loads CuRobo/CUDA. A request ACK means synchronous transport acceptance, not
robot arrival. No command is queued across an IPC failure or a control epoch.
"""
import copy
import hashlib
import os
from pathlib import Path
import pickle
import select
import socket
import struct
import subprocess
import sys
import threading
import time

from v3_1_panel_transport import PanelTransport, StalePlanStart, ControlSessionCancelled

MAX_MESSAGE = 1_000_000
REQUEST_AGE = .10
RPC_TIMEOUT = .30


def publisher_fingerprint():
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in (
        'v3_1_panel_process.py', 'v3_1_panel_worker.py', 'planner_lifecycle.py',
        'v3_1_board_client.py', 'v3_1_panel_transport.py', 'v3_1_seldon.py',
        '../seldon_v31/packet_trace.py'))).hexdigest()


def process_identity(pid):
    fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[1]), fields[19]


def verify_publisher(report, planner_pid):
    """Read-only proof that the socket owner is this planner's current child."""
    worker_pid = report.get('publisher_pid')
    if type(worker_pid) is not int or worker_pid != report['pid'] or worker_pid == planner_pid:
        raise ValueError('缺少独立UDP通信子进程，请重启当前board Planner')
    parent, worker_start = process_identity(worker_pid)
    _, planner_start = process_identity(planner_pid)
    if (parent != planner_pid or report.get('planner_pid') != planner_pid
            or worker_start != report.get('publisher_start')
            or planner_start != report.get('planner_start')):
        raise ValueError('UDP通信子进程与当前Planner身份不匹配')
    if report.get('publisher_sha256') != publisher_fingerprint():
        raise ValueError('运行中的UDP通信子进程版本已过期，请重启当前board Planner')
    argv = Path(f'/proc/{worker_pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if str(Path(__file__).with_name('v3_1_panel_worker.py')) not in argv:
        raise ValueError('UDP通信子进程入口不匹配')


def write_message(sock, value, deadline):
    raw = pickle.dumps(value, protocol=4)
    if len(raw) > MAX_MESSAGE:
        raise ValueError('Panel IPC message too large')
    sock.settimeout(max(.001, deadline-time.monotonic()))
    if time.monotonic() >= deadline:
        raise TimeoutError('Panel IPC send expired')
    sock.sendall(struct.pack('!I', len(raw))+raw)


def read_message(sock, deadline):
    def read_exact(size):
        result = bytearray()
        while len(result) < size:
            remaining = deadline-time.monotonic()
            # The planner can hold the GIL after the child has already replied.
            # Drain a buffered reply without waiting; its original feedback
            # timestamp still decides freshness. Never renew from read time.
            if remaining <= 0 and not select.select([sock], [], [], 0)[0]:
                raise TimeoutError('Panel IPC response timeout')
            sock.settimeout(max(.001, remaining))
            block = sock.recv(size-len(result))
            if not block:
                raise EOFError('Panel IPC peer closed')
            result.extend(block)
        return result
    size = struct.unpack('!I', read_exact(4))[0]
    if size > MAX_MESSAGE:
        raise ValueError('Panel IPC message too large')
    # Both endpoints are our processes over an inherited private socketpair;
    # this socket is never exposed as a path or network listener.
    return pickle.loads(read_exact(size))


class ProcessBoardPanel:
    def __init__(self, local, remote, timeout=10):
        self.lock = threading.RLock()
        self._rpc_lock = threading.Lock()
        self._closed = False
        self.error = None
        self.data = self.stamp = self.packet = None
        self.remote = remote
        self.transport_state = {}
        self._request_id = 0
        self.sock, child = socket.socketpair()
        worker = Path(__file__).with_name('v3_1_panel_worker.py')
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)
        env['OPENBLAS_NUM_THREADS'] = env['OMP_NUM_THREADS'] = '1'
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-u', str(worker), '--fd', str(child.fileno()),
                 '--parent-pid', str(os.getpid())],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, env=env)
        except BaseException:
            self.sock.close()
            raise
        finally:
            child.close()
        try:
            self.rpc('open', local=local, remote=remote, timeout=timeout,
                     _timeout=timeout+5.)
        except BaseException:
            self.terminate()
            raise

    def _abort(self, reason):
        self.error = reason
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def rpc(self, operation, _timeout=RPC_TIMEOUT, **arguments):
        started = time.monotonic()
        deadline = started+_timeout
        if not self._rpc_lock.acquire(timeout=_timeout):
            self._abort('Panel IPC caller stalled; command channel closed')
            raise RuntimeError(self.error)
        try:
            if self._closed:
                raise RuntimeError(self.error or 'Panel IPC is closed')
            self._request_id += 1
            request = dict(id=self._request_id, op=operation, sent_at=started, args=arguments)
            lock_wait_ms = (time.monotonic()-started)*1000
            phase = 'send'
            try:
                write_message(self.sock, request, deadline)
                phase = 'receive'
                response = read_message(self.sock, deadline)
                if response['id'] != request['id']:
                    raise ValueError('Panel IPC reply identity changed')
            except (OSError, EOFError, ValueError, KeyError) as error:
                self._abort('Panel IPC failed; no automatic restart: '+str(error)+
                            f'; op={operation}, request_id={self._request_id}, phase={phase}, '
                            f'lock_wait_ms={lock_wait_ms:.3f}, '
                            f'elapsed_ms={(time.monotonic()-started)*1000:.3f}')
                raise RuntimeError(self.error) from error
            with self.lock:
                state = response.get('panel', {})
                self.data, self.stamp, self.packet = state.get('data'), state.get('stamp'), state.get('packet')
                self.transport_state = response.get('transport', {})
            error = response.get('error')
            if error:
                cls = {'StalePlanStart': StalePlanStart,
                       'ControlSessionCancelled': ControlSessionCancelled}.get(error['type'], RuntimeError)
                raise cls(error['message'])
            return response.get('result')
        finally:
            self._rpc_lock.release()

    def receive_with_timestamp(self):
        self.rpc('feedback')
        with self.lock:
            old = self.stamp is not None and time.monotonic()-self.stamp > PanelTransport.FEEDBACK_AGE
        if old:
            # One refresh handles a reply buffered while CUDA held the parent
            # GIL. A genuinely stale board sample remains stale on this retry.
            self.rpc('feedback')
        with self.lock:
            return self.data, self.stamp

    def receive(self, max_age_s=None):
        data, stamp = self.receive_with_timestamp()
        if stamp is None or max_age_s is not None and not 0 <= time.monotonic()-stamp <= max_age_s:
            return None
        return data

    def send_strict(self, *args, **kwargs):
        raise RuntimeError('Process panel requires transport commit; direct send forbidden')

    def terminate(self):
        if not self._closed:
            try:
                self.rpc('close')
            except (RuntimeError, OSError):
                pass
            finally:
                self._abort(self.error or 'Panel IPC closed')
        process = getattr(self, 'process', None)
        if process is not None:
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                # This is only our child (not robot RT); signal normal shutdown.
                process.terminate()
                try:
                    process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    print('Panel IPC child stuck during shutdown; reaping only this owned child', flush=True)
                    process.kill()
                    process.wait(timeout=2.)


class ProcessBoardTransport:
    FEEDBACK_AGE = PanelTransport.FEEDBACK_AGE
    PRODUCER_LEASE = PanelTransport.PRODUCER_LEASE
    PLANNING_LEASE = PanelTransport.PLANNING_LEASE

    def __init__(self, panel, template, stop_state, hold_mode):
        self.remote_panel = panel
        panel.rpc('configure', template=template, stop_state=stop_state, hold_mode=hold_mode)

    def _call(self, name, **args):
        return self.remote_panel.rpc(name, **args)

    @property
    def feedback_wait(self):
        return self.remote_panel.transport_state.get('feedback_wait')

    @property
    def fault(self):
        return self.remote_panel.error or self.remote_panel.transport_state.get('fault')

    def start(self): return self._call('start')
    def begin_cycle(self): return self._call('begin_cycle')
    def ready_to_plan(self): return self._call('ready_to_plan')
    def begin_planning(self, start_joints=None): return self._call('begin_planning', start_joints=start_joints)
    def idle_hold(self, cancel_recovery=False): return self._call('idle_hold', cancel_recovery=cancel_recovery)
    def send(self, command): return self._call('send', command=copy.deepcopy(command))
    def terminate(self): return self.remote_panel.terminate()
