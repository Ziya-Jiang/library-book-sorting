"""One-command startup: connect only, or --multi/--plug/--baseline; offline --check."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
import uuid

import websockets
import grasp_cleanup
from model_startup import ModelStartup
import boot_progress
from startup_settling import StationaryWindow, TIMEOUT_SECONDS
from control_channel import dedicated_control
from control_timing import (ControlTiming, HEARTBEAT_PERIOD_S, HEARTBEAT_STATE_BUDGET_S,
                            HEARTBEAT_SEND_TIMEOUT_S, STATE_TIMEOUT_S)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = '/mnt/ssd/robotarm/env/robotram_wx/bin/python'
CUROBO = '/mnt/ssd/robotarm/env/curobo/bin/python'
AGENT = ROOT/'dual_v2_2_agent/multibook_grasp.py'
BASELINE_AGENT = ROOT/'dual_v2_2_agent/baseline_no_agent.py'
TASK_AGENT = ROOT/'dual_v2_2_agent/book_tasks_entry.py'
TASK_MODES = ('batch_right_to_left', 'shelf_right_neighbor')
PLANNER = ROOT/'dual_v2_2_control/cuarm_upper_planner/remote_planner.py'
HOST = 'erazer@10.30.3.155'
SOCKET = HERE/'runtime/ssh.sock'
RUNTIME = HERE/'runtime/book_grasp'
REMOTE_PYTHON = '/home/erazer/Seldon/keyboard_cockpit/.venv/bin/python'


def runtime_environment(python):
    environment=os.environ.copy()
    environment['PATH']=str(Path(python).parent)+os.pathsep+environment.get('PATH','')
    environment['PYTHONUNBUFFERED']='1'
    return environment


def ssh(*args, tty=False):
    return ['ssh', '-t' if tty else '-T', '-S', str(SOCKET), '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=8', HOST, *args]


def agent_command(*, baseline=False, task_mode=None):
    if task_mode is not None:
        if task_mode not in TASK_MODES:
            raise ValueError('Unknown book task mode')
        return [PYTHON, str(TASK_AGENT), '--task-mode', task_mode]
    return [PYTHON, str(BASELINE_AGENT if baseline else AGENT)]


def refuse_duplicate_agent(*, defer_orphans=False):
    names = {'multibook_grasp.py', 'cuart_book_grasp_v3_4_board_visual.py',
             'cuart_book_grasp_v3_4_board_numbered.py', 'baseline_no_agent.py', 'baseline_offset.py', 'book_tasks_entry.py'}
    paths={ROOT/'dual_v2_2_agent'/name for name in names}
    for item in grasp_cleanup.process_list():
        if grasp_cleanup.python_entry(item,paths):
            if defer_orphans and grasp_cleanup.orphan(item):
                continue # Inspect backend release before signalling this process.
            raise RuntimeError(f"已有抓取程序 PID={item['pid']}，未启动第二份；请先在原终端正常退出")


def precheck(*, baseline=False, task_mode=None):
    if not sys.stdin.isatty():
        raise RuntimeError('请在交互终端运行，需要输入SSH密码及使用p/s控制')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('10.30.3.100', 0))
    for file in (Path(PYTHON), Path(CUROBO), TASK_AGENT if task_mode else BASELINE_AGENT if baseline else AGENT, PLANNER):
        if not file.is_file():
            raise RuntimeError('缺少文件：'+str(file))
    if not shutil.which('rerun',path=runtime_environment(PYTHON)['PATH']):
        raise RuntimeError('robotram_wx环境缺少Rerun Viewer；未启动机器人')
    if not baseline:
        # Fail before opening the cockpit, RT, Planner or cameras. This check
        # reads the configured credential but never calls the model API.
        subprocess.run([PYTHON,str(AGENT),'--check-perception'],check=True,
                       timeout=10,env=runtime_environment(PYTHON))
    # Old clients are inspected only after the backend confirms release of the
    # previous control session. An interactive parent alone is no longer a veto.


def prepare_services(*, check_cameras=True):
    boot_progress.report('cockpit', '正在启动驾驶舱、连接155；Harness自动完成认证' if os.environ.get('DSH_BASELINE_AUTOAUTH') == '1' else '正在启动驾驶舱、连接155')
    subprocess.run(['bash', str(ROOT/'start_cockpit.sh'), '--managed-startup'], check=True, timeout=240)
    # Other launchers can reuse an older master path; create the fixed socket
    # explicitly without moving or destroying their existing session.
    if subprocess.run(['ssh', '-S', str(SOCKET), '-O', 'check', HOST],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError('固定SSH套接字未就绪；请检查统一驾驶舱runtime/ssh.sock')
    destination='/home/erazer/Seldon/keyboard_cockpit/runtime/prepare_book_robot.sh'
    for source, remote_name in (('prepare_book_robot.sh', 'prepare_book_robot.sh'),
                                ('can_health.py', 'book_can_health.py')):
        remote='/home/erazer/Seldon/keyboard_cockpit/runtime/'+remote_name
        subprocess.run(ssh('cat > '+shlex.quote(remote)), input=(HERE/source).read_bytes(), check=True)
    try:
        subprocess.run(ssh('bash '+shlex.quote(destination)+' --skip-chassis', tty=True), check=True)
    except subprocess.CalledProcessError as error:
        if error.returncode==42:
            raise RuntimeError('底盘CAN状态或错误计数异常，未继续启动双臂/抓取；请查看上方诊断文件') from error
        raise
    if check_cameras:
        subprocess.run([PYTHON, str(AGENT), '--check-cameras'], check=True, timeout=45)


class JsonSocket:
    def __init__(self, connection):
        self.connection=connection
        self.closed=False

    def __aiter__(self):
        return self.connection.__aiter__()

    async def send_json(self, value):
        await self.connection.send(json.dumps(value))

    async def close(self):
        self.closed=True
        await self.connection.close()


class PlannerNotStationary(RuntimeError):
    pass


class ArmSession:
    """Maintain the same websocket owner; never auto-rearm after an external stop."""
    def __init__(self, ws):
        self.ws=ws
        self.state=None
        self.stamp=0.
        self.error=None
        self.pending={}
        self.pending_actions={}
        self.armed=False
        self.control_epoch=None
        self.preparing=False
        self.enable_attempted=False
        self.stopping=False
        self.timing=ControlTiming()
        self.reader=asyncio.create_task(self.receive())
        self.heartbeater=asyncio.create_task(self.heartbeat_loop())

    def failed(self,error):
        if self.error is None:self.error=error
        for future in self.pending.values():
            if not future.done():future.set_exception(self.error)

    async def receive(self):
        try:
            async for msg in self.ws:
                if self.stopping:return
                if not isinstance(msg,str):
                    raise RuntimeError('驾驶舱连接已断开')
                data=json.loads(msg)
                if data.get('type')=='state':
                    stamp=time.monotonic()
                    self.timing.observe(data,stamp)
                    self.state=data; self.stamp=stamp
                    if self.armed:
                        self.check_owned()
                    # Drain received states without blocking on a send. Only
                    # the independent heartbeat task may use the latest state.
                elif data.get('type')=='error':
                    detail=data.get('stop_detail')
                    reason=data.get('message','驾驶舱拒绝请求')
                    if isinstance(detail,dict) and isinstance(detail.get('reason'),str):
                        # Error replies can precede the next periodic state.
                        # Preserve the backend's matched stop evidence without
                        # updating state age, ownership, epoch or heartbeat.
                        self.state=dict(self.state or {},stop_diagnostics={'recent':[detail]})
                        reason='控制停止原因：'+detail['reason']+'；'+reason
                    key=data.get('id')
                    # The deployed legacy rejection reply has no request ID.
                    # Before ownership, requests are serialized and no heartbeat
                    # is sent: only a sole pending enable may match this reply.
                    if key is None and len(self.pending)==1:
                        only=next(iter(self.pending))
                        if self.pending_actions.get(only) in ('planner_enable','prepared_planner_enable'):key=only
                    future=self.pending.get(key)
                    if (reason=='双臂尚未静止，不能切换 Planner 控制'
                            and self.pending_actions.get(key) in ('planner_enable','prepared_planner_enable')
                            and future is not None and not future.done()
                            and not self.armed and not detail):
                        # This explicit rejection guarantees no handoff occurred.
                        # All other errors retain the fatal session policy.
                        future.set_exception(PlannerNotStationary(reason))
                        continue
                    raise RuntimeError(reason)
                elif data.get('type')=='ok':
                    future=self.pending.get(data.get('id'))
                    if future and not future.done(): future.set_result(None)
            raise RuntimeError('驾驶舱连接已关闭')
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failed(error)

    async def heartbeat_loop(self):
        previous=None
        try:
            while not self.stopping:
                await asyncio.sleep(HEARTBEAT_PERIOD_S)
                if self.stopping:return
                if not (self.armed or self.preparing):continue
                self.check()
                if self.pending:continue  # Reservation can become control while ACK/new state is in flight.
                if self.armed:self.check_owned()
                elif not self.state.get('preparation',{}).get('mine'):
                    raise RuntimeError('启动准备控制权已丢失；未使能')
                data=self.state
                key=(data['epoch'],data['challenge'])
                now=time.monotonic()
                if (now-self.stamp>HEARTBEAT_STATE_BUDGET_S
                        or self.timing.extra_age(now)>HEARTBEAT_STATE_BUDGET_S):
                    self.timing.heartbeat('heartbeat_skipped_old_state',now)
                    continue
                if key==previous:continue
                self.timing.heartbeat('heartbeat_send_started',now)
                try:
                    await asyncio.wait_for(self.ws.send_json(dict(action='heartbeat' if self.armed else 'prepare_heartbeat',
                        challenge=data['challenge'],epoch=data['epoch'])),HEARTBEAT_SEND_TIMEOUT_S)
                except asyncio.TimeoutError as error:
                    raise RuntimeError('控制心跳发送阻塞超过200ms；停止本会话') from error
                done=time.monotonic()
                self.timing.heartbeat('heartbeat_sent',done,send_wait_ms=(done-now)*1000)
                previous=key
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failed(error)

    def check(self):
        if self.error: raise RuntimeError(str(self.error))
        if self.state is not None:
            now=time.monotonic()
            if now-self.stamp>STATE_TIMEOUT_S or self.timing.extra_age(now)>STATE_TIMEOUT_S:
                raise RuntimeError('驾驶舱状态更新已过期；不自动重新使能')

    def check_owned(self):
        s=self.state
        if not (s.get('fresh') and s.get('mine') and s.get('active') and s.get('planner',{}).get('enabled')):
            epoch=self.control_epoch
            recent=s.get('stop_diagnostics',{}).get('recent',[])
            stop=next((row for row in reversed(recent)
                       if type(epoch) is int and row.get('epoch')==epoch),None)
            if stop and isinstance(stop.get('reason'),str):
                raise RuntimeError('控制停止原因：'+stop['reason']+'；本次停止，不自动重新使能')
            raise RuntimeError('RT反馈或Planner控制权已丢失；本次停止，不自动重新使能')
        self.control_epoch=s.get('epoch')

    async def wait(self, predicate, seconds, description):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.check()
            if self.state is not None and predicate(self.state): return
            await asyncio.sleep(.05)
        raise RuntimeError(description+'超时；'+str((self.state or {}).get('message','无状态')))

    async def action(self, name, **kwargs):
        self.check()
        key=uuid.uuid4().hex
        future=asyncio.get_running_loop().create_future()
        self.pending[key]=future
        self.pending_actions[key]=name
        try:
            await asyncio.wait_for(self.ws.send_json(dict(action=name,id=key,challenge=self.state['challenge'],
                                         epoch=self.state['epoch'],**kwargs)), HEARTBEAT_SEND_TIMEOUT_S)
            await asyncio.wait_for(future,10)
            acknowledged=time.monotonic()
            # start_rt/connect/config can change epoch/state. Never use the
            # pre-command snapshot for the next guarded request.
            await self.wait(lambda _:self.stamp>acknowledged,2,'命令后状态更新')
        finally:
            self.pending.pop(key,None)
            self.pending_actions.pop(key,None)

    async def enable(self, guard=lambda:None):
        epoch=self.state['epoch']
        window=StationaryWindow()
        deadline=time.monotonic()+TIMEOUT_SECONDS
        last_report=0.
        print('等待'+('右臂' if self.state.get('rt_arm_mode') == 'right' else '双臂')+'连续停稳0.3秒（速度≤0.02rad/s，最多8秒）；尚未使能Planner。',flush=True)
        while time.monotonic()<deadline:
            self.check();guard()
            if (self.state.get('epoch')!=epoch or self.state.get('active')
                    or self.state.get('planner',{}).get('enabled')):
                raise RuntimeError('等待停稳期间控制会话变化；取消自动使能')
            now=time.monotonic()
            settled=window.observe(self.state,now)
            if now-last_report>=1. or settled:
                self.timing.events.append(dict(event='planner_startup_settling',
                    local_monotonic_s=now,latest=window.latest,peak=window.peak,
                    stable_samples=window.samples,settled=settled))
                if not settled:print('等待停稳：'+window.description(),flush=True)
                last_report=now
            if settled:
                self.enable_attempted=True
                try:
                    await asyncio.wait_for(self.action('prepared_planner_enable' if self.preparing else 'planner_enable'),
                                           max(.001,deadline-time.monotonic()))
                except PlannerNotStationary:
                    # The backend may see motion newer than our last web state.
                    window.reset()
                    print('接管瞬间仍有运动反馈，重新等待连续停稳。',flush=True)
                    continue
                except asyncio.TimeoutError:
                    raise RuntimeError('Planner接管响应超时；状态不确定，不重试使能') from None
                self.preparing=False
                await self.wait(lambda s:s.get('mine') and s.get('planner',{}).get('enabled'),3,'Planner使能')
                self.control_epoch=self.state.get('epoch')
                self.armed=True
                return
            await asyncio.sleep(.02)
        raise RuntimeError('双臂停稳等待超时；'+window.description()+'；未使能Planner')

    async def release_previous(self):
        """One startup handoff, not a runtime recovery or an automatic rearm."""
        self.check()
        previous=self.state
        if not (previous.get('active') or previous.get('planner',{}).get('enabled')):
            return
        record=dict(unix_s=time.time(),event='startup_control_release',
                    previous_epoch=previous.get('epoch'),status='requested')
        print('检测到旧双臂控制会话：请求停止并保持当前位置，等待后端确认释放。',flush=True)
        try:
            await self.action('stop')
            self.check()
            if (self.state.get('active') or self.state.get('planner',{}).get('enabled')
                    or self.state.get('epoch')==previous.get('epoch')):
                raise RuntimeError('旧控制会话停止未确认；未清理客户端、未启动新任务')
            # Observe more than an ACK. A competing client must not re-enable
            # while we clear the previous command producers.
            stamp=self.stamp
            def released(state):
                if state.get('active') or state.get('planner',{}).get('enabled'):
                    raise RuntimeError('旧会话释放后又出现活动控制；停止启动，不反复接管')
                return self.stamp>=stamp+.15
            await self.wait(released,2,'旧会话释放状态确认')
            record.update(status='released',released_epoch=self.state.get('epoch'))
            print('旧控制会话已释放；接下来清理旧抓取客户端。RT及夹爪保持状态保留。',flush=True)
        except BaseException as error:
            record.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            await asyncio.to_thread(record_session_release,record)

    async def close(self):
        # Send stop only for a control session we attempted to own. Closing this
        # websocket also invokes the existing backend owner-disconnect stop.
        self.armed=False
        self.stopping=True
        if self.preparing and not self.ws.closed and self.state:
            try:
                await asyncio.wait_for(self.ws.send_json(dict(action='prepare_cancel',
                    challenge=self.state['challenge'],epoch=self.state['epoch'])),1)
            except Exception:pass
        self.preparing=False
        if self.enable_attempted and not self.ws.closed and self.state and self.state.get('mine'):
            try:
                await asyncio.wait_for(self.ws.send_json(dict(action='stop')),1)
            except Exception:
                pass
        await self.ws.close()
        self.reader.cancel()
        self.heartbeater.cancel()
        await asyncio.gather(self.reader,self.heartbeater,return_exceptions=True)


CHILD_EXIT_TIMEOUT=8


async def terminate_child(proc, *, terminate_client=False):
    if proc is None or proc.returncode is not None:return
    try:proc.send_signal(signal.SIGINT)
    except ProcessLookupError:return
    try:await asyncio.wait_for(proc.wait(),CHILD_EXIT_TIMEOUT)
    except asyncio.TimeoutError:
        if not terminate_client:
            print(f'进程{proc.pid}尚未退出，请检查；未强杀。',flush=True)
            return
        # Only our grasp client, after the owning control connection closed.
        # Never escalate signals for RT, a reused planner or other services.
        print(f'本会话控制连接已关闭；终止未响应中断的抓取客户端 PID={proc.pid}。',flush=True)
        try:proc.terminate()
        except ProcessLookupError:return
        try:await asyncio.wait_for(proc.wait(),CHILD_EXIT_TIMEOUT)
        except asyncio.TimeoutError:
            print(f'抓取客户端 PID={proc.pid}仍未退出；未使用SIGKILL，请检查。',flush=True)


def planner_ready():
    sys.path.insert(0,str(ROOT/'dual_v2_2_agent'))
    import cuart_book_grasp_v3_4_remote as remote
    # Match the same entry as cuart_book_grasp_v3_4_board.main().
    previous=remote.PLANNER_ENTRY
    remote.PLANNER_ENTRY=remote.PLANNER_ROOT/'dual_v2_2_upper_planner_board.py'
    try:
        remote.require_remote_planner('10.30.3.155')
        planner_ready.last_error=''
        planner_ready.restart_required=False
        return True
    except (ValueError, RuntimeError, OSError) as error:
        planner_ready.last_error=str(error)
        planner_ready.restart_required=type(error).__name__=='PlannerRestartRequired'
        return False
    finally:
        remote.PLANNER_ENTRY=previous


def planner_exit_summary():
    """Report only the latest startup segment, never an earlier process fault."""
    path=HERE/'runtime/start_arms/planner.log'
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);size=stream.tell()
            # Find the latest boundary even after hours of trajectory output;
            # retain only a small error tail, never cite an earlier startup.
            marker=b'\nstart_arms startup unix_s='
            position=size;overlap=b'';boundary=None
            while position:
                start=max(0,position-65536);stream.seek(start)
                chunk=stream.read(position-start)+overlap
                index=chunk.rfind(marker)
                if index>=0:
                    boundary=start+index+1;break
                if start==0 and chunk.startswith(marker[1:]):boundary=0;break
                overlap=chunk[:len(marker)];position=start
            if boundary is not None:
                stream.seek(max(boundary,size-12000))
                lines=stream.read().decode('utf-8',errors='replace').splitlines()
        if boundary is None:
            return ' 日志尾部未找到本次启动分界；未引用历史错误，请查看start_arms/planner.log。'
        current=[line for line in lines if not line.startswith('start_arms startup unix_s=')]
        errors=[line.strip() for line in current if line.startswith(
            ('RuntimeError:','ValueError:','OSError:','ImportError:','ModuleNotFoundError:',
             'PermissionError:','FileNotFoundError:','MemoryError','KeyboardInterrupt',
             'Ctrl+C detected. Exiting'))]
        if errors:
            error=errors[-1]
            if error.startswith(('KeyboardInterrupt','Ctrl+C detected. Exiting')):
                error='初始化/运行被中断（SIGINT）；可能来自启动等待超时后的清理或人工中断。'
            return ' 本次Planner启动日志：'+error[:350]
        last=next((line.strip() for line in reversed(current) if line.strip()),'尚无输出')
        return ' 本次Planner日志尚无明确退出原因；最后输出：'+last[:200]
    except OSError:return ''


async def wait_existing_planner(arm,timeout=100.):
    """Program-only waits for an existing initializer; it never starts services."""
    if await asyncio.to_thread(planner_ready):return
    import start_arms
    identity=await asyncio.to_thread(start_arms.matching_planner)
    instruction='请先运行 bash /mnt/ssd/user/V/baseline_start.sh，看到“服务准备完成”后再重试当前抓取命令。'
    if identity is None:
        raise RuntimeError('Planner未运行；'+instruction+planner_exit_summary())
    deadline=time.monotonic()+timeout
    print(f'既有Planner PID={identity["pid"]}尚未就绪，等待本次初始化；不启动第二份、不使能。',flush=True)
    while True:
        arm.check()
        if arm.state.get('active') or arm.state.get('planner',{}).get('enabled'):
            raise RuntimeError('等待Planner期间出现活动控制；取消本次程序启动')
        if not await asyncio.to_thread(grasp_cleanup.same,identity):
            raise RuntimeError('等待中的Planner已退出；'+instruction+planner_exit_summary())
        if getattr(planner_ready,'restart_required',False):
            raise RuntimeError(str(getattr(planner_ready,'last_error','Planner代码已修改，需要重启')))
        if await asyncio.to_thread(planner_ready):return
        if time.monotonic()>=deadline:
            raise RuntimeError('既有Planner就绪等待超时；'+instruction+
                               ' 详情：'+str(getattr(planner_ready,'last_error','')))
        await asyncio.sleep(.4)


async def cleanup_previous(arm):
    def guard():
        arm.check()
        if arm.state is None or arm.state.get('active') or arm.state.get('planner',{}).get('enabled'):
            raise RuntimeError('已有控制会话占用双臂；未清理或接管原控制')
    guard()
    targets=await asyncio.to_thread(grasp_cleanup.inventory,ROOT,RUNTIME/'children.json',
                                   released_clients=True)
    await grasp_cleanup.cleanup(targets,guard,RUNTIME/'children.json',audit=RUNTIME/'cleanup.jsonl')
    await asyncio.to_thread(refuse_duplicate_agent)
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1',10175))
    print(f'启动前残留检查完成：清理{len(targets)}个进程，抓取端口已释放。',flush=True)


async def cleanup_program_orphans(arm):
    """Only released orphan agents; never sweep the shared Planner registry."""
    def guard():
        arm.check()
        state=arm.state
        if (not state or state.get('active') or state.get('preparation_active')
                or state.get('planner',{}).get('enabled')):
            raise RuntimeError('控制会话尚未释放；不清理孤儿抓取程序')
    guard()
    targets=await asyncio.to_thread(grasp_cleanup.program_orphans,ROOT)
    if targets:
        print('检测到父启动器已退出的孤儿抓取程序；控制已释放，正常清理后继续。',flush=True)
        await grasp_cleanup.cleanup(targets,guard,RUNTIME/'program_orphans.json',
                                    audit=RUNTIME/'cleanup.jsonl')
    await asyncio.to_thread(refuse_duplicate_agent)


_registry_lock=threading.Lock()


def record_session_release(record):
    with (RUNTIME/'session_cleanup.jsonl').open('a') as file:
        file.write(json.dumps(record,ensure_ascii=False)+'\n')


def record_control_failure(record):
    with (RUNTIME/'control_timing.jsonl').open('a') as file:
        file.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def remember_children(*children):
    with _registry_lock:
        grasp_cleanup.remember(RUNTIME/'children.json',children)


async def record_children(get_children):
    # Disk I/O and /proc enumeration must never run on the heartbeat loop.
    # One worker at a time; the lock also serializes the final shutdown record.
    while True:
        await asyncio.to_thread(remember_children,*get_children())
        await asyncio.sleep(1.)


async def run_robot(token, *, check_cameras=False, control_url='ws://127.0.0.2:8780/ws', plug=False, baseline=False, program_only=False, task_mode=None, shelf_confirmation=None):
    baseline = baseline or task_mode is not None
    if plug and baseline:raise ValueError('baseline与plug不能同时运行')
    command=agent_command(baseline=baseline, task_mode=task_mode)
    if shelf_confirmation is not None:
        command += ['--shelf-confirmation', str(shelf_confirmation)]
    planner=None; grasp=None; log=None; arm=None; gate=None; recorder=None; failure=None
    async with websockets.connect(control_url,proxy=None,compression=None,
                                  open_timeout=5,ping_interval=2,close_timeout=2) as connection:
        ws=JsonSocket(connection)
        await ws.send_json(dict(token=token))
        arm=ArmSession(ws)
        try:
            await arm.wait(lambda s:True,5,'驾驶舱连接')
            if arm.state.get('simulation'): raise RuntimeError('驾驶舱处于仿真模式')
            if program_only:
                boot_progress.report('planner_check', '核验现有RT、Planner与控制权')
                if arm.state.get('active') or arm.state.get('planner',{}).get('enabled'):
                    raise RuntimeError('已有活动控制；请先在原程序按s或Ctrl+C退出，再启动baseline程序')
                # Do not sweep the old full-launch registry: it can contain a
                # Planner that must now remain alive across Python restarts.
                await cleanup_program_orphans(arm)
                with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
                    probe.bind(('127.0.0.1',10175))
                if not (arm.state.get('rt_running') and arm.state.get('fresh')
                        and arm.state.get('rt_arm_mode')=='both'
                        and arm.state.get('state',{}).get('system') in (1,2,3)):
                    raise RuntimeError('双臂RT未就绪；先运行 bash /mnt/ssd/user/V/baseline_start.sh')
                await wait_existing_planner(arm)
            else:
                await arm.release_previous()
                await cleanup_previous(arm)
            if check_cameras:
                boot_progress.report('cameras', '正在检查相机与当前抓书配置')
                await asyncio.to_thread(subprocess.run,[*command,'--check-cameras'],
                                        check=True,timeout=45,env=runtime_environment(PYTHON))
            arm.check()
            if arm.state.get('active') or arm.state.get('planner',{}).get('enabled'):
                raise RuntimeError('准备期间出现活动控制会话；停止启动')
            if arm.state.get('rt_running') and arm.state.get('rt_arm_mode','both') != 'both':
                raise RuntimeError('当前RT为仅右臂模式；需要双臂模式，未自动切换')
            if not program_only:
                print('启动/复用双臂RT，保留严格初态校验。',flush=True)
                if not arm.state.get('rt_running'):
                    await arm.action('start_rt',reference='strict')
                else:
                    await arm.action('connect',ip='127.0.0.1',port=12201)
                await arm.wait(lambda s:s.get('fresh') and s.get('state',{}).get('system') in (1,2,3),45,'RT反馈')
                await arm.action('planner_config',ip='10.30.3.100',port=30241,local_ip='0.0.0.0',local_port=30240)
                if not await asyncio.to_thread(planner_ready):
                    for endpoint in [('127.0.0.1',10176),('10.30.3.100',30241)]:
                        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:probe.bind(endpoint)
                    log=(RUNTIME/'planner.log').open('ab',buffering=0)
                    planner=await asyncio.create_subprocess_exec(CUROBO,str(PLANNER),
                        stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True,
                        env=runtime_environment(CUROBO))
                    await asyncio.to_thread(remember_children,planner)
                print('等待Planner；日志：'+str(RUNTIME/'planner.log'),flush=True)
                end=time.monotonic()+100
                reported_reason=None
                while not await asyncio.to_thread(planner_ready):
                    arm.check()
                    reason=getattr(planner_ready,'last_error','')
                    if reason and reason!=reported_reason:
                        print('Planner尚未就绪：'+reason,flush=True)
                        reported_reason=reason
                    if planner is not None and planner.returncode is not None: raise RuntimeError('Planner退出，请查看planner.log')
                    if time.monotonic()>end:raise RuntimeError('Planner就绪超时，请查看planner.log')
                    await asyncio.sleep(.3)
                await arm.wait(lambda s:s.get('planner',{}).get('connected') and s.get('fresh'),10,'Planner握手')
            else:
                await arm.wait(lambda s:s.get('planner',{}).get('connected') and s.get('fresh'),10,'既有Planner握手')
                print('复用现有双臂RT和Planner；只启动baseline Python及本次控制会话。',flush=True)
            if baseline:
                # VisualIPC is created by the bridge tick only after
                # planner_config opens its UDP socket. Checking before that
                # reads an absent/stale descriptor from the previous process.
                # Installation remains before prepare_start/control enable.
                boot_progress.report('worker', '准备SAM3.1隔离worker，核对视觉IPC')
                print('Planner握手完成；准备SAM3.1隔离worker并核对当前视觉IPC，尚未使能运动。',flush=True)
                await asyncio.to_thread(subprocess.run,[*command,'--install-worker'],
                                        check=True,timeout=90,env=runtime_environment(PYTHON))
                arm.check()
                if arm.state.get('active') or arm.state.get('planner',{}).get('enabled'):
                    raise RuntimeError('基线准备期间出现活动控制；未自动接管')
            # The agent initializes its camera/models while control is disabled,
            # and blocks at its first task provider until this session grants GO.
            await arm.action('prepare_start')
            arm.preparing=True
            epoch=arm.state.get('epoch')
            gate=ModelStartup()
            manual_start=boot_progress.manual_enabled()
            boot_progress.report('models_loading', '正在初始化模型与相机；尚未使能运动')
            print('初始化抓取模型；Planner控制保持未使能，初始化完成后才允许执行任务。',flush=True)
            grasp=await asyncio.create_subprocess_exec(*command,
                *(['--assume-head-camera-calibrated'] if baseline else ['--plug'] if plug else []),
                **gate.spawn_options(runtime_environment(PYTHON)))
            gate.spawned()
            recorder=asyncio.create_task(record_children(lambda:(planner,grasp)))

            def check_preparation():
                arm.check()
                if not arm.state.get('preparation',{}).get('mine'):
                    raise RuntimeError('启动准备控制权已丢失；未自动使能')
                if arm.state.get('active') or arm.state.get('planner',{}).get('enabled'):
                    raise RuntimeError('初始化期间出现活动控制会话；停止启动，不接管')
                if arm.state.get('epoch')!=epoch:
                    raise RuntimeError('初始化期间控制状态已改变；取消自动使能')
                if planner is not None and planner.returncode is not None:
                    raise RuntimeError('模型初始化期间Planner已退出')
                if recorder.done():recorder.result()

            await gate.wait_ready(grasp,check_preparation)
            ready_stamp=arm.stamp
            await arm.wait(lambda s:arm.stamp>ready_stamp,2,'模型就绪后控制状态更新')
            check_preparation()
            if not (arm.state.get('fresh') and arm.state.get('planner',{}).get('connected')
                    and arm.state.get('state',{}).get('system') in (1,2,3)):
                raise RuntimeError('模型就绪，但RT/Planner反馈未就绪；未使能')
            if grasp.returncode is not None:
                raise RuntimeError('抓取程序已退出；未使能')
            if manual_start:
                def check_manual_preparation():
                    check_preparation()
                    if not (arm.state.get('fresh') and arm.state.get('planner',{}).get('connected')
                            and arm.state.get('state',{}).get('system') in (1,2,3)):
                        raise RuntimeError('等待手动确认期间RT/Planner未就绪；取消准备')
                print('模型已就绪；请在Harness界面点击“开始抓取”。等待期间不使能、不执行任务；Ctrl+C取消。',flush=True)
                await gate.wait_confirmation(grasp,check_manual_preparation,boot_progress.confirmation_requested)
                ready_stamp=arm.stamp
                await arm.wait(lambda s:arm.stamp>ready_stamp,2,'手动确认后控制状态更新')
                check_manual_preparation()
            boot_progress.report('enabling', '已收到手动开始，正在重新核验并使能' if manual_start else '正在核验控制使能')
            await arm.enable(check_preparation)

            def check_execution():
                arm.check();arm.check_owned()
                if planner is not None and planner.returncode is not None:
                    raise RuntimeError('Planner已退出')

            await gate.release(check_execution)
            boot_progress.report('running', '已放行抓取任务；可在程序终端按Ctrl+C停止本次会话')
            print('Planner已允许；'+('逐本重新观察，从右到左抓放；目标不确定或失败时暂停。' if task_mode=='batch_right_to_left' else
                  '固定源格，向同层右邻格逐本放置；场景或空间校验失败时暂停。' if task_mode=='shelf_right_neighbor' else
                  'SAM3.1单书基线：先视觉对齐，再固定Y/Z和姿态沿X前进；保留原抓放循环。' if baseline else
                  '插槽两轮：先右移，再重新识别从左到右左移。' if plug else
                  '头部识别一次后，从右到左自动逐本抓放，无需输入编号。')+'Ctrl+C停止本启动会话。',flush=True)
            while grasp.returncode is None:
                check_execution()
                if recorder.done():recorder.result()
                await asyncio.sleep(.1)
            if grasp.returncode:raise RuntimeError(f'抓取程序退出码：{grasp.returncode}')
        except Exception as error:
            failure=arm.timing.snapshot(error,arm.state)
            # Print the initiating error BEFORE interrupting the client so its
            # later KeyboardInterrupt is not mistaken for a model failure.
            print('启动器停止原因：'+str(error),flush=True)
            raise
        finally:
            control_closed=False
            try:
                if arm:
                    await arm.close()
                    control_closed=True
            finally:
                if arm:
                    arm.armed=False
                    arm.stopping=True
                    arm.reader.cancel()
                    arm.heartbeater.cancel()
                    await asyncio.gather(arm.reader,arm.heartbeater,return_exceptions=True)
                if failure:
                    try:
                        await asyncio.to_thread(record_control_failure,failure)
                        print('控制收发时序已保存：'+str(RUNTIME/'control_timing.jsonl'),flush=True)
                    except Exception as error:
                        print('控制收发时序记录失败：'+str(error),flush=True)
                if gate:gate.close()
                if recorder:
                    recorder.cancel()
                    await asyncio.gather(recorder,return_exceptions=True)
                try:
                    # Capture descendants before their parent exits/reparents.
                    try:await asyncio.to_thread(remember_children,planner,grasp)
                    except Exception as error:print('残留身份记录失败：'+str(error),flush=True)
                    if gate and not getattr(gate,'released',False) and grasp and grasp.returncode is None:
                        # Closing the startup gate already asks the idle client
                        # to exit. Let cleanup finish before sending SIGINT.
                        try:await asyncio.wait_for(grasp.wait(),CHILD_EXIT_TIMEOUT)
                        except asyncio.TimeoutError:pass
                    await terminate_child(grasp,terminate_client=control_closed)
                finally:
                    try:await terminate_child(planner)
                    finally:
                        if log:log.close()
            print('本次控制会话结束；'+('保留既有RT、Planner、相机和网页' if program_only else '保留RT、相机、底盘和网页')+'，不自动松爪或复位。',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description='一键启动：--connect-only仅连接；--multi/--plug/--baseline会允许抓取运动。暂不启动底盘CAN/ROS。')
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--multi',action='store_true',help='识别一次，从右到左逐本抓放')
    modes.add_argument('--plug',action='store_true',help='抓取后抽出并插入最近空槽；先右移一轮，再左移一轮')
    modes.add_argument('--baseline',action='store_true',help='SAM3.1单书基线：失败后DeepSeek快速语义备用，先对齐后水平直进')
    modes.add_argument('--baseline-batch',action='store_true',help='逐本重新观察、从右到左抓放；复用baseline服务和运动门控')
    modes.add_argument('--baseline-shelf',action='store_true',help='固定源格向同层右邻格放置；仅已发布模式可以启动')
    modes.add_argument('--connect-only',action='store_true',help='打开本机驾驶舱，启动/复用155 RT和本机CuRobo；不接管、不使能、不运行抓取')
    modes.add_argument('--baseline-services',action='store_true',help='只准备baseline网页、相机、双臂RT和Planner，不启动抓取或使能')
    parser.add_argument('--program-only',action='store_true',help='仅baseline系列：复用已启动的RT/Planner，退出保留服务')
    parser.add_argument('--check',action='store_true',help='仅离线检查文件和配置，不连接机器人')
    parser.add_argument('--shelf-confirmation',type=Path,help='已确认的源格和实测夹爪/书本几何文件，仅路线2')
    args=parser.parse_args(argv)
    task_mode = 'batch_right_to_left' if args.baseline_batch else 'shelf_right_neighbor' if args.baseline_shelf else None
    baseline = args.baseline or task_mode is not None
    if baseline or args.baseline_services:
        # The baseline owns this model root; unrelated suction experiments may
        # use CUARM_CONFIG_ROOT in their own launchers, never in this one.
        os.environ['CUARM_CONFIG_ROOT']=str(ROOT/'dual_v2_2_configuration')
    if args.program_only and not baseline:parser.error('--program-only只能与baseline系列模式一起使用')
    if task_mode and not args.program_only:parser.error('新任务必须使用--program-only并先准备baseline服务')
    if args.shelf_confirmation is not None and not args.baseline_shelf:
        parser.error('--shelf-confirmation仅用于--baseline-shelf')
    if args.baseline_shelf and not args.check and args.shelf_confirmation is None:
        parser.error('路线2需要--shelf-confirmation；先完成观测编号和源格/几何确认')
    if args.baseline_services:
        if args.check:
            for name in ('baseline_start.sh','baseline_no_agent.sh','start_arms.sh'):
                subprocess.run(['bash','-n',str(ROOT/name)],check=True)
            print('baseline服务入口离线检查通过；未连接机器人。');return
        print('只启动/复用网页、相机、双臂RT和Planner；不使能、不启动抓取。',flush=True)
        prepare_services(check_cameras=False)
        boot_progress.report('services_planner', '正在准备双臂RT和本机CuRobo Planner')
        subprocess.run(['bash',str(ROOT/'start_arms.sh')],check=True)
        boot_progress.report('services_ready', '基础服务准备完成')
        print('服务准备完成；运行 bash /mnt/ssd/user/V/baseline_no_agent.sh 启动程序。',flush=True)
        return
    if args.connect_only:
        # Deliberately return before grasp cleanup, ownership, Agent or GO gate.
        if args.check:
            import ast
            for name in ('book_grasp.sh','start_cockpit.sh','start_arms.sh'):
                subprocess.run(['bash','-n',str(ROOT/name)],check=True)
            for name in ('book_grasp.py','cockpit.py','start_arms.py'):
                ast.parse((HERE/name).read_text())
            print('仅连接入口离线检查通过；未连接或启动机器人。');return
        print('仅连接模式：打开本机驾驶舱，核验155双臂RT和本机CuRobo；不运行抓取、不使能运动。',flush=True)
        subprocess.run(['bash',str(ROOT/'start_cockpit.sh')],check=True)
        subprocess.run(['bash',str(ROOT/'start_arms.sh')],check=True)
        print('连接就绪；本机界面：http://127.0.0.2:8798/；未启动抓书。',flush=True)
        return
    if args.check:
        subprocess.run(['bash','-n',str(ROOT/'book_grasp.sh')],check=True)
        if args.baseline:subprocess.run(['bash','-n',str(ROOT/'baseline_no_agent.sh')],check=True)
        if task_mode:subprocess.run(['bash','-n',str(ROOT/('baseline_batch.sh' if args.baseline_batch else 'baseline_shelf.sh'))],check=True)
        subprocess.run(['bash','-n',str(HERE/'prepare_book_robot.sh')],check=True)
        subprocess.run([*agent_command(baseline=baseline, task_mode=task_mode),'--check-config',*(['--plug'] if args.plug else [])],check=True)
        print('离线入口检查通过；未启动机器人。');return
    RUNTIME.mkdir(parents=True,exist_ok=True)
    if args.program_only:refuse_duplicate_agent(defer_orphans=True)
    precheck(baseline=baseline, task_mode=task_mode)
    with (RUNTIME/'lock').open('a') as lock:
        if args.program_only:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('已有抓取启动会话，拒绝重复启动；原会话保持运行，未接管或发送停止') from None
            refuse_duplicate_agent(defer_orphans=True) # Active parents remain a veto.
        else:grasp_cleanup.acquire_startup_lock(lock,ROOT)
        if not args.program_only:print('将启动相机、'+'双臂RT'+'、Planner和'+('SAM3.1单书基线' if args.baseline else '多书抓取')+'；暂不启动底盘CAN/ROS。SSH密码按终端提示输入。',flush=True)
        if not args.program_only:prepare_services(check_cameras=False)
        else:print('程序模式：不启停相机、RT或Planner；修改Python后可重复运行本入口。',flush=True)
        # Token remains in memory and is never printed or stored by this launcher.
        code="import json; from pathlib import Path; from urllib.parse import urlsplit,parse_qs; print(parse_qs(urlsplit(json.loads(Path('/home/erazer/Seldon/dual_arm_cockpit/runtime/web.json').read_text())['url']).fragment)['token'][0])"
        token=subprocess.check_output(ssh(shlex.join([REMOTE_PYTHON,'-c',code])),text=True,timeout=10).strip()
        with dedicated_control() as control_url:
            asyncio.run(run_robot(token,check_cameras=True,control_url=control_url,plug=args.plug,baseline=baseline,program_only=args.program_only,task_mode=task_mode,shelf_confirmation=args.shelf_confirmation))


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        boot_progress.report('failed','用户已中断本次启动')
        print('已中断。',file=sys.stderr);sys.exit(130)
    except Exception as error:
        boot_progress.report('failed','启动/运行未完成：'+str(error))
        print('启动/运行未完成：'+str(error),file=sys.stderr);sys.exit(1)
