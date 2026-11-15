"""J6-only persistent control lifetime. No automatic reconnect or re-arm."""
import asyncio
import copy
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import partial
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
import sys
import threading
import time
from urllib.request import Request, ProxyHandler, build_opener
from feedback_recovery import FeedbackGate, feedback_ages

_CURRENT = threading.local()


class RoundClosing(RuntimeError):
    pass


def detached_error(error):
    """Do not expose a live worker traceback to the caller's exception handler."""
    try:
        result = copy.copy(error)
        # Custom errors can reconstruct their message from a payload in __init__.
        result.args = error.args
    except Exception:
        result = RuntimeError(f'{type(error).__name__}: {error}')
        result.__dict__.update(getattr(error, '__dict__', {}))
    result.__cause__ = None
    result.__context__ = None
    return result.with_traceback(None)


class GuardedUdp:
    def __init__(self, udp, guard):
        self._udp, self._guard = udp, guard

    def send(self, command):
        self._guard()
        return self._udp.send(command)

    def __getattr__(self, name):
        return getattr(self._udp, name)


class ClientPool:
    def __init__(self, guard, record, before_send=None):
        self.robot = None
        self.guard, self.record = guard, record
        self.closed = False
        self.before_send = before_send or guard

    def acquire(self, factory, evidence):
        self.guard()
        if self.closed:
            raise RuntimeError('本轮UDP客户端已关闭，禁止重新建立')
        if self.robot is None:
            began = time.monotonic()
            robot = factory()
            try:
                self.guard()
                if robot.udp is None:
                    raise RuntimeError('无法建立本轮UDP客户端')
                robot.udp = GuardedUdp(robot.udp, self.before_send)
            except BaseException:
                robot.shutdown()
                raise
            robot._motion_tracking = getattr(self,'tracking',None)
            robot._trial_evidence = evidence
            self.robot = robot
            if self.before_send != self.guard:
                robot._continuous_wait_feedback = self.before_send
            self.record('udp_created', elapsed_s=time.monotonic()-began)
        self.robot._trial_evidence = evidence
        return self.robot

    def close(self):
        self.closed = True
        if self.robot is not None:
            self.robot.shutdown()
            self.robot = None


class RoundControl:
    def __init__(self, motion, report, out):
        self.motion, self.report, self.out = motion, report, out
        self.summary = dict(status='not_started', connection_count=0, udp_count=0, stages=[], events=[])
        report['round_control'] = copy.deepcopy(self.summary)
        self.cancelled = threading.Event()
        self.closing = threading.Event()
        self.ready = Future()
        self.thread = None
        self.arm = None
        self.error = None
        self.epoch = None
        self.pool = ClientPool(self.send_guard, self.record, self.wait_feedback)
        self.resources = ExitStack()
        self.tracking = None
        self.event_lock = threading.Lock()
        self.pending = None
        self.last_completion = None
        self.used = False
        self.closed = False
        self.feedback_lock = threading.RLock()
        self.feedback_sample = None
        self.feedback_gate = FeedbackGate(
            lambda: self.motion.SettledJoints(self.motion.app.StationaryWindow(),self.read_arrays,self.epoch),
            self.record, loss_seconds=self.motion.app.STATE_TIMEOUT_S,
            settle_seconds=self.motion.app.TIMEOUT_SECONDS)

    def record(self, event, **fields):
        row = dict(event=event, unix_s=time.time(), monotonic_s=time.monotonic(), **fields)
        if event == 'feedback_pause':
            print('反馈短时缺帧：'+','.join(fields.get('sources',[]))+
                  '；暂停新目标，等待新鲜反馈和连续停稳。',flush=True)
        elif event == 'feedback_recovered':
            print('反馈已恢复并重新确认连续停稳；允许继续后续目标。',flush=True)
        with self.event_lock:
            self.summary['events'].append(row)
            if event == 'udp_created':
                self.summary['udp_count'] += 1
            try:
                with (self.out/'round_control_timing.jsonl').open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
            except OSError as error:
                self.summary['timing_write_error'] = str(error)

    def send_guard(self):
        if self.closing.is_set() and self.error is None:
            raise RoundClosing('本轮控制正常结束，禁止发送新目标')
        if self.cancelled.is_set():
            raise RuntimeError(str(self.error or '本轮控制已结束，禁止发送新目标'))
        if self.arm is not None:
            self.arm.check()
            # Pin the original ownership epoch; check_owned may otherwise update it.
            if self.epoch is not None and self.arm.state.get('epoch') != self.epoch:
                raise RuntimeError('本轮控制epoch已改变；不自动重新接管')
            self.arm.check_owned()

    def check(self):
        self.send_guard()
        if self.tracking is not None and self.arm is not None:
            context=getattr(self.pool.robot,'_tracking_context',{}) if self.pool.robot else {}
            self.tracking.observe_rt(self.arm.state,self.arm.stamp,context)
        if self.pool.robot is None:
            return True
        with self.feedback_lock:
            # Bypass the worker's blocking wait here: this runs on the WS event
            # loop too, which must keep receiving states and sending heartbeats.
            sample = self.pool.robot._continuous_raw_state()
            now = time.monotonic()
            detail = feedback_ages(sample,self.arm,now)
            detail['stage']=(getattr(self.pool.robot,'_trial_evidence',{}).get('stages') or [{}])[0].get('stage','first_reset')
            if not detail['sources']:
                try:
                    self.motion.validate_pose_sample(sample)
                except ValueError as error:
                    # The clock may cross 250ms between the two checks. Re-observe
                    # that same frame rather than losing the recovery transition.
                    if str(error)!='Baseline preflight requires fresh measured feedback':raise
                    now=time.monotonic()
                    detail=feedback_ages(sample,self.arm,now)
                    if not detail['sources']:raise
            ready = self.feedback_gate.update(detail,sample,self.arm.state,now)
            self.feedback_sample = sample if ready else None
            return ready

    def wait_feedback(self):
        """Worker/caller only; block new sends, never the WS/heartbeat event loop."""
        if self.thread is not None and threading.current_thread() is self.thread:
            raise RuntimeError('反馈等待不得阻塞控制事件循环')
        while True:
            with self.feedback_lock:
                if self.check():
                    return self.feedback_sample
            time.sleep(.01)

    async def wait_feedback_async(self):
        while not self.check():
            await asyncio.sleep(.02)

    def fail(self, error):
        if self.error is None:
            self.error = detached_error(error)
            self.summary.update(status='failed', reason=f'{type(error).__name__}: {error}')
            self.record('round_failed', reason=str(error))
            if self.arm is not None:
                try:
                    diagnostic=self.arm.timing.snapshot(error,self.arm.state)
                    diagnostic['feedback_timeout_detail'] = getattr(error,'feedback_detail',None)
                    diagnostic['feedback_last_observation'] = self.feedback_gate.latest
                    try:
                        if self.pool.robot is not None:
                            diagnostic['feedback_at_failure']=feedback_ages(
                                self.pool.robot._continuous_raw_state(),self.arm,time.monotonic())
                    except Exception as capture_error:
                        diagnostic['feedback_capture_error']=str(capture_error)
                    self.summary['control_failure_diagnostics']=diagnostic
                    (self.out/'control_failure.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2)+'\n')
                except Exception as diagnostic_error:
                    self.summary['control_diagnostic_error']=type(diagnostic_error).__name__
        self.cancelled.set()

    def execute(self, evidence, token, read_arrays, first_reset):
        if self.closed or self.error is not None:
            raise RuntimeError(str(self.error or '本轮会话已关闭，不能重用'))
        if not self.used:
            self.used = True
            self.token, self.read_arrays = token, read_arrays
            try:
                self._start(evidence)
            except BaseException as error:
                self.fail(error)
                raise
        elif token != self.token or read_arrays is not self.read_arrays:
            error = RuntimeError('整轮控制接口身份改变，禁止复用')
            self.fail(error)
            raise error
        self.check()
        if self.pending is not None:
            raise RuntimeError('本轮已有未完成动作，禁止并发运动')
        result = Future()
        self.pending = result
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (evidence, first_reset, result))
        try:
            while True:
                try:
                    return result.result(timeout=.1)
                except FutureTimeout:
                    if self.error is not None:
                        raise self.error
        except BaseException as error:
            self.fail(error)
            raise
        finally:
            self.pending = None

    def _start(self, evidence):
        m = self.motion
        # hkstp_arm -> utils imports torch. First import can hold the GIL;
        # do it before the owner/heartbeat exists, never inside first_reset.
        sys.path.insert(0, str(m.start.V/'dual_v2_2_agent/RobotRAM'))
        os.environ['CUARM_CONFIG_ROOT'] = str(m.start.ROOT/'configuration')
        began=time.monotonic()
        m.preload_motion_dependencies()
        self.record('motion_dependencies_ready',elapsed_s=time.monotonic()-began,
                    policy='imports_only_before_owner_no_udp')
        began = time.monotonic()
        pid = m.start.existing_planner()
        if pid is None:
            raise RuntimeError('J6 Planner未运行；未接管')
        m.start.wait_planner(pid, None)
        self.record('planner_verified', elapsed_s=time.monotonic()-began, pid=pid)
        sys.path.insert(0, str(m.start.V/'dual_v2_2_agent/RobotRAM'))
        os.environ['CUARM_CONFIG_ROOT'] = str(m.start.ROOT/'configuration')
        m.app.RUNTIME.mkdir(parents=True, exist_ok=True)
        lock = self.resources.enter_context((m.app.RUNTIME/'lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有抓取会话；未接管') from None
        m.app.refuse_duplicate_agent()
        from motion_tracking import TrackingJournal
        self.tracking=TrackingJournal(self.out)
        self.pool.tracking=self.tracking
        began = time.monotonic()
        url = self.resources.enter_context(m.app.dedicated_control(remote_port=8810))
        self.summary['connection_count'] += 1
        self.record('ssh_connected', elapsed_s=time.monotonic()-began)
        self.thread = threading.Thread(target=self._thread_main, args=(url,evidence), daemon=True,
                                       name='j6-round-control')
        self.thread.start()
        while True:
            try:
                self.ready.result(timeout=.1)
                return
            except FutureTimeout:
                if self.error is not None:
                    raise self.error

    def _thread_main(self, url, evidence):
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="j6-motion") as self.worker_pool:
                asyncio.run(self._serve(url, evidence))
        except BaseException as error:
            self.fail(error)
        finally:
            if not self.ready.done():
                self.ready.set_exception(self.error or RuntimeError('本轮控制初始化未完成'))
            pending = self.pending
            if pending is not None and not pending.done():
                pending.set_exception(self.error or RuntimeError('本轮控制已停止'))

    async def _serve(self, url, initial):
        m = self.motion
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        worker = None
        async with m.app.websockets.connect(url, proxy=None, compression=None,
                    open_timeout=5, ping_interval=2, close_timeout=1) as connection:
            ws = m.app.JsonSocket(connection)
            await ws.send_json(dict(token=self.token))
            self.arm = m.app.ArmSession(ws)
            try:
                await self.arm.wait(lambda s:s.get('fresh') and s.get('planner',{}).get('connected'),
                                    5, '双夹爪Planner反馈')
                if self.arm.state.get('active') or self.arm.state['planner'].get('enabled'):
                    raise RuntimeError('已有活动控制，未接管')
                def guard():
                    if self.cancelled.is_set() or self.closing.is_set():
                        raise RuntimeError('本轮接管已取消')
                    m.check_start(initial,self.arm.state,self.read_arrays)
                began = time.monotonic()
                await self.arm.enable(guard)
                self.epoch = self.arm.state['epoch']
                # Enable legitimately changes epoch once, after the original
                # snapshot was checked. Never rebind an epoch on later stages.
                initial['acquisition_epoch'] = initial['epoch']
                initial['epoch'] = self.epoch
                self.summary.update(status='owned', epoch=self.epoch)
                self.record('owner_enabled', elapsed_s=time.monotonic()-began, epoch=self.epoch)
                print('J6整轮控制已建立；后续阶段复用连接和UDP，保留到位、停稳和失联保护。', flush=True)
                self.ready.set_result(None)
                while not self.closing.is_set():
                    self.check()  # Also runs while GPT/CPU work blocks the caller.
                    try:
                        job = await asyncio.wait_for(self.queue.get(), timeout=.05)
                    except asyncio.TimeoutError:
                        continue
                    evidence, first_reset, result = job
                    await self.wait_feedback_async()
                    m.check_start(evidence,self.arm.state,self.read_arrays)
                    label = 'first_reset' if first_reset else evidence['stages'][0]['stage']
                    began = time.monotonic()
                    row = dict(stage=label, started_monotonic_s=began, status='running',
                               idle_before_s=None if self.last_completion is None else began-self.last_completion)
                    self.summary['stages'].append(row)
                    self.record('stage_started', **row)
                    if self.pool.robot is not None:self.pool.robot._tracking_context={'stage':label}
                    def read_grip():
                        self.wait_feedback()
                        return self.read_arrays(self.arm.state)[2]
                    async def settle(value, at_start):
                        window=m.SettledJoints(m.app.StationaryWindow(),self.read_arrays,self.epoch)
                        generation=self.feedback_gate.generation
                        def observe(state):
                            nonlocal window,generation
                            ready=self.check()
                            if not ready or generation!=self.feedback_gate.generation:
                                window=m.SettledJoints(m.app.StationaryWindow(),self.read_arrays,self.epoch)
                                generation=self.feedback_gate.generation
                            if not ready:return False
                            return window.observe(state,time.monotonic())
                        t=time.monotonic()
                        try:
                            await self.arm.wait(observe,m.app.TIMEOUT_SECONDS,
                                '本段执行前连续停稳' if at_start else '本段到位后连续停稳')
                            if not at_start:m.check_settled_j7(evidence,window.latest['joints_rad'])
                            value['settled_feedback']=window.latest
                            return value
                        finally:
                            evidence['start_settling' if at_start else 'completion_settling']=window.latest
                            self.record('start_settling' if at_start else 'completion_settling',
                                stage=label,elapsed_s=time.monotonic()-t,
                                confirmed=bool(window.latest and window.latest.get('settled')))
                    def confirmed(value, *, at_start=False):
                        f=asyncio.run_coroutine_threadsafe(settle(value,at_start),self.loop)
                        try:return f.result(timeout=m.app.TIMEOUT_SECONDS+2.)
                        except BaseException:
                            f.cancel()
                            raise
                    worker=self.loop.run_in_executor(self.worker_pool, partial(m.drive,evidence,self.cancelled,
                        first_reset,read_grip,confirmed,
                        lambda:confirmed({},at_start=True)['settled_feedback'],client_pool=self.pool))
                    try:
                        while not worker.done():
                            self.check()
                            await asyncio.sleep(.02)
                        value=worker.result()
                        await self.wait_feedback_async()
                        self.last_completion=time.monotonic()
                        row.update(status='confirmed',elapsed_s=self.last_completion-began)
                        value['round_session']=dict(epoch=self.epoch,reused=True,released=False)
                        self.record('stage_completed',stage=label,elapsed_s=row['elapsed_s'])
                        result.set_result(value)
                    except BaseException as error:
                        row.update(status='failed',elapsed_s=time.monotonic()-began,reason=str(error))
                        if not result.done():result.set_exception(detached_error(error))
                        raise
                    finally:
                        # A worker must finish before resources/targets may be reused.
                        if worker.done():worker=None
            except RoundClosing:
                if self.pending is not None or worker is not None:
                    self.fail(RuntimeError("动作未完成时控制会话被关闭"))
                    raise self.error
            except BaseException as error:
                self.fail(error)
                raise
            finally:
                self.cancelled.set()
                try:
                    await self.arm.close()  # STOP/release once, before waiting for a worker.
                    self.record('owner_close_requested')
                finally:
                    if worker is not None:
                        await asyncio.gather(worker,return_exceptions=True)
                    self.pool.close()
                    self.record('udp_closed')

    def close(self):
        if self.closed:return
        self.closed=True
        self.closing.set()
        self.cancelled.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=self.motion.app.TIMEOUT_SECONDS+5.)
                if self.thread.is_alive():
                    self.fail(RuntimeError('控制线程尚未退出；禁止下一轮重用'))
                    # Closing our private transport revokes ownership even if a worker hangs.
                    self.resources.close()
                    raise self.error
                self._confirm_release()
        finally:
            self.resources.close()
            if self.tracking is not None:self.summary['joint_tracking']=self.tracking.close()
            self.report['round_control'] = copy.deepcopy(self.summary)
        if self.used and self.error is None:
            self.summary['status']='completed_and_released'
        self.record('round_closed',status=self.summary['status'])
        self.report['round_control'] = copy.deepcopy(self.summary)
        if self.error is not None:raise self.error

    def _confirm_release(self):
        m=self.motion
        request=Request(m.start.GUI+'/api/state',headers={'X-Arm-Token':self.token})
        opener=build_opener(ProxyHandler({}))
        deadline=time.monotonic()+3.
        try:
            while time.monotonic()<deadline:
                with opener.open(request,timeout=1.) as response:s=json.load(response)
                self.read_arrays(s)
                if not s.get('active') and not s.get('planner',{}).get('enabled'):
                    self.summary['released_epoch']=s.get('epoch')
                    self.record('release_confirmed',epoch=s.get('epoch'))
                    return
                time.sleep(.05)
            raise RuntimeError('整轮控制释放未确认')
        except Exception as error:
            self.summary['release_error']=str(error)
            if self.error is None:self.fail(error)


def current():
    return getattr(_CURRENT,'session',None)


@contextmanager
def motion_round(report, out):
    import right_approach_motion as motion
    if current() is not None:raise RuntimeError('不允许嵌套运动会话')
    session=RoundControl(motion,report,out)
    _CURRENT.session=session
    primary=None
    try:
        yield session
    except BaseException as error:
        primary=error
        session.fail(error)
        raise
    finally:
        try:
            session.close()
        except BaseException:
            if primary is None:raise
        finally:
            _CURRENT.session=None
