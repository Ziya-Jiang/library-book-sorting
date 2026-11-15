import copy
from .rt_logs import OutputCapture, log_path
import ipaddress
import math
from pathlib import Path
import socket
import subprocess
import threading
import time
import sys
from .protocol_v31 import DEFAULT_GRIPPER_JOINT_SIZES, pack_command, unpack_state
from .calibration import JointCalibration
from .planner_bridge_v31 import PlannerBridge
from .v31_timing import ControlLoopTiming
from .stop_diagnostics import StopJournal, capture as capture_stop
from .heartbeat_policy import HeartbeatJournal, expired_heartbeat

from .active_arm_policy import required_joints_enabled, validate_manual_action

INITIAL_POSE_DEGREES = (90, 80, 10, 14, 10, -10, 0)

class Controller:
    def __init__(self, root, simulation=True, panel_port=12101, rt_port=12201, model_limits=None, calibration=None, kinematics=None):
        self.root, self.simulation = Path(root), simulation
        self.panel_port, self.rt_port = panel_port, rt_port
        self.lock = threading.RLock()
        self.sock = None
        self.endpoint = ('127.0.0.1', rt_port)
        self.state = None
        self.received = 0
        self.active = False
        self.owner = None
        self.heartbeat = 0
        self.epoch = 0
        self.sequence = 1
        self.target = [[0.0]*7 for _ in range(2)]
        self.gripper_target = [[0.0]*size for size in DEFAULT_GRIPPER_JOINT_SIZES]
        self.duration = 5.0
        self.message = '尚未连接 rt_control'
        self.rt_arm_mode = "both"
        self.process = None
        self.log = None
        self.closed = False
        self.exit_reported = False
        self.rt_log = ""
        self.log_read_at = 0
        self.model_limits = model_limits
        self.calibration = calibration or JointCalibration()
        self.kinematics = kinematics
        self.last_jog = 0.
        self.loop_timing = ControlLoopTiming()
        self.planner = PlannerBridge(self)
        self.stop_journal = StopJournal(self.root)
        self.heartbeat_journal = HeartbeatJournal(self.root)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def fresh(self):
        return self.state is not None and time.monotonic()-self.received < .3

    def stop(self, reason='控制已停止', *, source=None):
        with self.lock:
            caller=sys._getframe(1)
            source=source or f'{Path(caller.f_code.co_filename).name}:{caller.f_code.co_name}:{caller.f_lineno}'
            del caller
            self._stop_context=capture_stop(self,reason,source)
            try:
                self.planner.disarm()
            finally:
                journal=getattr(self,'stop_journal',None)
                if journal is not None:journal.record(self._stop_context)
                self._stop_context=None
            self.active, self.owner = False, None
            self.epoch += 1
            if self.state:
                self.target = [a['position'][:] for a in self.state['arms']]
                if len(self.state.get('grippers', [])) == len(self.gripper_target):
                    self.gripper_target = [g['position'][:] for g in self.state['grippers']]
            self.message = reason

    def connect(self, ip='127.0.0.1', port=None):
        ip = str(ipaddress.IPv4Address(ip))
        port = int(port or self.rt_port)
        if not 1024 <= port <= 65535:
            raise ValueError('RT 端口必须为 1024–65535')
        with self.lock:
            if self.active:
                raise ValueError('先停止控制再切换连接')
            if self.sock and self.endpoint == (ip, port):
                return
            if self.sock:
                self.sock.close()
            self.state, self.received = None, 0
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                self.sock.bind(('127.0.0.1' if ipaddress.ip_address(ip).is_loopback else '0.0.0.0', self.panel_port))
                self.sock.setblocking(False)
            except Exception:
                self.sock.close()
                self.sock = None
                raise
            self.endpoint = (ip, port)
            self.message = '等待 RT 反馈'

    def check_live(self):
        if not self.fresh():
            raise ValueError('没有新鲜的 RT 反馈，禁止执行')
        if self.state['system'] in (0, 4, 5, 6) or self.state['diagnostics'] & 0xffff0000:
            raise ValueError('RT 状态未就绪或存在硬件故障')

    def verify(self, owner, message, require_owner=True):
        now = time.monotonic()
        challenge = message.get('challenge')
        if not isinstance(challenge, int) or not 0 <= now-challenge*.05 <= .3:
            raise ValueError('控制请求已过期')
        if message.get('epoch') != self.epoch:
            raise ValueError('停止后请重新接管控制')
        if require_owner and self.owner != owner:
            raise ValueError('请先接管关节控制')

    def action(self, owner, m, *, received_monotonic=None):
        with self.lock:
            action = m.get('action')
            if action == 'stop':
                self.stop(source='action:stop')
                return
            if action == 'heartbeat':
                discarded=expired_heartbeat(self,owner,m,received_monotonic=received_monotonic)
                if discarded is not None:return discarded
                self.heartbeat = time.monotonic()
                return
            self.verify(owner, m, require_owner=False)
            validate_manual_action(self, m)
            if action == 'planner_config':
                self.planner.configure(m)
                return
            if action == 'planner_disconnect':
                self.planner.disconnect()
                return
            if action == 'planner_enable':
                if self.active:
                    raise ValueError('先停止当前控制，再允许 Planner')
                self.check_live()
                self._check_gripper_protocol()
                if not self.simulation and not required_joints_enabled(self):
                    raise ValueError('请检查全部关节使能状态')
                self.planner.arm()
                self.target = [a['position'][:] for a in self.state['arms']]
                self.gripper_target = [g['position'][:] for g in self.state['grippers']]
                self.owner, self.active, self.heartbeat = owner, True, time.monotonic()
                self.message = 'Planner 控制已开放；停止按钮或断流会撤销'
                return
            if self.planner.enabled:
                raise ValueError('Planner 控制中，请先停止再手动控制')
            if action == 'connect':
                self.connect(m.get('ip','127.0.0.1'), m.get('port'))
                return
            if action == 'start_rt':
                self.start_rt(m.get('reference','strict'), m.get('arm_mode','both'), m.get('left_static_confirmed',False))
                return
            if action == 'stop_rt':
                self.stop('RT 停止请求已发送')
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                return
            if action == 'claim':
                self.check_live()
                self._check_gripper_protocol()
                if self.owner and self.owner != owner:
                    raise ValueError('另一个页面正在控制')
                if not self.simulation and not required_joints_enabled(self):
                    raise ValueError('驱动未启用全部关节；请检查驱动状态')
                # Seed ALL joints from measured feedback, never from a default zero pose.
                self.target = [a['position'][:] for a in self.state['arms']]
                self.gripper_target = [g['position'][:] for g in self.state['grippers']]
                self.owner, self.active, self.heartbeat = owner, True, time.monotonic()
                self.message = '关节控制已接管；输入角度后点击执行'
                return
            self.verify(owner, m)
            self.check_live()
            if action == 'cartesian':
                if self.kinematics is None:
                    raise ValueError('末端控制未开放；启动上位机时添加 --cartesian')
                now = time.monotonic()
                if now-self.last_jog < .045:
                    raise ValueError('末端指令过于频繁')
                deltas = m.get('deltas')
                if not isinstance(deltas, list) or len(deltas) != 2:
                    raise ValueError('需要左右臂各一个 XYZ 增量')
                targets = []
                for a, delta in enumerate(deltas):
                    q = [self.calibration.to_v2(a, j, value) for j, value in enumerate(self.state['arms'][a]['position'])]
                    bounds = [self.calibration.bounds(a, j, pair) for j, pair in enumerate(self.state['arms'][a]['soft_position'])]
                    if self.model_limits:
                        bounds = [(max(lo, model[0]), min(hi, model[1])) for (lo, hi), model in zip(bounds, self.model_limits[a])]
                    result = self.kinematics.jog(a, q, delta, bounds)
                    targets.append([self.calibration.to_rt(a, j, value) for j, value in enumerate(result)])
                # Commit both arms atomically; small jogs retain the existing RT interpolator.
                self.target, self.duration = targets, 2.
                self.last_jog = self.heartbeat = now
                self.message = '手柄末端微动 · V2 上身坐标系'
                return
            if action == 'gripper':
                self._check_gripper_protocol()
                index = m.get('gripper')
                if type(index) is not int or not 0 <= index < len(self.gripper_target):
                    raise ValueError('末端编号无效：0=左手，1=右夹爪')
                gesture = m.get('gesture')
                presets = ([[0], [1]], [[0], [1]])
                if gesture is not None and gesture not in ('open', 'close'):
                    raise ValueError('未知末端动作')
                values = (presets[index][int(gesture == 'close')] if gesture is not None
                          else m.get('positions', m.get('position')))
                if isinstance(values, (int, float)):
                    values = [values] * len(self.gripper_target[index])
                if not isinstance(values, list) or len(values) != len(self.gripper_target[index]):
                    raise ValueError('末端目标关节数无效')
                values = [float(value) for value in values]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                    raise ValueError('末端目标必须是 0–1 之间的有限数')
                for value, soft, hard in zip(values, self.state['grippers'][index]['soft_position'], self.state['grippers'][index]['hard_position']):
                    if not max(0, soft[0], hard[0]) <= value <= min(1, soft[1], hard[1]):
                        raise ValueError('末端目标超出 RT 限位')
                self.gripper_target[index] = values
                self.heartbeat = time.monotonic()
                self.message = f'{"左手" if index == 0 else "右夹爪"}目标已更新'
                return
            if action in ('initial_pose', 'zero_pose'):
                duration = float(m.get('duration', 5))
                if not math.isfinite(duration) or not 2 <= duration <= 30:
                    raise ValueError('执行时间为 2–30 秒')
                pose_degrees = INITIAL_POSE_DEGREES if action == 'initial_pose' else [0.0] * 7
                pose_name = '立正' if action == 'initial_pose' else 'Go Home（全关节零点）'
                targets = []
                for a in range(2):
                    arm_targets = []
                    for j, degrees in enumerate(pose_degrees):
                        target = math.radians(degrees)
                        lo, hi = self.calibration.bounds(a, j, self.state['arms'][a]['soft_position'][j])
                        if self.model_limits:
                            lo, hi = max(lo, self.model_limits[a][j][0]), min(hi, self.model_limits[a][j][1])
                        if not lo <= target <= hi:
                            raise ValueError(f'{"左" if a == 0 else "右"}臂 J{j+1} 目标角 {degrees:g}° 超出限位 {math.degrees(lo):.1f}°～{math.degrees(hi):.1f}°')
                        raw = self.calibration.to_rt(a, j, target)
                        measured = self.state['arms'][a]['position'][j]
                        if not math.isfinite(raw) or not math.isfinite(measured):
                            raise ValueError('关节反馈或目标无效')
                        duration = max(duration, math.ceil(abs(raw-measured)/math.radians(10)))
                        arm_targets.append(raw)
                    targets.append(arm_targets)
                if duration > 30:
                    raise ValueError(f'当前位置到{pose_name}需要超过 30 秒，请先分步调整关节')
                # Validate all 14 joints before committing either arm's target.
                self.target, self.duration = targets, duration
                self.heartbeat = time.monotonic()
                values = ', '.join(f'{value:g}' for value in pose_degrees)
                self.message = f'双臂 → {pose_name} [{values}]° · {duration:g} 秒'
                return
            if action == 'joint':
                a, j = m.get('arm'), m.get('joint')
                if type(a) is not int or type(j) is not int or not 0 <= a < 2 or not 0 <= j < 7:
                    raise ValueError('关节编号无效')
                degrees, duration = float(m['degrees']), float(m.get('duration', 5))
                if not math.isfinite(degrees) or not math.isfinite(duration) or not 2 <= duration <= 30:
                    raise ValueError('角度必须有限，执行时间为 2–30 秒')
                target = math.radians(degrees)
                lo, hi = self.calibration.bounds(a,j,self.state['arms'][a]['soft_position'][j])
                if self.model_limits:
                    lo, hi = max(lo, self.model_limits[a][j][0]), min(hi, self.model_limits[a][j][1])
                if not lo <= target <= hi:
                    raise ValueError(f'目标超出限位 {math.degrees(lo):.1f}°～{math.degrees(hi):.1f}°')
                target = self.calibration.to_rt(a,j,target)
                if abs(target-self.state['arms'][a]['position'][j]) / duration > math.radians(10):
                    raise ValueError('平均速度超过 10°/s，请增加执行时间')
                # Every single-joint command holds the other 13 joints at current measured pose.
                self.target = [arm['position'][:] for arm in self.state['arms']]
                self.target[a][j], self.duration = target, duration
                self.heartbeat = time.monotonic()
                self.message = f'{"左" if a == 0 else "右"}臂 J{j+1} → {degrees:.1f}°'
            else:
                raise ValueError('未知控制操作')

    def start_rt(self, reference, arm_mode="both", left_static_confirmed=False):
        if arm_mode not in ("both", "right"):
            raise ValueError("Unknown RT arm mode")
        if arm_mode == "right" and (reference != "strict" or left_static_confirmed is not True):
            raise ValueError("Right-only requires strict startup and confirmed unchanged left pose")
        if reference not in ('strict', 'accept-current'):
            raise ValueError('未知启动参考模式')
        if self.process and self.process.poll() is None:
            raise ValueError('本包 RT 已运行')
        # Do not run a second hardware driver alongside legacy processes.
        if not self.simulation:
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    exe = path.read_bytes().split(b'\0')[0].decode(errors='replace')
                except (OSError, PermissionError):
                    continue
                if Path(exe).name in ('dual_v2_2_rt_control', 'seldon_rt_control', 'dual_v2_2_upper_software'):
                    raise ValueError('已有双臂驱动/旧上位机运行，请先关闭它')
        binary = self.root/'runtime/build/seldon_rt_control'
        if not binary.is_file():
            raise ValueError('请先运行 ./build_rt.sh')
        self.stop('正在启动 RT；等待反馈')
        self.connect('127.0.0.1', self.rt_port)
        self.state, self.received = None, 0
        config_dir = self.root/'runtime/config'
        encoder = self.root/'runtime/encoder_last_position.bin'
        if self.log:
            self.log.close()
        self.log = OutputCapture(log_path('rt.log'))
        self.exit_reported = False
        self.rt_arm_mode = arm_mode
        try:
            self.process = subprocess.Popen([str(binary), '--config-dir', str(config_dir),
                '--encoder-file', str(encoder), '--simulation' if self.simulation else '--hardware',
                '--startup-reference', reference] + (['--right-arm-only'] if arm_mode == 'right' else []), stdout=self.log, stderr=subprocess.STDOUT,
                cwd=self.root/'runtime', start_new_session=True)
        except BaseException:
            self.log.close()
            self.log = None
            raise

    def snapshot(self, owner=None):
        with self.lock:
            fresh = self.fresh()
            return dict(fresh=fresh, state=self.calibration.state_for_web(self.state) if fresh else None,
                age_ms=round((time.monotonic()-self.received)*1000) if self.received else None,
                active=self.active, mine=self.owner == owner and self.active, epoch=self.epoch,
                challenge=int(time.monotonic()/.05), simulation=self.simulation,
                heartbeat_diagnostics=self.heartbeat_journal.snapshot(),
                endpoint=f'{self.endpoint[0]}:{self.endpoint[1]}', panel_port=self.panel_port,
                message=self.message, target=self.calibration.positions(self.target),
                initial_pose_degrees=list(INITIAL_POSE_DEGREES),
                grippers=self.state.get('grippers', []) if fresh and self.state else [],
                gripper_target=self.gripper_target,
                rt_assembly='dual_gripper',
                gripper_installed=[True, True],
                gripper_feedback_sources=['command_echo' if not self.simulation else 'simulation',
                                          'command_echo' if not self.simulation else 'simulation'],
                capabilities={'cartesian': self.kinematics is not None,
                              'gripper': fresh and self._gripper_protocol_ok()},
                planner=self.planner.snapshot(),
                control_loop_timing=self.loop_timing.snapshot(),
                stop_diagnostics=self.stop_journal.snapshot(),
                coordinate_system="V2 calibrated", calibration=self.calibration.manifest(),
                right_only_supported=True, rt_arm_mode=getattr(self,"rt_arm_mode","both"),
                left_pose_source="saved_static_reference" if getattr(self,"rt_arm_mode","both") == "right" else "measured",
                rt_running=self.process is not None and self.process.poll() is None,
                rt_exit=self.process.poll() if self.process else None, rt_log=self.rt_log)

    def run(self):
        while not self.closed:
            cycle_requested = time.monotonic()
            with self.lock:
                self.loop_timing.begin(cycle_requested, time.monotonic())
                try:
                    if self.sock:
                        # Bounded drain protects the watchdog from packet floods.
                        for _ in range(32):
                            try:
                                raw, sender = self.sock.recvfrom(65536)
                                # Legacy RT uses a separate ephemeral source port for its sender.
                                if sender[0] != self.endpoint[0]:
                                    continue
                            except (BlockingIOError, ConnectionRefusedError):
                                break
                            try:
                                state = unpack_state(raw)
                            except (ValueError, UnicodeError):
                                continue
                            if self.state and state['timestamp'] <= self.state['timestamp']:
                                continue
                            self.state, self.received = state, time.monotonic()
                            self.sequence = max(self.sequence, state['sequence']+1)
                            if not self._gripper_protocol_ok():
                                if self.active:
                                    self.stop('RT 末端协议已改变，控制已停止')
                                self.message = 'RT 末端协议不匹配；需要 left gripper(1) + right gripper(1)，请重启 RT 加载新配置'
                        self.loop_timing.mark("rt_receive")
                        if self.active and not self.fresh():
                            self.stop('RT 反馈超时，控制已停止')
                        elif (self.active and not self.planner.enabled and
                              time.monotonic()-self.heartbeat > .3):
                            self.stop('浏览器心跳超时，手动控制已停止')
                        if self.active and self.state and (self.state['system'] in (4,6) or self.state['diagnostics'] & 0xffff0000):
                            self.stop('RT 报告故障，控制已停止')
                        self.loop_timing.mark("rt_safety")
                        self.planner.tick()
                        self.loop_timing.mark("planner")
                        self.planner.check_control_deadline()
                        target = self.target if self.active else ([a['position'] for a in self.state['arms']] if self.state else self.target)
                        # Before first telemetry: receive only. Avoid overwriting driver enable state with defaults.
                        if self.state and self._gripper_protocol_ok():
                            self.sequence += 1
                            raw = pack_command(time.time_ns()//1000, self.sequence, self.state, target,
                                self.active, self.simulation, self.duration,
                                self.gripper_target if self.active else [g['position'] for g in self.state['grippers']],
                                waypoints=self.planner.waypoints if self.planner.enabled else None,
                                waypoint_dt=self.planner.waypoint_dt,
                                planner_mode=self.planner.mode if self.planner.enabled else 0,
                                planner_ratio=self.planner.ratio if self.planner.enabled else .1)
                            self.loop_timing.mark("rt_pack")
                            was_active = self.active
                            self.planner.check_control_deadline()
                            if was_active and not self.active:
                                # Do not send the active payload built before
                                # the deadline trip. Next cycle sends measured hold.
                                raw = None
                            try:
                                if raw is not None:
                                    self.sock.sendto(raw, self.endpoint)
                            except ConnectionRefusedError:
                                pass
                            self.loop_timing.mark("rt_send")
                    if not self.sock:
                        self.planner.tick()
                        self.loop_timing.mark("planner_without_rt")
                        self.planner.check_control_deadline()
                    if self.process and time.monotonic()-self.log_read_at > .5:
                        self.log_read_at = time.monotonic()
                        if self.log:
                            self.rt_log = self.log.tail()
                    self.loop_timing.mark("rt_log")
                    if self.process and self.process.poll() is not None and not self.exit_reported:
                        self.exit_reported = True
                        self.stop(f'RT 已退出（代码 {self.process.returncode}）；详情见启动日志')
                except Exception as exc:
                    self.stop('通信错误：'+str(exc))
                finally:
                    self.loop_timing.end()
            time.sleep(.01)

    def _gripper_protocol_ok(self):
        return (self.state is not None and
                [g.get('joint_size') for g in self.state.get('grippers', [])] == list(DEFAULT_GRIPPER_JOINT_SIZES))

    def _check_gripper_protocol(self):
        if not self._gripper_protocol_ok():
            actual = [g.get('joint_size') for g in (self.state or {}).get('grippers', [])]
            raise ValueError(f'RT 末端协议不匹配：需要 left gripper(1)+right gripper(1)，当前为 {actual or "no_gripper"}')

    def close(self):
        self.stop('程序退出')
        self.planner.disconnect()
        time.sleep(.05)
        self.closed = True
        self.thread.join(timeout=2)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                # Do not SIGKILL a hardware shutdown path.
                pass
        if self.sock:
            self.sock.close()
        self.stop_journal.close()
        self.heartbeat_journal.close()
        if self.log:
            self.log.close()
