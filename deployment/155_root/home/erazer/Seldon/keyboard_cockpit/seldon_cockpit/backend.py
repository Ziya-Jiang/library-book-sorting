"""50 Hz control in a separate process, independent of Qt and camera decoding."""
import json
import os
import math
import queue
import tempfile
import time
import uuid
from pathlib import Path
from .control import Motion, target, stop_reason
from .angles import AngleFeedback


def latest(q, fallback=None):
    for _ in range(8):
        try:
            fallback = q.get_nowait()
        except queue.Empty:
            break
    return fallback


def offer(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        pass  # Never block a watchdog or UI on a slow consumer.


class DriveFeedback:
    """Physical readiness from fresh CiA402 PDOs, never a service-success flag."""
    AXES = {'propel_fr': 3, 'propel_rl': 3, 'steer_fr': 1, 'steer_rl': 1}

    def __init__(self, stopped_velocity_threshold=.25):
        self.values = {axis: {} for axis in self.AXES}
        self.stopped_velocity_threshold = float(stopped_velocity_threshold)

    def object(self, axis, index, value, now):
        if index in (0x6041, 0x6061):
            self.values[axis][index] = (int(value), now)

    def joint(self, axis, velocity, now):
        self.values[axis]['velocity'] = (float(velocity), now)

    def reason(self, now, since=0., enabled=None):
        for axis, mode in self.AXES.items():
            data = self.values[axis]
            for key in (0x6041, 0x6061, 'velocity'):
                value, stamp = data.get(key, (float('nan'), -1e9))
                if not math.isfinite(value) or not 0 <= now-stamp <= .3 or stamp < since:
                    return f'{axis} 状态反馈未更新'
            word = data[0x6041][0]
            if enabled is not False and word & 8:
                return f'{axis} 驱动故障 (0x6041=0x{word:04x})'
            if enabled is True:
                if word & 0x6f != 0x27:
                    return f'{axis} 尚未使能 (0x6041=0x{word:04x})'
                if data[0x6061][0] & 0xff != mode:
                    return f'{axis} 模式未就绪 (需要 {mode}，当前 {data[0x6061][0] & 0xff})'
            if enabled is False:
                if (word & 0x6f == 0x27
                        or abs(data['velocity'][0]) > self.stopped_velocity_threshold):
                    return f'{axis} 等待禁用及静止确认'
        return ''

    def snapshot(self):
        return {axis: {'statusword': data.get(0x6041, (None,))[0],
                       'mode': data.get(0x6061, (None,))[0],
                       'velocity': data.get('velocity', (None,))[0]}
                for axis, data in self.values.items()}


class RosPort:
    def __init__(self, prefix="", config=None):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_srvs.srv import Trigger
        from canopen_interfaces.msg import COData
        from sensor_msgs.msg import JointState
        from rcl_interfaces.srv import GetParameters
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        rclpy.init(args=[])
        self.rclpy, self.Twist, self.Trigger = rclpy, Twist, Trigger
        self.name = f"seldon_keyboard_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.prefix = prefix
        self.node = rclpy.create_node(self.name)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub = self.node.create_publisher(Twist, prefix+"/cmd_vel_joy", qos)
        self.clients = {name: self.node.create_client(Trigger, prefix+"/chassis/"+name)
                        for name in ("enable", "disable", "set_mode_0", "set_mode_1")}
        self.accepted_at = 0.0
        self.accepted_count = 0
        self.accepted = (0., 0., 0.)
        self.measured, self.odom_at = None, 0.0
        self.node.create_subscription(Twist, prefix+"/chassis/accepted_cmd_vel", self.on_accepted, 1)
        self.node.create_subscription(Odometry, prefix+"/odom", self.on_odom, 1)
        threshold = (config or {}).get('initialization_speed_tolerance', .25)
        self.drives = DriveFeedback(threshold)
        self.angles = AngleFeedback()
        self.angle_parameters = self.node.create_client(GetParameters, prefix+'/chassis_controller_node/get_parameters')
        self.angle_parameter_request = GetParameters.Request()
        self.angle_parameter_request.names = ['modules.steer_drivers', 'modules.steer_zero_offset', 'modules.x', 'modules.y']
        self.angle_parameter_future, self.angle_parameter_at = None, -1e9
        self.inhibit_safe_since = None
        for axis in self.drives.AXES:
            self.node.create_subscription(COData, prefix+'/'+axis+'/rpdo',
                lambda msg, axis=axis: self.drives.object(axis, msg.index, msg.data, time.monotonic())
                if msg.subindex == 0 else None, 10)
            self.node.create_subscription(JointState, prefix+'/'+axis+'/joint_states',
                lambda msg, axis=axis: self.on_joint(axis, msg), 1)
        self.created = time.monotonic()

    def on_joint(self, axis, msg):
        now = time.monotonic()
        if msg.velocity:
            self.drives.joint(axis, msg.velocity[0], now)
        self.angles.position(axis, msg.position[0] if msg.position else None, now)

    def poll_angle_parameters(self):
        now = time.monotonic()
        if self.angle_parameter_future is not None:
            if self.angle_parameter_future.done():
                try:
                    p = self.angle_parameter_future.result().values
                    self.angles.calibration(list(p[0].string_array_value), list(p[1].double_array_value),
                                            list(p[2].double_array_value), list(p[3].double_array_value), now)
                except Exception:
                    self.angles.geometry = {}
                self.angle_parameter_future = None
            elif now-self.angle_parameter_at > 2.:
                self.angle_parameter_future.cancel()
                self.angle_parameter_future = None
        if self.angle_parameter_future is None and now-self.angle_parameter_at >= 2. and self.angle_parameters.service_is_ready():
            self.angle_parameter_future = self.angle_parameters.call_async(self.angle_parameter_request)
            self.angle_parameter_at = now

    def on_accepted(self, msg):
        self.accepted_at = time.monotonic()
        self.accepted_count += 1
        self.accepted = msg.linear.x, msg.linear.y, msg.angular.z

    def on_odom(self, msg):
        self.measured = msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z
        self.odom_at = time.monotonic()

    def spin(self):
        # Four multiplexed PDO feeds exceed 50 callbacks/s. Drain callbacks
        # within a bounded budget without starving the independent watchdog.
        deadline = time.monotonic()+.006
        for _ in range(256):
            self.rclpy.spin_once(self.node, timeout_sec=0)
            if time.monotonic() >= deadline:
                break
        self.poll_angle_parameters()

    def preflight(self):
        if self.inhibit_reason():
            return self.inhibit_reason()
        if time.monotonic()-self.created < 1.0:
            return "ROS 发现中，请稍后重试"
        for topic in ("/cmd_vel", "/cmd_vel_joy"):
            others = [p.node_name for p in self.node.get_publishers_info_by_topic(self.prefix+topic)
                      if p.node_name != self.name]
            if others:
                return f"控制源冲突 {topic}: {', '.join(others)}"
        if self.pub.get_subscription_count() == 0:
            return "未连接底盘速度订阅者"
        if not all(c.service_is_ready() for c in self.clients.values()):
            return "底盘服务未就绪"
        return self.drives.reason(time.monotonic())

    def inhibit_reason(self):
        if not self.prefix and (Path(__file__).resolve().parent.parent/'runtime/drive-inhibit.json').exists():
            return '安全锁定：初始化出现非预期轮速，待现场排查；禁止接管'
        return ''

    def auto_clear_inhibit(self, now):
        """Clear a stale startup lock only after fresh disabled zero feedback.

        The lock is never removed merely because the file is old.  All four
        drives must report fresh feedback, no fault, a disabled CiA402 state,
        and near-zero velocity continuously for one second.
        """
        lock = Path(__file__).resolve().parent.parent/'runtime/drive-inhibit.json'
        if self.prefix or not lock.exists():
            self.inhibit_safe_since = None
            return True
        safe = True
        for data in self.drives.values.values():
            try:
                statusword, status_at = data[0x6041]
                velocity, velocity_at = data['velocity']
                _, mode_at = data[0x6061]
                safe &= (now-status_at <= .3 and now-velocity_at <= .3 and now-mode_at <= .3
                         and not (statusword & 8)
                         and (statusword & 0x6f) != 0x27
                         and math.isfinite(velocity) and abs(velocity) <= .02)
            except (KeyError, TypeError, ValueError):
                safe = False
        if not safe:
            self.inhibit_safe_since = None
            return False
        if self.inhibit_safe_since is None:
            self.inhibit_safe_since = now
            return False
        if now-self.inhibit_safe_since < 1.0:
            return False
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return False
        self.inhibit_safe_since = None
        return not lock.exists()

    def latch_inhibit(self, reason):
        """Persist a live initialization anomaly across worker/web restarts."""
        if self.prefix:
            return
        runtime = Path(__file__).resolve().parent.parent/'runtime'
        runtime.mkdir(exist_ok=True)
        payload = {
            'reason': reason,
            'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'drives': self.drives.snapshot(),
            'clear_policy': ('禁止普通接管及地面行驶；仅可在轮子离地、现场急停人员就位后，'
                             '由操作员明确授权受控诊断。'),
        }
        fd, temporary = tempfile.mkstemp(prefix='.drive-inhibit.', suffix='.tmp', dir=runtime)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, runtime/'drive-inhibit.json')
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def drive_reason(self, now, since=0., enabled=None):
        return self.drives.reason(now, since, enabled)

    def publish(self, velocity):
        m = self.Twist()
        m.linear.x, m.linear.y, m.angular.z = map(float, velocity)
        self.pub.publish(m)

    def call(self, name):
        return self.clients[name].call_async(self.Trigger.Request())

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


class Controller:
    def __init__(self, config, port=None):
        self.config, self.port = config, port
        self.motion = Motion(config)
        self.state, self.message = "LOCKED", "按 E 接管；J 停车"
        self.owned = False
        self.arm_id = 0
        self.future = None
        self.disable_future = None
        self.disable_at = -100.0
        self.disable_confirmed = False
        self.stage_at = 0.0
        self.active_at = 0.0
        self.stopped = False
        self.last_graph = 0.0
        self.gear = "normal"
        self.phase = ''
        self.ready_since = None
        self.enable_since = 0.
        self.verify_count = 0
        self.needs_disable = False
        self.unexpected_speed_since = {}
        self.network_paused = False
        self.arm_block_reason = ''

    def stable(self, good, now, duration=.3):
        if not good:
            self.ready_since = None
            return False
        if self.ready_since is None:
            self.ready_since = now
        return now-self.ready_since >= duration

    def start_call(self, name, state, phase, message, now):
        if name in ('enable', 'disable'):
            self.needs_disable = True
        self.future = self.port.call(name)
        self.state, self.phase, self.message = state, phase, message
        self.stage_at, self.ready_since = now, None

    def advance_startup(self, now):
        """Block the joy input during asynchronous init; prove drives + zero ACKs."""
        if self.phase in ('RESET', 'BLOCK_JOY', 'ENABLE_REQUEST', 'MANUAL'):
            if not self.future.done():
                if now-self.stage_at > self.config['service_timeout']:
                    self.stop('底盘服务超时；保持零速并请求禁用')
                return
            try:
                result = self.future.result()
                success, message = result.success, result.message
            except Exception as exc:
                success, message = False, str(exc)
            if not success:
                self.stop('底盘拒绝: '+message)
                return
            previous = self.phase
            self.stage_at, self.ready_since = now, None
            if previous == 'RESET':
                self.phase, self.message = 'DISABLED', '等待四轴禁用及静止确认'
            elif previous == 'BLOCK_JOY':
                self.enable_since = now
                self.start_call('enable', 'ENABLE', 'ENABLE_REQUEST', '初始化四轴；尚未开放驾驶', now)
            elif previous == 'ENABLE_REQUEST':
                self.phase, self.message = 'DRIVES', '等待四轴使能及模式确认'
            else:
                self.state, self.phase = 'ENABLE', 'ZERO_ACK'
                self.verify_count = self.port.accepted_count
                self.message = '验证连续零速指令回执'
            return
        if self.phase == 'DISABLED':
            error = self.port.drive_reason(now, self.stage_at, enabled=False)
            self.message = error or '确认四轴已停止'
            if self.stable(not error, now):
                self.phase, self.stage_at = 'ZERO_HOLD', now
                self.message = '建立零速窗口；保持禁用模式'
            elif now-self.stage_at > self.config['service_timeout']:
                self.stop('禁用确认超时: '+error)
        elif self.phase == 'ZERO_HOLD':
            if now-self.stage_at >= self.config['zero_hold_seconds']:
                # set_mode_1 changes only the input gate and brakes; unlike
                # disable it does not invoke Motor402 before its first init.
                # preflight forbids every other /cmd_vel or /cmd_vel_joy source.
                self.start_call('set_mode_1', 'MODE', 'BLOCK_JOY', '隔离驾驶输入；准备初始化', now)
        elif self.phase == 'DRIVES':
            error = self.port.drive_reason(now, self.enable_since, enabled=True)
            self.message = error or '四轴已就绪；等待状态稳定'
            if self.stable(not error, now):
                self.start_call('set_mode_0', 'MODE', 'MANUAL', '四轴就绪，切换手动模式', now)
            elif now-self.stage_at > self.config['service_timeout']:
                self.stop('四轴就绪确认超时: '+error)
        elif self.phase == 'ZERO_ACK':
            error = self.port.drive_reason(now, self.enable_since, enabled=True)
            zero = all(math.isfinite(v) and abs(v) < 1e-6 for v in self.port.accepted)
            fresh = self.port.accepted_at > self.stage_at and 0 <= now-self.port.accepted_at < .15
            if fresh and not zero:
                self.stop('零速验证收到非零回执')
            elif self.stable(not error and zero and fresh, now) and self.port.accepted_count-self.verify_count >= 5:
                self.state, self.phase, self.active_at = 'ACTIVE', '', now
                self.message = '四轴及零速回执已确认；已接管'
                self.network_paused = False
            elif now-self.stage_at > 2.:
                self.stop('零速回执验证失败: '+(error or '回执不连续'))

    def stop(self, reason):
        was_active = self.state == 'ACTIVE'
        self.motion.stop()
        self.network_paused = False
        self.state, self.message = "LOCKED", reason
        # Before any enable request, confirmed-disabled cold drives must not
        # receive /disable: Motor402 rejects it before /init, poisoning the
        # existing adapter's software state. Never skip stopping an active drive.
        self.stopped = self.owned and (self.needs_disable or was_active)
        if self.port and self.owned:
            self.port.publish((0., 0., 0.))

    def tick(self, intent, now, dt):
        intent_age = now-intent["time"] if intent else math.inf
        if self.state == "ACTIVE":
            if intent_age > self.config["heartbeat_timeout"] or intent.get("pause"):
                self.network_paused = True
            elif self.network_paused and not intent.get("keys"):
                self.network_paused = False
        reason = stop_reason(intent, now, self.config, self.port is not None)
        if reason and self.state != "LOCKED":
            self.stop(reason)

        # A failed takeover leaves the controller safely locked. Keep its
        # read-only readiness message current so a repaired CAN link or cleared
        # drive fault does not leave a stale error on the cockpit indefinitely.
        if (self.state == "LOCKED" and self.port
                and now-self.last_graph >= 0.1):
            self.last_graph = now
            self.arm_block_reason = self.port.preflight()
            if not self.owned:
                self.message = self.arm_block_reason or "按 E 接管；J 停车"

        request = intent.get("arm_id", 0) if intent else 0
        if request != self.arm_id:
            self.arm_id = request  # Requests are consumed even if rejected.
            pending = self.future is not None and not self.future.done()
            disabling = self.disable_future is not None and not self.disable_future.done()
            if self.state == "LOCKED" and not reason:
                error = self.port.preflight() if self.port else ""
                if intent.get("keys"):
                    error = "请松开所有运动键后按 E"
                if pending or disabling or (self.stopped and self.port):
                    error = "等待底盘停止确认，再按 E"
                if error:
                    self.message = error
                else:
                    self.owned, self.stopped = True, False
                    self.state, self.stage_at = "ZEROING", now
                    self.message = "建立零速窗口"
                    if self.port:
                        if not self.port.drive_reason(now, enabled=False):
                            self.phase, self.ready_since = 'DISABLED', None
                            self.message = '确认四轴已禁用及静止；不重复禁用'
                        else:
                            self.start_call('disable', 'ZEROING', 'RESET', '停止当前驱动，确认禁用', now)

        if self.owned and self.state != "LOCKED" and self.port and now-self.last_graph >= 0.1:
            self.last_graph = now
            error = self.port.preflight()
            if error:
                self.stop(error)

        if self.state not in ('LOCKED', 'ACTIVE') and intent and intent.get('keys'):
            self.stop('接管期间请松开所有运动键；准备好后重新按 E')
        if self.state not in ('LOCKED', 'ACTIVE') and self.port and hasattr(self.port, 'drives'):
            tolerance = self.config['initialization_speed_tolerance']
            hard_limit = self.config['initialization_speed_hard_limit']
            grace = self.config['initialization_speed_grace']
            for axis, data in self.port.drives.snapshot().items():
                velocity = data.get('velocity')
                # Steering can move while its position mode settles. Only
                # propulsion feedback indicates unintended chassis travel.
                if not axis.startswith('propel_') or velocity is None or not math.isfinite(velocity):
                    continue
                speed = abs(velocity)
                if speed <= tolerance:
                    self.unexpected_speed_since.pop(axis, None)
                    continue
                since = self.unexpected_speed_since.setdefault(axis, now)
                if speed >= hard_limit or now-since >= grace:
                    self.needs_disable = True
                    fault = f'初始化非预期轮速: {axis}={velocity:.12g} rad/s，已停止并锁定'
                    if hasattr(self.port, 'latch_inhibit'):
                        try:
                            self.port.latch_inhibit(fault)
                        except Exception:
                            # The live stop must never depend on filesystem I/O.
                            pass
                    self.stop(fault)
                    break
        if self.state in ('ZEROING', 'MODE', 'ENABLE'):
            if self.port:
                self.advance_startup(now)
            elif now-self.stage_at >= self.config['zero_hold_seconds']:
                self.state, self.active_at = "ACTIVE", now
                self.message = "离线演示 · 不连接底盘"
                self.network_paused = (intent_age > self.config['heartbeat_timeout']
                                       or bool(intent and intent.get('pause')))

        if self.state == "ACTIVE":
            error = self.port.drive_reason(now, self.enable_since, enabled=True) if self.port else ''
            if error:
                self.stop('四轴状态中断: '+error)
            elif self.port and now-max(self.port.accepted_at, self.active_at) > 0.6:
                self.stop("底盘指令回执中断")
            elif self.network_paused:
                self.motion.stop()
                self.message = '网络延迟，速度归零；松开运动键后自动恢复'
            else:
                desired, self.gear = target(intent.get("keys", []), self.config["speeds"])
                self.motion.step(desired, dt)
                self.message = '四轴及零速回执已确认；已接管' if self.port else '离线演示 · 不连接底盘'
                if not intent.get('focused'):
                    self.message = '已接管，失焦保持零速；回到驾驶页按运动键即可'
        else:
            self.motion.stop()

        if self.port and self.owned:
            self.port.publish(self.motion.velocity)
            # Continue until outstanding arm call AND a subsequent disable have
            # completed. A late enable response can never re-arm a stopped UI.
            if self.stopped:
                pending_arm = self.future is not None and not self.future.done()
                if self.disable_future is not None and self.disable_future.done():
                    try:
                        self.disable_confirmed = self.disable_future.result().success
                    except Exception:
                        self.disable_confirmed = False
                    self.disable_future = None
                if (self.disable_confirmed and not pending_arm and self.disable_at >= self.stage_at
                        and not self.port.drive_reason(now, self.disable_at, enabled=False)):
                    self.stopped = False
                if self.stopped and self.disable_future is None and now-self.disable_at >= .8:
                    self.disable_confirmed = False
                    self.disable_future = self.port.call("disable")
                    self.disable_at = now
                # An earlier disable response may race a late enable. While
                # enable is pending, keep issuing disables after its completion.
                if pending_arm:
                    self.stage_at = now

        return {"time": now, "state": self.state, "message": self.message,
                "arm_id": self.arm_id,
                "input_focused": bool(intent and intent.get('focused')),
                "arm_block_reason": self.arm_block_reason,
                "arm_ready": (self.state == 'LOCKED' and not self.arm_block_reason
                              and not (self.future is not None and not self.future.done())
                              and not (self.disable_future is not None and not self.disable_future.done())
                              and not (self.stopped and self.port)),
                "velocity": self.motion.velocity, "gear": self.gear,
                "accepted": self.port.accepted if self.port else None,
                "accepted_age": now-self.port.accepted_at if self.port else None,
                "phase": self.phase, "network_paused": self.network_paused,
                "drives": self.port.drives.snapshot() if self.port and hasattr(self.port, 'drives') else None,
                "angles": self.port.angles.snapshot(now) if self.port and hasattr(self.port, 'angles') else None,
                "measured": self.port.measured if self.port and now-self.port.odom_at < 0.5 else None}


def run_control(config, real, intents, statuses, shutdown):
    port = None
    controller = None
    try:
        port = RosPort(config=config) if real else None
        controller = Controller(config, port)
        intent, previous = None, time.monotonic()
        while not shutdown.is_set():
            started = time.monotonic()
            if port:
                port.spin()
            intent = latest(intents, intent)
            now = time.monotonic()
            status = controller.tick(intent, now, now-previous)
            if port and port.inhibit_reason() and not port.auto_clear_inhibit(now):
                status.update(state='ERROR', message=port.inhibit_reason())
            offer(statuses, status)
            previous = now
            shutdown.wait(max(0., 0.02-(time.monotonic()-started)))
    except Exception as exc:
        offer(statuses, {"time": time.monotonic(), "state": "ERROR", "message": str(exc),
                         "velocity": (0., 0., 0.), "gear": "normal", "measured": None})
    finally:
        if controller and port and controller.owned:
            controller.stop("退出驾驶舱")
            deadline = time.monotonic()+config["service_timeout"]+1.0
            while time.monotonic() < deadline:
                try:
                    port.spin()
                    controller.tick(None, time.monotonic(), 0.02)
                    if not controller.stopped:
                        break
                    time.sleep(0.02)
                except Exception:
                    break
        if port:
            port.close()
