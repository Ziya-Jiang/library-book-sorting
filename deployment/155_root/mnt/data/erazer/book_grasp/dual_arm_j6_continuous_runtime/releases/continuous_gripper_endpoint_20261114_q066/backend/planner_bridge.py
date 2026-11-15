"""Original V2 RemotePanelCommand -> Seldon RT bridge.

Connection/telemetry and motion permission are separate. No driver startup.
No socket opens until a person applies an address from the authenticated page.
"""
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import socket
import time
from .protocol import Reader


PLANNER_MAX_JOINT_SPEED_RAD_S = math.radians(30.0)


def decode_command(raw):
    r = Reader(raw)
    duration, ratio = r.take(), r.take()
    if r.take(int) != 0 or r.take(int) != 0:
        raise ValueError('Planner 仅支持位置控制')
    mode = r.take(int)
    if mode not in (0, 1):
        raise ValueError('拒绝 RAW/无插值 Planner 命令')
    if r.take(int) != 2 or r.array(2, int) != [7, 7]:
        raise ValueError('Planner 需要 2×7 关节')
    target = [r.array(7), r.array(7)]
    n, dt = r.take(int), r.take()
    if not 0 <= n <= 64 or (mode == 1 and n < 2) or (mode == 0 and n != 0):
        raise ValueError('Planner 路径点数量无效（2–64）')
    points = [[r.array(7), r.array(7)] for _ in range(n)]
    if not n:
        if r.take() != 0:  # canonical Python codec encodes [] as one zero
            raise ValueError('Invalid empty waypoint field')
    if r.take(int) != 2 or r.array(2, int) != [1, 1]:
        raise ValueError('Planner 需要左右各 1 个夹爪关节')
    grippers = [r.array(1), r.array(1)]
    r.end()
    if not 2 <= duration <= 30 or not 0 < ratio <= .1:
        raise ValueError('Planner 时间或速度比例超限')
    if n and (not .05 <= dt <= 5 or n*dt > 30):
        raise ValueError('Planner 轨迹时间超限')
    if any(not 0 <= x <= 1 for g in grippers for x in g):
        raise ValueError('Planner 手爪目标必须在 0–1 内')
    return dict(target=target, points=points, dt=dt, duration=duration,
                grippers=grippers, mode=mode)


def encode_state(controller, started):
    # Source is the calibrated V2 feedback, never legacy FK. Original planner
    # derives tool poses from joint feedback; its converter ignores ee/tool_pose.
    s = controller.state
    q = controller.calibration.positions([a['position'] for a in s['arms']])
    f = [int(started), 2, 7, 7]
    f += [x for row in q for x in row]
    for key in ('velocity', 'torque'):
        f += [controller.calibration.gain[a][j]*x for a, arm in enumerate(s['arms']) for j, x in enumerate(arm[key])]
    f += [-1.]*28  # unavailable EE/tool fields, NOT invented V2 poses
    f += [x for row in controller.calibration.positions(controller.target) for x in row]
    f += [2, 1, 1]
    f += [x for g in s['grippers'] for x in g['position']]
    return ('#'.join(map(str, f))+'#').encode('ascii')


class PlannerBridge:
    def __init__(self, controller):
        self.c = controller
        self.sock = None
        self.settings = dict(ip='', port=30241, local_ip='0.0.0.0', local_port=30240)
        self.enabled = False
        self.last_rx = self.last_tx = self.armed_at = 0.
        self.rx_count = self.tx_count = self.rejected = 0
        self.last_digest = None
        self.waypoints = []
        self.waypoint_dt = .5
        self.status = '未连接 Planner；请手动填写地址'
        self.path = controller.root/'runtime/planner.json'
        if self.path.is_file():
            try:
                self.settings = self.validate_settings(json.loads(self.path.read_text()))
            except (OSError, ValueError, TypeError, KeyError):
                self.status = '已保存的 Planner 配置无效，请重新填写'

    def validate_settings(self, data):
        ip = ipaddress.IPv4Address(str(data['ip']))
        local = ipaddress.IPv4Address(str(data.get('local_ip', '0.0.0.0')))
        if ip.is_multicast or ip.is_unspecified or str(ip) == '255.255.255.255':
            raise ValueError('请输入具体的 Planner IPv4 地址')
        if local.is_multicast:
            raise ValueError('本机监听地址无效')
        port, local_port = int(data['port']), int(data['local_port'])
        if not 1024 <= port <= 65535 or not 1024 <= local_port <= 65535:
            raise ValueError('Planner 端口必须为 1024–65535')
        if local_port in (self.c.panel_port, self.c.rt_port):
            raise ValueError('Planner 端口不能占用 RT 端口')
        return dict(ip=str(ip), port=port, local_ip=str(local), local_port=local_port)

    def configure(self, data):
        if self.c.active:
            raise ValueError('请先停止控制，再修改 Planner 地址')
        config = self.validate_settings(data)
        self.disconnect()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((config['local_ip'], config['local_port']))
            sock.setblocking(False)
        except Exception:
            sock.close()
            raise
        self.sock, self.settings = sock, config
        try:
            self.path.parent.mkdir(exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(config, indent=2)+'\n')
            tmp.chmod(0o600)
            tmp.replace(self.path)
        except Exception:
            self.disconnect()
            raise
        self.status = '已监听；仅发送实测状态，尚未允许 Planner 控制'

    def disarm(self):
        self.enabled = False
        self.waypoints = []
        self.last_digest = None

    def disconnect(self):
        if self.enabled:
            self.c.stop('Planner 连接已关闭')
        self.disarm()
        if self.sock:
            self.sock.close()
        self.sock = None
        self.last_rx = self.last_tx = 0.
        self.rx_count = self.tx_count = self.rejected = 0
        self.status = 'Planner 未连接'

    def arm(self):
        if not self.sock:
            raise ValueError('先应用 Planner 地址并连接')
        self.c.check_live()
        self.c._check_gripper_protocol()
        if any(abs(v) > .02 for a in self.c.state['arms'] for v in a['velocity']):
            raise ValueError('双臂尚未静止，不能切换 Planner 控制')
        # Empty all queued datagrams before START. A flood prevents arming.
        for _ in range(128):
            try:
                self.sock.recvfrom(65536)
            except BlockingIOError:
                break
        else:
            raise ValueError('Planner 接收队列拥塞，拒绝接管')
        self.enabled = True
        self.armed_at = time.monotonic()
        self.last_rx = 0.
        self.last_digest = None
        self.waypoints = []
        self.status = '已允许 Planner；等待第一条新目标（不超时）'

    def validate_targets(self, command):
        c = self.c
        paths = command['points'] or [command['target']]
        converted = []
        previous = [a['position'][:] for a in c.state['arms']]
        required_waypoint_dt = float(command['dt'])
        for point in paths:
            row_out = []
            for a in range(2):
                vals = []
                for j in range(7):
                    arm = c.state['arms'][a]
                    soft = c.calibration.bounds(a, j, arm['soft_position'][j])
                    hard = c.calibration.bounds(a, j, arm['hard_position'][j])
                    lo, hi = max(soft[0], hard[0]), min(soft[1], hard[1])
                    if c.model_limits:
                        lo, hi = max(lo, c.model_limits[a][j][0]), min(hi, c.model_limits[a][j][1])
                    if not lo <= point[a][j] <= hi:
                        raise ValueError(f'Planner 臂 {a} J{j+1} 超过 V2/RT 限位')
                    raw = c.calibration.to_rt(a, j, point[a][j])
                    interval = command['dt'] if command['points'] else command['duration']
                    peak = math.pi*.5*abs(raw-previous[a][j])/interval
                    speed_bound = min(PLANNER_MAX_JOINT_SPEED_RAD_S, arm['soft_velocity'][j][1], -arm['soft_velocity'][j][0],
                                      arm['hard_velocity'][j][1], -arm['hard_velocity'][j][0])
                    if command['points']:
                        required_waypoint_dt = max(
                            required_waypoint_dt,
                            math.pi*.5*abs(raw-previous[a][j])/speed_bound,
                        )
                    elif peak > speed_bound:
                        raise ValueError('Planner 目标速度过大且没有可延长的轨迹')
                    vals.append(raw)
                row_out.append(vals)
            converted.append(row_out)
            previous = row_out
        # Validate final target too, even though waypoints are authoritative.
        for a in range(2):
            for j in range(7):
                if command['points'] and abs(command['target'][a][j]) > 20:
                    raise ValueError('Invalid Planner instantaneous target')
        for values, g in zip(command['grippers'], c.state['grippers']):
            for x, soft, hard in zip(values, g['soft_position'], g['hard_position']):
                if not max(0, soft[0], hard[0]) <= x <= min(1, soft[1], hard[1]):
                    raise ValueError('Planner 手爪目标超过驱动限位')
        if command['points'] and required_waypoint_dt > command['dt']:
            required_waypoint_dt *= 1.05
        if command['points'] and (
            required_waypoint_dt > 5.0
            or len(command['points']) * required_waypoint_dt > 30.0
        ):
            raise ValueError('Planner 轨迹即使降速也会超过 30 秒安全上限')
        return converted, required_waypoint_dt

    def tick(self):
        if not self.sock:
            return
        now = time.monotonic()
        if self.enabled and (not self.c.active or not self.c.fresh()):
            self.c.stop('Planner: RT 反馈或控制权丢失')
        # Planner UDP packets are complete, discrete trajectories rather than
        # a continuous command stream. Long gaps are expected while MotionGen
        # previews candidates, so packet silence must not revoke control. RT
        # feedback/control-loss checks above and the manual stop remain active.
        for _ in range(16):
            try:
                raw, sender = self.sock.recvfrom(60001)
            except BlockingIOError:
                break
            if sender[0] != self.settings['ip']:
                self.rejected += 1
                continue  # original planner uses an ephemeral source UDP port
            try:
                command = decode_command(raw)
                self.rx_count += 1
                if not self.enabled:
                    self.status = '收到 Planner 报文；控制未开放，目标已丢弃'
                    continue
                self.c.check_live()
                digest = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).digest()
                # MotionGen instant target changes along an unchanged trajectory.
                identity = dict(command)
                if command['points']:
                    identity['target'] = []
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()
                if digest != self.last_digest:
                    converted, safe_waypoint_dt = self.validate_targets(command)
                    self.c.target = converted[-1]
                    self.c.duration = command['duration']
                    self.c.gripper_target = command['grippers']
                    self.waypoints = converted if command['points'] else []
                    self.waypoint_dt = safe_waypoint_dt
                    if command['points'] and safe_waypoint_dt > command['dt']:
                        self.status = (
                            'Planner 控制中；轨迹已自动降速 '
                            f'{command["dt"]:.3f}s→{safe_waypoint_dt:.3f}s/点'
                        )
                    else:
                        self.status = 'Planner 控制中；所有目标通过限位与速度检查'
                    self.last_digest = digest
                self.last_rx = now
            except (ValueError, UnicodeError, TypeError, KeyError, OverflowError) as exc:
                self.rejected += 1
                reason = str(exc)
                if self.enabled:
                    # Reject only this target. Keep the Planner lease alive so
                    # a later valid plan can recover without manual re-arming.
                    # Holding measured feedback ensures an invalid target or
                    # stale trajectory is never forwarded to RT.
                    self.waypoints = []
                    self.last_digest = None
                    if self.c.state:
                        self.c.target = [
                            arm['position'][:] for arm in self.c.state['arms']
                        ]
                        if len(self.c.state.get('grippers', [])) == len(
                            self.c.gripper_target
                        ):
                            self.c.gripper_target = [
                                gripper['position'][:]
                                for gripper in self.c.state['grippers']
                            ]
                    self.c.duration = 2.0
                    self.last_rx = now
                    self.c.message = (
                        'Planner 当前目标已拒绝；控制保持开放：' + reason
                    )
                    self.status = (
                        '拒绝当前目标；Planner 控制仍开放：' + reason
                    )
                else:
                    self.status = '拒绝：' + reason
        if now-self.last_tx >= .05 and self.c.fresh() and self.c._gripper_protocol_ok():
            self.sock.sendto(encode_state(self.c, self.enabled), (self.settings['ip'], self.settings['port']))
            self.last_tx = now
            self.tx_count += 1

    def snapshot(self):
        return dict(**self.settings, connected=self.sock is not None, enabled=self.enabled,
                    status=self.status, received=self.rx_count, sent=self.tx_count, rejected=self.rejected,
                    rx_age_ms=round((time.monotonic()-self.last_rx)*1000) if self.last_rx else None)
