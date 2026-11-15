"""Local-network cockpit. Camera and 50 Hz safety controller are separate processes.

Authenticated E enables client retries until J. No server-generated heartbeats.
Use on a trusted LAN, Tailscale, or an SSH tunnel; not a public Internet service.
"""
import argparse
import asyncio
from collections import OrderedDict
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import time

from aiohttp import web, WSMsgType
from .backend import latest, offer, run_control
from .camera import run_camera
from .control import validate_config

ROOT = Path(__file__).resolve().parent.parent
KEYS = {'up', 'down', 'left', 'right', 'w', 's', 'a', 'd', 'shift', 'space'}
UNIFIED_PARENT_PORTS = {8798, 42377, 8790, 8795}


def origin_allowed(origin, request):
    if not origin:
        return True
    if origin == f'{request.scheme}://{request.host}':
        return True
    from urllib.parse import urlsplit
    parsed = urlsplit(origin)
    request_host = request.host.rsplit(':', 1)[0].strip('[]')
    return parsed.hostname == request_host and parsed.port in UNIFIED_PARENT_PORTS


class InputGate:
    """Single pilot lease; monotonic action IDs survive client reconnects."""
    def __init__(self, config, real, require_head_camera=True):
        self.config, self.real = config, real
        self.require_head_camera = require_head_camera
        self.owner = None
        self.arm_id = 0
        self.latched = True
        self.hold_start = None
        self.key_event = None
        self.network_paused = False
        self.stop_revision = 0
        self.stop_message = '已锁定'

    def stop(self, now, reason='控制连接中断；等待自动恢复'):
        self.latched = True
        self.stop_message = reason
        self.hold_start = self.key_event = None
        self.network_paused = False
        return dict(time=now, focused=False, stop=True, keys=[], arm_id=self.arm_id,
                    frame_time=0, pause=False, stop_reason=self.stop_message)

    def pause(self, client, now, frame_time):
        """Keep the pilot lease while forcing zero until an empty fresh input arrives."""
        if self.owner != client:
            return None
        self.hold_start = self.key_event = None
        self.network_paused = True
        return dict(time=now, focused=True, stop=self.latched, keys=[], arm_id=self.arm_id,
                    frame_time=frame_time, pause=True, stop_reason=self.stop_message)

    def accept(self, client, data, now, frame_time):
        action = data.get('action', 'heartbeat')
        if action == 'stop':
            self.stop_revision += 1
            return self.stop(now, 'J 停车；自动接管已关闭')
        if action == 'hold':
            return self.stop(now, '页面或画面暂停；等待自动恢复') if self.owner == client else None
        if self.owner not in (None, client):
            raise ValueError('另一窗口正在控制；请先在原窗口退出')
        if action not in ('arm', 'resume', 'heartbeat', 'pause'):
            raise ValueError('未知操作')
        if action == 'resume' and data.get('stop_revision') != self.stop_revision:
            raise ValueError('停车指令已更新；请重新按 E')
        keys = data.get('keys', [])
        if not isinstance(keys, list) or len(keys) > 10 or any(not isinstance(k, str) or k not in KEYS for k in keys):
            raise ValueError('非法按键')
        focused = data.get('focused') is True
        # Visible, empty heartbeats are valid zero-speed input even without
        # keyboard focus. Re-pausing them forever mislabeled blur as a network
        # outage and prevented the paused state from clearing.
        visible_idle = action in ('resume', 'heartbeat') and data.get('visible') is True and not keys
        if action == 'resume' and data.get('visible') is False:
            return self.stop(now, '页面隐藏；等待页面可见') if self.owner == client else None
        if action == 'pause':
            return self.pause(client, now, frame_time)
        if not focused and not visible_idle:
            # A visible browser can briefly lose focus while its websocket and
            # watchdog remain healthy. Release every key and keep a zero-speed
            # pilot lease; hidden pages and lost heartbeats still hard-stop.
            if keys:
                return self.stop(now) if self.owner == client else None
            if self.owner != client:
                return None
            return self.pause(client, now, frame_time)
        if self.require_head_camera and self.real and not 0 <= now-frame_time <= self.config['camera_timeout']:
            if self.owner == client:
                self.latched = True
            raise ValueError('相机画面未就绪或延迟过高')
        if action in ('arm', 'resume'):
            if keys:
                raise ValueError('请松开运动键后接管')
            self.owner = client
            self.arm_id += 1
            self.latched = False
            self.network_paused = False
        if self.owner != client:
            return None
        # A delayed movement packet cannot restart the chassis. Recovery first
        # requires one current heartbeat with every movement key released.
        if self.network_paused:
            if keys:
                return self.pause(client, now, frame_time)
            self.network_paused = False
        if keys:
            self.hold_start = now if self.hold_start is None else self.hold_start
            # Browser keydown/repeat sequence, not the network heartbeat sequence.
            event = data.get('key_event')
            if not isinstance(event, int) or isinstance(event, bool) or event < 0:
                raise ValueError('非法键盘事件')
            if self.key_event is None or event != self.key_event[0]:
                self.key_event = (event, now)
            if now-self.hold_start > self.config['max_hold_seconds'] or now-self.key_event[1] > self.config['key_repeat_timeout']:
                return self.stop(now)
        else:
            self.hold_start = self.key_event = None
        return dict(time=now, focused=focused, stop=self.latched, keys=keys,
                    arm_id=self.arm_id, frame_time=frame_time, pause=False, stop_reason=self.stop_message)


class Cockpit:
    def __init__(self, config, real, source, chassis_source, token):
        self.config, self.real = config, real
        self.source, self.chassis_source, self.token = source, chassis_source, token
        self.gate = InputGate(config, real, require_head_camera=source != 'none')
        self.ctx = mp.get_context('spawn')
        self.intents, self.statuses, self.frames, self.chassis_frames = [
            self.ctx.Queue(n) for n in (2, 2, 1, 1)]
        self.shutdown = self.ctx.Event()
        self.status = dict(state='LOCKED', message='等待控制进程', velocity=[0, 0, 0], gear='normal', time=0)
        self.frame, self.frame_id, self.frame_times = None, 0, OrderedDict()
        self.chassis_frame, self.chassis_frame_id = None, 0
        self.camera_error = ('等待 305 彩色画面' if source != 'none' else '头部相机未由驾驶舱打开') if real else '离线演示 · 不连接硬件'
        self.chassis_camera_error = '等待 335L 底盘画面' if chassis_source != 'none' else '底盘相机未启用'
        self.clients, self.sessions = {}, set()
        self.client_states = {}

    def stop(self, reason='控制连接中断；等待自动恢复'):
        # Clear queued movement before submitting the stop. Independent worker
        # still stops within heartbeat_timeout if the event loop is unavailable.
        latest(self.intents)
        offer(self.intents, self.gate.stop(time.monotonic(), reason))

    async def startup(self, app):
        self.workers = [self.ctx.Process(target=run_control,
            args=(self.config, self.real, self.intents, self.statuses, self.shutdown))]
        if self.source != 'none':
            self.workers.append(self.ctx.Process(target=run_camera,
                args=(self.source, self.config, self.frames, self.shutdown)))
        if self.chassis_source != 'none':
            self.workers.append(self.ctx.Process(target=run_camera,
                args=(self.chassis_source, self.config, self.chassis_frames, self.shutdown)))
        for worker in self.workers:
            worker.start()
        self.pump_task = asyncio.create_task(self.pump())

    async def cleanup(self, app):
        self.stop()
        for ws in list(self.clients):
            await ws.close(code=1001, message=b'Server stopping')
        self.pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.pump_task
        self.shutdown.set()
        for worker in self.workers:
            await asyncio.to_thread(worker.join, self.config['service_timeout']+2)
            if worker.is_alive():
                worker.terminate()
                await asyncio.to_thread(worker.join, 2)

    async def pump(self):
        while True:
            self.status = latest(self.statuses, self.status)
            frame = latest(self.frames)
            if frame:
                if 'jpeg' in frame:
                    self.frame_id += 1
                    self.frame = frame
                    self.camera_error = ''
                    self.frame_times[self.frame_id] = frame['time']
                    while len(self.frame_times) > 64:
                        self.frame_times.popitem(last=False)
                else:
                    self.camera_error = frame.get('error', '相机错误')
            chassis_frame = latest(self.chassis_frames)
            if chassis_frame:
                if 'jpeg' in chassis_frame:
                    self.chassis_frame_id += 1
                    self.chassis_frame = chassis_frame
                    self.chassis_camera_error = ''
                else:
                    self.chassis_camera_error = chassis_frame.get('error', '底盘相机错误')
            await asyncio.sleep(.02)

    def authorized(self, request):
        if request.cookies.get('seldon_session') not in self.sessions:
            raise web.HTTPUnauthorized(text='请使用完整驾驶舱链接打开')

    async def session(self, request):
        if not origin_allowed(request.headers.get('Origin'), request):
            raise web.HTTPForbidden()
        data = await request.json()
        token = data.get('token', '')
        if not isinstance(token, str) or not secrets.compare_digest(token, self.token):
            raise web.HTTPUnauthorized(text='访问口令不正确')
        if len(self.sessions) >= 32:
            raise web.HTTPTooManyRequests()
        cookie = secrets.token_urlsafe(32)
        self.sessions.add(cookie)
        response = web.json_response({'ok': True})
        response.set_cookie('seldon_session', cookie, httponly=True, samesite='Strict')
        return response

    async def frame_response(self, request):
        self.authorized(request)
        if not self.frame or time.monotonic()-self.frame['time'] > self.config['camera_timeout']:
            raise web.HTTPServiceUnavailable(text=self.camera_error or '画面超时')
        return web.Response(body=self.frame['jpeg'], content_type='image/jpeg',
                            headers={'X-Frame-ID': str(self.frame_id), 'Cache-Control': 'no-store'})

    async def chassis_frame_response(self, request):
        self.authorized(request)
        if (not self.chassis_frame or
                time.monotonic()-self.chassis_frame['time'] > self.config['chassis_camera_timeout']):
            raise web.HTTPServiceUnavailable(text=self.chassis_camera_error or '底盘画面超时')
        return web.Response(body=self.chassis_frame['jpeg'], content_type='image/jpeg',
                            headers={'X-Frame-ID': str(self.chassis_frame_id), 'Cache-Control': 'no-store'})

    async def socket(self, request):
        self.authorized(request)
        if not origin_allowed(request.headers.get('Origin'), request):
            raise web.HTTPForbidden()
        if len(self.clients) >= 8:
            raise web.HTTPTooManyRequests()
        ws = web.WebSocketResponse(max_msg_size=4096, heartbeat=2)
        await ws.prepare(request)
        challenges = OrderedDict()
        challenge_limit = max(10, int(self.config['network_lock_timeout']/.05)+6)
        self.clients[ws] = challenges
        seq = -1

        async def send_status():
            while not ws.closed:
                now = time.monotonic()
                nonce = secrets.token_hex(12)
                challenges[nonce] = now
                while len(challenges) > challenge_limit:
                    challenges.popitem(last=False)
                status = dict(self.status)
                status.update(real=self.real, challenge=nonce, worker_age=now-status['time'],
                    camera_age=now-self.frame['time'] if self.frame else None,
                    camera_error=self.camera_error,
                    chassis_camera_age=now-self.chassis_frame['time'] if self.chassis_frame else None,
                    chassis_camera_error=self.chassis_camera_error, pilot=self.gate.owner == ws,
                    occupied=self.gate.owner not in (None, ws), speeds=self.config['speeds'],
                    stop_revision=self.gate.stop_revision,
                    pilot_ui=self.client_states.get(self.gate.owner))
                await asyncio.wait_for(ws.send_json(status), .3)
                await asyncio.sleep(.05)

        sender = asyncio.create_task(send_status())
        # A blocked downstream client must also close its input lease.
        def sender_done(task):
            if not task.cancelled() and task.exception():
                asyncio.create_task(ws.close())
        sender.add_done_callback(sender_done)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError('非法数据')
                    now = time.monotonic()
                    # A manual stop is always honored, even when its challenge
                    # expired in transit. It revokes every prior auto-resume.
                    if data.get('action') == 'stop':
                        intent = self.gate.accept(ws, data, now, 0)
                        latest(self.intents)
                        offer(self.intents, intent)
                        await ws.send_json({'stop_ack': self.gate.stop_revision})
                        continue
                    incoming = data.get('seq')
                    if not isinstance(incoming, int) or isinstance(incoming, bool) or incoming <= seq:
                        raise ValueError('重复或乱序输入')
                    seq = incoming
                    # No delayed/replayed heartbeat can make old motion fresh.
                    issued = challenges.get(data.get('challenge'), -1e9)
                    latency = now-issued
                    if data.get('action') == 'probe':
                        self.client_states[ws] = {
                            'visible': data.get('visible') is True,
                            'focused': data.get('focused') is True,
                            'auto_takeover': data.get('auto_takeover') is True,
                            'version': str(data.get('ui_version', 'unknown'))[:64], 'time': now}
                        await ws.send_json({'probe_ack': incoming,
                                            'probe_fresh': 0 <= latency <= self.config['heartbeat_timeout']})
                        continue
                    if not 0 <= latency <= self.config['network_lock_timeout']:
                        raise ValueError('网络连接中断；已停止')
                    frame_time = self.frame_times.get(data.get('frame_id'), 0)
                    if data.get('action') == 'hold':
                        intent = self.gate.accept(ws, data, now, frame_time)
                        if intent:
                            latest(self.intents)
                            offer(self.intents, intent)
                        continue
                    if latency > self.config['heartbeat_timeout']:
                        intent = self.gate.pause(ws, now, frame_time)
                        if intent:
                            latest(self.intents)
                            offer(self.intents, intent)
                        await asyncio.wait_for(ws.send_json(
                            {'error': '网络延迟偏高；速度已归零，稳定后自动恢复'}), .3)
                        continue
                    if data.get('action') in ('arm', 'resume'):
                        if now-self.status['time'] > .3 or self.status['state'] != 'LOCKED':
                            await ws.send_json({'error': '控制进程未就绪或正在接管', 'request_seq': incoming})
                            continue
                    intent = self.gate.accept(ws, data, now, frame_time)
                    if intent:
                        if intent['stop']:
                            latest(self.intents)
                        offer(self.intents, intent)
                        if data.get('action') in ('arm', 'resume'):
                            await ws.send_json({'arm_ack': incoming, 'arm_id': intent['arm_id']})
                except (ValueError, TypeError, KeyError) as exc:
                    if self.gate.owner == ws:
                        self.stop(str(exc))
                    await asyncio.wait_for(ws.send_json({'error': str(exc)}), .3)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
            self.clients.pop(ws, None)
            self.client_states.pop(ws, None)
            if self.gate.owner == ws:
                self.stop()
                self.gate.owner = None
        return ws


def make_app(cockpit, allowed_hosts):
    @web.middleware
    async def protection(request, handler):
        if request.host not in allowed_hosts:
            raise web.HTTPForbidden(text='Host not allowed')
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        # The standalone page remains same-origin only.  The unified cockpit
        # is a presentation shell around this unchanged control page, so its
        # known local entry origins are allowed to embed it as an iframe.
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'self' http://127.0.0.1:* http://127.0.0.2:* http://localhost:* http://10.30.3.155:* http://100.99.201.2:*; base-uri 'none'"
        return response
    app = web.Application(middlewares=[protection], client_max_size=4096)
    app.router.add_post('/session', cockpit.session)
    app.router.add_get('/frame.jpg', cockpit.frame_response)
    app.router.add_get('/chassis-frame.jpg', cockpit.chassis_frame_response)
    app.router.add_get('/ws', cockpit.socket)
    for url, file in [('/', 'index.html'), ('/app.js', 'app.js'), ('/style.css', 'style.css')]:
        async def serve(request, filename=file):
            return web.FileResponse(ROOT/'web'/filename)
        app.router.add_get(url, serve)
    app.on_startup.append(cockpit.startup)
    app.on_cleanup.append(cockpit.cleanup)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ros', action='store_true', help='Connect real chassis; still locked until E')
    parser.add_argument('--camera', choices=['orbbec', 'ros', 'none'], default='none')
    parser.add_argument('--chassis-camera', choices=['uvc', 'none'])
    parser.add_argument('--external-head-camera', action='store_true',
                        help='真机运行，但不打开头部 Orbbec；图像由本机抓书程序读取')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--allow-host', action='append', default=['127.0.0.1', '127.0.0.2', 'localhost', '10.30.3.155', '100.99.201.2'])
    args = parser.parse_args()
    if args.ros and args.camera == 'none' and not args.external_head_camera:
        parser.error('真机模式必须指定 --camera orbbec 或 ros；头部相机交给本机时加 --external-head-camera')
    (ROOT/'runtime').mkdir(exist_ok=True)
    lock = open(ROOT/'runtime'/f'web-{args.port}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = validate_config(json.loads((ROOT/'config.json').read_text()))
    chassis_source = args.chassis_camera or ('uvc' if args.ros else 'none')
    token = secrets.token_urlsafe(24)
    links = {host: f'http://{host}:{args.port}/#token={token}' for host in args.allow_host}
    fd = os.open(ROOT/'runtime'/f'web-{args.port}.json', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as info:
        json.dump({'pid': os.getpid(), 'real': args.ros,
                   'chassis_camera': chassis_source, 'links': links}, info)
    cockpit = Cockpit(config, args.ros, args.camera, chassis_source, token)
    allowed = {f'{host}:{args.port}' for host in args.allow_host}
    print(f'Seldon cockpit: http://127.0.0.1:{args.port}/ (access link in runtime/web-{args.port}.json)', flush=True)
    web.run_app(make_app(cockpit, allowed), host=args.host, port=args.port, access_log=None)


if __name__ == '__main__':
    main()
