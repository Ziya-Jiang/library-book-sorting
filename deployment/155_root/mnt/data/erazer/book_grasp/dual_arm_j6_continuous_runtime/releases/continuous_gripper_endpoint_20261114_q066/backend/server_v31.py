import argparse
import asyncio
import fcntl
import json
import ipaddress
import os
import signal
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from urllib.parse import urlsplit
import webbrowser
from aiohttp import web
import yaml
from .controller_v31 import Controller
from .model import Model, JOINTS
from .calibration import JointCalibration
from .runtime_config import prepare_runtime_config
from .stop_diagnostics import rejection
from .heartbeat_policy import action_response

ROOT = Path(__file__).resolve().parents[1]
UNIFIED_PARENT_PORTS = {8798, 42377, 8790, 8795}


def startup_urls(host, port, token):
    """Display usable addresses, never the wildcard bind address."""
    hosts = [host]
    if host in ('0.0.0.0', '::'):
        hosts = ['127.0.0.1' if host == '0.0.0.0' else '::1']
        try:
            addresses = subprocess.check_output(
                ['hostname', '-I'], text=True, timeout=1).split()
            for address in addresses:
                try:
                    ip = ipaddress.ip_address(address)
                except ValueError:
                    continue
                if (not ip.is_loopback and not ip.is_unspecified
                        and not ip.is_link_local
                        and (host == '::' or ip.version == 4)):
                    hosts.append(str(ip))
        except (OSError, subprocess.SubprocessError):
            pass  # Local URL remains available if interface discovery fails.
    return [f'http://{("[" + h + "]") if ":" in h else h}:{port}/#token={token}'
            for h in dict.fromkeys(hosts)]


def origin_allowed(origin, request):
    if not origin:
        return True
    if origin == f'{request.scheme}://{request.host}':
        return True
    parsed = urlsplit(origin)
    request_host = request.host.rsplit(':', 1)[0].strip('[]')
    return parsed.hostname == request_host and parsed.port in UNIFIED_PARENT_PORTS

async def main():
    p = argparse.ArgumentParser(description='Seldon V3.1 dual-arm cockpit (RAW + leased planner)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8780)
    p.add_argument('--panel-port', type=int, default=12101)
    p.add_argument('--rt-port', type=int, default=12201)
    p.add_argument('--hardware', action='store_true', help='Allow explicitly starting real RT hardware from the page')
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--cartesian', action='store_true', default=True, help='Enable conservative V2 Cartesian jogging (no collision planner)')
    p.add_argument('--legacy', default='/home/erazer/cuarm_panel_control')
    p.add_argument('--mesh-root', type=Path)
    p.add_argument('--urdf', type=Path)
    args = p.parse_args()
    runtime = ROOT/'runtime'
    runtime.mkdir(exist_ok=True)
    runtime.chmod(0o700)
    lock = (runtime/'server.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('双臂上位机已运行；请使用 runtime/web.json 的链接')
    legacy = Path(args.legacy)
    base = legacy/'cuarm_configuration/dual_v2_2'
    calibration = JointCalibration()
    profile_path = ROOT/'v2_profile.json'
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
    if profile:
        calibration = JointCalibration.from_configs(profile['legacy_hardware'], profile['v2_hardware'])
        if args.urdf is None and args.mesh_root is None:
            args.urdf, args.mesh_root = Path(profile['urdf']), Path(profile['mesh_root'])
    synced = ROOT/'assets/cuarm_mesh/dual_v2_2'
    mesh_root = args.mesh_root or (synced if synced.exists() else base/'share')
    urdf = args.urdf
    if urdf is None and mesh_root == synced:
        candidates = sorted(synced.rglob('*.urdf'))
        candidates = [u for u in candidates if 'no_gripper' in str(u)] or candidates
        candidates = [u for u in candidates if u.name == 'dual_arm.urdf'] or candidates
        if len(candidates) != 1:
            raise SystemExit('同步模型包含多个 URDF，请用 --urdf 明确选择 no_gripper 双臂模型')
        urdf = candidates[0]
    model = Model(urdf or base/'share/no_gripper/dual_arm.urdf', mesh_root, urdf is not None)
    if not args.hardware:
        calibration = JointCalibration()
    limits_by_name = {j['name']: j['limits'] for j in model.manifest['joints']}
    controller = Controller(ROOT, not args.hardware, args.panel_port, args.rt_port,
        [[limits_by_name[n] for n in arm] for arm in JOINTS], calibration)
    if args.cartesian:
        from .kinematics import Kinematics
        controller.kinematics = Kinematics(model.manifest)
    cfg = runtime/'config'
    cfg.mkdir(exist_ok=True)
    prepare_runtime_config(ROOT, base, cfg, args.panel_port, args.rt_port, not args.hardware)
    encoder = runtime/'encoder_last_position.bin'
    original = base/'rt_control/encoder_last_position.bin'
    if not encoder.exists() and original.is_file():
        shutil.copy2(original, encoder)
    token = secrets.token_urlsafe(32)
    url = f'http://127.0.0.1:{args.port}/#token={token}'
    info = runtime/'web.json'
    info.write_text(json.dumps({'url':url, 'port':args.port, 'hardware':args.hardware, 'pid':os.getpid()}, indent=2))
    info.chmod(0o600)
    sockets = set()

    @web.middleware
    async def auth(request, handler):
        if request.path.startswith('/api/'):
            if not secrets.compare_digest(request.headers.get('X-Arm-Token',''), token):
                raise web.HTTPUnauthorized()
        response = await handler(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    async def index(request):
        return web.FileResponse(ROOT/'web/index.html')
    async def manifest(request):
        published = ROOT/'web/display-model.json'
        if published.is_file():
            return web.FileResponse(published)
        return web.json_response(model.manifest)
    async def mesh(request):
        path = model.files.get(request.match_info['key'])
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path)
    async def status(request):
        return web.json_response(controller.snapshot())
    async def websocket(request):
        origin = request.headers.get('Origin')
        if not origin_allowed(origin, request):
            raise web.HTTPForbidden()
        ws = web.WebSocketResponse(max_msg_size=8192, heartbeat=2)
        await ws.prepare(request)
        try:
            first = await asyncio.wait_for(ws.receive_json(), 5)
            if not secrets.compare_digest(str(first.get('token','')), token):
                await ws.close(code=1008)
                return ws
        except Exception:
            await ws.close(code=1008)
            return ws
        owner = secrets.token_hex(12)
        sockets.add(ws)
        async def publish():
            while not ws.closed:
                await ws.send_json({'type':'state', **controller.snapshot(owner)})
                await asyncio.sleep(.05)
        sender = asyncio.create_task(publish())
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = None
                    received_monotonic=time.monotonic()
                    try:
                        data = json.loads(msg.data)
                        if not isinstance(data, dict):
                            raise ValueError('Invalid message')
                        result=controller.action(owner,data,received_monotonic=received_monotonic)
                        response=action_response(data,result)
                        if response is not None:
                            await ws.send_json(response)
                    except (ValueError, TypeError, KeyError, OSError) as exc:
                        with controller.lock:
                            reply = rejection(controller, data, exc)
                        await ws.send_json(reply)
        finally:
            sender.cancel()
            sockets.discard(ws)
            with controller.lock:
                if controller.owner == owner:
                    controller.stop('控制页面已断开')
        return ws

    app = web.Application(middlewares=[auth], client_max_size=8192)
    app.router.add_get('/', index)
    app.router.add_get('/model.json', manifest)
    app.router.add_get('/mesh/{key}', mesh)
    app.router.add_get('/api/state', status)
    app.router.add_get('/ws', websocket)
    app.router.add_static('/static/', ROOT/'web', show_index=False)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    print('Seldon 双臂上位机已启动，完整访问网址：', flush=True)
    for access_url in startup_urls(args.host, args.port, token):
        print(access_url, flush=True)
    print(f'链接记录：{info}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, shutdown.set)
    try:
        await shutdown.wait()
    finally:
        controller.close()
        for ws in list(sockets):
            await ws.close()
        await runner.cleanup()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
