"""Run on155: start RT using stationary measured positions as its reference."""
import argparse
import asyncio
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import aiohttp

ROOT = Path(os.environ.get('DUAL_GRIPPER_PROFILE_ROOT', Path(__file__).resolve().parent)).resolve()
# A versioned launcher can run against the unchanged active model/backend release.
sys.path.insert(0, str(ROOT))
PYTHON = Path('/home/erazer/Seldon/keyboard_cockpit/.venv/bin/python')
MODULE = 'backend.server_board'
BASE = 'http://127.0.0.1:8810'
RT_LOG = Path('/mnt/data/erazer/book_grasp/dual_arm_control_runtime/logs/console/rt.log')


def identity_matches(pid):
    try:
        process = Path('/proc') / str(int(pid))
        argv = (process/'cmdline').read_bytes().split(b'\0')
        return ((process/'cwd').resolve() == ROOT and
                any(argv[i:i+2] == [b'-m', MODULE.encode()] for i in range(len(argv)-1)))
    except (OSError, ValueError, TypeError):
        return False


def backend_info():
    try:
        info = json.loads((ROOT/'runtime/web.json').read_text())
    except (OSError, ValueError):
        return None
    if not identity_matches(info.get('pid')):
        return None
    parsed = urlsplit(info['url'])
    if (parsed.hostname != '127.0.0.1' or parsed.port != 8810 or
            parsed.scheme != 'http' or not info.get('hardware')):
        raise RuntimeError('现有双臂后端地址或硬件模式不匹配，未替换服务')
    return info


async def state(client, token):
    async with client.get(BASE+'/api/state', headers={'X-Arm-Token':token}) as response:
        response.raise_for_status()
        return await response.json()


def rt_ready(s):
    return (not s.get('simulation') and s.get('fresh') and
            isinstance(s.get('age_ms'), (int,float)) and 0 <= s['age_ms'] <= 250 and
            s.get('state', {}).get('system') in (1,2,3) and
            not (s.get('state', {}).get('diagnostics',0) & 0xffff0000))


def needed_action(s):
    if s.get('simulation'):
        raise RuntimeError('当前是仿真后端，未启动硬件')
    if rt_ready(s):
        return None  # Reuse feedback, including when somebody else is controlling.
    if s.get('active') or s.get('planner',{}).get('enabled') or s.get('preparation',{}).get('active'):
        raise RuntimeError('已有控制/准备会话且反馈未就绪，未抢占或重连')
    return 'connect' if s.get('rt_running') else 'start_rt'


async def receive_before(ws, deadline, stage):
    remaining=deadline-time.monotonic()
    try:
        if remaining<=0:raise asyncio.TimeoutError
        return await ws.receive_json(timeout=remaining)
    except asyncio.TimeoutError as error:
        raise RuntimeError(stage+'超时；未自动重试或使能') from error


async def wait_ack(ws, command, seconds, stage):
    # This private socket has exactly one pending command and no heartbeats.
    # Deployed backends omit the request ID on rejection; an ID-less error is
    # fatal, but an ID-less success must never acknowledge this command.
    deadline=time.monotonic()+seconds
    while True:
        reply=await receive_before(ws,deadline,stage)
        if reply.get('type')=='error' and reply.get('id') in (None,command['id']):
            raise RuntimeError(reply.get('message') or reply.get('error') or stage+'被拒绝')
        if reply.get('type')=='ok' and reply.get('id')==command['id']:
            return


async def prepare_feedback(client, token):
    async with client.ws_connect(BASE+'/ws', heartbeat=2) as ws:
        await ws.send_json({'token':token})
        deadline=time.monotonic()+5
        while True:
            msg=await receive_before(ws,deadline,'等待双臂后台初始状态')
            if msg.get('type')=='state':break
            if msg.get('type')=='error':raise RuntimeError(msg.get('message','双臂身份验证失败'))
        action=needed_action(msg)
        if action is None:return msg
        command=dict(action=action,id=uuid.uuid4().hex,epoch=msg['epoch'],challenge=msg['challenge'])
        if action=='start_rt':
            command['reference']='accept-current'
            print('启动双臂RT：以当前有限、停稳的关节读数更新启动参考，不因旧位置记录不同退出；保留硬件/限位保护。',flush=True)
        else:
            command.update(ip='127.0.0.1',port=12201)
            print('复用已运行RT，连接本机反馈。',flush=True)
        await ws.send_json(command)
        await wait_ack(ws,command,10,'等待RT启动应答' if action=='start_rt' else '等待RT连接应答')
    # We never acquire an owner, so closing this socket leaves RT and other sessions intact.
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        s=await state(client,token)
        if rt_ready(s):return s
        if not s.get('rt_running') and s.get('rt_exit') is not None:
            raise RuntimeError('RT已退出；查看 '+str(RT_LOG))
        await asyncio.sleep(.2)
    raise RuntimeError('未取得新鲜RT反馈；查看 '+str(RT_LOG)+'；未自动重试或使能')


PC_PLANNER = dict(ip='10.30.3.100',port=30241,local_ip='0.0.0.0',local_port=30240)


def planner_settings(host):
    address=ipaddress.IPv4Address(host)
    if address.is_unspecified or address.is_loopback or address.is_multicast or str(address)=='255.255.255.255':
        raise ValueError('Planner必须使用本机可达LAN地址，例如10.30.3.100')
    return dict(PC_PLANNER,ip=str(address))


def planner_action(s, settings=None):
    settings=PC_PLANNER if settings is None else settings
    planner=s.get('planner',{})
    if not rt_ready(s):raise RuntimeError('RT反馈未就绪，未打开Planner接口')
    if planner.get('connected') and all(planner.get(k)==v for k,v in settings.items()):
        return None
    if s.get('active') or planner.get('enabled') or s.get('preparation',{}).get('active'):
        raise RuntimeError('现有控制会话正在使用Planner地址，未改动或接管')
    return dict(action='planner_config',**settings)


async def open_planner(client,token,settings):
    async with client.ws_connect(BASE+'/ws',heartbeat=2) as ws:
        await ws.send_json({'token':token});deadline=time.monotonic()+5
        while True:
            s=await receive_before(ws,deadline,'等待Planner接口初始状态')
            if s.get('type')=='state':break
            if s.get('type')=='error':raise RuntimeError('Planner接口身份验证失败')
        command=planner_action(s,settings)
        if command is not None:
            command.update(id=uuid.uuid4().hex,epoch=s['epoch'],challenge=s['challenge'])
            await ws.send_json(command)
            await wait_ack(ws,command,5,'等待Planner接口配置应答')
        result=await state(client,token)
        if not (result.get('planner',{}).get('connected') and
                all(result['planner'].get(k)==v for k,v in settings.items()) and rt_ready(result)):
            raise RuntimeError('Planner接口或新鲜RT反馈尚未就绪')
        return result


async def status_json():
    """Read-only, credential-free status for the PC launcher."""
    info=backend_info()
    if info is None:raise RuntimeError('双臂后台未运行')
    token=parse_qs(urlsplit(info['url']).fragment)['token'][0]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4),trust_env=False) as client:
        s=await state(client,token)
    p=s.get('planner',{})
    return dict(rt_ready=bool(rt_ready(s)),age_ms=s.get('age_ms'),active=bool(s.get('active')),
                preparation_active=bool(s.get('preparation',{}).get('active')),
                planner={k:p.get(k) for k in (*PC_PLANNER,'connected','enabled','sent','received')})


async def run(check=False,planner_host=None):
    settings=planner_settings(planner_host) if planner_host else None
    required=[PYTHON,ROOT/(MODULE.replace('.','/')+'.py'),ROOT/'runtime/build/seldon_rt_control']
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError('缺少双臂运行文件：'+', '.join(missing))
    from backend.physical_profile import verify as verify_profile
    verify_profile(ROOT)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().split(b'\0'); cwd=path.parent.joinpath('cwd').resolve()
        except (OSError, PermissionError):continue
        if cwd != ROOT and any(x in args for x in (b'backend.server_board',b'book_grasp_arm.server_board_visual',b'seldon_arm.server_board_visual',b'seldon_arm.server_v31')):
            raise RuntimeError('其他双臂后台仍运行；请正常停止后再启动独立双夹爪配置')
    if check:
        print('155双臂启动文件检查通过；未启动服务或RT。')
        return
    from backend.rt_logs import detached_output, log_path
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'start_arms.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4),trust_env=False) as client:
            info=backend_info()
            if info is None:
                try:
                    async with client.get(BASE+'/'):
                        raise RuntimeError('8810端口已有其他服务，未重复启动')
                except aiohttp.ClientConnectorError:
                    pass
                print('启动155双臂后台。',flush=True)
                with detached_output(log_path('backend.log')) as log:
                    process=subprocess.Popen([str(PYTHON),'-m',MODULE,'--hardware','--no-browser',
                        '--host','127.0.0.1','--port','8810'],cwd=ROOT,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+35
                while time.monotonic()<deadline:
                    if process.poll() is not None:raise RuntimeError('双臂后台退出；查看 '+str(log_path('backend.log')))
                    info=backend_info()
                    if info:
                        token=parse_qs(urlsplit(info['url']).fragment)['token'][0]
                        try:
                            await state(client,token)
                            break
                        except aiohttp.ClientConnectorError:pass
                    await asyncio.sleep(.2)
                else:raise RuntimeError('双臂后台等待超时；未重复启动')
            token=parse_qs(urlsplit(info['url']).fragment)['token'][0]
            s=await prepare_feedback(client,token)
            if settings:
                s=await open_planner(client,token,settings)
                print(f"155反馈接口已配置：10.30.3.155:30240 → {settings['ip']}:30241。")
                print('CuRobo在PC运行，由PC启动器继续核验进程、模型契约和持续反馈。')
            print(f"双臂反馈已就绪：age={s['age_ms']}ms，active={s.get('active')}，Planner={s.get('planner',{}).get('enabled',False)}。")
            print('左右夹爪已接入：共享串口，左ID=1、右ID=2；未发送开合或机械臂目标。')
            print('本脚本未接管运动；RT保持运行，退出终端不停止RT。')
            print('155本机双臂页面：'+info['url'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='仅检查运行文件，不启动硬件')
    parser.add_argument('--planner-host',default='10.30.3.100',help='本机CuRobo的LAN地址')
    parser.add_argument('--rt-only',action='store_true',help='只启动RT，不修改Planner地址')
    parser.add_argument('--status-json',action='store_true',help='只读输出不含令牌的实时状态')
    args=parser.parse_args()
    try:
        if args.status_json:print(json.dumps(asyncio.run(status_json()),ensure_ascii=False,allow_nan=False))
        else:asyncio.run(run(args.check,None if args.rt_only else args.planner_host))
    except (Exception,KeyboardInterrupt) as error:
        print('双臂启动未完成：'+(str(error) or type(error).__name__));raise SystemExit(1)
