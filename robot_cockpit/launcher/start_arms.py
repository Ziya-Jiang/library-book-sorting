"""Start155 strict arm RT plus the existing PC CuRobo; never acquire motion control."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

import grasp_cleanup

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'runtime/start_arms'
PLANNER_ROOT=ROOT/'dual_v2_2_control/cuarm_upper_planner'
PLANNER=PLANNER_ROOT/'remote_planner.py'
BOARD=PLANNER_ROOT/'dual_v2_2_upper_planner_board.py'
CUROBO=Path('/mnt/ssd/robotarm/env/curobo/bin/python')
HOST='erazer@10.30.3.155'
ROBOT_IP='10.30.3.155'
PC_IP='10.30.3.100'
REMOTE='/home/erazer/start_arms.sh'
EXPECTED=dict(ip=PC_IP,port=30241,local_ip='0.0.0.0',local_port=30240)


def precheck():
    for path in (CUROBO,PLANNER,BOARD):
        if not path.is_file():raise RuntimeError('缺少文件：'+str(path))
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
        try:probe.bind((PC_IP,0))
        except OSError as error:raise RuntimeError(f'本机尚未配置{PC_IP}；未启动机器人') from error
    route=json.loads(subprocess.check_output(['ip','-j','route','get',ROBOT_IP],text=True))
    if not route or route[0].get('prefsrc')!=PC_IP:
        raise RuntimeError(f'到155的路由源地址不是{PC_IP}；请检查有线网卡与路由')


def ssh_master():
    # Reuse live authenticated connections; never delete another launcher's socket.
    candidates=[HERE/'runtime/ssh.sock',RUNTIME/'ssh.sock',Path('/tmp/arms-start-155.sock')]
    candidates.extend(sorted(RUNTIME.glob('ssh-*.sock'),reverse=True))
    for control in candidates:
        if control.exists() and subprocess.run(['ssh','-S',str(control),'-O','check',HOST],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5).returncode==0:
            return control
    control=RUNTIME/f'ssh-{os.getpid()}.sock'
    print('连接155，必要时输入SSH密码。',flush=True)
    subprocess.run(['ssh','-M','-S',str(control),'-fNT','-o','ConnectTimeout=8',
        '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ControlPersist=600',
        HOST],check=True,timeout=120)
    return control


def remote(control,*args,capture=False):
    command=shlex.join(['bash',REMOTE,*args])
    return subprocess.run(['ssh','-T','-S',str(control),'-o','BatchMode=yes','-o','ConnectTimeout=8',
        HOST,command],check=True,text=True,stdout=subprocess.PIPE if capture else None,timeout=100)


def remote_status(control):
    value=json.loads(remote(control,'--status-json',capture=True).stdout)
    if not value.get('rt_ready'):raise RuntimeError('155实时反馈未就绪；未使能Planner')
    p=value.get('planner',{})
    if not p.get('connected') or any(p.get(k)!=v for k,v in EXPECTED.items()):
        raise RuntimeError('155反馈目的地址已变化；应为10.30.3.100:30241')
    return value


def matching_planner():
    entries={PLANNER,BOARD,PLANNER_ROOT/'dual_v2_2_upper_planner_v3_1.py'}
    found=[p for p in grasp_cleanup.process_list() if grasp_cleanup.python_entry(p,entries)]
    if len(found)>1:raise RuntimeError('存在多个Planner进程；未启动或终止任何旧进程')
    if not found:return None
    p=found[0]
    argv=p['argv'][1:]
    while argv and argv[0] in ('-u','-B'):argv=argv[1:]
    entry=(Path(p['cwd'])/argv[0]).resolve()
    if entry==PLANNER and len(argv)==1:return p
    if entry!=BOARD:raise RuntimeError('现有Planner入口不匹配，未替换：PID='+str(p['pid']))
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False,exit_on_error=False)
    parser.add_argument('--robot-host');parser.add_argument('--planner-bind',default='0.0.0.0')
    try:args,extra=parser.parse_known_args(argv[1:])
    except argparse.ArgumentError as error:raise RuntimeError('现有Planner参数无法核验') from error
    if extra or args.robot_host!=ROBOT_IP or args.planner_bind!=PC_IP:
        raise RuntimeError('现有Planner地址不匹配，未替换：PID='+str(p['pid']))
    return p


def cleanup_orphan_publishers():
    scripts = str(PLANNER_ROOT/'scripts')
    if scripts not in sys.path: sys.path.insert(0, scripts)
    from planner_lifecycle import reap_orphan_publishers
    reap_orphan_publishers()


def require_book_planner(identity):
    """Do not reuse an already-running Planner from a different robot profile."""
    if identity is None:return
    try:
        raw=(Path('/proc')/str(identity['pid'])/'environ').read_bytes()
        env=dict(item.split(b'=',1) for item in raw.split(b'\0') if b'=' in item)
        selected=Path(env.get(b'CUARM_CONFIG_ROOT',str(ROOT/'dual_v2_2_configuration').encode()).decode())
    except (OSError,ValueError,UnicodeError) as exc:
        raise RuntimeError('无法核验现有Planner模型配置；未复用或终止它') from exc
    if selected.resolve()!=(ROOT/'dual_v2_2_configuration').resolve():
        raise RuntimeError('现有Planner属于其他模型配置；抓书入口未复用或终止它：'+str(selected))


def available_ports():
    for address in [('127.0.0.1',10176),(PC_IP,30241)]:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
            try:probe.bind(address)
            except OSError as error:raise RuntimeError(f'UDP {address} 已被占用，未启动第二份Planner') from error


def planner_probe():
    """Reuse the existing code/model/process/worker identity checks, no UDP target."""
    agent=str(ROOT/'dual_v2_2_agent')
    if agent not in sys.path:sys.path.insert(0,agent)
    module=importlib.import_module('cuart_book_grasp_v3_4_remote')
    old=module.PLANNER_ENTRY
    module.PLANNER_ENTRY=BOARD
    try:module.require_remote_planner(ROBOT_IP)
    finally:module.PLANNER_ENTRY=old
    readiness=importlib.import_module('v3_1_readiness')
    receipt=json.loads(readiness.receipt_path(10176).read_text())
    report=json.loads(module.REMOTE_LINK_REPORT.read_text())
    # Validate the exact report used below, including its child/parent identity.
    from v3_1_panel_process import verify_publisher
    verify_publisher(report,receipt['pid'])
    if (report['error'] is not None or report['remote']!=[ROBOT_IP,30240] or
            report['adapter_sha256']!=hashlib.sha256(module.REMOTE_ADAPTER.read_bytes()).hexdigest()):
        raise RuntimeError('Planner反馈版本、地址或状态不匹配')
    if not 0 <= time.monotonic()-report['updated_monotonic'] <= 5:
        raise RuntimeError('Planner链路诊断已过期')
    return dict(pid=receipt['pid'],start=receipt['start'],worker=report['publisher_pid'],
                updated=report['updated_monotonic'],received=report['received'])


def progress(previous,current):
    return (previous is not None and all(previous[k]==current[k] for k in ('pid','start','worker'))
            and current['updated']>previous['updated'] and current['received']>previous['received'])


def wait_planner(process,identity,probe=planner_probe,timeout=100.,sleep=time.sleep,clock=time.monotonic):
    scripts = str(PLANNER_ROOT/'scripts')
    if scripts not in sys.path:sys.path.insert(0,scripts)
    from v3_1_readiness import PlannerRestartRequired
    deadline=clock()+timeout;previous=None;last_reason=None
    while clock()<deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError('CuRobo已退出；查看 '+str(RUNTIME/'planner.log'))
        if identity and not grasp_cleanup.same(identity):raise RuntimeError('原Planner进程已退出或被替换')
        try:
            current=probe()
            if identity and (current['pid']!=identity['pid'] or current['start']!=identity['start']):
                raise RuntimeError('就绪记录不属于本次核验的Planner')
            if progress(previous,current):return current
            previous=current
            reason='CuRobo已初始化，等待155反馈计数继续增长'
        except PlannerRestartRequired:
            raise  # This live process cannot load changed code by waiting.
        except (ValueError,OSError,RuntimeError) as error:
            previous=None;reason=str(error)
        if reason!=last_reason:print('等待Planner：'+reason,flush=True);last_reason=reason
        sleep(.4)
    raise RuntimeError('Planner就绪等待超时：'+str(last_reason)+'；日志：'+str(RUNTIME/'planner.log'))


def stop_own(process,identity):
    if process is None or process.poll() is not None:return
    if identity is None or not grasp_cleanup.same(identity):return
    grasp_cleanup.signal_matching(identity,signal.SIGINT)
    try:process.wait(timeout=6)
    except subprocess.TimeoutExpired:print(f'本次Planner PID={process.pid} 尚未退出；未强杀。',flush=True)


def run(*,check=False,rt_only=False):
    os.environ['CUARM_CONFIG_ROOT']=str(ROOT/'dual_v2_2_configuration')
    if not rt_only:precheck()
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'startup.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('另一个start_arms正在启动，请等待其结果')
        existing=None if rt_only else matching_planner()
        require_book_planner(existing)
        if not rt_only and existing is None:
            if not check: cleanup_orphan_publishers()
            available_ports()
        control=ssh_master()
        if check:
            remote(control,'--check','--rt-only' if rt_only else '--planner-host',*(() if rt_only else (PC_IP,)))
            if not rt_only:
                subprocess.run([str(CUROBO),'-c',
                    'import torch, curobo; assert torch.cuda.is_available(), "NVIDIA CUDA unavailable"; print("本机CuRobo/CUDA导入检查通过")'],
                    check=True,timeout=45)
            print('启动依赖检查通过；未启动RT或Planner。');return
        remote(control,*(['--rt-only'] if rt_only else ['--planner-host',PC_IP]))
        if rt_only:return
        state=remote_status(control)
        process=None;identity=existing
        try:
            if existing is None:
                if state.get('active') or state.get('preparation_active') or state['planner'].get('enabled'):
                    raise RuntimeError('双臂已有活动控制，未创建新的Planner')
                available_ports()
                env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
                env['PATH']=str(CUROBO.parent)+os.pathsep+env.get('PATH','')
                env['BOOK_PACKET_TRACE_DIR']=str(RUNTIME/'packet_trace')
                for name,value in dict(CUARM_V31_TARGET_LOCAL_IP='127.0.0.1',CUARM_V31_TARGET_REMOTE_IP='127.0.0.1',
                                       CUARM_V31_TARGET_LOCAL_PORT='10176',CUARM_V31_TARGET_REMOTE_PORT='10175').items():env[name]=value
                with (RUNTIME/'planner.log').open('ab',buffering=0) as log:
                    log.write(f'\nstart_arms startup unix_s={time.time():.6f}\n'.encode())
                    process=subprocess.Popen([str(CUROBO),str(PLANNER)],cwd=PLANNER_ROOT,env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=grasp_cleanup.snapshot(process.pid)
                print(f'已启动本机CuRobo PID={process.pid}；日志：{RUNTIME/"planner.log"}',flush=True)
            else:print(f'核验已有Planner PID={existing["pid"]}，不启动第二份。',flush=True)
            ready=wait_planner(process,identity)
            state=remote_status(control)
            record=dict(unix_s=time.time(),planner=ready,robot=state,motion_control_acquired=False)
            (RUNTIME/'last_start.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            print(f'就绪：155双臂RT反馈{state["age_ms"]}ms；本机CuRobo PID={ready["pid"]}；模型/代码契约通过，155→PC反馈持续到达。')
            print('反馈：155:30240 → 10.30.3.100:30241；任务接口：127.0.0.1:10175/10176。')
            print('未接管运动、未启动抓取；退出本命令后RT和Planner继续运行。')
        except BaseException:
            stop_own(process,identity)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='只检查文件、网络地址和CuRobo依赖，不启动硬件')
    parser.add_argument('--rt-only',action='store_true',help='只启动155 RT，不启动本机CuRobo或修改Planner地址')
    args=parser.parse_args()
    try:run(check=args.check,rt_only=args.rt_only)
    except (Exception,KeyboardInterrupt) as error:
        print('双臂/Planner启动未完成：'+(str(error) or '用户中断'),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
