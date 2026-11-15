"""Identity-checked cleanup of released local grasp sessions, never robot RT."""
import asyncio
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import platform
import re
import time


def snapshot(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z' or path.stat().st_uid!=os.getuid():return None
        return dict(pid=int(pid),start=fields[19],ppid=int(fields[1]),
                    argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v],
                    cwd=str((path/'cwd').resolve()))
    except (OSError,ValueError,IndexError):return None


def identity(item):
    return dict(pid=item['pid'],start=item['start'])


def same(item):
    current=snapshot(item['pid'])
    return current is not None and current['start']==item['start']


def python_entry(item, paths):
    argv=item['argv']
    if len(argv)<2 or not Path(argv[0]).name.startswith('python'):return False
    # Only the script argument, not arbitrary strings in Python -c or worker data.
    args=argv[1:]
    while args and args[0] in ('-u','-B'):args=args[1:]
    if not args or args[0].startswith('-'):return False
    entry = (Path(item['cwd'])/args[0]).resolve()
    if entry in paths:
        return True
    # A task mode may exec its independently retained PC release. Recognize
    # only the fixed release tree and only its task entry, never a basename
    # match in arbitrary experiment/diagnostics directories.
    for standard in paths:
        if standard.name != 'book_tasks_entry.py':
            continue
        releases = (standard.parent/'diagnostics/book_task_pc_releases').resolve()
        try:
            relative = entry.relative_to(releases)
        except ValueError:
            continue
        if (len(relative.parts) == 3 and re.fullmatch('[0-9a-f]{64}', relative.parts[0])
                and relative.parts[1:] == ('dual_v2_2_agent', 'book_tasks_entry.py')):
            return True
    return False


def orphan(item):
    if item['ppid']==1:return True
    parent=snapshot(item['ppid'])
    return bool(parent and parent['argv'] and Path(parent['argv'][0]).name=='systemd' and '--user' in parent['argv'])


def descendants(pid, processes):
    result=[];pending=[pid]
    while pending:
        parent=pending.pop()
        for item in processes:
            if item['ppid']==parent and item['pid'] not in {v['pid'] for v in result}:
                result.append(item);pending.append(item['pid'])
    return result


def process_list():
    return [v for p in Path('/proc').glob('[0-9]*') if (v:=snapshot(int(p.name)))]


def program_orphans(root):
    """Program-only restart: ignore full-launch registry and unrelated services."""
    paths={root/'dual_v2_2_agent'/name for name in (
        'multibook_grasp.py','cuart_book_grasp_v3_4_board_visual.py',
        'cuart_book_grasp_v3_4_board_numbered.py','baseline_no_agent.py','baseline_offset.py','book_tasks_entry.py')}
    processes=process_list()
    agents=[p for p in processes if python_entry(p,paths)]
    live=[p for p in agents if not orphan(p)]
    if live:
        raise RuntimeError(f"已有抓取程序 PID={live[0]['pid']}，父进程仍在；未清理或启动第二份")
    targets={p['pid']:p for p in agents}
    for p in agents:
        targets.update({child['pid']:child for child in descendants(p['pid'],processes)})
    return list(targets.values())


def inventory(root, registry, *, released_clients=False):
    paths={root/'dual_v2_2_agent'/name for name in (
        'multibook_grasp.py','cuart_book_grasp_v3_4_board_visual.py','cuart_book_grasp_v3_4_board_numbered.py',
        'baseline_no_agent.py','baseline_offset.py','book_tasks_entry.py')}
    processes=process_list()
    agents=[v for v in processes if python_entry(v,paths)]
    live=[v for v in agents if not orphan(v)]
    if live and not released_clients:
        raise RuntimeError(f"抓取程序仍有活动父进程 PID={live[0]['pid']}，请先在原终端正常退出")
    targets=list(agents)
    for item in agents:targets.extend(descendants(item['pid'],processes))
    # Durable identities cover workers orphaned before the next startup.
    registered=[]
    if registry.exists():
        data=json.loads(registry.read_text())
        for saved in data.get('children',[]):
            current=snapshot(saved['pid'])
            if current and current['start']==saved['start']:
                registered.append(current)
        covered={v['pid'] for v in targets+registered}
        for current in registered:
            if not orphan(current) and current['ppid'] not in covered:
                # Descendants of a discovered orphan are already covered.
                if current['pid'] not in {v['pid'] for v in targets}:
                    raise RuntimeError('上次会话进程仍有活动父进程；不自动接管')
            targets.append(current)
            targets.extend(descendants(current['pid'],processes))
    unique={v['pid']:v for v in targets}
    return list(unique.values())


def startup_lock_owner(lock, root):
    """Resolve the actual flock holder, not a name search or a stale PID file."""
    stat=os.fstat(lock.fileno())
    for line in Path('/proc/locks').read_text().splitlines():
        fields=line.split()
        if len(fields)<8 or fields[1:4]!=['FLOCK','ADVISORY','WRITE']:
            continue
        major,minor,inode=fields[5].split(':')
        if (int(major,16),int(minor,16),int(inode))!=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
            continue
        item=snapshot(int(fields[4]))
        if (item and item['pid']!=os.getpid()
                and python_entry(item,{root/'robot_cockpit/launcher/book_grasp.py',
                                      root/'robot_cockpit/launcher/baseline_offset.py'})):
            return item
        raise RuntimeError('启动锁由无法确认归属的进程占用；未发送停止')
    return None


def acquire_startup_lock(lock, root, *, timeout=35.):
    """Ask the previous launcher to finish its own stop/finally before reuse."""
    def acquire():
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    if acquire():return
    owner=startup_lock_owner(lock,root)
    if owner is None:
        if acquire():return
        raise RuntimeError('启动锁尚未释放，无法确认旧启动器；未启动第二份')
    print(f"检测到旧启动器 PID={owner['pid']}：请求正常退出，等待其停止控制并释放启动锁。",flush=True)
    try:signal_matching(owner,signal.SIGINT)
    except ProcessLookupError:pass
    deadline=time.monotonic()+timeout
    while True:
        if acquire():return
        if time.monotonic()>=deadline:
            raise RuntimeError(f'旧启动器未在{timeout:g}秒内正常退出；未强杀、未启动第二份')
        # Never send another stop to a different launcher racing for the lock.
        current=startup_lock_owner(lock,root)
        if current is not None and identity(current)!=identity(owner):
            raise RuntimeError('启动锁已由另一个新会话持有；本次不继续接管')
        time.sleep(.1)


def remember(registry, processes):
    values={}
    if registry.exists():
        for item in json.loads(registry.read_text()).get('children',[]):
            if same(item):values[item['pid']]=item
    tree=process_list()
    for process in processes:
        if process is not None and (item:=snapshot(process.pid)):
            values[item['pid']]=identity(item)
            for child in descendants(item['pid'],tree):
                values[child['pid']]=identity(child)
    if not values and not registry.exists():return
    tmp=registry.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(children=list(values.values())),indent=2))
    tmp.replace(registry)


def pidfd_call(number, *args):
    # Linux x86_64/aarch64 syscall ABI; older conda Python/glibc lacks wrappers.
    # Do not fall back to kill(pid), which reintroduces PID reuse races.
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','aarch64'):
        raise RuntimeError('当前平台不支持安全进程清理，停止启动')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(number),*args)
    if result<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return result


def open_pidfd(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    return pidfd_call(434,ctypes.c_int(pid),ctypes.c_uint(0))


def send_pidfd(fd,sig):
    if hasattr(signal,'pidfd_send_signal'):return signal.pidfd_send_signal(fd,sig)
    return pidfd_call(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))


def signal_matching(item, sig):
    if not same(item):return
    # Pin the process while signalling; PID reuse cannot hit a replacement.
    fd=open_pidfd(item['pid'])
    try:
        if same(item):send_pidfd(fd,sig)
    except ProcessLookupError:pass
    finally:os.close(fd)


async def cleanup(targets, guard, registry, *, grace=8., audit=None):
    record=dict(unix_s=time.time(),targets=[identity(v) for v in targets],events=[],status='pending')
    try:
        guard()
        # Persist descendants before stopping their parents.
        tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(dict(children=record['targets']),indent=2));tmp.replace(registry)
        for sig in (signal.SIGINT,signal.SIGTERM):
            remaining=[v for v in targets if same(v)]
            if not remaining:break
            guard()
            for item in remaining:
                guard()
                try:signal_matching(item,sig)
                except ProcessLookupError:continue
                record['events'].append(dict(pid=item['pid'],signal=sig.name))
                print(f"清理上次抓取残留 PID={item['pid']}：{sig.name}",flush=True)
            deadline=time.monotonic()+grace
            while any(same(v) for v in remaining) and time.monotonic()<deadline:
                guard();await asyncio.sleep(.1)
        remaining=[v for v in targets if same(v)]
        if remaining:raise RuntimeError('残留进程未退出，停止启动；未使用SIGKILL')
        guard();registry.write_text(json.dumps(dict(children=[])))
        record['status']='cleared'
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if audit:
            with audit.open('a') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')
