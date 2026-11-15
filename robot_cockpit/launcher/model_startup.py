"""Parent end of the per-process initialization barrier (no robot commands)."""
import asyncio
import json
import socket
import time

ENV_FD = 'BOOK_GRASP_STARTUP_FD'
MODEL_STARTUP_TIMEOUT = 300.


class ModelStartup:
    def __init__(self):
        self.parent,self.child = socket.socketpair()
        self.parent.setblocking(False)
        self.ready = False
        self.released = False

    def spawn_options(self, environment):
        env = environment.copy()
        env[ENV_FD] = str(self.child.fileno())
        return dict(env=env,pass_fds=(self.child.fileno(),))

    def spawned(self):
        self.child.close()

    async def wait_ready(self, process, guard, timeout=MODEL_STARTUP_TIMEOUT):
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        next_report = started+10.
        payload = bytearray()
        while time.monotonic()-started<timeout:
            guard()
            if process.returncode is not None:
                raise RuntimeError(f'模型初始化期间抓取程序退出：{process.returncode}；未使能Planner')
            try:
                chunk = await asyncio.wait_for(loop.sock_recv(self.parent,512),.1)
            except asyncio.TimeoutError:
                if time.monotonic()>=next_report:
                    print(f'模型初始化中（{time.monotonic()-started:.0f}s）；Planner控制尚未使能。',flush=True)
                    next_report = time.monotonic()+10.
                continue
            if not chunk:
                raise RuntimeError('模型初始化通道已关闭；未使能Planner')
            payload.extend(chunk)
            if len(payload)>512:
                raise RuntimeError('模型就绪消息过长；未使能Planner')
            if b'\n' in payload:
                try:
                    message = json.loads(payload)
                except (ValueError,UnicodeDecodeError) as error:
                    raise RuntimeError('模型就绪消息无效；未使能Planner') from error
                if message != dict(stage='models_ready',pid=process.pid):
                    raise RuntimeError('模型就绪消息与当前抓取进程不匹配；未使能Planner')
                guard()
                self.ready = True
                return
        raise RuntimeError(f'模型初始化超过{timeout:g}秒；未使能Planner')

    async def release(self, guard):
        if not self.ready:
            raise RuntimeError('模型未就绪，禁止放行任务')
        guard()
        await asyncio.wait_for(asyncio.get_running_loop().sock_sendall(self.parent,b'GO\n'),1.)
        self.released = True

    async def wait_confirmation(self, process, guard, confirmed, timeout=1800.):
        """Keep the existing private barrier closed while a human decides.

        WAIT renews only the child's startup deadline; no control enable or GO.
        The normal ownership/freshness guard still runs every 100 ms.
        """
        if not self.ready or self.released:
            raise RuntimeError('当前模型屏障不能等待手动确认')
        deadline = time.monotonic() + timeout
        next_check = 0.
        while time.monotonic() < deadline:
            guard()
            if process.returncode is not None:
                raise RuntimeError('等待手动确认期间抓取程序已退出；未使能')
            if time.monotonic() >= next_check:
                if confirmed():
                    guard()
                    return
                await asyncio.wait_for(asyncio.get_running_loop().sock_sendall(self.parent, b'WAIT\n'), 1.)
                next_check = time.monotonic() + 1.
            await asyncio.sleep(.1)
        raise RuntimeError('等待手动开始超过30分钟；取消本次准备，未使能')

    def close(self):
        self.parent.close()
        self.child.close()
