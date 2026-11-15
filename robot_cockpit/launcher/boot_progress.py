"""Optional local boot UI status; never issues robot commands or records stdout."""
import json
import os
from pathlib import Path
import time
import re


def context():
    directory = os.environ.get('BOOK_GRASP_BOOT_DIR')
    run_id = os.environ.get('BOOK_GRASP_BOOT_RUN')
    stage = os.environ.get('BOOK_GRASP_BOOT_STAGE')
    if not directory or not run_id or stage not in ('services', 'program'):
        return None
    return Path(directory), run_id, stage


def report(phase, message):
    ctx = context()
    if ctx is None:
        return
    directory, run_id, stage = ctx
    target = directory / (stage + '-progress.json')
    message = re.sub(r'(?i)([?&#](?:token|access_token|api_key)=)[^\s&#]+', r'\1[redacted]', str(message))
    payload = dict(runId=run_id, stage=stage, phase=phase, message=message[:1500],
                   pid=os.getpid(), updated=time.time())
    temporary = target.with_suffix('.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)


def manual_enabled():
    if os.environ.get('BOOK_GRASP_MANUAL_START') != '1':
        return False
    ctx = context()
    if ctx is None or ctx[2] != 'program':
        raise RuntimeError('手动启动缺少本次启动会话；拒绝自动运行')
    return True


def confirmation_requested():
    directory, run_id, _ = context()
    report('awaiting_confirmation', '模型与相机已就绪，等待手动开始；尚未使能运动')
    target = directory / 'start.json'
    try:
        payload = json.loads(target.read_text())
    except FileNotFoundError:
        return False
    if payload != dict(runId=run_id, action='start'):
        raise RuntimeError('手动确认与本次启动会话不匹配')
    target.unlink()
    return True
