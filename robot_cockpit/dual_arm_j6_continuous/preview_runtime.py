"""Optional visualization isolation and bounded startup storage selection."""
from pathlib import Path
import json
import shutil
import sys
import threading
import uuid

MIN_FREE = 512 * 1024 * 1024
FALLBACK_ROOT = Path('/tmp/V-right-j6-continuous')


def choose_output(default, explicit=None):
    if explicit is not None:
        return Path(explicit)
    default=Path(default)
    ancestor=default.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free >= MIN_FREE:
        return default
    FALLBACK_ROOT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(FALLBACK_ROOT).free < MIN_FREE:
        raise OSError('运行日志所在分区均不足512MB；未启动任务')
    result=FALLBACK_ROOT/default.name
    print('SSD剩余空间不足512MB，本轮诊断临时保存到：'+str(result),flush=True)
    return result


def save_report(out, report):
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    target=Path(out)/'result.json'
    try:target.write_text(text)
    except OSError as error:
        target=Path('/tmp')/('V-right-approach-result-'+uuid.uuid4().hex+'.json')
        try:target.write_text(text)
        except OSError:
            print('报告写入失败；原始任务状态：'+text,file=sys.stderr)
            return None
        print('原日志目录写入失败('+type(error).__name__+')；报告改存：'+str(target),file=sys.stderr)
    return target


class OptionalPreview:
    """Catch only errors originating in visualization, never motion errors."""
    def __init__(self, viewer, report):
        self.viewer=viewer;self.report=report;self.disabled=False;self.lock=threading.Lock()

    def __getattr__(self, name):
        def call(*args, **kwargs):
            if self.disabled and name!='close':return None
            try:return getattr(self.viewer,name)(*args,**kwargs)
            except Exception as error:
                with self.lock:
                    self.disabled=True
                    self.report.setdefault('preview_errors',[]).append(dict(method=name,type=type(error).__name__,reason=str(error)))
                stop=getattr(self.viewer,'_stop',None)
                if stop is not None:stop.set()
                print('可视化已停用，仅影响显示：'+type(error).__name__+': '+str(error),file=sys.stderr,flush=True)
                return None
        return call
