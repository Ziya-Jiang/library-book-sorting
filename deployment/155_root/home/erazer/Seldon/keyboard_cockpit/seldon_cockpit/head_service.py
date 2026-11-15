"""Standalone shared RGB-D owner; no chassis/arm control imports or writes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
from .shared_rgbd_v34 import run_shared_head


def main():
    root=Path(__file__).resolve().parents[1]
    runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'shared_head.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Shared head owner already running')
        config=json.loads((root/'config.json').read_text())
        if config.get('orbbec_serial')!='CV2TC5100060' or config.get('shared_rgbd_port')!=8792:
            raise SystemExit('Unexpected head serial/port; refusing fallback')
        shutdown=threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:shutdown.set())
        info=runtime/'shared_head.json'
        info.write_text(json.dumps(dict(pid=os.getpid(),port=8792,serial=config['orbbec_serial'])))
        try:run_shared_head(config,lambda *_:None,shutdown)
        finally:
            try:
                if json.loads(info.read_text()).get('pid')==os.getpid():info.unlink()
            except (OSError,ValueError):pass

if __name__=='__main__':main()
