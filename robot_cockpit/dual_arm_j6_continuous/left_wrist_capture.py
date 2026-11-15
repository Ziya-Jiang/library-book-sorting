"""Read-only on-155 D405 exposure bracket. Invoked over existing SSH; no control."""
import io
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit, parse_qs
from urllib.request import urlopen, Request
import numpy as np


def capture(root):
    info = json.loads((Path(root)/'runtime/web.json').read_text())
    token = parse_qs(urlsplit(info['url']).fragment)['token'][0]
    def state():
        req = Request('http://127.0.0.1:8810/api/state', headers={'X-Arm-Token': token})
        with urlopen(req, timeout=2) as r: s = json.load(r)
        if (not s.get('fresh') or s.get('simulation') is not False or not s.get('rt_running')
                or s.get('age_ms', 999)>100 or s.get('rt_assembly')!='dual_gripper'):
            raise ValueError('D405 needs fresh dual-gripper RT feedback')
        arm=s['state']['arms'][0]
        q=np.asarray(arm['rt_position'],float); vel=np.asarray(arm['velocity'],float)
        if q.shape!=(7,) or vel.shape!=(7,) or not np.isfinite(np.r_[q,vel]).all() or max(abs(vel))>.01:
            raise ValueError('Left arm must remain stationary for D405 exposure')
        return dict(rt_position=q.tolist(),velocity=vel.tolist(),epoch=s['epoch'],calibration=s['calibration'],
                    observed_unix_s=time.time(),rt_age_ms=s['age_ms'])
    before=state(); deadline=time.monotonic()+3
    while True:
        with urlopen('http://127.0.0.1:8793/frame',timeout=2) as r: raw=r.read()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            h=json.loads(str(z['header'])); bgr=z['bgr'].copy(); depth=z['depth'].copy()
        if h['captured_unix_s']>=before['observed_unix_s']: break
        if time.monotonic()>deadline: raise ValueError('No new left wrist frame')
        time.sleep(.02)
    after=state(); stamp=h['captured_unix_s']
    if (not before['observed_unix_s']<=stamp<=after['observed_unix_s']
            or after['observed_unix_s']-before['observed_unix_s']>1
            or not 0<=time.time()-stamp<=.5 or before['epoch']!=after['epoch']
            or before['calibration']!=after['calibration']
            or max(abs(np.array(before['rt_position'])-after['rt_position']))>np.deg2rad(.1)):
        raise ValueError('Left wrist exposure/pose binding invalid')
    h['joint_binding']=dict(method='stationary_before_after_bracket_not_hardware_sync',before=before,after=after)
    out=io.BytesIO();np.savez_compressed(out,bgr=bgr,depth=depth,header=json.dumps(h))
    return out.getvalue()


if __name__=='__main__':
    try: sys.stdout.buffer.write(capture(sys.argv[1]))
    except Exception as e:
        # Never echo web URL or authorization header.
        print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(1)
