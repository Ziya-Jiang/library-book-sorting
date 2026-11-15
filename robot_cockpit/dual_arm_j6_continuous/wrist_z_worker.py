"""155-only camera/feature worker. No robot-command interface."""
import base64
import io
import json
from pathlib import Path
import sys
import time
import uuid
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener
import cv2
import numpy as np
from horizontal_spine import SpineTracker
from wrist_line_measurement import annotate

DEVICE = '/dev/v4l/by-id/usb-LRCP_G-720P_LRCP_G-720P_SN0001-video-index0'


def main():
    directory = Path('/mnt/data/erazer/book_grasp/dual_arm_control_runtime/wrist_z_runs')/uuid.uuid4().hex
    directory.mkdir(parents=True)
    opener = build_opener(ProxyHandler({}))
    tracker = None
    selected = None
    last_token = None
    def capture():
        nonlocal last_token
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            url='http://127.0.0.1:8792/wrist?'+urlencode(dict(device=DEVICE,width=1280,height=720))
            with opener.open(url,timeout=3) as response:
                payload=response.read(16*1024*1024)
            with np.load(io.BytesIO(payload),allow_pickle=False) as packet:
                rgb=packet['rgb'].copy();header=json.loads(str(packet['header']))
            age=time.time()-float(header['captured_unix_s'])
            metadata=header['metadata']
            if (header['protocol']!='wrist-rgb-v1' or metadata['device']!=DEVICE or
                rgb.shape!=(720,1280,3) or rgb.dtype!=np.uint8):
                raise ValueError('Wrist identity or resolution changed')
            token=header['token']
            if 0<=age<=.5 and token!=last_token:
                last_token=token
                return rgb,header
            time.sleep(.025)
        raise ValueError('No fresh wrist frame')
    for line in sys.stdin:
        started=time.monotonic()
        try:
            request=json.loads(line);op=request['op']
            if op=='capture':
                rgb,header=capture();selected=(rgb,header)
                cv2.imwrite(str(directory/'selection_rgb.png'),rgb[:,:,::-1])
                ok,jpg=cv2.imencode('.jpg',rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
                if not ok:raise ValueError('Wrist image encoding failed')
                result=dict(jpeg=base64.b64encode(jpg).decode(),frame_token=header['token'],width=1280,height=720)
            elif op=='init':
                if selected is None or request['frame_token']!=selected[1]['token']:
                    raise ValueError('Spine selection frame mismatch')
                rgb,header=selected;points=np.asarray(request['corners_px'],float)
                if (points.shape!=(4,2) or not np.isfinite(points).all() or np.any(points<0) or
                    np.any(points>=[1280,720]) or not cv2.isContourConvex(points.astype(np.int32))):
                    raise ValueError('Invalid wrist spine polygon')
                mask=np.zeros((720,1280),np.uint8);cv2.fillPoly(mask,[points.astype(np.int32)],1)
                tracker=SpineTracker(rgb,mask.astype(bool),header['token'])
                (directory/'selection.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
                result=dict(initialized=True)
            elif op=='observe':
                if tracker is None:raise ValueError('Tracker not initialized')
                rgb,header=capture();result=tracker.observe(rgb,header['token'])
                overlay=annotate(rgb,result,request.get('axis','Z'))
                ok,jpg=cv2.imencode('.jpg',overlay[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,85])
                if not ok:raise ValueError('Wrist overlay encoding failed')
                result.update(jpeg=base64.b64encode(jpg).decode(),
                              source_age_s=time.time()-header['captured_unix_s'])
                cv2.imwrite(str(directory/'latest_centerline.png'),overlay[:,:,::-1])
                cv2.imwrite(str(directory/'latest_rgb.png'),rgb[:,:,::-1])
            elif op=='close':break
            else:raise ValueError('Unknown read-only worker operation')
            result.update(ok=True,nonce=request['nonce'],runtime_host='155',directory=str(directory),
                          processing_s=time.monotonic()-started)
        except Exception as error:
            result=dict(ok=False,nonce=request.get('nonce') if 'request' in locals() else None,error=str(error))
        with (directory/'observations.jsonl').open('a') as log:
            log.write(json.dumps({k:v for k,v in result.items() if k!='jpeg'},ensure_ascii=False)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
