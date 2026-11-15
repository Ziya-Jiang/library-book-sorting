"""On155: print complete JSON/NDJSON camera records (including image_base64)."""
import argparse
import json
import httpx

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['snapshot','stream'])
    p.add_argument('--channels', default='[0,0,0,0,0,0,0,0]')
    p.add_argument('--depth-channels',default=None)
    p.add_argument('--instruction', default='')
    p.add_argument('--metadata-only', action='store_true', help='omit image bytes in printed diagnostics')
    a = p.parse_args()
    body = dict(channels=json.loads(a.channels), instruction=a.instruction)
    if a.depth_channels is not None:body['depth_channels']=json.loads(a.depth_channels)
    def emit(event):
        if a.metadata_only:
            for frame in event.get('frames',[]): frame.pop('image_base64',None)
            for frame in event.get('depth_frames',[]):
                if frame:frame.pop('depth_npz_base64',None)
        print(json.dumps(event,ensure_ascii=False),flush=True)
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(10., read=None)) as client:
        if a.mode=='snapshot':
            r=client.post('http://127.0.0.1:8796/v1/snapshot',json=body)
            r.raise_for_status()
            emit(r.json())
        else:
            with client.stream('POST','http://127.0.0.1:8796/v1/stream',json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line: emit(json.loads(line))
if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: pass
