"""Independent D405 owner, compatible with the head-rgbd-v1 frame transport."""
import fcntl,hashlib,json,signal,sys,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import parse_qs,urlsplit
import cv2
import numpy as np
from transport import Latest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'realsense_sdk'))
import pyrealsense2 as rs
SERIAL='261122273825'
PORT=8793

def capture(latest,shutdown):
    pipe=rs.pipeline();cfg=rs.config();cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.depth,848,480,rs.format.z16,30)
    cfg.enable_stream(rs.stream.color,848,480,rs.format.rgb8,30)
    started=False
    try:
        profile=pipe.start(cfg);started=True
        device=profile.get_device()
        if device.get_info(rs.camera_info.serial_number)!=SERIAL:raise ValueError('Wrong D405 identity')
        scale=device.first_depth_sensor().get_depth_scale()
        align=rs.align(rs.stream.color)
        previous=-1.;offset=None;last_publish=time.monotonic();warmup=15
        while not shutdown.is_set():
            frames=pipe.wait_for_frames(2000)
            color=frames.get_color_frame();depth=frames.get_depth_frame()
            if time.monotonic()-last_publish>5.:raise TimeoutError('No fresh synchronized RGB-D pair')
            if not color or not depth:continue
            stamp=color.get_timestamp()/1000.;dstamp=depth.get_timestamp()/1000.
            if stamp<=previous or abs(stamp-dstamp)>.05:continue
            previous=stamp
            if warmup:warmup-=1;continue
            now=time.time();offset=min(offset if offset is not None else now-stamp,now-stamp)
            captured=stamp+offset
            if now-captured>.3:raise TimeoutError('D405 camera backlog exceeds 300 ms')
            frames=align.process(frames)
            color=frames.get_color_frame();depth=frames.get_depth_frame()
            bgr=cv2.cvtColor(np.asanyarray(color.get_data()),cv2.COLOR_RGB2BGR)
            depth_mm=np.asanyarray(depth.get_data()).astype(np.float32)*(scale*1000.)
            if bgr.shape!=(480,848,3) or depth_mm.shape!=(480,848):raise ValueError('D405 aligned shape mismatch')
            intr=depth.profile.as_video_stream_profile().get_intrinsics()
            metadata=dict(serial=SERIAL,usb_descriptor_serial='262843072089',model='RealSense D405',
                camera_role='left_suction_d405',width=848,height=480,native_width=848,native_height=480,
                K=[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],D=list(intr.coeffs),
                distortion_model=str(intr.model),depth_alignment='rs.align(color)',fps=30,
                depth_scale_m=scale,calibration_source='device_factory_intrinsics',
                extrinsics_calibrated=False,T_base_camera=None,
                firmware=device.get_info(rs.camera_info.firmware_version),
                usb_type=device.get_info(rs.camera_info.usb_type_descriptor),
                device_color_timestamp_s=stamp,device_depth_timestamp_s=dstamp,
                timestamp_domain=str(color.get_frame_timestamp_domain()))
            latest.publish(dict(bgr=bgr,depth=depth_mm,native_bgr=bgr,native_depth=depth_mm),captured,metadata)
            last_publish=time.monotonic()
    finally:
        if started:pipe.stop()

def make_server(latest):
    limit=threading.BoundedSemaphore(3)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            self.connection.settimeout(2.)
            if not limit.acquire(False):self.send_error(503,'Camera subscribers busy');return
            try:
                parsed=urlsplit(self.path);params=parse_qs(parsed.query)
                if parsed.path=='/status':
                    state=latest.status()
                    payload=json.dumps(dict(profile='left_suction_d405',serial=SERIAL,live=state['live'],camera=state)).encode();mime='application/json'
                elif parsed.path=='/frame':
                    payload=latest.get(after=params.get('after',[''])[0]);mime='application/octet-stream'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
            except (ValueError,TimeoutError) as exc:self.send_error(503,str(exc))
            except OSError:pass
            finally:limit.release()
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler);server.daemon_threads=True
    return server

def main():
    cv2.setNumThreads(1)
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'owner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('D405 service owner already running')
        shutdown=threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:shutdown.set())
        latest=Latest();server=make_server(latest)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        print(f'D405 service 127.0.0.1:{PORT}; SDK serial={SERIAL}',flush=True)
        try:
            while not shutdown.is_set():
                try:capture(latest,shutdown)
                except Exception as exc:
                    latest.fail(exc);print(f'D405 capture unavailable: {exc}',flush=True);shutdown.wait(2.)
        finally:server.shutdown();server.server_close();thread.join(3.)
if __name__=='__main__':main()
