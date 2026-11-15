"""Camera-only sharing inside the existing cockpit head-camera child."""
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

SERIAL = 'CV2TC5100060'
WRIST = '/dev/v4l/by-id/usb-LRCP_G-720P_LRCP_G-720P_SN0001-video-index0'
MAX_AGE = .5
MAX_BYTES = 64*1024*1024


def profiles_list(profiles):
    return [profiles[index].as_video_stream_profile() for index in range(profiles.get_count())]


def choose_color(profiles, formats):
    choices = profiles_list(profiles)
    maximum = max(profile.get_width()*profile.get_height() for profile in choices)
    usable = [profile for profile in choices if profile.get_width()*profile.get_height() == maximum
              and profile.get_format() in formats]
    if not usable:
        raise ValueError('Maximum native color profile has no supported decoder')
    return max(usable, key=lambda profile: (profile.get_fps() <= 30,
               profile.get_fps() if profile.get_fps() <= 30 else -profile.get_fps(),
               profile.get_format() == formats[0]))


def calibration_metadata(path, native_size):
    raw = Path(path).read_bytes()
    document = json.loads(raw)['left']
    transform = np.eye(4)
    transform[:3, :3] = document['R']
    transform[:3, 3] = np.asarray(document['T']).reshape(3)
    values = [np.asarray(document[key], float) for key in ('K', 'D', 'R', 'T')]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError('Nonfinite camera calibration')
    if np.asarray(document['K']).shape != (3, 3) or document['w'] <= 0 or document['h'] <= 0:
        raise ValueError('Invalid camera calibration dimensions')
    return dict(K=document['K'], D=document['D'], T_dummy_world_camera_mm=transform.tolist(),
                width=document['w'], height=document['h'], serial=SERIAL,
                calibration_path=str(path), calibration_sha256=hashlib.sha256(raw).hexdigest(),
                native_width=native_size[0], native_height=native_size[1], native_is_maximum=True)


class Latest:
    def __init__(self):
        self.condition = threading.Condition()
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.frame = None
        self.error = 'Waiting for camera'

    def publish(self, arrays, stamp, metadata):
        if not 0 <= time.time()-stamp <= MAX_AGE:
            raise ValueError('Camera frame is stale')
        with self.condition:
            self.sequence += 1
            self.frame = (f'{self.session}:{self.sequence}', stamp, arrays, metadata)
            self.error = ''
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = str(error)
            self.frame = None
            self.condition.notify_all()

    def get(self, after='', newer_than=0., timeout=1.):
        with self.condition:
            def ready():
                return (self.frame is not None and self.frame[0] != after and self.frame[1] >= newer_than
                        and 0 <= time.time()-self.frame[1] <= MAX_AGE)
            if not self.condition.wait_for(ready, timeout):
                raise TimeoutError(self.error or 'Waiting for fresh camera frame')
            token, stamp, arrays, metadata = self.frame
        header = dict(protocol='head-rgbd-v1' if 'depth' in arrays else 'wrist-rgb-v1',
                      token=token, captured_unix_s=stamp, metadata=metadata,
                      color_encoding='bgr8' if 'depth' in arrays else 'rgb8',
                      depth_encoding='32FC1', depth_units='mm')
        output = io.BytesIO()
        np.savez(output, header=np.array(json.dumps(header, allow_nan=False)), **arrays)
        if time.time()-stamp > MAX_AGE:
            raise TimeoutError('Camera frame expired during encoding')
        payload = output.getvalue()
        if len(payload) > MAX_BYTES:
            raise ValueError('Camera payload exceeds bounded transport')
        return payload

    def status(self):
        with self.condition:
            age = time.time()-self.frame[1] if self.frame else None
            return dict(sequence=self.sequence, frame_age_s=age, error=self.error,
                        live=age is not None and 0 <= age <= MAX_AGE,
                        metadata=self.frame[3] if self.frame else None)


class WristOwner:
    def __init__(self, shutdown):
        self.shutdown = shutdown
        self.latest = Latest()
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()

    def run(self):
        while not self.shutdown.is_set():
            capture = None
            try:
                if not Path(WRIST).exists():
                    raise ValueError('155 wrist stable by-id device missing')
                capture = cv2.VideoCapture(WRIST, cv2.CAP_V4L2)
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    raise ValueError('Cannot open shared wrist camera')
                warmup = 5
                while not self.shutdown.is_set():
                    started = time.time()
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.shape != (720, 1280, 3):
                        raise ValueError('Shared wrist frame dimensions/read invalid')
                    if warmup:
                        warmup -= 1
                        continue
                    self.latest.publish(dict(rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), started,
                                        dict(device=WRIST, width=1280, height=720))
            except Exception as error:
                self.latest.fail(error)
            finally:
                if capture is not None:
                    capture.release()
            self.shutdown.wait(1.)


def wrist_payload(latest, params, requested):
    if params.get('latest') != ['1']:
        return latest.get(newer_than=requested, timeout=2.)
    since=float(params.get('since',['nan'])[0])
    if not math.isfinite(since) or not requested-MAX_AGE-.01<=since<=requested+.005:
        raise ValueError('Invalid wrist pose-history time window')
    return latest.get(after=params.get('after',[''])[0],newer_than=since,timeout=2.)


def make_server(head, wrist, port=8792):
    limit = threading.BoundedSemaphore(3)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.connection.settimeout(2.)
            if not limit.acquire(blocking=False):
                self.send_error(503, 'Camera subscribers busy')
                return
            try:
                parsed = urlsplit(self.path)
                params = parse_qs(parsed.query)
                if parsed.path == '/status':
                    payload = json.dumps(dict(head=head.status(), wrist=wrist.latest.status())).encode()
                    mime = 'application/json'
                elif parsed.path == '/frame':
                    payload = head.get(after=params.get('after', [''])[0])
                    mime = 'application/octet-stream'
                elif parsed.path == '/wrist':
                    if (params.get('device') != [WRIST] or params.get('width') != ['1280']
                            or params.get('height') != ['720']):
                        self.send_error(400, 'Wrist device/calibrated size mismatch')
                        return
                    requested = time.time()
                    wrist.start()
                    payload = wrist_payload(wrist.latest, params, requested)
                    mime = 'application/octet-stream'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
            except (TimeoutError, ValueError) as error:
                self.send_error(503, str(error))
            except OSError:
                pass
            finally:
                limit.release()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def capture_head(config, send, shutdown, latest):
    import pyorbbecsdk as ob
    if config.get('orbbec_serial') != SERIAL:
        raise ValueError('Shared head serial mismatch')
    context = ob.Context()
    device = context.query_devices().get_device_by_serial_number(SERIAL)
    pipeline = ob.Pipeline(device)
    try:
        color = choose_color(pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR),
                             (ob.OBFormat.MJPG, ob.OBFormat.RGB, ob.OBFormat.BGR))
        depth_choices = profiles_list(pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR))
        depth_choices = [profile for profile in depth_choices if profile.get_format() == ob.OBFormat.Y16
                         and profile.get_fps() == color.get_fps()]
        if not depth_choices:
            raise ValueError('No depth profile matching maximum color FPS')
        depth = max(depth_choices, key=lambda profile: (
            (profile.get_width(), profile.get_height()) == (color.get_width(), color.get_height()),
            profile.get_width()*profile.get_height()))
        metadata = calibration_metadata(config['shared_rgbd_calibration'],
                                        (color.get_width(), color.get_height()))
        failures = []
        for mode in (ob.OBAlignMode.HW_MODE, ob.OBAlignMode.SW_MODE):
            settings = ob.Config()
            settings.enable_stream(color)
            settings.enable_stream(depth)
            settings.set_align_mode(mode)
            try:
                pipeline.enable_frame_sync()
                pipeline.start(settings)
                metadata['depth_alignment'] = str(mode)
                break
            except Exception as error:
                failures.append(str(error))
                try:
                    pipeline.stop()
                except Exception:
                    pass
        else:
            raise ValueError('Native D2C failed: '+ '; '.join(failures))
        previous_color, previous_depth, offset = -1., -1., None
        last_received, last_preview = time.monotonic(), 0.
        while not shutdown.is_set():
            packet = pipeline.wait_for_frames(200)
            color_frame = packet.get_color_frame() if packet else None
            depth_frame = packet.get_depth_frame() if packet else None
            if color_frame is None or depth_frame is None:
                if time.monotonic()-last_received > 5.:
                    raise ValueError('No complete RGB-D pair for five seconds')
                continue
            color_stamp = color_frame.get_timestamp_us()/1e6
            depth_stamp = depth_frame.get_timestamp_us()/1e6
            if color_stamp <= previous_color or depth_stamp <= previous_depth:
                continue
            if abs(color_stamp-depth_stamp) > .05:
                continue
            previous_color, previous_depth = color_stamp, depth_stamp
            last_received = time.monotonic()
            now = time.time()
            offset = min(offset if offset is not None else now-color_stamp, now-color_stamp)
            captured = color_stamp+offset
            if now-captured > .3:
                raise ValueError('Head camera backlog exceeds 300ms')
            data = np.frombuffer(color_frame.get_data(), np.uint8)
            shape = (color_frame.get_height(), color_frame.get_width(), 3)
            if color_frame.get_format() == ob.OBFormat.MJPG:
                native_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif color_frame.get_format() == ob.OBFormat.RGB:
                native_bgr = cv2.cvtColor(data.reshape(shape), cv2.COLOR_RGB2BGR)
            else:
                native_bgr = data.reshape(shape).copy()
            native_depth = np.frombuffer(depth_frame.get_data(), np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()).astype(np.float32)*depth_frame.get_depth_scale()
            if (native_bgr is None or native_bgr.shape != (metadata['native_height'], metadata['native_width'], 3)
                    or native_depth.shape != native_bgr.shape[:2]):
                raise ValueError('Native color/D2C depth dimensions mismatch')
            native_depth[~np.isfinite(native_depth) | (native_depth <= 0) | (native_depth > 5000)] = 0
            size = (metadata['width'], metadata['height'])
            arrays = dict(bgr=cv2.resize(native_bgr, size),
                          depth=cv2.resize(native_depth, size, interpolation=cv2.INTER_NEAREST),
                          native_bgr=native_bgr, native_depth=native_depth)
            latest.publish(arrays, captured, metadata)
            if time.monotonic()-last_preview >= 1./config.get('camera_fps', 15):
                send(native_bgr, time.monotonic()-(time.time()-captured))
                last_preview = time.monotonic()
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def run_shared_head(config, send, shutdown):
    cv2.setNumThreads(1)
    head = Latest()
    wrist = WristOwner(shutdown)
    server = make_server(head, wrist, config.get('shared_rgbd_port', 8792))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while not shutdown.is_set():
            try:
                capture_head(config, send, shutdown, head)
            except Exception as error:
                head.fail(error)
                print('Shared head RGB-D: '+str(error), flush=True)
                shutdown.wait(1.)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3.)
        if wrist.thread is not None:
            wrist.thread.join(3.)
