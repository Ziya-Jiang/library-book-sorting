"""155 camera transport with bounded age, SSH identity and no local USB fallback."""
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import tempfile
import time
import uuid

MAX_BYTES = 64*1024*1024
WORKER = Path(__file__).with_name('cart_book_grasp_v3_4_remote_worker.py')
HEAD_MAX_AGE_SECONDS = .5
HEAD_STARTUP_MAX_ATTEMPTS = 3
# Existing first-read bounds: up to 5s for the SSH worker, then 3s for a frame.
# Retries share this budget rather than receiving a fresh startup timeout.
HEAD_STARTUP_BUDGET_SECONDS = 8.


class RemoteHeadFrameTooOld(ValueError):
    """A fully validated frame exceeded the age limit; never return its pixels."""

    def __init__(self, token, age_seconds, decode_seconds):
        self.token = token
        self.age_seconds = age_seconds
        self.decode_seconds = decode_seconds
        super().__init__(f'Remote head frame too old after decoding '
                         f'(age={age_seconds:.3f}s, decode={decode_seconds:.3f}s, '
                         f'limit={HEAD_MAX_AGE_SECONDS:.3f}s)')


class CameraTransport:
    def __init__(self, settings, config):
        self.settings, self.config = dict(settings), dict(config)
        self.process = None
        self.log = None
        self.buffer = b''

    def _start(self):
        from cart_book_grasp_v3_4_remote_worker import zstd_library
        zstd_library()
        command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
        if self.settings.get('ssh_control'):
            command += ['-S', str(self.settings['ssh_control'])]
        command += [self.settings['ssh_user']+'@'+self.settings['host'],
                    shlex.join([self.settings['remote_python'], '-u', '-c', WORKER.read_text()])]
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, bufsize=0)
        self.process.stdin.write((json.dumps(self.config)+'\n').encode())

    def _read(self, size, deadline, checkpoint):
        while len(self.buffer) < size:
            checkpoint()
            if time.monotonic() >= deadline:
                raise TimeoutError('155 camera response timeout; no local fallback')
            ready, _, _ = select.select([self.process.stdout], [], [], .025)
            if ready:
                chunk = os.read(self.process.stdout.fileno(), min(65536, size-len(self.buffer)))
                if not chunk:
                    self.log.seek(0)
                    raise OSError('155 camera SSH worker exited: '+self.log.read(2000).decode(errors='replace'))
                self.buffer += chunk
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result

    def request(self, timeout, checkpoint=lambda: None, after=''):
        checkpoint()
        started = time.monotonic()
        try:
            if self.process is None:
                self._start()
                ready = b''
                while not ready.endswith(b'\n'):
                    if len(ready) >= 8192:
                        raise ValueError('Oversized camera readiness message')
                    ready += self._read(1, started+5., checkpoint)
                if json.loads(ready) != {'ready': True}:
                    raise ValueError('Remote camera worker not ready: '+ready.decode(errors='replace'))
                started = time.monotonic()
            nonce = uuid.uuid4().hex
            self.process.stdin.write((json.dumps(dict(nonce=nonce, after=after))+'\n').encode())
            deadline = started+timeout
            line = b''
            while not line.endswith(b'\n'):
                if len(line) >= 8192:
                    raise ValueError('Oversized camera response header')
                line += self._read(1, deadline, checkpoint)
            header = json.loads(line)
            if 'error' in header:
                raise ValueError(header['error'])
            count = header.get('bytes')
            if type(count) is not int or not 0 < count <= MAX_BYTES:
                raise ValueError('Invalid camera payload size')
            payload = self._read(count, deadline, checkpoint)
            ended = time.monotonic()
            if header.get('nonce') != nonce or hashlib.sha256(payload).hexdigest() != header.get('sha256'):
                raise ValueError('Camera response identity or checksum mismatch')
            if header.get('encoding', 'raw') == 'zstd':
                from cart_book_grasp_v3_4_remote_worker import decompress_payload
                payload = decompress_payload(payload)
            elif header.get('encoding', 'raw') != 'raw':
                raise ValueError('Unknown camera payload encoding')
            ended = time.monotonic()
            age = float(header['source_age_s'])
            processing = float(header.get('processing_s', 0.))
            elapsed = ended-started
            if (not math.isfinite(age) or age < 0 or elapsed < 0 or not math.isfinite(processing)
                    or processing < 0 or processing > elapsed+.002):
                raise ValueError('Invalid remote camera age')
            checkpoint()
            return payload, age+max(0., elapsed-processing)+(.002 if processing else 0.), started
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            from cart_book_grasp_v3_camera_stream import stop_capture_process
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                if not stop_capture_process(self.process):
                    return False
            self.process.stdout.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.buffer = b''
        return True


class RemoteHeadSubscriber:
    def __init__(self, settings):
        self.transport = CameraTransport(settings, dict(kind='head', head_port=settings['head_port']))
        self.token = ''

    def read(self, timeout=3.):
        import cv2
        import numpy as np
        from head_rgbd.protocol import decode
        payload, age, started = self.transport.request(timeout, after=self.token)
        decode_started = time.monotonic()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            original = json.loads(str(archive['header']))
            if original.get('transport_baseline_from_native'):
                metadata = dict(original['metadata'])
                width, height = int(metadata['width']), int(metadata['height'])
                native_color, native_depth = archive['native_bgr'], archive['native_depth']
                native_shape = (int(metadata['native_height']), int(metadata['native_width']))
                if (not 0 < width*height <= MAX_BYTES//7 or min(width, height) <= 0
                        or native_color.shape != (*native_shape, 3) or native_color.dtype != np.uint8
                        or native_depth.shape != native_shape or native_depth.dtype != np.float32):
                    raise ValueError('Invalid native remote head frame')
                if (original.get('protocol') != 'head-rgbd-v1' or original.get('color_encoding') != 'bgr8'
                        or original.get('depth_encoding') != '32FC1' or original.get('depth_units') != 'mm'):
                    raise ValueError('Unsupported native remote RGB-D format')
                bgr = cv2.resize(native_color, (width, height))
                depth = cv2.resize(native_depth, (width, height), interpolation=cv2.INTER_NEAREST)
                header = dict(original, native_bgr=native_color, native_depth=native_depth)
        remote_stamp = float(original['captured_unix_s'])
        if not np.isfinite(remote_stamp) or not np.isfinite(age) or age < 0:
            raise ValueError('Invalid remote head timestamp or age')
        if not original.get('transport_baseline_from_native'):
            bgr, depth, header = decode(payload, now=remote_stamp+age)
        if header['token'] == self.token:
            raise ValueError('Repeated remote head frame')
        if not header['metadata'].get('native_is_maximum') or 'native_bgr' not in header:
            raise ValueError('155 head source lacks highest native RGB-D')
        if header['metadata'].get('serial') != 'CV2TC5100060':
            raise ValueError('Wrong remote head camera')
        if not np.isfinite(depth).all() or not np.isfinite(header['native_depth']).all():
            raise ValueError('Nonfinite remote head depth')
        decode_seconds = time.monotonic()-decode_started
        age += decode_seconds
        if age > HEAD_MAX_AGE_SECONDS:
            raise RemoteHeadFrameTooOld(header['token'], age, decode_seconds)
        header['remote_captured_unix_s'] = remote_stamp
        header['captured_unix_s'] = time.time()-age
        self.token = header['token']
        return bgr, depth, header

    def close(self):
        return self.transport.close()


class RemoteWristCapture:
    def __init__(self, settings, config):
        self.config = dict(config)
        self.transport = CameraTransport(settings, dict(kind='wrist', head_port=settings.get('head_port', 8792), **self.config))

    def capture(self, checkpoint=lambda: None):
        import numpy as np
        payload, age, started = self.transport.request(self.config['capture_timeout_seconds'], checkpoint)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            rgb = archive['rgb'].copy()
            source_header = json.loads(str(archive['header']))
        if age > 1. or time.monotonic()-started > 1.:
            raise ValueError('Remote wrist frame too old after transport')
        if rgb.dtype != np.uint8 or rgb.shape != (self.config['height'], self.config['width'], 3):
            raise ValueError('Remote wrist frame differs from calibrated resolution')
        checkpoint()
        return dict(rgb=rgb, captured_at=started, read_started_at=started,
                    pair_gap_seconds=time.monotonic()-started, source_header=source_header)

    def close(self):
        return self.transport.close()


@contextmanager
def remote_cameras(settings):
    import cart_book_grasp_v3_camera as wrist
    import head_rgbd.client as head
    original_wrist, original_head = wrist.PersistentWristCapture, head.CalibratedCamera
    sources = []

    def wrist_factory(config):
        source = RemoteWristCapture(settings, config)
        sources.append(source)
        return source

    def head_factory(calibration, warmup_frames=5, *, prefer_native=True):
        source = RemoteHeadSubscriber(settings)
        sources.append(source)
        return original_head(calibration, warmup_frames=warmup_frames, subscriber=source, prefer_native=prefer_native)

    wrist.PersistentWristCapture, head.CalibratedCamera = wrist_factory, head_factory
    try:
        yield
    finally:
        wrist.PersistentWristCapture, head.CalibratedCamera = original_wrist, original_head
        for source in reversed(sources):
            source.close()


def _startup_head_frame(head):
    """Retry only startup age rejection, requesting a frame after the discarded one.

    Runtime read() still fails immediately. Identity, checksum, source-service,
    decode and repeated-token errors are not retryable here.
    """
    deadline = time.monotonic() + HEAD_STARTUP_BUDGET_SECONDS
    for attempt in range(1, HEAD_STARTUP_MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Head startup freshness check exceeded its total time budget')
        try:
            frame = head.read(timeout=min(3., remaining))
        except RemoteHeadFrameTooOld as error:
            if attempt == HEAD_STARTUP_MAX_ATTEMPTS or time.monotonic() >= deadline:
                raise ValueError(f'Head startup freshness check failed after {attempt} attempt(s); '
                                 f'no frame within {HEAD_MAX_AGE_SECONDS:.3f}s: {error}') from error
            # read() has validated this frame's identity but has not exposed it.
            # Advancing only this startup subscriber prevents reusing its token.
            head.token = error.token
            print(f'头部启动检查：丢弃超龄帧（{error.age_seconds:.3f}s > '
                  f'{HEAD_MAX_AGE_SECONDS:.3f}s），获取更新帧 '
                  f'({attempt + 1}/{HEAD_STARTUP_MAX_ATTEMPTS})。', flush=True)
        else:
            if time.monotonic() > deadline:
                raise TimeoutError('Head startup freshness check exceeded its total time budget')
            if attempt > 1:
                print('头部启动检查：更新帧通过原有新鲜度与身份校验。', flush=True)
            return frame


def check_sources(settings, include_wrist=False):
    head = RemoteHeadSubscriber(settings)
    try:
        _, _, header = _startup_head_frame(head)
        metadata = header['metadata']
        print(f'155 head RGB-D: {metadata["serial"]}, native '
              f'{metadata["native_width"]}x{metadata["native_height"]}', flush=True)
    finally:
        head.close()
    if include_wrist:
        from cart_book_grasp_v3_camera import WristCamera
        with remote_cameras(settings):
            camera = WristCamera()
            try:
                frame = camera.capture()
                print(f'155 wrist RGB: {camera.config["device"]}, shape={frame["rgb"].shape}', flush=True)
            finally:
                camera.close()
