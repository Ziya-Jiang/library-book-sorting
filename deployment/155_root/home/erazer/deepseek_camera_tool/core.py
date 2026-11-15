"""Robot-local, read-only channel broker. No motor or model API dependency."""
from __future__ import annotations
import asyncio
import base64
import io
import json
import math
import time
from dataclasses import dataclass, field, replace
from typing import AsyncIterator

import cv2
import httpx
import numpy as np

DEFAULT_CHANNELS = [0] * 8
WRIST = '/dev/v4l/by-id/usb-LRCP_G-720P_LRCP_G-720P_SN0001-video-index0'
NAMES = {0: 'head_global', 1: 'right_wrist'}
MAX_BYTES = 32 * 1024 * 1024


def channels_mask(value=None):
    if value is None:
        return DEFAULT_CHANNELS.copy()
    if not isinstance(value, list) or not 2 <= len(value) <= 64:
        raise ValueError('channels must be an array of 2..64 integer bits')
    if any(type(bit) is not int or bit not in (0, 1) for bit in value):
        raise ValueError('channels contains a non-integer bit (only 0 or 1)')
    if any(value[2:]):
        raise ValueError('Only channel 0 (head) and 1 (right wrist) are connected')
    return value.copy()


def depth_mask(value, rgb_mask):
    mask=[0]*len(rgb_mask) if value is None else channels_mask(value)
    if len(mask)!=len(rgb_mask):raise ValueError('depth_channels length must match channels')
    if any(mask[1:]):raise ValueError('Only head channel0 has measured depth')
    return mask


@dataclass(frozen=True)
class Frame:
    channel: int
    token: str
    captured_unix_s: float
    received_unix_s: float
    width: int
    height: int
    jpeg: bytes
    timestamp_basis: str
    timing_ms: dict = field(default_factory=dict)
    depth_mm: np.ndarray | None = None
    calibration: dict = field(default_factory=dict)
    depth_error: str | None = None

    def depth_wire(self, sent_unix_s):
        if self.depth_mm is None:
            raise ValueError(self.depth_error or 'No paired measured depth in this frame')
        began=time.monotonic();buffer=io.BytesIO()
        np.savez_compressed(buffer,depth_mm=self.depth_mm)
        return dict(channel=self.channel,source_frame_id=self.token,frame_id=self.token,
            captured_unix_s=self.captured_unix_s,sent_unix_s=sent_unix_s,
            width=self.width,height=self.height,encoding='npz_float32_mm',units='mm',
            invalid='nonfinite or <=0',depth_npz_base64=base64.b64encode(buffer.getvalue()).decode('ascii'),
            calibration=self.calibration,paired_with_rgb=True,
            timing_ms={'depth_encode_ms':(time.monotonic()-began)*1000})

    def wire(self, sent_unix_s=None):
        sent = time.time() if sent_unix_s is None else sent_unix_s
        encode_started = time.monotonic()
        encoded = base64.b64encode(self.jpeg).decode('ascii')
        return dict(channel=self.channel, name=NAMES[self.channel], frame_id=self.token,
                    captured_unix_s=self.captured_unix_s, received_unix_s=self.received_unix_s,
                    sent_unix_s=sent, frame_age_ms=(sent-self.captured_unix_s)*1000,
                    timestamp_basis=self.timestamp_basis, clock='155_unix_seconds',
                    width=self.width, height=self.height, mime_type='image/jpeg',
                    image_base64=encoded, timing_ms={**self.timing_ms,
                        'wire_base64_encode':(time.monotonic()-encode_started)*1000})


def decode_frame(payload, channel, received, quality=90):
    started = time.monotonic()
    if len(payload) > MAX_BYTES:
        raise ValueError('Shared camera payload too large')
    depth=None;calibration={};depth_error=None
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        header = json.loads(str(data['header'].item()))
        meta = header['metadata']
        if channel == 0:
            if header['protocol'] != 'head-rgbd-v1' or meta.get('serial') != 'CV2TC5100060':
                raise ValueError('Head camera identity/protocol mismatch')
            if not meta.get('native_is_maximum'):
                raise ValueError('Head source did not confirm maximum native resolution')
            bgr = data['native_bgr'].copy()
            expected = (int(meta['native_height']), int(meta['native_width']), 3)
            basis = 'head_device_timestamp_mapped_to_host_clock'
            try:
                if header.get('depth_units')!='mm' or header.get('depth_encoding')!='32FC1':
                    raise ValueError('Missing explicit float32/mm depth contract')
                depth=data['native_depth'].copy()
                if depth.shape!=expected[:2] or depth.dtype!=np.float32:
                    raise ValueError('Paired native RGB/depth size or dtype mismatch')
                sx=expected[1]/int(meta['width']);sy=expected[0]/int(meta['height'])
                scale=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
                k=scale@np.asarray(meta['K'],float)
                calibration=dict(K=k.tolist(),D=meta['D'],
                    T_base_camera_mm=meta['T_dummy_world_camera_mm'],
                    base_frame='dummy_world',camera_serial=meta['serial'],
                    calibration_sha256=meta['calibration_sha256'],depth_alignment=meta['depth_alignment'],
                    image_geometry='native_distorted_color_with_D2C_depth',
                    intrinsic_scaling='pixel_center_resize_from_calibration_resolution')
            except (ValueError,KeyError,TypeError) as error:
                depth=None;depth_error=str(error)

        else:
            if header['protocol'] != 'wrist-rgb-v1' or meta.get('device') != WRIST:
                raise ValueError('Wrist camera identity/protocol mismatch')
            rgb = data['rgb']
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            expected = (720, 1280, 3)
            basis = 'host_time_before_shared_camera_read_not_hardware_exposure'
        if bgr.dtype != np.uint8 or bgr.shape != expected:
            raise ValueError('Camera image shape/type disagrees with metadata')
    stamp = float(header['captured_unix_s'])
    if not math.isfinite(stamp) or not -.01 <= received - stamp <= .5:
        raise ValueError('Shared camera frame is stale or future dated')
    token = header['token']
    if not isinstance(token, str) or not token:
        raise ValueError('Missing source frame identity')
    decoded = time.monotonic()
    ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    jpeg = jpg.tobytes()
    return Frame(channel, token, stamp, received, expected[1], expected[0], jpeg, basis,
                 dict(npz_decode_validate_color=(decoded-started)*1000,
                      jpeg_encode=(time.monotonic()-decoded)*1000),depth,calibration,depth_error)


class SharedSource:
    def __init__(self, url='http://127.0.0.1:8792', quality=90):
        self.url = url.rstrip('/')
        self.quality = quality
        self.client = httpx.AsyncClient(timeout=2.5, trust_env=False,
                                       limits=httpx.Limits(max_connections=2))

    async def read(self, channel, after):
        started = time.monotonic()
        if channel == 0:
            endpoint, params = '/frame', {'after': after}
        else:
            endpoint = '/wrist'
            params = dict(device=WRIST, width='1280', height='720', latest='1',
                          since=str(time.time()-.49), after=after)
        async with self.client.stream('GET', self.url+endpoint, params=params) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError('Shared camera payload too large')
        received = time.time()
        transferred = time.monotonic()
        frame = await asyncio.to_thread(decode_frame, bytes(data), channel, received, self.quality)
        done = time.monotonic()
        return replace(frame, timing_ms={**frame.timing_ms,
            'source_http_wait_transfer':(transferred-started)*1000,
            'decode_thread_queue_return':max(0.,(done-transferred)*1000-sum(frame.timing_ms.values())),
            'source_read_total':(done-started)*1000})

    async def close(self):
        await self.client.aclose()


class Subscription:
    """At most one pending event per channel; slow readers never build a frame FIFO."""
    def __init__(self, mask):
        self.mask = mask
        self.pending = {}
        self.ready = asyncio.Event()
        self.dropped = [0] * len(mask)

    def offer(self, channel, item):
        if self.mask[channel]:
            if channel in self.pending:
                self.dropped[channel] += 1
            self.pending[channel] = item
            self.ready.set()

    async def take(self, timeout=1.):
        try:
            await asyncio.wait_for(self.ready.wait(), timeout)
        except asyncio.TimeoutError:
            return {}
        batch, self.pending = self.pending, {}
        self.ready.clear()
        return batch


class CameraBroker:
    def __init__(self, source, fps=10.):
        if not 0 < fps <= 30:
            raise ValueError('fps must be in (0,30]')
        self.source, self.fps = source, fps
        self.subscriptions = set()
        self.wake = [asyncio.Event(), asyncio.Event()]
        self.tasks = []
        self.frames = [0, 0]
        self.errors = [0, 0]

    async def start(self):
        self.tasks = [asyncio.create_task(self._worker(c), name=f'camera-channel-{c}') for c in (0, 1)]

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.source.close()

    def subscribe(self, channels):
        sub = Subscription(channels_mask(channels))
        self.subscriptions.add(sub)
        for c in (0, 1):
            if sub.mask[c]:
                self.wake[c].set()
        return sub

    def unsubscribe(self, sub):
        self.subscriptions.discard(sub)
        for c in (0, 1):
            if not any(s.mask[c] for s in self.subscriptions):
                self.wake[c].clear()

    async def _worker(self, channel):
        last = ''
        next_tick = 0.
        while True:
            await self.wake[channel].wait()
            await asyncio.sleep(max(0., next_tick - time.monotonic()))
            if not self.wake[channel].is_set():
                continue
            tick = time.monotonic()
            try:
                frame = await self.source.read(channel, last)
                if frame.token == last:
                    raise ValueError('Shared source repeated the same frame')
                last = frame.token
                item = frame
                self.frames[channel] += 1
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.errors[channel] += 1
                item = dict(channel=channel, error=f'{type(error).__name__}: {error}',
                            observed_unix_s=time.time())
            for sub in tuple(self.subscriptions):
                sub.offer(channel, item)
            # No catch-up bursts after a slow read. Errors back off without ending other channels.
            period = 1/self.fps if isinstance(item, Frame) else max(.5, 1/self.fps)
            next_tick = max(tick+period, time.monotonic())

    async def snapshot(self, channels=None, instruction='', timeout=3., depth_channels=None):
        started = time.monotonic()
        received = time.time()
        mask = channels_mask(channels)
        dmask=depth_mask(depth_channels,mask)
        wanted_mask=[int(a or b) for a,b in zip(mask,dmask)]
        sub = self.subscribe(wanted_mask)
        wanted = {c for c in (0, 1) if wanted_mask[c]}
        frames, errors = {}, {}
        deadline = time.monotonic()+timeout
        try:
            while wanted - frames.keys() and time.monotonic() < deadline:
                batch = await sub.take(max(.001, deadline-time.monotonic()))
                for c, item in batch.items():
                    if isinstance(item, Frame):
                        # Each image belongs to this command, not a pre-request cached exposure.
                        if item.captured_unix_s >= received and time.time()-item.captured_unix_s <= .5:
                            frames[c] = item
                            errors.pop(c, None)
                    else:
                        errors[c] = item['error']
            sent = time.time()
            # Never return an early good frame that became stale while another channel failed.
            frames = {c: f for c, f in frames.items() if -.01 <= sent-f.captured_unix_s <= .5}
            for c in wanted - frames.keys():
                errors.setdefault(c, 'No fresh post-request image before snapshot deadline')
            available = [int(mask[c] and c in frames) for c in range(len(mask))]
            ready = time.monotonic()
            wired = [frames[c].wire(sent) for c in sorted(frames) if mask[c]]
            depth_frames=[None]*len(mask);depth_errors=[]
            for c in wanted:
                if dmask[c]:
                    try:
                        if c not in frames:raise ValueError('No fresh paired depth frame')
                        depth_frames[c]=await asyncio.to_thread(frames[c].depth_wire,sent)
                    except ValueError as error:depth_errors.append(dict(channel=c,error=str(error)))
            return dict(kind='snapshot', instruction=instruction, channels=mask,
                        available_channels=available, depth_channels=dmask,depth_frames=depth_frames,
                        available_depth_channels=[int(x is not None) for x in depth_frames],depth_errors=depth_errors,
                        requested_unix_s=received, sent_unix_s=sent,
                        frames=wired, timing_ms=dict(wait_fresh_channels=(ready-started)*1000,
                            wire_base64_encode=(time.monotonic()-ready)*1000),
                        errors=[dict(channel=c, error=errors[c]) for c in sorted(errors)])
        finally:
            self.unsubscribe(sub)

    async def stream(self, channels=None, instruction='', depth_channels=None) -> AsyncIterator[dict]:
        mask = channels_mask(channels);dmask=depth_mask(depth_channels,mask)
        sub = self.subscribe([int(a or b) for a,b in zip(mask,dmask)])
        try:
            yield dict(kind='stream_open', channels=mask,depth_channels=dmask, instruction=instruction,
                       fps_per_channel=self.fps, sent_unix_s=time.time())
            while True:
                batch = await sub.take()
                sent = time.time()
                frames, errors = [], [];depth_frames=[None]*len(mask);depth_errors=[]
                for c, item in sorted(batch.items()):
                    if isinstance(item, Frame) and -.01 <= sent-item.captured_unix_s <= .5:
                        if mask[c]:frames.append(item.wire(sent))
                        if dmask[c]:
                            try:depth_frames[c]=await asyncio.to_thread(item.depth_wire,sent)
                            except ValueError as error:depth_errors.append(dict(channel=c,error=str(error)))
                    elif isinstance(item, Frame):
                        errors.append(dict(channel=c, error='Frame aged out before delivery'))
                    else:
                        errors.append(item)
                available = [0]*len(mask)
                for frame in frames:
                    available[frame['channel']] = 1
                yield dict(kind='frames' if frames else 'status', channels=mask,
                           available_channels=available,depth_channels=dmask,depth_frames=depth_frames,
                           available_depth_channels=[int(x is not None) for x in depth_frames],depth_errors=depth_errors,sent_unix_s=sent,
                           frames=frames, errors=errors, dropped_before_send=sub.dropped.copy())
        finally:
            self.unsubscribe(sub)

    def status(self):
        return dict(runtime='155',depth_capable_channels=[1,0,0,0,0,0,0,0],fps_per_channel=self.fps, subscribers=len(self.subscriptions),
                    enabled_channels=[int(w.is_set()) for w in self.wake]+[0]*6,
                    frames=self.frames, source_errors=self.errors)
