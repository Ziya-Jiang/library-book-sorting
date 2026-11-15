"""Left-wrist task segmentation and optional head live preview."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import threading
import uuid
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rb
import rerun_cli

import start
from preview_viewer import ensure_viewer


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError('Cannot save preview image: ' + str(path))


PANEL_TITLES = (
    ('live_head','Head camera | Live'),
    ('segmentation','Left wrist | Spine + edge 1-2'),
    ('book_count','Book count | Top to bottom'),
    ('grasp_y','Expected grasp | Base Y'),
)


def presentation_blueprint():
    views=[rb.Spatial2DView(origin='/'+entity,name=title,background=[18,23,31])
           for entity,title in PANEL_TITLES]
    return rb.Blueprint(
        rb.Vertical(rb.Horizontal(*views[:2],column_shares=[1,1]),
                    rb.Horizontal(*views[2:],column_shares=[1,1]),row_shares=[1,1]),
        rb.BlueprintPanel(state='hidden'),rb.SelectionPanel(state='hidden'),
        rb.TimePanel(state='hidden',timeline='log_time',play_state='Following'),auto_layout=False,auto_views=False)


def head_image(rgb):
    """One archetype for placeholder/live/stopped: avoid overlapping raw/JPEG layers."""
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError('Head preview JPEG encoding failed')
    return rr.EncodedImage(contents=encoded.tobytes(), media_type='image/jpeg')


class Preview:
    def __init__(self, out):
        self.out = Path(out)
        self._stop = threading.Event()
        self._live_thread = None
        self._reference_lock = threading.Lock()
        self._left_reference = None
        self._z_plan = None
        self._last_right_z = None
        self.rrd = None  # Live preview only: never append continuous video to disk.
        self.recording = rr.RecordingStream('GPT book segmentation | read only', recording_id=uuid.uuid4().hex)
        blueprint = presentation_blueprint()
        port = ensure_viewer(Path(rerun_cli.__file__).with_name('rerun'))
        self.recording.set_sinks(rr.GrpcSink(f'rerun+http://127.0.0.1:{port}/proxy'),
                                 default_blueprint=blueprint)
        rr.send_blueprint(blueprint, make_active=True, make_default=True, recording=self.recording)
        from presentation_panels import placeholder
        for entity,title in PANEL_TITLES:
            image = placeholder(title)
            self.recording.log(entity+'/image',head_image(image) if entity=='live_head' else rr.Image(image),static=entity!='live_head')
        self.status('Waiting for left-wrist segmentation. Head camera is preview only.')

    def status(self, message):
        # Task callers keep Chinese terminal messages; never send those glyphs to the viewer.
        english = message if message.isascii() else 'Task running. Detailed diagnostics are available in the terminal.'
        self.depth_message = english
        self.recording.log('task_status', rr.TextDocument(english), static=True)

    def _preview_event(self, event, **details):
        """Best-effort diagnostics must never interrupt task execution."""
        out = getattr(self, 'out', None)
        if out is None:
            return
        try:
            with (Path(out)/'head_preview_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(unix_s=time.time(), event=event, **details),
                                        ensure_ascii=True, allow_nan=False)+'\n')
        except Exception as error:
            print('Head preview diagnostic write failed: '+type(error).__name__, file=sys.stderr)

    def _live_display(self, image, status):
        # Serialize with close: a late camera response cannot replace Stream stopped.
        with self._live_display_lock:
            if self._stop.is_set():
                return
            self.recording.log('live_head/image', head_image(image), static=False)
            self.recording.log('live_status', rr.TextDocument(status), static=False)

    def start_live(self, right_z_reader=None):
        """Independent read-only preview channel; never delays motion or invokes GPT."""
        if self._live_thread is not None:
            return
        self._live_display_lock = threading.Lock()
        def loop():
            from cart_book_grasp_v3_4_remote_camera import CameraTransport, HEAD_MAX_AGE_SECONDS
            from tray_top.head import decode_frame
            settings = dict(host=start.ROBOT_IP, ssh_user='erazer', ssh_control=start.SOCKET,
                            remote_python=start.REMOTE_PYTHON)
            camera = None
            last = ''
            failures = 0
            displayed = 0
            def checkpoint():
                if self._stop.is_set():
                    raise InterruptedError('preview stopped')
            try:
                while not self._stop.is_set():
                    began = time.monotonic()
                    stage = 'connect' if camera is None else 'request'
                    age = None
                    try:
                        if camera is None:
                            camera = CameraTransport(settings, dict(kind='head', head_port=8792))
                        stage = 'request'
                        payload, age, _ = camera.request(1., checkpoint=checkpoint, after=last)
                        stage = 'decode'
                        decode_started = time.monotonic()
                        frame = decode_frame(payload, expected_serial='CV2TC5100060')
                        age += time.monotonic()-decode_started
                        stage = 'freshness'
                        if age > HEAD_MAX_AGE_SECONDS:
                            raise ValueError(f'Head frame too old: age={age:.3f}s limit={HEAD_MAX_AGE_SECONDS:.3f}s')
                        if frame.token == last:
                            raise ValueError('Repeated head frame: '+frame.token)
                        last = frame.token
                        if self._stop.is_set():break
                        stage = 'display'
                        from presentation_panels import banner
                        displayed += 1
                        wall = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%H:%M:%S.%f')[:-3]
                        live_image = banner(frame.rgb, [f'LIVE | Frame {displayed} | {wall} | Age {age*1000:.0f} ms'])
                        self._live_display(live_image,
                            f'Live preview: 5 Hz | Frame age: {age*1000:.0f} ms | {last}')
                        if displayed == 1 or displayed % 25 == 0:
                            self._preview_event('frame_displayed', frame_number=displayed,
                                                frame_token=last, age_s=age)
                        if failures:
                            self._preview_event('recovered', failed_attempts=failures,
                                                age_s=age, frame_token=last)
                            failures = 0
                    except InterruptedError:
                        break
                    except Exception as error:
                        if self._stop.is_set():break
                        failures += 1
                        self._preview_event('frame_failed', stage=stage,
                            error_type=type(error).__name__, reason=str(error), attempt=failures,
                            elapsed_s=time.monotonic()-began, age_s=age,
                            max_age_s=HEAD_MAX_AGE_SECONDS, previous_frame_token=last)
                        from presentation_panels import placeholder
                        try:
                            self._live_display(placeholder('HEAD CAMERA | LIVE',
                                'Stream unavailable - retrying. See head_preview_events.jsonl'),
                                'Preview unavailable: '+type(error).__name__+'; retrying. Task is unaffected.')
                        except Exception as display_error:
                            self._preview_event('display_failed', error_type=type(display_error).__name__,
                                                reason=str(display_error))
                        if camera is not None:
                            try:camera.close()
                            except Exception:pass
                        camera = None
                        last = ''
                    if right_z_reader is not None and not self._stop.is_set():
                        try:
                            z = float(right_z_reader())
                            if not np.isfinite(z):raise ValueError('invalid Z')
                            self.show_z_offset(z)
                        except Exception:
                            self.show_z_offset(None)
                    self._stop.wait(max(.02,.2-(time.monotonic()-began)))
            finally:
                if camera is not None:
                    try:camera.close()
                    except Exception:pass
        self._live_thread = threading.Thread(target=loop, name='right-trial-read-only-preview', daemon=True)
        self._live_thread.start()

    def show_left_failure(self, diagnostic):
        from presentation_panels import placeholder
        image=placeholder('LEFT WRIST | No validated result',
                          'Perception failed. See terminal for details.')
        for entity in ('segmentation','book_count','grasp_y'):
            self.recording.log(entity+'/image',rr.Image(image),static=True)
        self.status('Left-wrist perception failed. No validated grasp target.')

    def show_left_spine(self, reference):
        from presentation_panels import render_left_panels
        panels,marker=render_left_panels(self.out,reference)
        for entity,image in panels.items():
            save_rgb(self.out/('presentation_'+entity+'.png'),image)
            self.recording.log(entity+'/image',rr.Image(image),static=True)
        (self.out/'presentation_grasp_y.json').write_text(json.dumps(marker,indent=2)+'\n')
        with self._reference_lock:
            self._left_reference = dict(reference)
        self.status(f'Books: {reference["stack_book_count"]} | Lift: {reference["stack_lift_mm"]:g} mm | '
                    f'Grasp Y: {marker["target_grasp_y_mm"]:+.2f} mm')

    def show_front_forward(self, reference, current_x):
        target=float(reference['target_tcp_x_mm'])
        self.status(f'Target X: {target:.2f} mm | Advance: {target-current_x:.2f} mm')

    def show_right_wrist(self, observation):
        # Keep the four-pane presentation stable; this optional phase remains in diagnostics.
        self.status(f'Wrist alignment | Horizontal error: {observation["error_u_px"]:.2f} px')

    def show_z_offset(self, current_z, *, planned_offset=None):
        with self._reference_lock:
            self._last_right_z=current_z
            if planned_offset is not None:self._z_plan=float(planned_offset)
        # Do not replace the requested grasp-Y pane with transient Z diagnostics.

    def close(self):
        self._stop.set()
        if self._live_thread is not None:
            self._live_thread.join(timeout=5.)
        from presentation_panels import placeholder
        lock = getattr(self, '_live_display_lock', None)
        if lock is None:
            lock = threading.Lock()
        with lock:
            self.recording.log('live_head/image', head_image(placeholder('HEAD CAMERA | LIVE',
                'Stream stopped')), static=False)
            self.recording.log('live_status', rr.TextDocument(
                'Stream stopped. Task ended; segmentation results retained.'), static=False)
        self._preview_event('stopped', reason='task_ended')
        self.recording.flush(timeout_sec=2.)
        self.recording.disconnect()


def main():
    import right_approach as source
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
    out = args.out or start.RUNTIME/'gpt_segmentation'/f'{stamp}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='started', motion_enabled=False, camera_host='155', inference_client='100',
                  perception_source='left_wrist_rgbd_only', head_gpt_requests=0, selection={})
    preview = None
    try:
        preview = Preview(out)
        credential = source.api_key()
        start.ensure_tunnel()
        preview.start_live()
        from left_wrist_spine import run as perceive_left
        report['left_wrist_perception'] = {}
        print('只读预览：155左腕RGBD，100调用GPT；头部仅异步预览，不发送运动。', flush=True)
        reference = perceive_left(start, credential, out, out.name, report['left_wrist_perception'],
                                  require_far_edge=True)
        credential = None
        preview.show_left_spine(reference)
        report.update(status='viewer_started', first_left_spine=reference,
                      segmentation_total_s=time.monotonic()-began)
        print('Rerun已更新左腕书脊、远端1—2边和书堆编号。', flush=True)
    except Exception as error:
        report.update(status='failed', reason=str(error))
        raise
    finally:
        if preview is not None:
            preview.close()
        report['total_s'] = time.monotonic()-began
        (out/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print('预览结果：'+str(out), flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print('GPT分割预览未完成：'+str(error), file=sys.stderr)
        raise SystemExit(1)
