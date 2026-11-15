"""Read-only sampled telemetry. No polling sockets, command changes, or control waits.

RT target and measured joints come from ONE calibrated WebSocket RT snapshot.
UDP pose-monitor samples have a different clock/sample path and are separate events.
The RT target is the interpolated joint reference, not a motor-current command.
"""
import copy
import json
import math
import queue
import threading
import time
from pathlib import Path


def vector(value, size):
    values=[float(x) for x in value]
    if len(values)!=size or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid tracking vector')
    return values


def rt_sample(snapshot, received, context):
    state=snapshot.get('state')
    if not state:return None
    if state.get('coordinate_system')!='V2 calibrated':
        raise ValueError('Tracking requires calibrated joint positions and targets')
    arms=[]
    for arm in state['arms']:
        q=vector(arm['position'],7);target=vector(arm['target'],7)
        error=[a-b for a,b in zip(q,target)]
        arms.append(dict(target_rad=target,measured_rad=q,error_rad=error,
                         error_deg=[math.degrees(x) for x in error],
                         velocity_rad_s=vector(arm['velocity'],7)))
    if len(arms)!=2:raise ValueError('Tracking requires both arms')
    return dict(event='rt_joint_tracking',rt_timestamp_us=state['timestamp'],rt_sequence=state['sequence'],
        rt_echo=state.get('echo'),websocket_received_monotonic_s=received,
        rt_age_at_websocket_ms=snapshot.get('age_ms'),fresh=snapshot.get('fresh'),
        epoch=snapshot.get('epoch'),active=snapshot.get('active'),mine=snapshot.get('mine'),
        coordinate_system='V2 calibrated',unit='rad',arms=arms,
        pairing='target_and_measurement_same_RT_snapshot',
        target_source='RT interpolated_target; not endpoint or motor-current command',
        context=copy.deepcopy(context),context_binding='client_task_phase_not_RT_command_ack')


def monitor_sample(sample, context):
    if sample is None:return dict(event='monitor_missing',context=copy.deepcopy(context))
    q=vector([x for row in sample['joint_state'] for x in row],14) if len(sample['joint_state'])==2 else vector(sample['joint_state'],14)
    actual=vector(sample['right_arm']['position'],3)
    row=dict(event='pose_monitor',measured_joints_rad=q,right_tcp_m=actual,
        received_monotonic_s=sample.get('received_monotonic'),
        pose_sample_monotonic_s=sample.get('pose_sample_monotonic'),
        pose_fk_completed_monotonic_s=sample.get('pose_fk_completed_monotonic'),
        curobo_status=str(sample.get('curobo_status')),robot_status=str(sample.get('robot_status')),
        context=copy.deepcopy(context),pairing='UDP_monitor_sample_not_time_paired_with_websocket')
    if context.get('target_tcp_m') is not None:
        target=vector(context['target_tcp_m'],3);axes=context['fixed_axes']
        delta=[(a-b)*1000 for a,b in zip(actual,target)]
        error=math.sqrt(sum(delta[i]**2 for i in axes))
        row.update(target_tcp_m=target,tcp_error_mm=delta,fixed_axes=axes,
                   fixed_axis_error_mm=error,fixed_axis_limit_mm=3.,fixed_axis_exceeded=error>3.)
    return row


class TrackingJournal:
    def __init__(self,directory,capacity=4096,max_bytes=32*1024*1024):
        self.path=Path(directory)/'joint_tracking.jsonl'
        self.queue=queue.Queue(capacity);self.stop=threading.Event();self.max_bytes=max_bytes
        self.lock=threading.Lock();self.last_rt=None;self.last_monitor=None
        self.dropped=0;self.written=0;self.error=None;self.last_guard_sample=None
        self.thread=threading.Thread(target=self._write,name='j6-tracking-log',daemon=True);self.thread.start()

    def _error(self,error):
        if self.error is None:
            self.error=type(error).__name__+': '+str(error)
            print('Motion tracking log unavailable: '+self.error,flush=True)

    def emit(self,row):
        if self.stop.is_set() or self.error:return
        try:
            row=dict(row,logged_monotonic_s=time.monotonic(),logged_unix_s=time.time(),dropped_before=self.dropped)
            self.queue.put_nowait(row)
        except queue.Full:self.dropped+=1

    def observe_rt(self,snapshot,received,context):
        try:
            key=(snapshot.get('epoch'),(snapshot.get('state') or {}).get('timestamp'))
            with self.lock:
                if key==self.last_rt:return
                row=rt_sample(snapshot,received,context)
                if row is not None:self.last_rt=key;self.emit(row)
        except Exception as error:self._error(error)

    def observe_monitor(self,sample,context):
        try:
            row=monitor_sample(sample,context)
            with self.lock:
                if row.get('fixed_axis_exceeded'):self.last_guard_sample=copy.deepcopy(row)
                key=((sample or {}).get('received_monotonic'),context.get('request_id'))
                if key==self.last_monitor:return
                self.last_monitor=key;self.emit(row)
        except Exception as error:self._error(error)

    def _write(self):
        try:
            with self.path.open('x',buffering=1) as f:
                size=0
                while not self.stop.is_set() or not self.queue.empty():
                    try:row=self.queue.get(timeout=.05)
                    except queue.Empty:continue
                    data=json.dumps(row,allow_nan=False,separators=(',',':'))+'\n';size+=len(data.encode())
                    if size>self.max_bytes:raise OSError('32MB per-run tracking limit reached')
                    f.write(data);self.written+=1
        except Exception as error:self._error(error)

    def status(self):
        return dict(path=str(self.path),written=self.written,dropped=self.dropped,error=self.error,
                    pending=self.queue.qsize(),writer_alive=self.thread.is_alive(),
                    last_guard_sample=copy.deepcopy(self.last_guard_sample),
                    sample_rate='existing WebSocket snapshots (~20Hz), plus consumed UDP monitor samples; not every RT cycle')

    def close(self):
        self.stop.set();self.thread.join(.5)
        return self.status()
