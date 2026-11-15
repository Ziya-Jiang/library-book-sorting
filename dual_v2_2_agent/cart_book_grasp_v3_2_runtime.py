"""Real robot adapter for bounded V3.2 rolling targets. No devices on import."""
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from cart_book_grasp_v3_servo import ServoError
from cart_book_grasp_v3_centerline import measured_roll,roll_orientation,AxisResidualExceeded
from cart_book_grasp_v3_cotracker_points import draw_points
from cart_book_grasp_v3_2_stream import run_stream,translation_budget
from cart_book_grasp_v3_1_pose_time import PoseTimeline,PoseNotBracketed,PoseSupportUnavailable
from cart_book_grasp_v3_image_freshness import ImageFreshnessRecovery,StaleImage,ImageMotionExpired


def status_value(value):return getattr(value,'value',value)


class FeedbackStale(ServoError):
    """A well-formed feedback packet has aged out; never usable for control."""


class StreamTransport:
    def __init__(self,robot,base,c,s,clock=time.monotonic,log=lambda event:None):
        self.robot=robot;self.base=base;self.c=c;self.s=s;self.clock=clock
        self.log=log
        self.pending=None;self.sent_at=0.;self.fixed_angles=None;self.last_angles=np.asarray(base).copy()
        self.last_state=None;self.timeline=PoseTimeline()
        self.fixed_roll=None
        self.frame_stamp=None
        self.fixed_alignment_x=None
        self.axis_recovery=None

    def freeze_direct_pose(self,angles,roll):
        """Adopt the accepted switch pose once, never the latest drifting pose."""
        if self.fixed_angles is not None:
            raise ServoError('V3.2 direct orientation reference already frozen')
        angles=np.asarray(angles,float).copy();roll=float(roll)
        if angles.shape!=(3,) or not np.isfinite([*angles,roll]).all():
            raise ServoError('V3.2 invalid direct orientation reference')
        # The switch must still originate from a valid visual-axis pose.
        measured_roll(self.base,angles,self.c['max_orientation_error_deg'])
        angles.setflags(write=False)
        self.fixed_angles=angles;self.fixed_roll=roll
        self.fixed_alignment_x=None
        self.frame_stamp=None
        self.log(dict(event='direct_orientation_reference',reference_angles_deg=angles.tolist(),
                      reference_roll_deg=roll,stop_angle_deg=self.s['direct_stop_angle_deg']))

    def pose_roll(self,angles):
        if self.fixed_angles is None:
            return measured_roll(self.base,angles,self.c['max_orientation_error_deg'])
        relative=(Rotation.from_euler('xyz',self.fixed_angles,degrees=True).inv()
                  * Rotation.from_euler('xyz',angles,degrees=True))
        error=float(np.degrees(relative.magnitude()))
        if error>self.s['direct_stop_angle_deg']:
            self.log(dict(event='direct_orientation_stop',reference_angles_deg=self.fixed_angles.tolist(),
                          actual_angles_deg=np.asarray(angles).tolist(),orientation_error_deg=error,
                          stop_angle_deg=self.s['direct_stop_angle_deg']))
            raise ServoError(f'V3.2 direct-mode orientation hard limit: error={error:.3f}deg, '
                             f'limit={self.s["direct_stop_angle_deg"]:.3f}deg')
        # Roll is only a compatibility coordinate in direct mode. Commands use
        # the full frozen orientation; DirectPoseLock handles bounded recovery.
        matrix=relative.as_matrix()
        return self.fixed_roll+float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))

    def _read_feedback(self):
        state=self.robot.get_current_state();now=self.clock()
        try:
            stamp=float(state['received_monotonic']);arm=state['right_arm']
            q=np.asarray(arm['position'],float)*1000;quat=np.asarray(arm['orientation'],float)
            if q.shape!=(3,) or quat.shape!=(4,) or not np.isfinite([*q,*quat,stamp]).all():raise ValueError('pose')
            if stamp>now:raise ValueError('future feedback')
            angles=Rotation.from_quat(np.r_[quat[1:],quat[0]]).as_euler('xyz',degrees=True)
            planner_status=status_value(state['curobo_status'])
            failures={3:'PLANNING_FAILED',7:'IK_FAILED',8:'COLLISION_CHECK_FAILED'}
            if planner_status in failures:
                raise ServoError(f'V3.2 planner/IK/collision failure: status={planner_status} ({failures[planner_status]})')
            if status_value(state['robot_status']) not in (0,1):raise ServoError('V3.2 robot unavailable')
            if now-stamp>self.s['feedback_max_age_seconds']:
                raise FeedbackStale('V3.2 stale robot feedback')
            source=float(state['pose_sample_monotonic'])
            completed=float(state['pose_fk_completed_monotonic'])
            if not np.isfinite([source,completed]).all() or not 0<source<=completed<=stamp:
                raise ValueError('invalid pose provenance; restart V3.2 planner and agent')
            if now-source>self.s['feedback_max_age_seconds']:
                raise FeedbackStale(f'V3.2 stale pose sample: source_age={now-source:.3f}s, packet_age={now-stamp:.3f}s')
        except ServoError:
            # ServoError inherits ValueError; preserve recoverable stale packets
            # and explicit controller faults instead of erasing their types.
            raise
        except (TypeError,ValueError,KeyError,IndexError) as error:
            raise ServoError('V3.2 invalid or stale robot feedback') from error
        self.last_state=state
        return dict(q=q,
                    angles=angles,received_monotonic=stamp,pose_sample_monotonic=source,
                    pose_fk_completed_monotonic=completed)

    def read(self):
        result=self._read_feedback()
        try:result['roll']=self.pose_roll(result['angles'])
        except AxisResidualExceeded as error:
            if self.axis_recovery is None:raise
            # Recovery never returns an out-of-bound pose as usable feedback.
            self.axis_recovery.recover(self,result,error)
            raise AssertionError('Axis recovery must invalidate the current update')
        self.timeline.add(result)
        return result

    def pair_frame(self,stamp,checkpoint=lambda:None,sleep=time.sleep):
        deadline=self.clock()+self.s['max_pose_pair_gap_seconds']
        while True:
            checkpoint();self.read()
            try:
                paired=self.timeline.pair(stamp,self.s)
                try:paired['roll']=self.pose_roll(paired['angles'])
                except AxisResidualExceeded:
                    if self.axis_recovery is None:raise
                    # This is an old camera-time pose. Current feedback above
                    # already passed; discard the image, do not hold for history.
                    raise StaleImage('Camera-time pose outside tool axis; recapture')
                return paired
            except PoseNotBracketed:
                if self.clock()>=deadline:raise
                sleep(.005)


    def ready(self):
        try:self.read()
        except FeedbackStale:
            # Skip this update; the next observation enters the bounded
            # recapture/recovery loop. Never use a stale pose to acknowledge.
            return False
        if self.pending is None:return True
        state=self.last_state
        processed=np.asarray(state.get('processed_arm_tool_command',[]),float)
        same=processed.shape==self.pending.shape and np.allclose(processed,self.pending,atol=1e-5,rtol=0)
        if state['received_monotonic']>self.sent_at and same and status_value(state['curobo_status']) in (4,9):
            self.log(dict(event='stream_target_acknowledged',elapsed_seconds=self.clock()-self.sent_at))
            self.pending=None;return True
        if self.clock()-self.sent_at>self.s['ack_timeout_seconds']:
            status=status_value(state['curobo_status'])
            self.log(dict(event='stream_ack_timeout',planner_status=status,target_matches=bool(same),
                          elapsed_seconds=self.clock()-self.sent_at,
                          requested=self.pending.tolist(),processed=processed.tolist()))
            raise ServoError(f'V3.2 planner acknowledgement timeout: status={status}, target_matches={same}')
        return False

    def send(self,q,roll):
        began=self.clock()
        try:current=self.read()
        except FeedbackStale:
            # A sample can expire between ready() and send(). Do not send a
            # target computed from it; resume only through a new observation.
            return False
        if self.pending is not None:raise ServoError('V3.2 pending target must be acknowledged first')
        q=np.asarray(q,float).copy()
        # read/ready/send are separated by perception and scheduling. X hold
        # always means hold at the latest pose, never an older frame's X.
        if self.fixed_alignment_x is None:
            q[0]=max(q[0],current['q'][0])
        else:
            if abs(current['q'][0]-self.fixed_alignment_x)>self.c['max_pose_error_mm']:
                raise ServoError('Baseline fixed X drift at send')
            q[0]=self.fixed_alignment_x
        # Independent send-side bound: observations and planning can lag.
        if np.linalg.norm(q-current['q'])>translation_budget(self.c,self.s)+.05:
            raise ServoError('V3.2 target exceeds measured lookahead')
        angles=self.fixed_angles if self.fixed_angles is not None else roll_orientation(self.base,roll)
        if abs(roll-current['roll'])>self.s['lookahead_roll_deg']+.1:raise ServoError('V3.2 angular lookahead exceeded')
        self.robot._v3_servo_position_tolerance_mm=self.c['max_pose_error_mm']
        if self.robot.master_control(right_arm=[*q,*angles],action='book_v3_1_stream',object_category='book',wait_for_completion=False) is not True:
            raise ServoError('V3.2 target send failed')
        self.last_angles=np.asarray(angles).copy()
        self.sent_at=self.clock();self.pending=np.asarray(self.robot.target_command.arm_tool_command,float).copy()
        self.log(dict(event='stream_target_sent',requested=self.pending.tolist(),
                      send_seconds=self.sent_at-began,
                      source_age_at_send_seconds=None if self.frame_stamp is None else self.sent_at-self.frame_stamp))
        return True

    def finish(self,q,roll):
        self.read()
        angles=self.fixed_angles if self.fixed_angles is not None else roll_orientation(self.base,roll)
        self.robot._v3_servo_position_tolerance_mm=self.c['max_pose_error_mm']
        result=self.robot.master_control(right_arm=[*q,*angles],action='book_v3_xyz_servo',object_category='book')
        self.pending=None;self.last_angles=np.asarray(angles).copy()
        return result is True

    def abort(self):
        # Best effort planner-mediated hold. If feedback is stale, stop sending;
        # the last finite lookahead target remains the bounded fallback, not a
        # guaranteed emergency brake. Never open/reset the gripper here.
        # A separate cancel action lets the planner hold its newest measured
        # joints without planning another Cartesian path or changing Z plane.
        state=self.last_state
        if state is None:raise ServoError('V3.2 no state for hold request')
        arm=state['right_arm'];q=np.asarray(arm['position'])*1000
        quat=np.asarray(arm['orientation'])
        angles=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz',degrees=True)
        self.pending=None
        if self.robot.master_control(right_arm=[*q,*angles],action='book_v3_1_hold',
                object_category='book',wait_for_completion=False) is not True:
            raise ServoError('V3.2 hold request send failed')


def capture_paired_frame(transport,camera,checkpoint,log,last_frame,*,clock,sleep,total_deadline):
    """Suspend target generation until a newly captured frame has valid support.

    No send/hold/finish occurs here: the last finite target may finish moving.
    Missing/wide brackets and stale samples discard the frame and retry with
    the same precision limits. Invalid feedback, faults and camera failures
    still propagate immediately.
    """
    recovery_started=None;recovery_deadline=None;attempts=0;last_seen=last_frame
    def check_budget():
        checkpoint()
        if clock()>=total_deadline:raise ServoError('V3.2 total time budget exceeded')
        if recovery_deadline is not None and clock()>=recovery_deadline:
            log(dict(event='pose_pair_recovery_timeout',attempts=attempts,
                     elapsed_seconds=clock()-recovery_started))
            raise ServoError('V3.2 pose pairing recovery timed out; no new motion targets sent during recovery')
    def checked():
        check_budget();transport.read()
    while True:
        check_budget();began=clock();stamp=None;before=None;after=None
        try:
            before=transport.read()
            if hasattr(camera,'require_pose_after') and transport.timeline.samples:
                camera.require_pose_after(transport.timeline.samples[0][0])
            frame=camera.capture(checked);capture_wait=clock()-began
            after=transport.read();stamp=float(frame['captured_at'])
            check_budget()
            if not np.isfinite(stamp) or stamp<=0 or stamp>clock():
                raise ServoError('V3.2 invalid image timestamp')
            if stamp<=last_seen or clock()-stamp>transport.s['frame_max_age_seconds']:
                raise StaleImage('V3.2 frame age exceeded or duplicate frame at capture')
            last_seen=stamp
            paired=transport.pair_frame(stamp,check_budget,sleep)
            check_budget()
            if clock()-stamp>transport.s['frame_max_age_seconds']:
                raise StaleImage('V3.2 frame age exceeded after pairing')
            if recovery_started is not None:
                log(dict(event='pose_pair_recovered',attempts=attempts,
                         elapsed_seconds=clock()-recovery_started,frame_stamp=stamp,
                         pair_gap_seconds=paired['pair_gap_seconds']))
            return frame,paired,before,after,capture_wait
        except (PoseNotBracketed,PoseSupportUnavailable,FeedbackStale) as error:
            attempts+=1
            history=transport.timeline.samples
            log(dict(event='pose_pair_failed',reason=str(error),frame_stamp=stamp,
                     pose_oldest_stamp=history[0][0] if history else None,
                     pose_latest_stamp=history[-1][0] if history else None,
                     capture_wait_seconds=clock()-began,
                     acquisition_motion_mm=float(np.linalg.norm(after['q']-before['q']))
                         if before is not None and after is not None else None,
                     support=getattr(error,'details',{})))
            if recovery_started is None:
                recovery_started=clock()
                recovery_deadline=recovery_started+transport.s['pair_recovery_timeout_seconds']
                log(dict(event='pose_pair_recovery_start',timeout_seconds=transport.s['pair_recovery_timeout_seconds']))
            # Discard this image. A genuinely newer packet must arrive before
            # requesting a new post-request frame, not reusing a cached packet.
            wait_after=clock()
            while True:
                check_budget()
                try:fresh=transport.read()
                except FeedbackStale:fresh=None
                if fresh is not None and fresh['pose_sample_monotonic']>wait_after:break
                sleep(.005)


def run_stream_session(robot,camera,lock,target,goal_x,base,c,s,blocking_move,checkpoint,log,directory,display,*,started,geometry_recovery=None,verification_retry_seconds=0.,verification_only=False):
    transport=StreamTransport(robot,base,c,s,clock=time.monotonic,log=log);count=0;last_frame=-np.inf
    recovery_count=0;scale_multiplier=1.;reference_z=None
    from cart_book_grasp_v3_gpt_recovery import RecoveryProgress,TrackingRecoveryRejected
    recovery_progress=RecoveryProgress(time.monotonic,log)
    from cart_book_grasp_v3_convergence import VisualConvergence
    convergence=VisualConvergence((c['tolerance_px'],c['angle_tolerance_deg']),s['stall_seconds'])
    old_tolerance=getattr(robot,'_v3_servo_position_tolerance_mm',None)
    image_recovery=ImageFreshnessRecovery(s,time.monotonic,getattr(robot,'operator_sleep',time.sleep),
                                         checkpoint,log,started+c['max_seconds'])
    axis_recovery=None
    if c.get('axis_recovery') is not None:
        from baseline_axis_recovery import AxisRecovery
        axis_recovery=AxisRecovery(c['axis_recovery'],c['max_orientation_error_deg'],
            image_recovery,checkpoint,getattr(robot,'operator_sleep',time.sleep),log)
        transport.axis_recovery=axis_recovery
    def observe():
        nonlocal count,last_frame,lock,recovery_count,scale_multiplier,reference_z
        recovery_progress.check()
        began=time.monotonic()
        frame,paired,before,after_capture,capture_wait=capture_paired_frame(
            transport,camera,image_recovery.check,log,last_frame,clock=time.monotonic,
            sleep=getattr(robot,'operator_sleep',time.sleep),total_deadline=started+c['max_seconds'])
        paired_at=time.monotonic()
        stamp=float(frame['captured_at'])
        if axis_recovery is not None:
            axis_recovery.validate_frame(stamp)
            if axis_recovery.pending_visual_reset:convergence.reset()
        last_frame=stamp
        from cart_book_grasp_v3_feature_centerline import CenterlineGeometryError,FeaturePointsExhausted
        geometry_error=None;recovered=False
        # Optional baseline archive: only frames admitted to tracking after the
        # existing freshness/pose pairing checks, including failed tracking inputs.
        if hasattr(log,'used_frame'):log.used_frame(frame)
        try:observation=lock.observe(frame['rgb'])
        except (CenterlineGeometryError,FeaturePointsExhausted) as error:
            if axis_recovery is not None and axis_recovery.pending_visual_reset:
                raise ServoError('Axis recovery requires valid fresh same-target tracking; forward cancelled') from error
            if isinstance(error,FeaturePointsExhausted):raise
            geometry_error=error
        finally:
            # A point-loss exception must not bypass frame freshness and turn
            # a delayed/invalid observation into nonvisual forward motion.
            image_recovery.validate(stamp,stage='perception')
        if geometry_error is None:
            reason=convergence.observe(float(paired['q'][0]),
                (float(observation['uv'][0]-target[0]),float(observation['angle_deg'])),stamp)
            if reason:
                log(dict(event='visual_divergence',reason=reason,frame_stamp=stamp,
                         inliers=observation['inliers']))
                geometry_error=CenterlineGeometryError(reason)
        if geometry_error is not None:
            if geometry_recovery is None:raise geometry_error
            from cart_book_grasp_v3_gpt_recovery import recover_tracking
            recovery_progress.begin(transport.read()['q'][0])
            def capture_recovery(check):
                nonlocal last_frame
                data=capture_paired_frame(transport,camera,check,log,last_frame,clock=time.monotonic,
                    sleep=getattr(robot,'operator_sleep',time.sleep),total_deadline=started+c['max_seconds'])
                last_frame=float(data[0]['captured_at'])
                return data
            if reference_z is None:reference_z=float(transport.read()['q'][2])
            old_scale=lock.last_scale
            def recovery_checkpoint():
                checkpoint();recovery_progress.check()
            with image_recovery.held_tracking_recovery():
                while True:
                    recovery_checkpoint()
                    recovery_count+=1
                    folder=directory/f'gpt_recovery_{recovery_count:02d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(folder/'failed_wrist.png'),cv2.cvtColor(frame['rgb'],cv2.COLOR_RGB2BGR))
                    log(dict(event='gpt_tracking_recovery_start',attempt=recovery_count,
                             consecutive_failures=recovery_progress.failures,reason=str(geometry_error)))
                    try:
                        lock,data,observation=recover_tracking(lock,transport,capture_recovery,geometry_recovery,
                            recovery_checkpoint,folder,started+c['max_seconds'],reference_z=reference_z,
                            sleep=getattr(robot,'operator_sleep',time.sleep),
                            verification_retry_seconds=verification_retry_seconds)
                        break
                    except TrackingRecoveryRejected as error:
                        recovery_progress.rejected(error)
            scale_multiplier*=old_scale
            frame,paired,before,after_capture,capture_wait=data
            stamp=float(frame['captured_at']);recovered=True
            convergence.reset()
            log(dict(event='gpt_tracking_recovered',attempt=recovery_count,inliers=observation['inliers'],
                     directory=str(folder),frame_stamp=stamp))
        observation['scale']*=scale_multiplier
        tracking_completed=time.monotonic()
        state=transport.read()
        if reference_z is None:reference_z=float(state['q'][2])
        # Increasing command lookahead must not widen the image latency horizon.
        motion_delta=state['q']-paired['q']
        motion_mm=float(np.linalg.norm(motion_delta))
        motion_deg=float(abs(state['roll']-paired['roll']))
        motion_limit=min(5.,translation_budget(c,s))
        if motion_mm>motion_limit or motion_deg>s['lookahead_roll_deg']:
            log(dict(event='frame_motion_discarded',frame_stamp=stamp,
                     frame_age_seconds=time.monotonic()-stamp,
                     delta_xyz_mm=motion_delta.tolist(),motion_mm=motion_mm,
                     roll_delta_deg=motion_deg,translation_limit_mm=motion_limit,
                     rotation_limit_deg=s['lookahead_roll_deg'],
                     policy='pause_target_updates_recapture'))
            # Reuse the bounded image recovery loop. Do not send a new target
            # or a terminal hold that would latch a fault while recapturing.
            raise ImageMotionExpired(
                f'V3.2 frame-to-command motion exceeds prediction horizon: '
                f'{motion_mm:.3f}mm/{motion_limit:.3f}mm, '
                f'{motion_deg:.3f}deg/{s["lookahead_roll_deg"]:.3f}deg',
                tracking_recovered=recovered)
        error=np.array([observation['uv'][0]-target[0],observation['angle_deg']])
        if recovered:
            recovery_progress.check()
            recovery_progress.verified(state['q'][0],error,stamp)
        else:
            recovery_progress.observe(state['q'][0],error,stamp)
        if hasattr(log,'preview'):
            try:log.preview(frame['rgb'],observation,target,stamp=stamp)
            except Exception:pass # Visualization must never stop target generation.
        if count % c['snapshot_every']==0:
            path=directory/f'stream_{count:05d}.png'
            if hasattr(log,'snapshot'):
                log.snapshot(frame['rgb'],observation,target,path,display)
            else:
                image=draw_points(frame['rgb'],observation,target)
                ends=np.rint(observation['line']).astype(int)
                cv2.line(image,tuple(ends[0]),tuple(ends[1]),(0,220,100),2)
                cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                if display is not None:display.safe('log_image','book/images/spine',image)
        count+=1
        log(dict(event='stream_observation',inliers=observation['inliers'],xyz_mm=state['q'].tolist(),
                 image_xyz_mm=paired['q'].tolist(),image_roll_deg=paired['roll'],frame_stamp=stamp,
                 pose_before_stamp=paired['pose_before_stamp'],pose_after_stamp=paired['pose_after_stamp'],
                 pair_gap_seconds=paired['pair_gap_seconds'],pair_motion_bound_mm=paired['pair_motion_bound_mm'],
                 pair_angle_bound_deg=paired['pair_angle_bound_deg'],
                 timestamp_basis='host_read_completion_panel_receipt',
                 feedback_packet_stamp=state['received_monotonic'],
                 pose_sample_stamp=state['pose_sample_monotonic'],
                 pose_fk_completed_stamp=state['pose_fk_completed_monotonic'],
                 pose_age_seconds=time.monotonic()-state['pose_sample_monotonic'],
                 acquisition_motion_mm=float(np.linalg.norm(after_capture['q']-before['q'])),
                 capture_wait_seconds=capture_wait,
                 capture_timing=frame.get('capture_timing',{}),
                 capture_pair_checks_seconds=paired_at-began,
                 tracking_seconds=observation.get('inference_seconds'),
                 post_tracking_seconds=time.monotonic()-tracking_completed,
                 observation_seconds=time.monotonic()-began,
                 frame_to_command_mm=float(np.linalg.norm(state['q']-paired['q'])),
                 error_px=float(observation['uv'][0]-target[0]),angle_deg=float(observation['angle_deg'])))
        transport.frame_stamp=stamp
        axis_recovered=False;held_path=0.
        if axis_recovery is not None:
            axis_recovered=axis_recovery.consume_visual_reset()
            held_path=axis_recovery.recovery_path_mm
            axis_recovery.recovery_path_mm=0.
        return dict(state,image_q=paired['q'],image_roll=paired['roll'],
                    error=np.array([observation['uv'][0]-target[0],observation['angle_deg']]),
                    scale=observation['scale'],stamp=stamp,tracking_recovered=recovered or axis_recovered,
                    held_path_mm=held_path)
    def event(e):
        if e['event']=='baseline_alignment_start':
            transport.fixed_alignment_x=float(e['fixed_x_mm'])
        if e['event']=='jacobian_identification_start':
            convergence.active=False;convergence.reset()
        elif e['event'] in ('stream_start','jacobian_y_roll'):
            convergence.active=True;convergence.reset()
        if e['event'] in ('point_loss_direct_switch','aligned_direct_switch','visual_failure_direct_switch'):
            transport.freeze_direct_pose(e['fixed_angles_deg'],e['fixed_roll_deg'])
        log(e)
    try:
        result=run_stream(observe,transport.read,blocking_move,transport.send,transport.ready,transport.finish,
            goal_x,c,s,checkpoint=checkpoint,log=event,started=started,image_recovery=image_recovery,
            **({'verify_only':True} if verification_only else {}))
        result['final_angles_deg']=transport.last_angles.tolist()
        return result
    except BaseException:
        try:transport.abort()
        except BaseException as hold_error:
            log(dict(event='stream_hold_unconfirmed',reason=str(hold_error)))
        raise
    finally:
        if old_tolerance is None:
            if hasattr(robot,'_v3_servo_position_tolerance_mm'):del robot._v3_servo_position_tolerance_mm
        else:robot._v3_servo_position_tolerance_mm=old_tolerance
