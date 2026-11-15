"""Bounded rolling Cartesian targets, with no intermediate arrival waits.

The dedicated adapter follows visual targets with CPU differential joint servo.
Only one unacknowledged target may exist; a target lease holds on lost updates.
"""
import json
import time
from pathlib import Path
import numpy as np
from cart_book_grasp_v3_servo import ServoError
from cart_book_grasp_v3_feature_centerline import FeaturePointsExhausted,CenterlineGeometryError
from cart_book_grasp_v3_image_freshness import ImageFreshnessRecovery,StaleImage,ImageRecoveryTimedOut,PoseRecoveryRetry

class VisualJacobianError(ServoError):
    """Measured visual excitation/response cannot identify the Jacobian."""


CONFIG=Path(__file__).with_name('cart_book_grasp_v3_2_stream.json')
from cart_book_grasp_v3_2_lock import DirectPoseLock
from cart_book_grasp_v3_1_stream import DEFAULTS as BASE_DEFAULTS

DEFAULTS=dict(BASE_DEFAULTS, runtime_version="3.2", direct_recovery_seconds=2.0,
              direct_recovery_y_mm=1.0, direct_recovery_angle_deg=0.5,
              direct_stop_y_mm=4.0, direct_stop_angle_deg=2.0)

def load_stream_config(path=CONFIG):
    c=json.loads(Path(path).read_text())
    return validate_stream_config(c)


def validate_stream_config(c):
    if set(c)!=set(DEFAULTS) or c['runtime_version']!='3.2':
        raise ValueError('Invalid V3.2 stream config fields/version')
    limits=dict(gain=(.05,5.),update_hz=(1,20),lookahead_mm=(2.5,100.),
                lookahead_roll_deg=(.1,3),max_z_drift_mm=(2.1,4.),
                feedback_max_age_seconds=(.05,.5),frame_max_age_seconds=(.05,.5),
                ack_timeout_seconds=(.5,5),stall_seconds=(2,15),
                pair_recovery_timeout_seconds=(.2,3.),max_pose_pair_gap_seconds=(.02,.2),
                max_pose_pair_uncertainty_mm=(.1,3),max_pose_pair_uncertainty_deg=(.1,1))
    for key,(lo,hi) in limits.items():
        if key=='lookahead_mm' and c[key] is None:continue
        if isinstance(c[key],bool) or not isinstance(c[key],(int,float)) or not lo<=c[key]<=hi:
            raise ValueError('Invalid V3.2 setting: '+key)
    limits=dict(direct_recovery_seconds=(.2,2.),direct_recovery_y_mm=(.5,1.),
                direct_recovery_angle_deg=(.2,.5),direct_stop_y_mm=(2.,4.),
                direct_stop_angle_deg=(1.,2.))
    for key,(lo,hi) in limits.items():
        if isinstance(c[key],bool) or not isinstance(c[key],(int,float)) or not lo<=c[key]<=hi:
            raise ValueError('Invalid V3.2 setting: '+key)
    return c


def path_budget_exceeded(used, requested, limit):
    """Allow only floating-point roundoff, not a physical motion tolerance."""
    if not np.isfinite([used,requested,limit]).all() or min(used,requested,limit)<0:
        raise ServoError('Invalid path budget measurement')
    roundoff=64*np.spacing(max(1.,float(used),float(requested),float(limit)))
    return float(used)+float(requested)>float(limit)+roundoff


def translation_budget(c,s,path_mm=0.):
    """None removes the per-target cap; the remaining path budget stays finite."""
    remaining=float(c['max_path_mm'])-float(path_mm)
    configured=s['lookahead_mm']
    budget=remaining if configured is None else min(float(configured),remaining)
    if not np.isfinite(budget) or path_budget_exceeded(path_mm,0.,c['max_path_mm']):
        raise ServoError('V3.2 path budget exhausted')
    # Zero remaining travel can still confirm an already reached exact goal.
    # It cannot authorize a nonzero move or an unaligned correction.
    return max(0.,budget)


def run_stream(observe,read,blocking_move,send,ready,finish,goal_x,c,s,*,
               checkpoint=lambda:None,log=lambda e:None,clock=time.monotonic,sleep=time.sleep,started=None,
               image_recovery=None,align_then_straight=False,visual_failure_straight=False,
               direct_update_hz=None,handoff_only=False,verify_only=False):
    """Injected callbacks allow a moving-arm simulator without hardware imports.

    State: q (mm), roll (deg). Observation additionally supplies error [px,deg],
    scale and fresh frame stamp. send is asynchronous; ready means planner ACK,
    not arrival. Blocking moves are restricted to two identification probes.
    """
    if visual_failure_straight and not align_then_straight:
        raise ValueError("Visual failure straight policy is baseline-only")
    if direct_update_hz is not None:
        if (not align_then_straight or isinstance(direct_update_hz,bool)
                or not np.isfinite(direct_update_hz) or not 1 <= direct_update_hz <= 20):
            raise ValueError('Continuous direct update rate requires baseline and 1..20Hz')
    if (handoff_only or verify_only) and not align_then_straight:
        raise ValueError("Planned handoff is baseline-only")
    visual_failure_reason=None
    started=clock() if started is None else started
    if image_recovery is None:
        image_recovery=ImageFreshnessRecovery(s,clock,sleep,checkpoint,log,started+c['max_seconds'])
    last_reliable_error=None;approximate_alignment=False
    origin=None;previous=None;path=0.;stamp=-np.inf;direct=False;fixed=None;jac=None;reference_scale=1.
    last_sent=-np.inf;last_progress=clock();progress_q=None;progress_r=0.;final_samples=0
    recovering_z=False;recovery_started=None;pose_lock=None
    advancing=False;aligned_samples=0;furthest_x=None
    smooth=s.get('forward_lead_mm') is not None
    lead_policy=None
    if smooth:
        from cart_book_grasp_v3_4_policy import ForwardLead
        lead_policy=ForwardLead(s)
    z_stable_since=None;z_stable_stamp=-np.inf;z_stable_samples=0
    path_parts=dict(visual_mm=0.,direct_mm=0.,recovery_mm=0.)
    budget_logged=-np.inf
    tolerance=np.array([c['tolerance_px'],c['angle_tolerance_deg']])
    if s['max_z_drift_mm']<=c['max_pose_error_mm']:
        raise ServoError('V3.2 motion Z limit must exceed arrival tolerance')
    def fallback_to_straight(error):
        nonlocal direct,fixed,origin,pose_lock,visual_failure_reason
        nonlocal recovering_z,recovery_started,aligned_samples,final_samples,last_progress,progress_q
        checkpoint()
        o=read() # transport verifies real feedback freshness; failures propagate.
        q=np.asarray(o['q'],float);angles=np.asarray(o['angles'],float);roll=float(o['roll'])
        if q.shape!=(3,) or angles.shape!=(3,) or not np.isfinite([*q,*angles,roll]).all():
            raise ServoError('Invalid measured pose for visual fallback')
        if origin is not None:
            if abs(q[2]-origin[2])>s['max_z_drift_mm'] or abs(q[0]-origin[0])>c['max_pose_error_mm']:
                raise ServoError('Visual fallback cannot hide alignment pose drift')
        start=q if origin is None else origin
        if not -c['x_tolerance_mm']<=goal_x-start[0]<=c['max_forward_mm']:
            raise ServoError('Visual fallback endpoint outside approach budget')
        consumed=path+(float(np.linalg.norm(q-previous)) if previous is not None else 0.)
        if path_budget_exceeded(consumed,abs(goal_x-q[0]),c['max_path_mm']):
            raise ServoError('Visual fallback exceeds remaining path budget')
        fixed=(q.copy(),roll);pose_lock=DirectPoseLock(o,q,s,c,clock,log)
        visual_failure_reason=f'{type(error).__name__}: {error}'
        # Event freezes the exact full orientation in StreamTransport before hold.
        log(dict(event='visual_failure_direct_switch',reason=visual_failure_reason,
                 fixed_xyz_mm=q.tolist(),fixed_angles_deg=angles.tolist(),fixed_roll_deg=roll,
                 goal_x_mm=float(goal_x),alignment_verified=False,
                 policy='fixed_Y_Z_full_orientation_X_only'))
        if not finish(q,roll):raise ServoError('Visual fallback hold not confirmed')
        held=read();pose_lock.check(held,recover=False)
        if abs(held['q'][2]-q[2])>s['max_z_drift_mm']:
            raise ServoError('Visual fallback hold Z drift')
        direct=True
        if origin is None:origin=q.copy()
        origin[2]=q[2]
        recovering_z=False;recovery_started=None;aligned_samples=0;final_samples=0
        last_progress=clock();progress_q=None
        if smooth:lead_policy.update(clock(),0.,False)
        return held

    def _sample(visual=True):
        nonlocal origin,previous,path,stamp,direct,fixed,progress_q,progress_r,last_progress
        nonlocal last_reliable_error,approximate_alignment
        nonlocal recovering_z,recovery_started,pose_lock
        nonlocal z_stable_since,z_stable_stamp,z_stable_samples
        nonlocal advancing,aligned_samples,final_samples,last_sent
        checkpoint()
        if clock()-started>c['max_seconds']:raise ServoError('V3.2 total time budget exceeded')
        try:o=image_recovery.observe(observe,stamp) if visual and not direct else read()
        except (FeaturePointsExhausted,CenterlineGeometryError,ImageRecoveryTimedOut) as error:
            if visual_failure_straight and not direct:
                o=fallback_to_straight(error)
            else:
                if not isinstance(error,FeaturePointsExhausted):raise
                if align_then_straight:
                    accepted=(last_reliable_error is not None and
                              bool((np.abs(last_reliable_error)<=np.array([20.,10.])).all()))
                    log(dict(event='baseline_point_loss_decision',accepted=accepted,
                             last_reliable_error=None if last_reliable_error is None else last_reliable_error.tolist(),
                             limits=[20.,10.],remaining_points=error.count))
                    if not accepted:
                        raise ServoError('Baseline lost spine features before confirmed alignment; forward cancelled') from error
                    approximate_alignment=True
                o=read()
                # Losing features must not hide an already out-of-bound height by
                # silently resetting the plane to the drifting pose.
                if origin is not None and abs(o['q'][2]-origin[2])>s['max_z_drift_mm']:
                    raise ServoError('V3.2 fixed Z drift before point-loss hold')
                direct=True;fixed=(o['q'].copy(),float(o['roll']))
                if origin is None:origin=o['q'].copy()
                if align_then_straight:
                    origin[2]=fixed[0][2]
                    recovering_z=False;recovery_started=None
                else:fixed[0][2]=origin[2]
                log(dict(event='point_loss_direct_switch',remaining_points=error.count,minimum_points=error.minimum,
                         fixed_xyz_mm=fixed[0].tolist(),actual_xyz_mm=o['q'].tolist(),
                         fixed_roll_deg=float(o['roll']),fixed_angles_deg=o['angles'].tolist()))
                pose_lock=DirectPoseLock(o,fixed[0],s,c,clock,log)
                if not finish(fixed[0],o['roll']):raise ServoError('V3.2 point-loss hold not confirmed')
                o=read()  # Never rebase Y/roll after settling: transport holds the same pre-hold pose.
                if smooth:lead_policy.update(clock(),0.,False)
                log(dict(event='direct_reference_locked',reference_mm=fixed[0].tolist(),
                         reference_angles_deg=pose_lock.angles.tolist(),reference_roll_deg=fixed[1]))
        if o.get('tracking_recovered',False):
            # GPT runs only during a confirmed hold. Restart alignment progress,
            # not the fixed plane, endpoint, measured path or total-time budget.
            advancing=False;aligned_samples=0;final_samples=0;last_sent=-np.inf
            last_progress=clock();progress_q=None
            if smooth:lead_policy.update(clock(),0.,False)
        q=np.asarray(o['q'],float);r=float(o['roll'])
        if align_then_straight and not direct and origin is not None and abs(q[0]-origin[0])>c['max_pose_error_mm']:
            raise ServoError('Baseline fixed X drift during alignment')
        if q.shape!=(3,) or not np.isfinite([*q,r]).all():raise ServoError('Invalid V3.2 pose')
        if origin is None:origin=q.copy()
        if previous is not None:
            distance=float(np.linalg.norm(q-previous));path+=distance
            part='recovery_mm' if recovering_z or (direct and pose_lock.active) else 'direct_mm' if direct else 'visual_mm'
            path_parts[part]+=distance
        # Conservatively include measured hold travel, without resetting the
        # original path budget when a recovery hides intermediate samples.
        held_path=float(o.get('held_path_mm',0.))
        if not np.isfinite(held_path) or held_path<0:raise ServoError('Invalid held path')
        path+=held_path;path_parts['recovery_mm']+=held_path
        previous=q.copy()
        if path_budget_exceeded(path,0.,c['max_path_mm']):raise ServoError('V3.2 measured path budget exceeded')
        if not origin[0]-c['max_pose_error_mm']<=q[0]<=goal_x+c['x_tolerance_mm']:
            raise ServoError('V3.2 X outside endpoint interval')
        reference_z=origin[2]
        z_error=float(q[2]-reference_z)
        diagnostic=dict(reference_z_mm=float(reference_z),actual_z_mm=float(q[2]),
                        z_error_mm=z_error,motion_limit_mm=s['max_z_drift_mm'],
                        arrival_tolerance_mm=c['max_pose_error_mm'])
        if abs(z_error)>s['max_z_drift_mm']:
            log(dict(event='z_drift_stop',**diagnostic))
            raise ServoError(f"V3.2 fixed Z drift: reference={reference_z:.3f}mm, actual={q[2]:.3f}mm, "
                             f"delta={z_error:+.3f}mm, motion_limit={s['max_z_drift_mm']:.3f}mm")
        if not (align_then_straight and direct) and not recovering_z and abs(z_error)>min(2.,c['max_pose_error_mm']):
            recovering_z=True;recovery_started=clock()
            z_stable_since=None;z_stable_samples=0;z_stable_stamp=-np.inf
            log(dict(event='z_recovery_start',**diagnostic))
        elif recovering_z:
            inside=abs(z_error)<=.75*min(2.,c['max_pose_error_mm'])
            feedback_stamp=float(o.get('pose_sample_monotonic',clock()))
            if not inside:
                z_stable_since=None;z_stable_samples=0
            elif feedback_stamp>z_stable_stamp:
                if z_stable_since is None:z_stable_since=feedback_stamp
                z_stable_samples+=1
            z_stable_stamp=max(z_stable_stamp,feedback_stamp)
            settled=inside and (not smooth or (z_stable_samples>=c['stable_frames'] and
                feedback_stamp-z_stable_since>=s['recovery_stable_seconds']))
            if settled:
                recovering_z=False;recovery_started=None
                log(dict(event='z_recovery_complete',**diagnostic))
        if recovering_z and clock()-recovery_started>s['stall_seconds']:
            raise ServoError(f'V3.2 Z recovery timed out: delta={z_error:+.3f}mm')
        if not align_then_straight and abs(r)>c['max_roll_deg']+c['max_orientation_error_deg']:raise ServoError('V3.2 roll bound')
        if direct:
            pose_lock.check(o,recover=not align_then_straight)
        else:
            t=float(o['stamp'])
            stamp=t
            if not np.isfinite([*o['error'],o['scale']]).all() or not .1<=o['scale']<=20:
                if visual_failure_straight:
                    fallback_to_straight(VisualJacobianError('Invalid visual measurement'))
                    return sample(False)
                raise ServoError('V3.2 invalid visual measurement')
            last_reliable_error=np.asarray(o['error'],float).copy()
        if progress_q is None or np.linalg.norm(q-progress_q)>.3 or abs(r-progress_r)>.2:
            progress_q=q.copy();progress_r=r;last_progress=clock()
        if clock()-last_progress>s['stall_seconds']:
            raise ServoError(f'V3.2 no measured progress: elapsed_s={clock()-last_progress:.2f}, '
                             f'actual_mm={np.round(q,3).tolist()}, goal_x_mm={goal_x:.3f}, '
                             f'reference_z_mm={origin[2]:.3f}, direct={direct}')
        return o
    def sample(visual=True):
        nonlocal last_reliable_error
        while True:
            try:return _sample(visual)
            except PoseRecoveryRetry:
                # Recovery may interrupt a fallback's fresh pose read before
                # any direct-mode switch. Retry visual observation, never the
                # fallback transition or its stale held pose.
                last_reliable_error=None
                visual=True
                log(dict(event='axis_recovery_observation_discarded'))

    def handoff_result(observation):
        checkpoint()
        held=read()
        if not finish(held['q'],held['roll']):
            raise ServoError('Baseline planned handoff hold not confirmed')
        held=read()
        verified=visual_failure_reason is None and not approximate_alignment
        result=dict(completion_mode='alignment_handoff',alignment_verified=verified,
                    visual_failure_reason=visual_failure_reason,
                    approximate_alignment=approximate_alignment,
                    verified_error=None if last_reliable_error is None else last_reliable_error.tolist(),
                    frame_stamp=None if not np.isfinite(stamp) else float(stamp),
                    held_xyz_mm=np.asarray(held['q']).tolist(),
                    held_angles_deg=np.asarray(held['angles']).tolist(),goal_x_mm=float(goal_x),
                    path_mm=float(path),forward_completed=False)
        log(dict(event='baseline_alignment_handoff',**result))
        return result
    o=sample()
    if verify_only:
        if direct or last_reliable_error is None or (np.abs(last_reliable_error)>tolerance).any():
            raise ServoError('Baseline post-reshape fresh visual verification failed')
        return handoff_result(o)
    if handoff_only and direct:
        return handoff_result(o)
    if align_then_straight and not direct:log(dict(event='baseline_alignment_start',fixed_x_mm=float(origin[0])))
    if not np.isfinite(goal_x) or not -c['x_tolerance_mm']<=goal_x-o['q'][0]<=c['max_forward_mm']:
        raise ServoError('V3.2 endpoint outside approach budget')
    # Identify with actual settled displacements before permitting moving-camera feedback.
    def _identify(observation):
        nonlocal direct
        o=observation;jac=None;reference_scale=1.
        log(dict(event='jacobian_identification_start'))
        for identification_attempt in range(3):
            if not direct:
                df=[];dq=[];restart_identification=False
                for axis in (0,1):
                    try:image_recovery.validate(float(o['stamp']),stage='before_probe')
                    except StaleImage as error:
                        image_recovery.discard(error);o=sample()
                        if direct:break
                        if o.get('tracking_recovered',False):
                            restart_identification=True;break
                    old=o;goal=old['q'].copy();roll=float(old['roll'])
                    goal[2]=origin[2]
                    if align_then_straight:goal[0]=origin[0]
                    if axis==0:goal[1]+=c['probe_mm']
                    else:roll+=c['probe_roll_deg'] if align_then_straight or roll+c['probe_roll_deg']<=c['max_roll_deg'] else -c['probe_roll_deg']
                    if path_budget_exceeded(path,np.linalg.norm(goal-old['q']),c['max_path_mm']):raise ServoError('V3.2 probe path budget')
                    if not blocking_move(goal,roll):raise ServoError('V3.2 probe not confirmed')
                    o=sample()
                    if direct:break
                    if o.get('tracking_recovered',False):
                        restart_identification=True;break
                    dq.append([o.get('image_q',o['q'])[1]-old.get('image_q',old['q'])[1],
                               o.get('image_roll',o['roll'])-old.get('image_roll',old['roll'])]);df.append(o['error']-old['error'])
                if restart_identification:continue
                if not direct:
                    d=np.array(dq).T
                    if np.linalg.svd(d,compute_uv=False).min()<.5:raise VisualJacobianError('V3.2 probe displacement insufficient')
                    jac=np.array(df).T@np.linalg.inv(d);reference_scale=o['scale']
                    if not np.isfinite(jac).all() or np.linalg.cond(jac/tolerance[:,None])>40 or abs(jac[0,0])<.1 or abs(jac[1,1])<.1:
                        raise VisualJacobianError('V3.2 invalid Y/roll response')
                    log(dict(event='jacobian_y_roll',matrix=jac.tolist()))
            break
        else:
            raise ServoError('GPT recovery repeatedly interrupted Jacobian identification')
        return o,jac,reference_scale

    def identify(observation):
        try:return _identify(observation)
        except (VisualJacobianError,ImageRecoveryTimedOut) as error:
            if not visual_failure_straight:raise
            return fallback_to_straight(error),None,1.

    o,jac,reference_scale=identify(o)

    log(dict(event='stream_start',gain=s['gain'],update_hz=s['update_hz'],lookahead_mm=s['lookahead_mm']))
    # Time, travel, freshness and progress bound the loop; perception ticks are
    # not the old max_iterations waypoint count.
    while True:
        try:
            if o.get('tracking_recovered',False) and not direct:
                o,jac,reference_scale=identify(o)
            # Camera-time poses are for image pairing only. Rebase the command on
            # fresh feedback so a correction cannot ask to return to an old X.
            if handoff_only and direct:
                return handoff_result(o)
            fresh=sample(False) if direct else read()
            q=fresh['q'];r=fresh['roll'];remaining=goal_x-q[0]
            if smooth and clock()-budget_logged>=.5:
                minimum=float(np.linalg.norm(np.array([remaining,(fixed[0][1]-q[1]) if direct else 0.,origin[2]-q[2]])))
                log(dict(event='path_budget_status',used_mm=path,remaining_mm=c['max_path_mm']-path,
                         minimum_to_endpoint_mm=minimum,minimum_total_mm=path+minimum,
                         endpoint_reserve_sufficient=not path_budget_exceeded(path,minimum,c['max_path_mm']),
                         measured_x_mm=float(q[0]),goal_x_mm=float(goal_x),**path_parts))
                budget_logged=clock()
            recovering_pose=direct and pose_lock.active
            furthest_x=float(q[0]) if furthest_x is None else max(furthest_x,float(q[0]))
            lookahead=translation_budget(c,s,path)
            if not direct:
                effective=jac.copy();effective[:,0]*=o['scale']/reference_scale
                # Features belong to the paired image pose, not the later command
                # pose. Predict the small measured Y/roll change since that image.
                delta=np.array([q[1]-o.get('image_q',o['q'])[1],r-o.get('image_roll',o['roll'])])
                error=o['error']+effective@delta
            aligned=(not recovering_z and not recovering_pose) and (direct or bool((np.abs(error)<=tolerance).all()))
            if lookahead==0 and not (aligned and abs(remaining)<=c['x_tolerance_mm']):
                raise ServoError('V3.2 path budget exhausted before alignment/endpoint')
            aligned_samples=aligned_samples+1 if aligned else 0
            if align_then_straight and not direct and aligned_samples>=c['stable_frames'] and ready():
                # Confirm the stationary alignment on a NEW image before freezing
                # the measured full orientation. No forward target exists yet.
                try:image_recovery.validate(float(o['stamp']),stage='before_alignment_hold')
                except ImageRecoveryTimedOut as failure:
                    if not visual_failure_straight:raise
                    o=fallback_to_straight(failure);continue
                except StaleImage as failure:
                    image_recovery.discard(failure);o=sample();continue
                hold=q.copy();hold[0]=origin[0];hold[2]=origin[2]
                if path_budget_exceeded(path,np.linalg.norm(hold-q),c['max_path_mm']):
                    raise ServoError('Baseline alignment hold exceeds path budget')
                if not finish(hold,r):raise ServoError('Baseline alignment hold not confirmed')
                verified=sample()
                if direct:
                    o=verified;continue # Point-loss fallback already froze the pose.
                if verified.get('tracking_recovered',False):
                    aligned_samples=0;o=verified;continue
                if (np.abs(verified['error'])>tolerance).any():
                    aligned_samples=0;o=verified;continue
                settled=read()
                if np.linalg.norm(settled['q']-verified['q'])>c['max_pose_error_mm'] or abs(settled['roll']-verified['roll'])>c['max_orientation_error_deg']:
                    raise ServoError('Baseline pose changed after alignment verification')
                fixed=(settled['q'].copy(),float(settled['roll']))
                origin[2]=fixed[0][2]
                pose_lock=DirectPoseLock(settled,fixed[0],s,c,clock,log)
                log(dict(event='aligned_direct_switch',fixed_xyz_mm=fixed[0].tolist(),
                         fixed_angles_deg=settled['angles'].tolist(),fixed_roll_deg=fixed[1],
                         verified_error=verified['error'].tolist(),goal_x_mm=float(goal_x),
                         policy='X_only_fixed_Y_Z_full_orientation'))
                direct=True;aligned_samples=0;final_samples=0
                recovering_z=False;recovery_started=None
                log(dict(event="baseline_straight_advance",policy="fixed_Y_Z_orientation_no_recovery",hard_pose_limits_retained=True))
                last_progress=clock();progress_q=None
                o=sample(False)
                continue
            old_phase=advancing
            if recovering_z or recovering_pose or (not direct and (np.abs(error)>1.5*tolerance).any()):
                advancing=False
            elif direct or (not align_then_straight and aligned_samples>=c['stable_frames']):
                advancing=True
            if old_phase!=advancing:
                log(dict(event='stream_phase',phase='advance' if advancing else 'align',
                         measured_x_mm=float(q[0]),error=None if direct else error.tolist()))
            can_send=ready()
            if not direct:
                # Feedback/ACK reads can consume the remaining image lifetime.
                # Recapture before either a rolling target or final arrival command.
                try:image_recovery.validate(float(o['stamp']),stage='before_target')
                except ImageRecoveryTimedOut as failure:
                    if not visual_failure_straight:raise
                    o=fallback_to_straight(failure);continue
                except StaleImage as error:
                    image_recovery.discard(error);o=sample();continue
            if aligned and abs(remaining)<=c['x_tolerance_mm'] and can_send:
                final_samples+=1
                if final_samples>=c['stable_frames']:
                    goal=(fixed[0].copy() if direct else q.copy());goal[0]=goal_x
                    if not direct:goal[2]=origin[2]
                    roll=fixed[1] if direct else r
                    if path_budget_exceeded(path,np.linalg.norm(goal-q),path+lookahead):raise ServoError('V3.2 final target beyond lookahead')
                    if path_budget_exceeded(path,np.linalg.norm(goal-q),c['max_path_mm']):raise ServoError('V3.2 final path budget')
                    if not finish(goal,roll):raise ServoError('V3.2 final arrival not confirmed')
                    # Validate a new image after final settling before allowing the gripper.
                    o=sample()
                    if o.get('tracking_recovered',False):
                        final_samples=0;continue
                    if recovering_z or (direct and pose_lock.active):
                        final_samples=0;continue
                    if not direct and not (np.abs(o['error'])<=tolerance).all():
                        final_samples=0;continue
                    return dict(converged=not direct,distance_complete=True,visual_alignment_verified=not direct,
                                alignment_verified_before_advance=align_then_straight and direct and not approximate_alignment and visual_failure_reason is None,
                                approximate_alignment_accepted=approximate_alignment,
                                visual_failure_reason=visual_failure_reason,
                                completion_mode='visual_failure_then_straight' if visual_failure_reason is not None else (('approximate_point_loss_then_straight' if approximate_alignment else 'aligned_then_straight') if align_then_straight else 'direct_distance_after_point_loss') if direct else 'stream_visual',path_mm=path)
            else:final_samples=0
            if smooth:
                z_ratio=abs(q[2]-origin[2])/min(2.,c['max_pose_error_mm'])
                if direct:
                    from scipy.spatial.transform import Rotation
                    angle=float(np.rad2deg((Rotation.from_euler('xyz',fresh['angles'],degrees=True)*pose_lock.rotation.inv()).magnitude()))
                    ratios=[z_ratio,abs(q[1]-fixed[0][1])/s['direct_recovery_y_mm'],angle/s['direct_recovery_angle_deg']]
                else:
                    ratios=[z_ratio,*list(np.maximum(0.,np.abs(error)/tolerance-1.)/.5)]
                if align_then_straight and direct:ratios=[0.]
                forward_lead=lead_policy.update(clock(),max(ratios),advancing)
            update_hz=direct_update_hz if direct and direct_update_hz is not None else s['update_hz']
            if can_send and clock()-last_sent>=1/update_hz and not (aligned and abs(remaining)<=c['x_tolerance_mm']):
                goal=q.copy();goal[0]=min(goal_x,q[0] if recovering_pose or recovering_z else max(q[0],furthest_x))
                goal[2]=fixed[0][2] if direct else origin[2];roll=r
                if align_then_straight and not direct:goal[0]=origin[0]
                if recovering_z and not direct:
                    # Stop advancing X/Y/roll while returning toward the same TCP plane.
                    goal[2]=q[2]+np.clip(goal[2]-q[2],-lookahead,lookahead)
                elif direct:
                    goal[1]=fixed[0][1];roll=fixed[1]
                else:
                    a=effective/tolerance[:,None]
                    # Do not amplify already-acceptable residuals by gain=5; that
                    # otherwise knocks alignment back out of tolerance every tick.
                    b=np.where(np.abs(error)>tolerance,error,0.)/tolerance
                    correction=-s['gain']*np.linalg.solve(a.T@a+.02*np.eye(2),a.T@b)
                    correction/=max(1.,abs(correction[0])/lookahead,abs(correction[1])/s['lookahead_roll_deg'])
                    goal[1]+=correction[0];roll+=correction[1]
                lateral=float(np.linalg.norm(goal[1:]-q[1:]))
                if lateral>lookahead:
                    # Reserve the Z return first; scaling both axes changes the fixed plane.
                    y_budget=np.sqrt(max(0.,lookahead**2-(goal[2]-q[2])**2))
                    goal[1]=q[1]+np.clip(goal[1]-q[1],-y_budget,y_budget)
                    lateral=float(np.linalg.norm(goal[1:]-q[1:]))
                if advancing and remaining>c['x_tolerance_mm']:
                    # RAW filtering/feedback lag can carry the arm past the final
                    # endpoint. Reduce the remaining target lead over the last
                    # 20 mm; do not relax the measured endpoint boundary.
                    forward=remaining if remaining>20. else .35*remaining
                    if smooth:forward=min(forward,forward_lead)
                    goal[0]=q[0]+min(forward,np.sqrt(max(0.,lookahead**2-lateral**2)))
                if not align_then_straight and abs(roll)>c['max_roll_deg']:raise ServoError('V3.2 commanded roll bound')
                if path_budget_exceeded(path,np.linalg.norm(goal-q),c['max_path_mm']):
                    raise ServoError(f'V3.2 path budget: used_mm={path:.9f}, '
                                     f'requested_mm={np.linalg.norm(goal-q):.9f}, limit_mm={c["max_path_mm"]:.9f}, '
                                     f'excess_mm={path+np.linalg.norm(goal-q)-c["max_path_mm"]:.9f}')
                if smooth and not np.isfinite(forward_lead):raise ServoError('Invalid forward lead')
                if send(goal,roll) is not False:
                    now=clock()
                    if direct and direct_update_hz is not None:
                        log(dict(event='baseline_direct_target_timing',requested_hz=update_hz,
                                 interval_seconds=None if not np.isfinite(last_sent) else now-last_sent,
                                 measured_x_mm=float(q[0]),target_x_mm=float(goal[0]),
                                 lead_mm=float(goal[0]-q[0]),remaining_mm=float(remaining)))
                    last_sent=now
            # ACK is checked promptly; never add a whole target period after the
            # reads/IPC work. The send deadline above remains the rate limiter.
            sleep(.005 if not direct or direct_update_hz is not None else 1/s['update_hz'])
            o=sample()
        except PoseRecoveryRetry:
            # A feedback read can detect drift after image acquisition, during
            # ready/send/finish. The interrupted target must never be retried.
            advancing=False;aligned_samples=0;final_samples=0;last_sent=-np.inf
            last_reliable_error=None
            if smooth:lead_policy.update(clock(),0.,False)
            log(dict(event='axis_recovery_update_discarded',policy='fresh_image_new_jacobian'))
            o=sample()
            o['tracking_recovered']=True
