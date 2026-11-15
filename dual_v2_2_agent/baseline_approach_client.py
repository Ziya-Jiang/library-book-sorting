"""Baseline orchestration of a session-bound, prechecked CuRobo advance.

No sockets/services are created here. Uses the existing Planner command and
feedback channels; the local report is diagnostics, never executable input.
"""
import copy
import json
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation


def report_path(ticket):
    port=int(os.environ.get('CUARM_V31_TARGET_LOCAL_PORT','10176'))
    return Path(f'/tmp/baseline-approach-{os.getuid()}-{port}')/(ticket+'.json')


def pose(robot):
    s=robot.get_current_state();now=time.monotonic()
    if not s or not 0<=now-s.get('received_monotonic',-1e9)<=.25 or not 0<=now-s.get('pose_sample_monotonic',-1e9)<=.25:
        raise ValueError('Baseline preflight requires fresh measured feedback')
    a=s['right_arm']
    q=np.asarray(s['joint_state'],float).reshape(-1)[:14]
    xyz=np.asarray(a['position'],float)*1000
    quat=np.asarray(a['orientation'],float)
    if q.shape!=(14,) or not np.isfinite(np.r_[q,xyz,quat]).all():
        raise ValueError('Invalid baseline measured pose/joints')
    return s,q,xyz,Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz',degrees=True)


def preview(robot,ticket,xyz,angles,checkpoint,*,recheck=False):
    checkpoint()
    state,_,_,_=pose(robot)
    command=copy.deepcopy(robot.target_command)
    # Keep the wire enum type supplied by the live client.
    from cuarm_state import TargetCommandArmCommandType
    command.arm_command_type=TargetCommandArmCommandType.CARTESIAN
    command.arm_tool_command=np.asarray(command.arm_tool_command,float).copy()
    command.arm_tool_command[0]=state['left_arm']['position']+state['left_arm']['orientation']
    p,q=robot._convert_pos_rot_to_quaternion([*xyz,*angles])
    command.arm_tool_command[1]=np.r_[p,q]
    stage='recheck' if recheck else 'preflight'
    command.description=f'__preview__:baseline_{stage}:{ticket}'
    sent=time.monotonic()
    robot.udp.send(command)
    deadline=sent+10.
    while time.monotonic()<deadline:
        checkpoint()
        path=report_path(ticket)
        if path.exists():
            data=json.loads(path.read_text())
            if data.get('ticket')!=ticket or data.get('protocol')!=1:
                raise ValueError('Baseline preflight report identity mismatch')
            if data.get('time_monotonic',0)>=sent:
                if data.get('accepted') is False:
                    raise ValueError('Baseline preflight rejected: '+str(data.get('error',data.get('rejections'))))
                if data.get('stage')==stage and data.get('accepted') is True:
                    return data
        getattr(robot,'operator_sleep',time.sleep)(.02)
    raise ValueError('Baseline preflight produced no matching report within 10s')


def joint_arrival(robot,target,checkpoint,timeout=60.):
    deadline=time.monotonic()+timeout;anchor=None;stamp=-1.
    while time.monotonic()<deadline:
        checkpoint();state,q,_,_=pose(robot)
        current=state['pose_sample_monotonic']
        if current>stamp:
            stamp=current
            velocity=np.asarray(state.get('joint_velocity',[]),float).reshape(-1)
            if (velocity.shape==(14,) and np.isfinite(velocity).all() and
                    np.max(np.abs(velocity))<=.02 and np.max(np.abs(q-target))<=.002):
                if anchor is None:anchor=current
                if current-anchor>=.3:return
            else:anchor=None
        getattr(robot,'operator_sleep',time.sleep)(.02)
    raise ValueError('Baseline reshaped joints did not reach confirmed endpoint')


def complete(robot,result,checkpoint,log,verify):
    if result.get('completion_mode')!='alignment_handoff' or result.get('forward_completed') is not False:
        raise ValueError('Baseline worker did not return an alignment-only handoff')
    ticket=uuid.uuid4().hex
    _,joints,xyz,angles=pose(robot)
    goal=xyz.copy();goal[0]=float(result['goal_x_mm'])
    if np.linalg.norm(xyz-np.asarray(result['held_xyz_mm']))>3.:
        raise ValueError('Baseline moved after visual handoff')
    held_joints=np.asarray(result.get('held_joint_rad',[]),float)
    if held_joints.shape!=(14,) or np.max(np.abs(joints-held_joints))>.002:
        raise ValueError('Baseline joint configuration changed after visual handoff')
    held_angles=result.get('held_angles_deg')
    if held_angles is None or np.rad2deg((Rotation.from_euler('xyz',angles,degrees=True)*
            Rotation.from_euler('xyz',held_angles,degrees=True).inv()).magnitude())>.5:
        raise ValueError('Baseline orientation changed after visual handoff')
    log(dict(event='baseline_preflight_start',ticket=ticket,goal_xyz_mm=goal.tolist(),
             alignment_verified=result.get('alignment_verified',False),
             held_joint_rad=held_joints.tolist(),frame_stamp_155=result.get('frame_stamp'),
             held_feedback_stamp_155=result.get('held_feedback_stamp_155'),
             alignment_error=result.get('verified_error')))
    report=preview(robot,ticket,goal,angles,checkpoint)
    if report['needs_reshape']:
        checkpoint()
        if robot.master_control(right_arm=[*xyz,*angles],action='baseline_reshape:'+ticket,
                                object_category='book') is not True:
            raise ValueError('Baseline reshape not confirmed; forward cancelled')
        executed=json.loads(report_path(ticket).read_text())
        if executed.get('stage')!='reshape' or not executed.get('accepted'):
            raise ValueError('Missing reshape dispatch confirmation')
        joint_arrival(robot,np.asarray(executed['target_joint_rad']),checkpoint)
        verified=verify()
        if verified.get('alignment_verified') is not True:
            raise ValueError('Fresh post-reshape visual alignment not confirmed')
        report=preview(robot,ticket,goal,angles,checkpoint,recheck=True)
        if report['needs_reshape']:
            raise ValueError('Second reshape forbidden')
    checkpoint()
    if robot.master_control(right_arm=[*goal,*angles],action='baseline_advance:'+ticket,
                            object_category='book') is not True:
        raise ValueError('Baseline planned advance not confirmed; grasp cancelled')
    # Convert the completed trajectory to a measured stationary hold before
    # any gripper update, so its command cannot replay earlier waypoints.
    _,_,held,held_angles=pose(robot)
    if robot.master_control(right_arm=[*held,*held_angles],action='baseline_advance_hold:'+ticket,
                            object_category='book') is not True:
        raise ValueError('Baseline final hold not confirmed; grasp cancelled')
    log(dict(event='baseline_planned_advance_complete',ticket=ticket,actual_xyz_mm=held.tolist(),
             goal_xyz_mm=goal.tolist(),plan=report))
    return dict(result,distance_complete=True,forward_completed=True,converged=False,
                completion_mode='curobo_prechecked_advance',
                alignment_verified_before_advance=result.get('alignment_verified',False),
                final_angles_deg=held_angles.tolist())


@contextmanager
def ready_pose_installed():
    """Opt in only for a validated baseline candidate; never mutate shared reset."""
    from RobotRAM.controllor.hkstp_arm import RobotArm
    original=RobotArm.control_joint_position
    config=Path(__file__).with_name('baseline_ready_pose.json')
    def reset(robot,left_pos,right_pos,action,*,force=False):
        data=json.loads(config.read_text())
        if action!='first_reset' or left_pos or not right_pos or data.get('enabled') is not True:
            return original(robot,left_pos,right_pos,action,force=force)
        from cuarm_state import TargetCommandArmCommandType
        # The PC Planner verifies the candidate's contract, scores, same TCP,
        # unchanged left arm and fresh reset path before returning a ticket.
        state,q,_,_=pose(robot)
        q[7:]=np.deg2rad(robot.right_safe_joint_deg)
        request=copy.deepcopy(robot.target_command)
        request.arm_command_type=TargetCommandArmCommandType.JOINT
        request.arm_joint_size=[7,7];request.arm_joint_command=q.reshape(2,7)
        ticket=uuid.uuid4().hex
        request.description='__preview__:baseline_ready_preflight:'+ticket
        sent=time.monotonic();robot.udp.send(request)
        report=None
        while time.monotonic()-sent<10.:
            path=report_path(ticket)
            if path.exists():
                item=json.loads(path.read_text())
                if item.get('ticket')==ticket and item.get('time_monotonic',0)>=sent:
                    if not item.get('accepted'):raise ValueError('Ready preflight failed: '+str(item))
                    if item.get('stage')=='ready_preflight':report=item;break
            getattr(robot,'operator_sleep',time.sleep)(.02)
        if report is None:raise ValueError('Ready preflight timed out; no reset sent')
        target=np.asarray(report['goal_joint_rad'])
        robot.target_command=copy.deepcopy(request)
        robot.target_command.description='baseline_ready:'+ticket
        robot.target_command.arm_joint_command=target.reshape(2,7)
        robot.udp.send(robot.target_command)
        if robot._wait_for_robot_stop() is not True:return False
        joint_arrival(robot,target,lambda:None)
        print('Baseline准备位已确认：'+str(report.get('reason') or '优化构型'),flush=True)
        return True
    RobotArm.control_joint_position=reset
    try:yield
    finally:RobotArm.control_joint_position=original
