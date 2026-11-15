"""CuRobo path -> one bounded RT spline, validated before publication.

No robot I/O. This does not blend contact stages or turn an infeasible path
into executed submoves. IK and collision queries stay in the existing GPU engine.
"""
import json,time
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from j6_trajectory import panel_path,limits_for,collision_gate
from wrist_timing import derivative_peaks

ACTIONS={'book_plug_x':0,'book_plug_y':1,'book_plug_z':2}
COUNTS=(20,16,10)  # wire protocol capacity, not task stop points
MAX_DURATION=18.0
MAX_ATTEMPTS=12
BUDGET=5.0
X_ADVANCE_SPEED_FACTOR=.75  # half the previous forward-X request, other phases unchanged
X_ADVANCE_MAX_AVERAGE_M_S=.10
EXECUTION_SPEED_FACTOR=0.8  # desired speed relative to the original timing candidates

def smooth_points(source,start,count):
    source,start=np.asarray(source,float),np.asarray(start,float)
    if (source.ndim!=2 or source.shape[1]!=14 or len(source)<2 or
        start.shape!=(14,) or not np.isfinite(source).all() or not np.isfinite(start).all()
        or count not in COUNTS):raise ValueError('Invalid continuous path')
    # Stationary at both ends; smooth phase progression does not stop at knots.
    source=source.copy();source[:,:7]=start[:7]
    if np.max(abs(source[0]-start))>1e-7:source=np.vstack([start,source])
    else:source[0]=start
    curve=CubicSpline(np.linspace(0.,1.,len(source)),source,axis=0,bc_type='clamped')
    t=np.linspace(0.,1.,count);phase=10*t**3-15*t**4+6*t**5
    points=curve(phase);points[0]=start;points[-1]=source[-1];points[:,:7]=start[:7]
    return points.astype(np.float32).astype(float)

def geometry_gate(p,samples,start,bounds):
    a=np.asarray(samples,float)
    if a.ndim!=2 or a.shape[1]!=14 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Invalid continuous RT samples')
    if np.any(a<bounds[0]) or np.any(a>bounds[1]):raise ValueError('Continuous path joint limit')
    if np.max(abs(a[:,:7]-start[:7]))>.002:raise ValueError('Continuous path moves held left arm')
    xyz,quat=p._v31_bridge.tcp(a);axis=ACTIONS[p.target_description]
    target=np.asarray(p.target_arm_tool_position[1]);fixed=[i for i in range(3) if i!=axis]
    fixed_error=float(np.max(np.linalg.norm(xyz[:,1,fixed]-target[fixed],axis=1)))*1000
    rot=Rotation.from_quat(quat[:,1,[1,2,3,0]])
    ref=Rotation.from_quat(np.asarray(p.target_arm_tool_orientation[1])[[1,2,3,0]])
    angle=float(np.rad2deg((ref.inv()*rot).magnitude()).max())
    initial,_=p._v31_bridge.tcp(start);lo,hi=sorted((initial[0,1,axis],target[axis]))
    overshoot=max(0.,lo-xyz[:,1,axis].min(),xyz[:,1,axis].max()-hi)*1000
    endpoint=float(np.linalg.norm(xyz[-1,1]-target))*1000
    if fixed_error>3 or angle>1 or overshoot>3 or endpoint>3:
        raise ValueError(f'Continuous path geometry: fixed_mm={fixed_error:.4f}, angle_deg={angle:.4f}, overshoot_mm={overshoot:.4f}, endpoint_mm={endpoint:.4f}')
    return dict(fixed_axis_mm=fixed_error,orientation_deg=angle,overshoot_mm=overshoot,endpoint_mm=endpoint)

def duration_candidates(source_dt,source_count,count,speed_factor=EXECUTION_SPEED_FACTOR):
    duration=max(1.,float(source_dt)*source_count)
    dt=max(.125,np.ceil(duration/count/.125)*.125)/speed_factor
    return [float(np.float32(dt+i*.125)) for i in range(MAX_ATTEMPTS)
            if (dt+i*.125)*count<=MAX_DURATION]

def prepare(p,result):
    """Resample a successful GPU solution; failure returns a rejected result."""
    from book_planning_budget import FailedPlan
    from curobo.types.state import JointState
    began=time.monotonic();report=dict(event='continuous_cartesian',action=p.target_description,
        generator='CuRobo_GPU_MotionGen',execution='single_buffer_actual_RT_spline',
        requested_speed_factor=EXECUTION_SPEED_FACTOR,accepted=False,attempts=[])
    p._continuous_diagnostics=report
    def checkpoint():
        from planning_deadline import checkpoint as planning_checkpoint
        planning_checkpoint(p)
        if time.monotonic()-began>BUDGET:raise ValueError('Continuous validation budget exhausted')
        p.raise_if_state_thread_failed()
    try:
        names=list(p.total_arm_joint_names);start=np.asarray(p.arm_joint_position,float).reshape(14).copy()
        from x_refinement import is_advance,refine
        forward=is_advance(p)
        speed_factor=X_ADVANCE_SPEED_FACTOR if forward else EXECUTION_SPEED_FACTOR
        report['requested_speed_factor']=speed_factor
        initial_xyz,_=p._v31_bridge.tcp(start)
        forward_distance=max(0.,float(p.target_arm_tool_position[1,0]-initial_xyz[0,1,0])) if forward else 0.
        if forward:report['max_average_forward_m_s']=X_ADVANCE_MAX_AVERAGE_M_S
        if forward:
            if not bool(result.success.item()):raise ValueError('X refinement requires successful GPU motion planning')
            bounds,_=limits_for(p)
            report['source_geometry']={}
            for label,js in [('gpu_knots',result.optimized_plan),('gpu_interpolated',result.get_interpolated_plan())]:
                checkpoint()
                rows=p._motion_gen.get_full_js(js).get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(-1,14)
                rows=rows.copy();left_drift=float(np.max(abs(rows[:,:7]-start[:7])));rows[:,:7]=start[:7]
                report['source_geometry'][label+'_held_left_drift_rad']=left_drift
                try:report['source_geometry'][label]=geometry_gate(p,rows,start,bounds)
                except ValueError as error:report['source_geometry'][label]={'rejection':str(error)}
            result=refine(p,result,checkpoint)
            report['x_refinement']=result.x_refinement
        full=p._motion_gen.get_full_js(result.optimized_plan)
        source=full.get_ordered_joint_state(names).position.detach().cpu().numpy().reshape(-1,14)
        source_dt=float(result.optimized_dt.item())
        bounds,dynamics=limits_for(p)
        picked=None
        for count in COUNTS:
            points=smooth_points(source,start,count)
            for dt in duration_candidates(source_dt,len(source),count,speed_factor):
                checkpoint()
                panel=panel_path(p,points,dt,start)
                peaks=derivative_peaks(panel,dynamics);ratios=np.array([x['ratio'] for x in peaks])
                trial=dict(count=count,dt=dt,ratios=ratios.tolist(),samples=len(panel))
                report['attempts'].append(trial)
                if forward:
                    trial['average_forward_m_s']=forward_distance/max(.001,(len(panel)-1)*.001)
                    if trial['average_forward_m_s']>X_ADVANCE_MAX_AVERAGE_M_S:
                        trial['timing_rejection']='forward_average_speed';continue
                if not np.all(ratios<=[.95,.95,.9]):continue
                try:geometry=geometry_gate(p,panel,start,bounds)
                except ValueError as error:
                    trial['geometry_rejection']=str(error);continue
                collision_gate(p,np.concatenate([start[None],points,panel]),checkpoint)
                picked=(points,panel,dt,peaks,geometry);break
            if picked is not None:break
        if picked is None:raise ValueError('No continuous RT curve satisfies unchanged limits; no segmented execution fallback')
        points,panel,dt,peaks,geometry=picked
        checkpoint()
        # Preserve the gripper values and CuRobo joint ordering; dense commands
        # describe the same RT path, not an unvalidated secondary interpolation.
        template=full.position.detach().cpu().numpy().reshape(-1,len(full.joint_names))[0]
        def state(values,spacing):
            allq=np.repeat(template[None],len(values),axis=0)
            for j,name in enumerate(names):allq[:,full.joint_names.index(name)]=values[:,j]
            v=np.gradient(allq,spacing,axis=0);a=np.gradient(v,spacing,axis=0);jerk=np.gradient(a,spacing,axis=0)
            return JointState.from_numpy(list(full.joint_names),allq,v,a,jerk,tensor_args=p.tensor_args)
        indices=np.r_[np.arange(0,len(panel)-1,20),len(panel)-1]
        command=state(panel[indices],.02)
        out=SimpleNamespace(success=np.array(True),status=result.status,
            optimized_plan=state(points,dt),optimized_dt=p.tensor_args.to_device([dt]),
            get_interpolated_plan=lambda:command)
        for field in ('ik_time','graph_time','trajopt_time','finetune_time','solve_time','total_time'):
            setattr(out,field,float(getattr(result,field,0.)))
        report.update(accepted=True,waypoint_count=len(points),waypoint_dt=dt,
                      duration_s=(len(panel)-1)*.001,dynamics_peaks=peaks,geometry=geometry,
                      start_joints_rad=start.tolist(),requested_goal=p.target_arm_tool_position[1].tolist())
        out.continuous_diagnostics=report
        return out
    except Exception as error:
        report['reason']=f'{type(error).__name__}: {error}'
        return FailedPlan('continuous_rejected: '+report['reason'],time.monotonic()-began)
    finally:
        report['validation_seconds']=time.monotonic()-began
        print('Continuous Cartesian trajectory: '+json.dumps(report,allow_nan=False),flush=True)

def validate_publication(p,result):
    """Check what the base planner will actually publish, after all its gates."""
    began=time.monotonic()
    def checkpoint():
        from planning_deadline import checkpoint as planning_checkpoint
        planning_checkpoint(p)
        if time.monotonic()-began>3:raise ValueError('Continuous publication validation deadline')
        p.raise_if_state_thread_failed()
    if not getattr(result,'continuous_diagnostics',{}).get('accepted'):
        raise ValueError('Missing continuous candidate proof')
    points=result.optimized_plan.get_ordered_joint_state(p.total_arm_joint_names).position.detach().cpu().numpy()
    if (p.limited_waypoints.shape!=points.shape or
        not np.allclose(p.limited_waypoints,points,atol=1e-6,rtol=0) or
        float(p.waypoints_dt)!=float(result.optimized_dt.item())):
        raise ValueError('Published continuous trajectory changed after validation')
    actual=np.asarray(p.arm_joint_position,float).reshape(14)
    old=np.asarray(result.continuous_diagnostics['start_joints_rad'])
    if np.max(abs(actual-old))>.002:raise ValueError('Continuous trajectory start moved; discard')
    panel=panel_path(p,p.limited_waypoints,p.waypoints_dt,actual)
    bounds,dynamics=limits_for(p);geometry=geometry_gate(p,panel,actual,bounds)
    peaks=derivative_peaks(panel,dynamics)
    if any(x['ratio']>1 for x in peaks):raise ValueError('Continuous final RT dynamics exceeded: '+str(peaks))
    collision_gate(p,panel,checkpoint);checkpoint()
    p._continuous_diagnostics['publication']=dict(accepted=True,geometry=geometry,peaks=peaks)
