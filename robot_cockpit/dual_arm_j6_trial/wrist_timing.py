"""Bounded timing search on the exact published float32 interval and RT samples."""
import numpy as np
SAMPLE_DT=.001
MAX_ATTEMPTS=12
INTERVAL_STEP_S=.125
MAX_DURATION_S=18.
JERK_PLANNING_RATIO=.9  # headroom only; final physical limit stays 1.0
WAYPOINT_COUNTS=(10,20,16)  # protocol MAX_WAYPOINTS=20; original curve first


class TimingExhausted(ValueError):
    """Only exhausted timing candidates permit trying another waypoint density."""


def select_curve(start,goal,index,dynamics,speed_factor,points_for,sample,validate,checkpoint,report):
    """Resample the same quintic joint curve, never smooth away RT jerk peaks.

    Each density is sent through the same real RT interpolation and full
    derivative checks. An invalid position/collision/cancellation is not a
    timing failure and is never swallowed here.
    """
    trials=report.setdefault('curve_attempts',[])
    for count in WAYPOINT_COUNTS:
        checkpoint()
        points=points_for(start,goal,index,count).astype(np.float32).astype(float)
        validate(points)
        trial=dict(waypoint_count=count,retiming=[],accepted=False)
        trials.append(trial)
        try:
            panel,dt,ratio=select_timing(points,start,index,dynamics,speed_factor,
                lambda spacing:sample(points,spacing),validate,checkpoint,trial)
        except TimingExhausted as error:
            trial['reason']=str(error)
            continue
        trial['accepted']=True
        report.update(retiming=trial['retiming'],timing_policy=trial['timing_policy'],
                      curve_policy='same_quintic_new_waypoint_density_exact_RT_validation',
                      selected_waypoint_count=count)
        return points,panel,dt,ratio
    raise TimingExhausted('Wrist curve/timing search exhausted for waypoint counts '
                          +str(WAYPOINT_COUNTS)+'; dynamics limits remain enforced')


def derivative_peaks(samples,limits):
    derivative=np.pad(samples,((3,3),(0,0)),mode='edge')
    peaks=[]
    for order,limit in enumerate(limits,1):
        derivative=np.diff(derivative,axis=0)/SAMPLE_DT
        row,joint=np.unravel_index(np.argmax(np.abs(derivative)/limit),derivative.shape)
        peaks.append(dict(order=order,joint=('left' if joint<7 else 'right')+f' J{joint%7+1}',
                          sample=int(row),time_s=float((row-3)*SAMPLE_DT),
                          value=float(derivative[row,joint]),limit=float(limit[joint]),
                          ratio=float(abs(derivative[row,joint])/limit[joint])))
    return peaks


def select_timing(points,start,index,dynamics,speed_factor,sample,validate,checkpoint,report):
    if not 0<speed_factor<=1 or not np.isfinite(dynamics).all() or np.any(dynamics<=0):
        raise ValueError('Invalid wrist timing limits/speed')
    amount=abs(points[-1,index]-start[index])
    v,a,j=dynamics[:,index]*np.array([speed_factor,speed_factor**2,speed_factor**3])
    duration=max(1.875*amount/v,np.sqrt(5.774*amount/a),np.cbrt(60*amount/j),1.)
    dt=np.ceil(max(duration/(len(points)-1),.02)/INTERVAL_STEP_S)*INTERVAL_STEP_S
    report['timing_policy']=dict(max_attempts=MAX_ATTEMPTS,max_duration_s=MAX_DURATION_S,
                                jerk_planning_ratio=JERK_PLANNING_RATIO,interval_step_s=INTERVAL_STEP_S,
                                interval_precision='float32_publication')
    for attempt in range(MAX_ATTEMPTS):
        checkpoint()
        # CuRobo publishes float32 dt. Validate that exact value, not a nearby double.
        dt=float(np.float32(dt))
        if dt*len(points)>MAX_DURATION_S:break
        panel=sample(dt)
        validate(panel)
        if (len(panel)-1)*SAMPLE_DT>MAX_DURATION_S:break
        peaks=derivative_peaks(panel,dynamics)
        ratio=np.array([item['ratio'] for item in peaks])
        report['retiming'].append(dict(dt=dt,ratios=ratio.tolist(),samples=len(panel),peaks=peaks))
        if np.all(ratio<=np.array([1.,1.,JERK_PLANNING_RATIO])):
            return panel,dt,ratio
        # Actual discrete RT jerk is non-monotonic; validate every new candidate.
        dt+=INTERVAL_STEP_S
    raise TimingExhausted('Wrist timing search exhausted; velocity/acceleration/jerk checks remain enforced')
