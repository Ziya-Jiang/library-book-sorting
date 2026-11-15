"""Board-only bounded differential IK; never widens model or RT limits."""
from collections import deque
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from .kernel import DifferentialStep, norm_limit


def violation(q, bounds, phase):
    bad=np.flatnonzero((q<bounds[0])|(q>bounds[1])|~np.isfinite(q))
    if len(bad):
        i=int(bad[0]);arm='right' if i>=7 else 'left'
        raise ValueError(f'Direct servo joint limit: {phase}, {arm} J{i%7+1}, '
                         f'value_deg={np.degrees(q[i]):.6f}, '
                         f'limits_deg=[{np.degrees(bounds[0,i]):.6f}, '
                         f'{np.degrees(bounds[1,i]):.6f}]')


class JointConstraintError(ValueError):
    """Keep structured evidence; the bridge logs it before resetting the engine."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        j = diagnostics['joints'][0]
        super().__init__(
            f'Direct servo joint constraints infeasible: right J{j["joint_1based"]}, '
            f'velocity_bounds_rad_s=[{j["lower_rad_s"]:.6f},{j["upper_rad_s"]:.6f}], '
            f'lower_sources={j["lower_sources"]}, upper_sources={j["upper_sources"]}')


def braking_speed(distance, acceleration, dt):
    """v*dt + v^2/(2*a) <= distance, assuming a stationary boundary.

    Reserve this tick's travel AND a full braking distance. The rationalized
    root avoids cancellation near zero. A moving feedback window can still
    become infeasible; never conceal that by relaxing acceleration or lead.
    """
    distance = np.maximum(distance, 0.)
    root = np.sqrt((acceleration*dt)**2 + 2.*acceleration*distance)
    return 2.*acceleration*distance/(root + acceleration*dt)


def damper_braking_speed(distance, acceleration, dt, rate=2.):
    """Reserve the path needed to join v <= rate*distance under deceleration.

    Above a/rate, braking must begin before the linear damper becomes active.
    The tangent point is (distance=a/rate^2, velocity=a/rate). Include this
    tick's v*dt travel as well, rather than waiting for contradictory bounds.
    """
    distance=np.maximum(distance,0.)
    threshold=acceleration/(rate*rate)+acceleration*dt/rate
    slow=distance/(dt+1./rate)
    root=np.sqrt(np.maximum(0.,(acceleration*dt)**2+
                           2.*acceleration*distance-(acceleration/rate)**2))
    fast=(2.*acceleration*distance-(acceleration/rate)**2)/(root+acceleration*dt)
    return np.where(distance<=threshold,slow,fast)


def velocity_interval(command, measured, previous, bounds, c, dt):
    speed=c['joint_speed_rad_s'];acc=c['joint_acceleration_rad_s2']
    lead=c['joint_lead_rad'];lo,hi=bounds
    lower_terms = dict(speed=np.full(7,-speed), acceleration=previous-acc*dt,
        position=(lo-command)/dt, feedback_lead=(measured-lead-command)/dt,
        limit_damper=-2.*np.maximum(command-lo,0.))
    upper_terms = dict(speed=np.full(7,speed), acceleration=previous+acc*dt,
        position=(hi-command)/dt, feedback_lead=(measured+lead-command)/dt,
        limit_damper=2.*np.maximum(hi-command,0.))
    # Keep every original bound. Add an earlier braking envelope at both the
    # physical/model boundary and the current measured-feedback lead window.
    lower_terms['position_braking']=-braking_speed(command-lo,acc,dt)
    upper_terms['position_braking']=braking_speed(hi-command,acc,dt)
    lower_terms['lead_braking']=-braking_speed(command-(measured-lead),acc,dt)
    upper_terms['lead_braking']=braking_speed(measured+lead-command,acc,dt)
    lower_terms['damper_braking']=-damper_braking_speed(command-lo,acc,dt)
    upper_terms['damper_braking']=damper_braking_speed(hi-command,acc,dt)
    lower=np.maximum.reduce(list(lower_terms.values()))
    upper=np.minimum.reduce(list(upper_terms.values()))
    return lower,upper,lower_terms,upper_terms


def constraint_diagnostics(command, measured, previous, bounds, c, dt, interval):
    lower,upper,lower_terms,upper_terms=interval
    indices=sorted(np.flatnonzero(lower>upper+1e-12),key=lambda i:-(lower[i]-upper[i]))
    report=dict(event='joint_constraint_infeasible',dt_s=float(dt),
                speed_limit_rad_s=c['joint_speed_rad_s'],
                acceleration_limit_rad_s2=c['joint_acceleration_rad_s2'],
                lead_limit_rad=c['joint_lead_rad'],
                braking_assumption='stationary current feedback window; not a hardware stopping guarantee',joints=[])
    for i in indices:
        report['joints'].append(dict(joint_1based=int(i+1),
            measured_rad=float(measured[i]),command_rad=float(command[i]),
            command_minus_measured_rad=float(command[i]-measured[i]),
            previous_command_velocity_rad_s=float(previous[i]),
            joint_bounds_rad=bounds[:,i].tolist(),
            position_margins_rad=[float(command[i]-bounds[0,i]),float(bounds[1,i]-command[i])],
            lead_margins_rad=[float(command[i]-measured[i]+c['joint_lead_rad']),
                              float(measured[i]+c['joint_lead_rad']-command[i])],
            previous_braking_distance_rad=float(previous[i]**2/(2*c['joint_acceleration_rad_s2'])),
            lower_rad_s=float(lower[i]),upper_rad_s=float(upper[i]),
            excess_rad_s=float(lower[i]-upper[i]),
            lower_sources=[k for k,v in lower_terms.items() if abs(v[i]-lower[i])<=1e-10],
            upper_sources=[k for k,v in upper_terms.items() if abs(v[i]-upper[i])<=1e-10],
            lower_terms_rad_s={k:float(v[i]) for k,v in lower_terms.items()},
            upper_terms_rad_s={k:float(v[i]) for k,v in upper_terms.items()}))
    return report


def bounded_velocity(a, target, command, measured, previous, bounds, c, dt):
    """Constrained solve from the last dispatched command and velocity."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid board servo dt')
    interval=velocity_interval(command,measured,previous,bounds,c,dt)
    lower,upper=interval[:2]
    if (lower>upper+1e-12).any():
        raise JointConstraintError(constraint_diagnostics(command,measured,previous,bounds,c,dt,interval))
    lo,hi=bounds
    # Weak inward preference uses redundancy without dropping the TCP task.
    band=np.minimum(np.radians(10.),(hi-lo)/4.)
    bias=.1*(np.maximum(0.,1-(command-lo)/band)**2-
             np.maximum(0.,1-(hi-command)/band)**2)
    matrix=np.vstack([a,c['damping']*np.eye(7)])
    rhs=np.r_[target,c['damping']*bias]
    fixed=upper-lower<=1e-12;v=(lower+upper)/2
    if (~fixed).any():
        fit=lsq_linear(matrix[:,~fixed],rhs-matrix[:,fixed]@v[fixed],
                       bounds=(lower[~fixed],upper[~fixed]),method='bvls',tol=1e-10,max_iter=30)
        if not fit.success:raise ValueError('Direct servo bounded joint solve did not converge')
        v[~fixed]=fit.x
    return v


class JointLimitedStep(DifferentialStep):
    # Default path remains exactly the existing bounded solve. Only an explicit
    # baseline visual session selects the subclass with a secondary objective.
    solve_velocity = staticmethod(bounded_velocity)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.history=deque(maxlen=8)

    def reset(self):
        super().reset()
        self.history.clear()

    def step(self,q,goal,dt):
        c=self.c;q=np.asarray(q,float)
        if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid board servo dt')
        violation(q,self.bounds,'measured')
        if self.command is None:self.command=q.copy()
        violation(self.command,self.bounds,'previous command')
        pos,quat=self.tcp(q);position=pos[0,1]
        rotation=Rotation.from_quat(quat[0,1,[1,2,3,0]])
        eps=1e-4;perturbed=np.tile(q,(7,1));perturbed[np.arange(7),7+np.arange(7)]+=eps
        pp,rr=self.tcp(perturbed)
        jac=np.vstack(((pp[:,1]-position).T/eps,
            (Rotation.from_quat(rr[:,1][:,[1,2,3,0]])*rotation.inv()).as_rotvec().T/eps))
        error=np.r_[goal[:3]-position,
                    (Rotation.from_quat(goal[[4,5,6,3]])*rotation.inv()).as_rotvec()]
        twist=np.r_[norm_limit(c['position_gain']*error[:3],c['linear_speed_m_s']),
                    norm_limit(c['rotation_gain']*error[3:],c['angular_speed_rad_s'])]
        weight=np.array([1.,1.,1.,.15,.15,.15])
        self.history.append(dict(dt_s=float(dt),measured_rad=q[7:].tolist(),
            command_rad=self.command[7:].tolist(),previous_velocity_rad_s=self.velocity.tolist()))
        try:
            velocity=self.solve_velocity(jac*weight[:,None],twist*weight,
                self.command[7:],q[7:],self.velocity,self.bounds[:,7:],c,dt)
        except JointConstraintError as error:
            # Only include affected joints, keeping transport diagnostics bounded.
            indices=[j['joint_1based']-1 for j in error.diagnostics['joints']]
            error.diagnostics['history_joint_order']=[i+1 for i in indices]
            error.diagnostics['history']=[dict(dt_s=h['dt_s'],**{
                k:[h[k][i] for i in indices] for k in
                ('measured_rad','command_rad','previous_velocity_rad_s')}) for h in self.history]
            raise
        proposal=self.command.copy();proposal[7:]+=velocity*dt
        violation(proposal,self.bounds,'proposed')
        return proposal,velocity,error
