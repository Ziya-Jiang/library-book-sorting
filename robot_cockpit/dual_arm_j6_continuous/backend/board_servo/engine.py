"""50Hz measured-feedback controller inside the robot command owner."""
import copy
import json
import time
from types import SimpleNamespace
import numpy as np
from .kernel import DirectVisualServo
from .execution import ExecutionGuard
from .joint_limits import JointLimitedStep, JointConstraintError


class Engine:
    def __init__(self,model,*,joint_policy=None):
        from .baseline_j2 import POLICY, BaselineJ2Step
        if joint_policy not in (None, POLICY):raise ValueError('Unknown board joint policy')
        self.joint_policy=joint_policy
        self.model=model;self.request=None;self.accepted=0;self.identity=None
        self.last_target_id=0;self.snapshot=None;self.output=None;self.ticks=0
        self.computed_id=0
        self.constraint_diagnostics=None
        self.next_tick=None
        self.max_compute_ms=0.;self.started=None;self.fault=None;self.hold=None
        bridge=SimpleNamespace(tcp=model.tcp,serial=0,snapshot=self.get_snapshot)
        bridge.guard=ExecutionGuard(lambda cmd:None)
        p=SimpleNamespace(_v31_bridge=bridge,arm_soft_joint_limits=model.bounds.copy(),
            curobo_cmd=SimpleNamespace(),gripper_joint_position=[],set_gripper_command=lambda:None,
            reorder_input=lambda q,g,as_curobo=True:model.ordered(q,g))
        self.planner=p;self.kernel=DirectVisualServo(p,model.collision,copy.deepcopy(model.settings),success_status=4)
        stepper=BaselineJ2Step if joint_policy==POLICY else JointLimitedStep
        self.kernel.stepper=stepper(model.tcp,model.bounds.copy(),self.kernel.c)

    def get_snapshot(self,max_age=.1):
        now=time.monotonic()
        if self.snapshot is None or not 0<=now-self.snapshot['stamp']<=max_age:
            raise ValueError('Board servo stale local RT feedback')
        return self.snapshot

    def cancel(self):
        self.request=None;self.identity=None;self.output=None;self.hold=None
        self.next_tick=None
        self.constraint_diagnostics=None
        self.kernel.reset();self.planner._v31_bridge.guard.reset()
        # Keep last_target_id until session changes: old targets cannot return.

    def accept(self,request,now):
        if self.fault:raise ValueError(self.fault)
        key=request['target_id']
        if key<self.last_target_id:raise ValueError('Reordered Cartesian target')
        if key==self.last_target_id:
            if self.request is None or request!=self.identity:raise ValueError('Cartesian identity reused/changed')
            return
        lease=self.model.settings['stream_lease_s'] if request['action']=='book_v3_1_stream' else self.model.settings['probe_timeout_s']
        if now-request['target_echo']>min(.3,lease):raise ValueError('New Cartesian target already expired')
        self.request=copy.deepcopy(request);self.identity=copy.deepcopy(request)
        self.last_target_id=key

    def tick(self,snapshot,grippers,arm_bounds,ratio):
        began=time.monotonic();self.snapshot=snapshot
        if self.request is None:return None
        if self.fault:raise ValueError(self.fault)
        self.get_snapshot()
        p=self.planner;g=p._v31_bridge.guard
        g.update(snapshot)
        if g.fault:raise ValueError(g.fault)
        if self.next_tick is not None and began<self.next_tick:
            return self.output
        bounds=np.asarray(arm_bounds,float)
        self.kernel.stepper.bounds=np.array([np.maximum(self.model.bounds[0],bounds[0]),np.minimum(self.model.bounds[1],bounds[1])])
        if (self.kernel.stepper.bounds[0]>=self.kernel.stepper.bounds[1]).any():raise ValueError('Empty board joint bounds')
        p.gripper_joint_position=list(grippers)
        pose=np.asarray(self.request['pose']);p.target_arm_tool_position=pose[:,:3];p.target_arm_tool_orientation=pose[:,3:]
        p.target_description=self.request['action']
        self.kernel.c['raw_velocity_ratio']=min(self.model.settings['raw_velocity_ratio'],ratio)
        previous_tick=self.kernel.last_tick
        try:
            self.kernel.compute(self.request['target_echo'])
        except JointConstraintError as error:
            detail=copy.deepcopy(error.diagnostics)
            detail.update(target_id=self.request['target_id'],action=self.request['action'],
                time_monotonic=time.monotonic(),feedback_age_s=began-snapshot['stamp'],
                target_age_s=began-self.request['target_echo'],
                measured_velocity_rad_s=np.asarray(snapshot['velocity'])[7:].tolist(),
                elapsed_since_compute_s=None if previous_tick is None else began-previous_tick,
                target_right_tcp=self.request['pose'][1],
                paused=self.kernel.paused,recovering_z=self.kernel.recovering_z)
            self.constraint_diagnostics=detail
            print('Board servo joint constraints: '+json.dumps(detail,allow_nan=False,sort_keys=True),flush=True)
            raise
        elapsed=time.monotonic()-began
        self.max_compute_ms=max(self.max_compute_ms,elapsed*1000)
        if elapsed>.04:raise ValueError('Board servo compute exceeded 40ms; result discarded')
        command=self.kernel.proposed.copy()
        if self.started is None:self.started=began
        self.ticks+=1
        self.output=command
        self.computed_id=self.request['target_id']
        # Keep a phase deadline, rather than waiting another full period after
        # a late poll. Never burst to catch up after an overrun.
        self.next_tick=(began if self.next_tick is None else self.next_tick)+self.kernel.c['period_s']
        if self.next_tick<=time.monotonic():self.next_tick=time.monotonic()+self.kernel.c['period_s']
        return command

    def commit(self):
        self.kernel.dispatched(True)
        self.accepted=self.computed_id

    def status(self):
        result=dict(target_id=self.accepted,active=self.request is not None,
                    paused=self.kernel.paused,recovering_z=self.kernel.recovering_z,
                    fault=self.fault,ticks=self.ticks,max_compute_ms=self.max_compute_ms,
                    configured_hz=1/self.model.settings['period_s'])
        if self.joint_policy is not None:
            result['joint_policy']=self.joint_policy
            result['j2_avoidance']=self.kernel.stepper.j2_diagnostics
        return result
