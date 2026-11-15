"""CuRobo baseline transaction: preview -> optional reshape -> verified advance.

Only diagnostics cross the local file boundary; executable paths stay in this
Planner process and are session-bound, expiring, single-use transactions.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import numpy as np
from baseline_approach_path import Search, Rejected, angle

PROTOCOL = 1


def report_path(ticket):
    if not re.fullmatch('[a-f0-9]{32}', ticket):
        raise ValueError('Invalid baseline ticket')
    port = int(os.environ.get('CUARM_V31_TARGET_LOCAL_PORT', '10176'))
    return Path(f'/tmp/baseline-approach-{os.getuid()}-{port}')/(ticket+'.json')


def decode(description):
    match = re.fullmatch(r'(__preview__:baseline_preflight|__preview__:baseline_recheck|__preview__:baseline_ready_preflight|baseline_ready|baseline_reshape|baseline_advance|baseline_advance_hold):([a-f0-9]{32})', description)
    return (match[1].replace('__preview__:', ''), match[2]) if match else None


class Adapter:
    def __init__(self, planner):
        self.p = planner
        self.bounds = np.asarray(planner.arm_soft_joint_limits, float).copy()
        self.tcp = planner._v31_bridge.tcp
        self.grip = list(planner.target_gripper_joint)
        self.input_identity = self.command_identity(getattr(planner,'last_received_target_command',None))

    @staticmethod
    def command_identity(command):
        if command is None:return None
        # Include payload as well as UUID: a reused ID cannot replace the goal
        # while the main receive loop is occupied by a solve.
        fields=('description','arm_command_type','arm_joint_command','arm_tool_command',
                'gripper_command_type','gripper_joint_command','gripper_mode_command')
        return repr([(name,getattr(command,name,None)) for name in fields])

    def check(self):
        self.p.raise_if_state_thread_failed()
        latest = self.p._v31_bridge.snapshot(max_age=.25)
        if getattr(latest['panel'], '_seldon_session', None) != self.session:
            raise Rejected('Baseline planning session changed')
        if self.p.target_description != self.description:
            raise Rejected('Baseline planning target cancelled/replaced')
        from cuarm_state import PanelRemoteState
        if latest['panel'].state == PanelRemoteState.STOP:
            raise Rejected('Baseline planning cancelled by panel STOP')
        command, _ = self.p._target_udp.receive_with_timestamp()
        if command is None or self.command_identity(command) != self.input_identity:
            raise Rejected('Baseline input command replaced during solve')

    def arm_values(self, q):
        p = self.p
        order = np.asarray(p.reorder_input(np.arange(14), list(range(14,14+len(self.grip))), as_curobo=True), int)
        q = np.atleast_2d(q)
        return np.c_[q, np.broadcast_to(self.grip, (len(q),len(self.grip)))][:,order]

    def collision(self, samples, checkpoint):
        values = self.arm_values(samples)
        for i in range(0,len(values),512):
            checkpoint()
            with self.p._gpu_lock:
                result = self.p._motion_gen.rollout_fn.rollout_constraint(
                    self.p.tensor_args.to_device(np.ascontiguousarray(values[i:i+512])).unsqueeze(1),
                    use_batch_env=False)
                if not bool(result.feasible.all().item()):
                    raise Rejected('CuRobo collision/joint constraint failed on RT curve')

    def motion(self, start, goal, deadline):
        from curobo.types.state import JointState
        from book_post_lift_pose import execution_bounds
        self.check()
        p = self.p
        def state(q):
            position=p.tensor_args.to_device(self.arm_values(q))
            return JointState(position=position,velocity=position*0.,acceleration=position*0.,jerk=position*0.,
                              joint_names=p.curobo_input_joint_names)
        config = p._plan_config.clone()
        config.max_attempts = 1
        config.timeout = max(.01, deadline-time.monotonic())
        config.check_start_validity = True
        config.time_dilation_factor = getattr(p,'_baseline_speed_factor',None) or .4
        with p._gpu_lock, execution_bounds(p):
            result = p._motion_gen.plan_single_js(state(start),state(goal),config)
        if time.monotonic() >= deadline:
            raise TimeoutError('CuRobo returned after baseline planning deadline')
        self.check()
        if not bool(result.success.item()):
            raise Rejected('MotionGen: '+str(result.status))
        points = result.optimized_plan.get_ordered_joint_state(p.total_arm_joint_names).position.detach().cpu().numpy().reshape(-1,14)
        points[:, :7] = start[:7]
        dt = float(result.optimized_dt.detach().cpu().numpy().item())
        if not np.isfinite(points).all() or not np.isfinite(dt) or dt <= 0:
            raise Rejected('Invalid MotionGen trajectory')
        return points, dt

    def sample(self, start, points, dt, sample_dt):
        if len(points)*dt > 60:
            raise Rejected('Baseline constrained path duration exceeds 60s')
        b = self.p._v31_bridge
        arms = [b.panel_path.sample(start[i*7:i*7+7], points[:,i*7:i*7+7],
                dt,b.limits[i],sample_dt=sample_dt)[0] for i in range(2)]
        n = max(map(len,arms))
        return np.concatenate([np.pad(a,((0,n-len(a)),(0,0)),mode='edge') for a in arms],axis=1)

    def dynamics(self, samples, dt):
        from baseline_withdraw_path import dynamics_limits, dynamics_ratios
        if dt != .001:
            raise Rejected('Unexpected dynamics sampling interval')
        return dynamics_ratios(samples,dynamics_limits(self.p))


class Transactions:
    def __init__(self, planner, clock=time.monotonic):
        self.p, self.clock = planner, clock
        self.records = {}
        self.used_tickets = set()
        self.last_description = None
        self.last_session = None
        self.failed = False

    def write(self, ticket, **data):
        path = report_path(ticket)
        path.parent.mkdir(mode=0o700,exist_ok=True)
        data.update(protocol=PROTOCOL,ticket=ticket,planner_pid=os.getpid(),time_monotonic=self.clock())
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
        temporary.replace(path)
        archive=Path(__file__).resolve().parent.parent/'output/baseline_preflight'
        archive.mkdir(parents=True,exist_ok=True)
        with (archive/(ticket+'.jsonl')).open('a') as stream:
            stream.write(path.read_text())
        print('Baseline preflight: '+json.dumps(data,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)),flush=True)

    def claim(self,ticket,session):
        identity=(session,ticket)
        if identity in self.used_tickets:
            raise Rejected('Preflight ticket already used; old command replay rejected')
        self.used_tickets.add(identity)

    def prepare(self, ticket, recheck, snapshot):
        p = self.p
        a = Adapter(p)
        a.session = getattr(snapshot['panel'],'_seldon_session',None)
        a.description = p.target_description
        if a.session is None:
            raise Rejected('Baseline requires a session-bound board Planner')
        xyz = np.asarray(p.target_arm_tool_position[1],float).copy()
        quat = np.asarray(p.target_arm_tool_orientation[1],float).copy()
        old = self.records.get(ticket)
        if recheck:
            if not old or old['phase'] != 'reshaped' or old['session'] != a.session:
                raise Rejected('No same-session reshape for this recheck')
            if self.clock()>old['expires'] or np.linalg.norm(xyz-old['xyz'])>1e-8 or angle(quat,old['quat'])>1e-5:
                raise Rejected('Recheck changed frozen target or expired')
        elif old:
            raise Rejected('Preflight ticket already used')
        if not recheck:
            self.claim(ticket,a.session)
            self.records.clear()
        if np.max(np.abs(snapshot['velocity'])) > .02:
            raise Rejected('Preflight requires stationary arms')
        p._panel_begin_planning(snapshot['joints'].reshape(2,7).tolist())
        search = Search(a)
        from baseline_ready_pose import contract, json_value
        evidence=dict(start_joint_rad=snapshot['joints'],goal_xyz_m=xyz,goal_wxyz=quat,
                      gripper_state=a.grip,model_contract=contract(p),world_snapshot=p.world_cfg,
                      session_tag=hashlib.sha256(str(a.session).encode()).hexdigest()[:12])
        evidence=json.loads(json.dumps(evidence,default=json_value))
        self.write(ticket,accepted=None,stage='solving',**evidence)
        try:
            solution = search.solve(snapshot['joints'],xyz,quat,allow_reshape=not recheck)
        except Exception:
            self.write(ticket,accepted=False,rejections=search.rejections,stage='recheck' if recheck else 'preflight',
                       planning_seconds=search.clock()-(search.deadline-search.p.budget_s),**evidence)
            raise
        search.check()
        latest = p._v31_bridge.snapshot(max_age=.25)
        if np.max(np.abs(latest['joints']-snapshot['joints']))>.002:
            raise Rejected('Preflight start changed; discard unsent trajectories')
        solution.update(session=a.session,xyz=xyz,quat=quat,grip=a.grip,model_contract=evidence['model_contract'],
                        expires=self.clock()+30.,phase='ready',rechecked=recheck)
        self.records[ticket]=solution
        self.write(ticket,accepted=True,stage='recheck' if recheck else 'preflight',
            needs_reshape=solution['reshape'] is not None,rejections=search.rejections,
            planning_seconds=search.clock()-(search.deadline-search.p.budget_s),**evidence,
            advance=solution['advance']['metrics'],
            reshape=None if solution['reshape'] is None else solution['reshape']['metrics'])

    def ready_path(self,a,start,goal,deadline):
        from baseline_approach_path import Policy
        points,dt=a.motion(start,goal,deadline)
        for _ in range(6):
            a.check()
            if self.clock()>=deadline:raise TimeoutError('Ready planning budget exhausted')
            panel=a.sample(start,points,dt,.001)
            ratios=a.dynamics(panel,.001)
            if np.all(ratios<=1):break
            dt*=max(ratios[0],np.sqrt(ratios[1]),np.cbrt(ratios[2]))*1.1
        else:raise Rejected('Ready path dynamics failed')
        if not np.allclose(panel[:,:7],start[:7],atol=1e-8,rtol=0):
            raise Rejected('Ready path moved left arm')
        if np.any(panel<a.bounds[0]) or np.any(panel>a.bounds[1]):
            raise Rejected('Ready path joint bounds failed')
        if np.max(np.abs(points[-1]-goal))>.002:
            raise Rejected('Ready path endpoint joint mismatch')
        if np.min(np.minimum(points[-1,7:]-a.bounds[0,7:],a.bounds[1,7:]-points[-1,7:]))<np.deg2rad(1):
            raise Rejected('Ready endpoint reserve failed')
        def check():
            a.check()
            if self.clock()>=deadline:raise TimeoutError('Ready planning deadline exceeded')
        a.collision(panel,check);check()
        return dict(points=points,start=start.copy(),dt=dt,goal=goal.copy())

    def prepare_ready(self,ticket,snapshot):
        from baseline_ready_pose import candidate
        p=self.p;a=Adapter(p);a.session=getattr(snapshot['panel'],'_seldon_session',None)
        a.description=p.target_description
        if a.session is None or np.max(np.abs(snapshot['velocity']))>.02:
            raise Rejected('Ready preflight requires stationary session')
        self.claim(ticket,a.session)
        original=np.asarray(p.target_arm_joint,float).copy()
        if original.shape!=(14,):raise Rejected('Invalid ready joint request')
        original[:7]=snapshot['joints'][:7]
        p._panel_begin_planning(snapshot['joints'].reshape(2,7).tolist())
        deadline=self.clock()+8.;reason=None;chosen=None
        try:
            chosen=candidate(p,original)
        except (ValueError,KeyError,OSError) as error:reason=str(error)
        path=None
        if chosen is not None:
            try:path=self.ready_path(a,snapshot['joints'],chosen,deadline)
            except Rejected as error:reason=str(error)
        if path is None:
            path=self.ready_path(a,snapshot['joints'],original,deadline)
            reason=reason or 'No improved ready candidate configured; using original'
        self.records.clear()
        from baseline_ready_pose import contract
        self.records[ticket]=dict(ready_path=path,session=a.session,grip=a.grip,model_contract=contract(p),
                                 phase='ready',expires=self.clock()+30.)
        self.write(ticket,accepted=True,stage='ready_preflight',goal_joint_rad=path['goal'],
                   optimized=chosen is not None and np.allclose(path['goal'],chosen),reason=reason)

    def dispatch(self,ticket,phase,snapshot):
        p=self.p;record=self.records.get(ticket)
        if not record or self.clock()>record['expires']:
            raise Rejected('Unknown/expired baseline trajectory ticket')
        if record['session']!=getattr(snapshot['panel'],'_seldon_session',None):
            raise Rejected('Baseline trajectory session changed')
        if 'model_contract' in record:
            from baseline_ready_pose import contract
            if record['model_contract']!=contract(p):
                raise Rejected('Baseline model/world changed; trajectory discarded')
        if phase=='baseline_ready':
            if record['phase']!='ready' or 'ready_path' not in record:
                raise Rejected('Ready ticket consumed/wrong phase')
            path=record['ready_path']
            if (not np.allclose(p.target_arm_joint,path['goal'],atol=1e-8,rtol=0)
                    or list(p.target_gripper_joint)!=record['grip']
                    or np.max(np.abs(snapshot['joints']-path['start']))>.002
                    or np.max(np.abs(snapshot['velocity']))>.02):
                raise Rejected('Ready command/start identity changed')
            a=Adapter(p);a.session=record['session'];a.description=p.target_description
            # Revalidate actual interpolation at the fresh start; don't solve
            # a different path after approval of the candidate.
            p._panel_begin_planning(snapshot['joints'].reshape(2,7).tolist())
            validation=Search(a)
            samples=a.sample(snapshot['joints'],path['points'],path['dt'],.001)
            if np.any(samples<a.bounds[0]) or np.any(samples>a.bounds[1]):
                raise Rejected('Ready splice exceeds joint limits')
            a.collision(samples,validation.check)
            if np.any(a.dynamics(samples,.001)>1):raise Rejected('Ready splice dynamics failed')
            validation.check()
            newest=p._v31_bridge.snapshot(max_age=.25)
            if np.max(np.abs(newest['joints']-snapshot['joints']))>.002:
                raise Rejected('Ready start moved during final validation')
            from curobo.types.state import JointState
            p._move_arm=[False,True];p.waypoints=path['points'].copy();p.limited_waypoints=path['points'].copy()
            p.waypoints_dt=path['dt'];p._cmd_idx=0
            p._cmd_plan=JointState(position=p.tensor_args.to_device(samples),joint_names=p.total_arm_joint_names)
            p.curobo_cmd.waypoints=path['points'].reshape(-1,2,7).tolist()
            p.curobo_cmd.waypoints_size=len(path['points']);p.curobo_cmd.waypoints_dt=path['dt']
            p.curobo_cmd.cmd_plan_size=len(samples);p._panel_replan_pending=False
            record['phase']='ready_dispatched'
            self.write(ticket,accepted=True,stage='ready',goal_joint_rad=path['goal'])
            return
        if phase=='baseline_advance_hold':
            if record['phase']!='consumed':
                raise Rejected('Baseline hold before advance completion')
            pos,quat=p._v31_bridge.tcp(snapshot['joints'])
            if (np.linalg.norm(pos[0,1]-record['xyz'])>.003 or angle(quat[0,1],record['quat'])>.5
                    or np.max(np.abs(snapshot['velocity']))>.02):
                raise Rejected('Baseline final hold not at verified endpoint')
            if (np.linalg.norm(p.target_arm_tool_position[1]-pos[0,1])>.001
                    or angle(p.target_arm_tool_orientation[1],quat[0,1])>.25):
                raise Rejected('Baseline final hold identity mismatch')
            p._move_arm=[False,True]
            p._cmd_plan=None
            p.curobo_cmd.arm_joint_command=snapshot['joints'].reshape(2,7).tolist()
            p.curobo_cmd.waypoints=[p.curobo_cmd.arm_joint_command]*2
            p.curobo_cmd.waypoints_size=2;p.curobo_cmd.waypoints_dt=.2;p.curobo_cmd.cmd_plan_size=0
            record['phase']='holding'
            record['hold_pose']=np.c_[p.target_arm_tool_position,p.target_arm_tool_orientation].copy()
            self.write(ticket,accepted=True,stage='hold',no_trajectory_replay=True)
            return
        if record['phase']!='ready':
            raise Rejected('Baseline trajectory consumed or wrong phase')
        reshape=phase=='baseline_reshape'
        if reshape != (record['reshape'] is not None):
            raise Rejected('Visual recheck required before advance / no reshape planned')
        path=record['reshape' if reshape else 'advance']
        if (np.linalg.norm(p.target_arm_tool_position[1]-path['xyz'])>1e-8
                or angle(p.target_arm_tool_orientation[1],path['quat'])>1e-5
                or list(p.target_gripper_joint)!=record['grip']):
            raise Rejected('Baseline trajectory target/gripper identity changed')
        if np.max(np.abs(snapshot['joints']-path['start']))>.002 or np.max(np.abs(snapshot['velocity']))>.02:
            raise Rejected('Baseline execution start changed; discard trajectory')
        a=Adapter(p);a.session=record['session'];a.description=p.target_description
        p._panel_begin_planning(snapshot['joints'].reshape(2,7).tolist())
        validation=Search(a)
        samples=a.sample(snapshot['joints'],path['points'],path['dt'],.001)
        metrics=validation.metrics(snapshot['joints'],samples,path['xyz'],path['quat'],reshape=reshape)
        if np.any(a.dynamics(samples,.001)>1):
            raise Rejected('Spliced RT dynamics failed')
        validation.check()
        newest=p._v31_bridge.snapshot(max_age=.25)
        if np.max(np.abs(newest['joints']-snapshot['joints']))>.002:
            raise Rejected('Start moved during final validation')
        from curobo.types.state import JointState
        points=path['points']
        p._move_arm=[False,True]
        p.waypoints=points.copy();p.limited_waypoints=points.copy();p.waypoints_dt=path['dt']
        p._cmd_plan=JointState(position=p.tensor_args.to_device(samples),joint_names=p.total_arm_joint_names)
        p._cmd_idx=0
        p.curobo_cmd.waypoints=points.reshape(-1,2,7).tolist()
        p.curobo_cmd.waypoints_size=len(points);p.curobo_cmd.waypoints_dt=path['dt']
        p.curobo_cmd.cmd_plan_size=len(samples)
        p._panel_replan_pending=False
        record['phase']='reshaped' if reshape else 'consumed'
        # Allow time for the known adjustment trajectory plus a fresh image;
        # this is a plan ticket lifetime, never a motion/watchdog lease.
        record['expires']=self.clock()+min(90.,len(points)*path['dt']+30.)
        self.write(ticket,accepted=True,stage='reshape' if reshape else 'advance',metrics=metrics,
                   target_joint_rad=points[-1],validated_start_joint_rad=snapshot['joints'])

    def compute(self):
        from cuarm_state import CuroboStatus
        p=self.p;parsed=decode(p.target_description)
        if parsed is None:
            return False
        phase,ticket=parsed
        try:
            guard=getattr(p._v31_bridge,'guard',None)
            if guard is not None and guard.fault:
                raise Rejected('Baseline cannot take over a faulted visual guard: '+str(guard.fault))
            snapshot=p._v31_bridge.snapshot(max_age=.25)
            session=getattr(snapshot['panel'],'_seldon_session',None)
            if self.last_description==p.target_description:
                if session!=self.last_session:
                    raise Rejected('Baseline executing session changed')
                if getattr(p,'_panel_replan_pending',False):
                    raise Rejected('Baseline transport rejected stale start; new preflight required')
                record=self.records.get(ticket)
                if record and record['phase'] in ('ready_dispatched','ready_holding'):
                    goal=record['ready_path']['goal']
                    if not np.allclose(p.target_arm_joint,goal,atol=1e-8,rtol=0):
                        raise Rejected('Ready execution goal changed')
                    if (np.max(np.abs(snapshot['joints']-goal))<=.002 and
                            np.max(np.abs(snapshot['velocity']))<=.02):
                        if record['phase']=='ready_dispatched':
                            record['phase']='ready_holding';p._cmd_plan=None
                            held=snapshot['joints'].reshape(2,7).tolist()
                            p.curobo_cmd.arm_joint_command=held;p.curobo_cmd.waypoints=[held,held]
                            p.curobo_cmd.waypoints_size=2;p.curobo_cmd.waypoints_dt=.2;p.curobo_cmd.cmd_plan_size=0
                        p.set_gripper_command()
                        return True
                if record and record['phase']=='holding':
                    if not np.allclose(np.c_[p.target_arm_tool_position,p.target_arm_tool_orientation],record['hold_pose'],atol=1e-8,rtol=0):
                        raise Rejected('Baseline completed hold pose changed')
                    p.set_gripper_command()
                    return True
                if record and list(p.target_gripper_joint)!=record['grip']:
                    raise Rejected('Gripper changed before final measured hold')
                if record and phase in ('baseline_advance','baseline_reshape'):
                    path=record['advance' if phase=='baseline_advance' else 'reshape']
                    if np.linalg.norm(p.target_arm_tool_position[1]-path['xyz'])>1e-8 or angle(p.target_arm_tool_orientation[1],path['quat'])>1e-5:
                        raise Rejected('Executing baseline target identity changed')
                if not self.failed and phase in ('baseline_advance','baseline_reshape','baseline_ready'):
                    p._get_next_planned_arm_cmd()
                return True
            self.last_description=p.target_description;self.last_session=session;self.failed=False
            if guard is not None:guard.reset()
            p._clear_planned_result();p._hold_current_arm_command()
            p.status=CuroboStatus.PLANNING;p._curobo_planning_status_planning_count=0
            if phase=='baseline_ready_preflight':
                p._panel_replan_pending=False
                self.prepare_ready(ticket,snapshot)
            elif phase in ('baseline_preflight','baseline_recheck'):
                p._panel_replan_pending=False
                self.prepare(ticket,phase=='baseline_recheck',snapshot)
            else:
                self.dispatch(ticket,phase,snapshot)
                p._get_next_planned_arm_cmd()
            p.status=CuroboStatus.PLANNING_SUCCEEDED
            p._prev_target_tool_position=p.target_arm_tool_position.copy()
            p._prev_target_tool_orientation=p.target_arm_tool_orientation.copy()
            p._prev_target_arm_joint=(p.target_arm_joint.copy() if phase=='baseline_ready' else snapshot['joints'].copy())
            p.set_gripper_command()
        except Exception as error:
            self.failed=True
            self.records.pop(ticket,None)
            p._clear_planned_result();p._hold_current_arm_command();p.status=CuroboStatus.PLANNING_FAILED
            self.write(ticket,accepted=False,stage=phase,error=f'{type(error).__name__}: {error}')
        return True


def compute(planner):
    if decode(planner.target_description) is None:
        # Invalidate tickets across unrelated actions, cancellation or a new
        # grasp. A visual hold between reshape and recheck is the sole exception.
        instance=getattr(planner,'_baseline_approach_transactions',None)
        if instance and planner.target_description not in ('book_v3_1_stream','book_v3_xyz_servo'):
            instance.records.clear()
            instance.last_description=None
        return False
    if not hasattr(planner,'_baseline_approach_transactions'):
        planner._baseline_approach_transactions=Transactions(planner)
    return planner._baseline_approach_transactions.compute()


def observe_visual(planner):
    """A visual producer interrupts any native plan; only reshape recheck survives."""
    instance=getattr(planner,'_baseline_approach_transactions',None)
    if instance is not None:
        instance.last_description=None
        instance.records={key:record for key,record in instance.records.items() if record['phase']=='reshaped'}
