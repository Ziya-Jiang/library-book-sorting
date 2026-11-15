"""Current continuous profile only: bound GPU + validation before the 10s lease.

Worker owns scratch command buffers; no socket/transport writes. A late worker
is quarantined, not cancelled mid CUDA, and can never publish its result.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
from book_planning_budget import BookPlanningBudget, FailedPlan, freeze_inputs

SOLVER_SECONDS = 4.0
RETURN_SECONDS = 7.5
MAX_ATTEMPTS = 3
MINIMUM_DT = .0625  # exact float32; RT wire requires interval >= .05 seconds

# Only validated command outputs may cross from the scratch planner to main.
COMMAND_FIELDS = ('_cmd_plan', '_cmd_idx', 'waypoints', 'waypoints_velocity',
    'waypoints_acceleration', 'waypoints_jerk', 'waypoints_dt', 'limited_waypoints',
    'status', '_v31_retryable_height', '_book_post_lift_diagnostics', '_book_lower_path_diagnostics', '_continuous_diagnostics',
    '_lift_continuity_diagnostics')
WIRE_FIELDS = ('cmd_plan_size', 'waypoints_size', 'waypoints_dt', 'waypoints')


def time_options(kwargs):
    kwargs['minimum_trajectory_dt'] = MINIMUM_DT
    maximum = 7.0 / kwargs['trajopt_tsteps']
    if not 0 < MINIMUM_DT < maximum:
        raise ValueError('Invalid continuous solver dt interval')
    return dict(minimum_dt_s=MINIMUM_DT, maximum_dt_s=maximum,
                trajectory_horizon_s=7.0, control_points=kwargs['trajopt_tsteps'])


def checkpoint(planner):
    deadline = getattr(planner, '_continuous_planning_deadline', None)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('continuous_planning_return_deadline')


class PlanningBudget(BookPlanningBudget):
    def __init__(self, planner):
        super().__init__(planner, None, Path(__file__).parent/'runtime/planning_diagnostics')

    def begin(self):
        super().begin()
        self.planner._book_plan_failure = None
        self.planner._planning_deadline_report = None

    def configure(self, config):
        result = config.clone()
        result.timeout = min(float(config.timeout), SOLVER_SECONDS)
        result.max_attempts = min(int(config.max_attempts), MAX_ATTEMPTS)
        return result

    def run(self, kind, work):
        started = self.started if self.started is not None else time.monotonic()
        snapshot = freeze_inputs(self.planner)
        # Shared freeze_inputs deliberately shares ordinary live objects. This
        # profile also validates into scratch output, so isolate these explicitly.
        snapshot.curobo_cmd = copy.deepcopy(self.planner.curobo_cmd)
        snapshot._trial_request = copy.deepcopy(getattr(self.planner, '_trial_request', None))
        snapshot._continuous_planning_deadline = started + RETURN_SECONDS
        job = dict(request_id=(snapshot._trial_request or {}).get('request_id',str(time.time_ns())),
            action=snapshot.target_description, kind=kind, started_monotonic_s=started,
            solver_timeout_s=SOLVER_SECONDS, return_budget_s=RETURN_SECONDS,
            max_attempts=MAX_ATTEMPTS, planning_lease_s=10., unix_s=time.time())
        snapshot._deadline_report = job
        def calculate():
            try:
                result = work(snapshot)
                job.update(elapsed_s=time.monotonic()-started, status=str(result.status),
                    late_result_discarded=time.monotonic() >= started+RETURN_SECONDS)
                return result
            except Exception as error:
                job.update(reason=f'{type(error).__name__}: {error}',elapsed_s=time.monotonic()-started)
                raise
            finally:
                job['worker_finished_monotonic_s'] = time.monotonic()
                self.save(job)  # worker only; disk cannot extend the main wait
        result, error = self.call.run(calculate, started+RETURN_SECONDS)
        if error:
            detail=dict(job, reason=error, elapsed_s=time.monotonic()-started,
                        failure_source='continuous_planning_deadline', late_result_discarded=True)
            self.planner._book_plan_failure = detail
            print('Continuous planning failed; controlled hold; '+json.dumps(detail),flush=True)
            return FailedPlan(error,time.monotonic()-started)
        self.planner._planning_deadline_report = copy.deepcopy(job)
        return result

    def invoke(self, snapshot, function, joint_names, start, goal=None):
        checkpoint(snapshot)
        began=time.monotonic()
        result=function()
        from book_planning_diagnostics import result_summary, summary
        report=snapshot._deadline_report
        report.update(solver=result_summary(result), solver_wall_s=time.monotonic()-began)
        report['dt_limits'] = {}
        for name in ('trajopt_solver','finetune_trajopt_solver'):
            config=getattr(getattr(snapshot._motion_gen,name,None),'traj_evaluator_config',None)
            if config is not None:
                report['dt_limits'][name] = dict(min_dt=float(config.min_dt),max_dt=float(config.max_dt))
        report['optimizer_dt']={}
        for name,stage in (getattr(result,'debug_info',None) or {}).items():
            if name in ('trajopt_result','finetune_trajopt_result'):
                report['optimizer_dt'][name]=dict(dt=summary(getattr(stage,'optimized_dt',None)),
                    exception=str((getattr(stage,'debug_info',None) or {}).get('dt_exception')))
        checkpoint(snapshot)
        return result


def install(planner):
    from v3_1_panel_transport import PanelTransport
    if not SOLVER_SECONDS < RETURN_SECONDS < PanelTransport.PLANNING_LEASE:
        raise ValueError('Continuous planning budget must precede transport lease')
    if planner._motion_gen is None:return
    planner._book_plan_budget=PlanningBudget(planner)
    planner._motion_gen.store_debug_in_result=True
    print(f'Continuous planning: solver {SOLVER_SECONDS}s; return including path checks {RETURN_SECONDS}s; lease unchanged {PanelTransport.PLANNING_LEASE}s',flush=True)


def validate_base(snapshot,result,original_retrieve):
    # The shared endpoint check assumes a main-loop caller. Serialize it with
    # the readonly preparation service when running in the bounded worker.
    original_check=snapshot._check_target_arm_collision
    def locked_check(*args,**kwargs):
        with snapshot._gpu_lock:
            return original_check(*args,**kwargs)
    snapshot._check_target_arm_collision=locked_check
    try:return original_retrieve(snapshot,result)
    finally:snapshot._check_target_arm_collision=original_check


def validated_result(snapshot,result,original_retrieve):
    """All expensive checks happen in the bounded worker, with scratch outputs."""
    from continuous_trajectory import prepare,validate_publication
    checkpoint(snapshot)
    if not bool(result.success.item()):return result
    from descent_refinement import is_descent,refine
    if is_descent(snapshot):
        result=refine(snapshot,result)
        snapshot._deadline_report['descent_refinement']=result.descent_refinement
    result=prepare(snapshot,result)
    if not bool(result.success.item()):return result
    checkpoint(snapshot)
    accepted=validate_base(snapshot,result,original_retrieve)
    if not accepted:return FailedPlan('continuous_base_path_rejected',result.total_time)
    validate_publication(snapshot,result)
    checkpoint(snapshot)
    return seal(snapshot,result)


def validate_wire_timing(command):
    count=int(command.waypoints_size)
    dt=float(command.waypoints_dt)
    if count < 2 or count > 64 or not np.isfinite(dt) or not .05 <= dt <= 5. or count*dt > 30.:
        raise ValueError(f'prepublication_wire_timing: count={count}, interval_s={dt}; RT requires 0.05..5s and total<=30s')


def seal(snapshot,result):
    validate_wire_timing(snapshot.curobo_cmd)
    result._continuous_validated = dict(
        outputs={k:getattr(snapshot,k) for k in COMMAND_FIELDS if hasattr(snapshot,k)},
        wire={k:copy.deepcopy(getattr(snapshot.curobo_cmd,k)) for k in WIRE_FIELDS},
        start=np.array(snapshot.arm_joint_position,copy=True),
        grippers=np.array(snapshot.target_gripper_joint,copy=True),
        request=copy.deepcopy(snapshot._trial_request),deadline=snapshot._continuous_planning_deadline)
    return result


def validated_joint_result(snapshot,result,original_retrieve):
    checkpoint(snapshot)
    if not bool(result.success.item()):return result
    from j6_only import moving_joint,check_path,validate_result_before_publish
    from preparation_connection import ACTION as PREP_ACTION,validate_before_publish as validate_preparation
    if str(snapshot.target_description).split(':',1)[0]==PREP_ACTION:
        validate_preparation(snapshot,result)
    joint=moving_joint(snapshot.target_description)
    if joint is not None:
        names=list(snapshot.total_arm_joint_names)
        for candidate in (result.get_interpolated_plan(),result.optimized_plan):
            full=snapshot._motion_gen.get_full_js(candidate).get_ordered_joint_state(names)
            check_path(full.position.detach().cpu().numpy(),snapshot.arm_joint_position,names,joint)
        validate_result_before_publish(snapshot,result)
    checkpoint(snapshot)
    if not validate_base(snapshot,result,original_retrieve):return FailedPlan('joint_path_validation_rejected',result.total_time)
    checkpoint(snapshot)
    return seal(snapshot,result)


def commit(planner,result):
    proof=result._continuous_validated
    if time.monotonic()>=proof['deadline']:raise TimeoutError('validated_result_expired_before_commit')
    if getattr(planner,'_trial_request',None)!=proof['request']:raise ValueError('validated_request_changed')
    planner.raise_if_state_thread_failed()
    if np.max(abs(np.asarray(planner.arm_joint_position)-proof['start']))>.002:
        raise ValueError('validated_start_changed')
    if not np.array_equal(planner.target_gripper_joint,proof['grippers']):
        raise ValueError('validated_gripper_target_changed')
    for key,value in proof['outputs'].items():setattr(planner,key,value)
    for key,value in proof['wire'].items():setattr(planner.curobo_cmd,key,value)
    # The existing publisher still verifies fresh feedback, lease and session.
    return True
