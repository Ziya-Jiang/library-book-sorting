"""Book board Planner budgets. Never publish from the solver thread.

CuRobo's timeout is checked between attempts, so it is not a hard wall-clock
deadline. The caller also stops waiting after 8s. A late CUDA result is discarded;
until its thread drains no other solver or motion request may use that engine.
The live main loop, not this worker, maintains the existing short hold lease.
Feedback/session/STOP checks and the publisher's 10s watchdog are unchanged.
"""
import copy
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import numpy as np

SOLVER_SECONDS = 6.0
RETURN_SECONDS = 8.0
MAX_ATTEMPTS = 3


class FailedPlan:
    def __init__(self, reason, elapsed):
        self.success = np.array(False)
        self.status = SimpleNamespace(name=reason)
        self.ik_time = self.graph_time = self.trajopt_time = self.finetune_time = 0.
        self.solve_time = self.total_time = elapsed


class BoundedCall:
    """One isolated computation; a timed-out result can never become successful."""
    def __init__(self):
        self.thread = None
        self.expired = False

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def run(self, work, deadline):
        if self.busy:
            return None, 'previous_solver_still_running'
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            return None, 'planning_return_budget_exhausted'
        done = threading.Event()
        box = {}
        self.expired = False

        def solve():
            try:
                box['result'] = work()
            except Exception as error:
                box['error'] = f'{type(error).__name__}: {error}'
            finally:
                box['finished'] = time.monotonic()
                done.set()

        self.thread = threading.Thread(target=solve, daemon=True, name='book-bounded-solver')
        self.thread.start()
        if not done.wait(max(0., deadline-time.monotonic())) or box['finished'] >= deadline:
            self.expired = True
            return None, 'planning_return_budget_exhausted'
        if 'error' in box:
            return None, 'solver_exception: '+box['error']
        return box['result'], None


class SolverSnapshot:
    """Bind solver methods without inheriting the live Planner's destructor.

    copy.copy(CuroboPlanner) would share sockets/threads and call terminate()
    when the snapshot is collected. This data-only facade owns neither.
    """
    def __init__(self, planner):
        self.__dict__.update(vars(planner))
        self._source_type = type(planner)

    def __getattr__(self, name):
        method = getattr(self._source_type, name)
        return method.__get__(self, self._source_type) if hasattr(method, '__get__') else method


def freeze_inputs(planner):
    """Share the serialized GPU engine, freeze all ordinary solver inputs."""
    snapshot = SolverSnapshot(planner)
    for key, value in vars(planner).items():
        if isinstance(value, np.ndarray):
            setattr(snapshot, key, value.copy())
    for key in ('target_link_names', 'total_arm_joint_names', 'curobo_input_joint_names',
                'arm_joint_size', 'target_gripper_joint', 'last_received_target_command'):
        if hasattr(planner, key):
            setattr(snapshot, key, copy.deepcopy(getattr(planner, key)))
    if hasattr(planner, '_v31_bridge'):
        snapshot._v31_bridge = copy.copy(planner._v31_bridge)
        snapshot._v31_bridge.p = snapshot
    return snapshot


class BookPlanningBudget:
    def __init__(self, planner, collision, directory=None):
        self.planner = planner
        self.collision = collision
        self.call = BoundedCall()
        self.started = None
        self.serial = 0
        self.directory = Path(directory or Path(__file__).resolve().parents[3] /
                              'robot_cockpit/launcher/runtime/start_arms/planning_diagnostics')

    def begin(self):
        self.started = time.monotonic()

    def configure(self, config):
        result = config.clone()
        result.timeout = min(float(config.timeout), SOLVER_SECONDS)
        result.max_attempts = min(int(config.max_attempts), MAX_ATTEMPTS)
        # No constraint, interpolation, speed, or collision settings changed.
        return result

    def run(self, kind, work):
        started = self.started if self.started is not None else time.monotonic()
        self.serial += 1
        snapshot = freeze_inputs(self.planner)
        snapshot._book_budget_job = dict(request_id=f'{os.getpid()}-{time.time_ns()}-{self.serial}',
            action=snapshot.target_description, kind=kind, started_monotonic_s=started,
            unix_s=time.time(), solver_timeout_s=SOLVER_SECONDS,
            return_budget_s=RETURN_SECONDS, max_attempts=MAX_ATTEMPTS,
            target_xyz_m=np.asarray(snapshot.target_arm_tool_position).tolist(),
            target_wxyz=np.asarray(snapshot.target_arm_tool_orientation).tolist())
        result, error = self.call.run(lambda: work(snapshot), started+RETURN_SECONDS)
        if error:
            result = FailedPlan(error, time.monotonic()-started)
            self.planner._book_plan_failure = dict(snapshot._book_budget_job,
                reason=error, elapsed_s=time.monotonic()-started)
            # No CUDA access, result validation or filesystem I/O on this path.
            print('Book planning failed: '+json.dumps(self.planner._book_plan_failure)+
                  '; hold current pose; late result will not execute', flush=True)
        return result

    def block_while_draining(self):
        if not (self.call.expired and self.call.busy):
            return False
        from cuarm_state import CuroboStatus
        self.planner.status = CuroboStatus.PLANNING_FAILED
        self.planner._hold_current_arm_command()
        # Explicitly reject a newly received request; never execute it later just
        # because an older timed-out CUDA job finally released the engine.
        for previous, current in (('_prev_target_tool_position', 'target_arm_tool_position'),
                ('_prev_target_tool_orientation', 'target_arm_tool_orientation'),
                ('_prev_target_joint', 'target_arm_joint'),
                ('_prev_target_arm_joint', 'target_arm_joint'),
                ('_prev_target_description', 'target_description')):
            setattr(self.planner, previous, copy.deepcopy(getattr(self.planner, current)))
        return True

    def invoke(self, snapshot, function, joint_names, start, goal=None):
        """Runs under GPU lock and the actual temporary constraint contexts."""
        from book_planning_diagnostics import constraints, failure_evidence, result_summary
        job = snapshot._book_budget_job
        began = time.monotonic()
        limits = constraints(snapshot._motion_gen)
        try:
            result = function()
            try:
                summary = result_summary(result)
            except Exception as error:
                summary = dict(status=str(result.status), diagnostic_error=str(error))
            report = dict(job, solver_wall_s=time.monotonic()-began, **summary)
            report['diagnostic_source'] = 'actual GPU constraints + CPU geometry; diagnostic only'
            if not bool(result.success.item()):
                report['constraints'] = limits
                report.update(failure_evidence(snapshot, self.collision, joint_names, start, goal, result))
                print('Book planning failed: '+str(result.status)+'; holding; diagnostics='+
                      str(self.directory/(job['request_id']+'.json')), flush=True)
            report['elapsed_s'] = time.monotonic()-job['started_monotonic_s']
            report['late_result_not_executable'] = report['elapsed_s'] >= RETURN_SECONDS
            self.save(report)
            return result
        except Exception as error:
            self.save(dict(job, reason=f'{type(error).__name__}: {error}', constraints=limits,
                           elapsed_s=time.monotonic()-job['started_monotonic_s']))
            raise

    def save(self, report):
        # Only the solver worker writes. Disk delays cannot extend the caller's budget.
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory/(report['request_id']+'.json')
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            temp.replace(path)
        except Exception as error:
            print(f'Book planning diagnostics unavailable: {type(error).__name__}: {error}', flush=True)


def install(planner):
    """Only called by the book-profile board entry, never legacy/v4.1 entries."""
    expected = Path(__file__).resolve().parents[3]/'dual_v2_2_configuration'
    if Path(os.environ.get('CUARM_CONFIG_ROOT', expected)).resolve() != expected.resolve():
        return
    from v3_1_panel_transport import PanelTransport
    if not SOLVER_SECONDS < RETURN_SECONDS < PanelTransport.PLANNING_LEASE:
        raise ValueError('Book solver budgets must precede the unchanged planning lease')
    planner._book_plan_budget = BookPlanningBudget(planner, planner._v31_direct.collision)
    # Request existing final optimizer metrics; do not enable iteration history.
    planner._motion_gen.store_debug_in_result = True
    print(f'Book planning budget: solver={SOLVER_SECONDS:g}s, attempts={MAX_ATTEMPTS}, '
          f'return={RETURN_SECONDS:g}s, watchdog={PanelTransport.PLANNING_LEASE:g}s; '
          'failed/late plans hold without publishing motion', flush=True)
