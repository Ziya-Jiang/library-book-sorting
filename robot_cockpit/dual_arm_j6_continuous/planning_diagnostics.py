"""Independent profile diagnostics and explicitly selected path tolerance."""
import json
import os
from pathlib import Path
import time


def install(base):
    # Process-local override: leave the shared module's file and old planners
    # untouched. User selected 1 degree for planned paths on 2026-11-13.
    import book_lower_pose
    book_lower_pose.PLANNED_ANGLE_DEG = 1.0
    print('Dual-arm planned path orientation limit: 1.0deg; fixed-axis limit: 3mm.', flush=True)
    import book_post_lift_pose
    from lift_continuity import install as install_lift_continuity
    install_lift_continuity(book_post_lift_pose)
    # Continuous profile never replaces a failed GPU path with CPU IK pieces.
    import baseline_withdraw_path
    baseline_withdraw_path.refine = lambda planner: False
    original = base.CuroboPlanner

    class DiagnosticPlanner(original):
        def _retrieve_target_command(self, command):
            from planning_receipt import decode
            clean, request = decode(command)
            previous = getattr(self, '_trial_request', None)
            if request and previous and request['request_id'] == previous['request_id'] and request != previous:
                raise ValueError('Trial request identity reused for a different target')
            accepted = super()._retrieve_target_command(clean)
            if accepted:
                if request != previous:
                    self._trial_ever_accepted = False
                    if request is not None:
                        self._panel_replan_pending = True
                self._trial_request = request
            return accepted

        def __init__(self, *args, **kwargs):
            # GPU optimization has more internal control points than old J6.
            # Ordinary joint paths trim to <=20 wire points; Cartesian phases
            # are explicitly resampled and checked by continuous_trajectory.
            kwargs['trajopt_tsteps'] = 20
            kwargs['trim_steps'] = [2, -2]
            from planning_deadline import time_options
            print('Continuous dt policy: '+json.dumps(time_options(kwargs)),flush=True)
            # Independent trial only: user-selected timing factor; solver limits unchanged.
            if 'time_dilation_factor' in kwargs:
                kwargs['time_dilation_factor'] = 1.0
                print(f"Right trial timing: time_dilation_factor={kwargs['time_dilation_factor']:g}; dynamics limits unchanged.", flush=True)
            self._trial_z_hold_weight = 20.0
            super().__init__(*args, **kwargs)
            from planning_deadline import install as install_deadline
            install_deadline(self)
            from j6_only import publish_capability
            from gpu_preparation import PreparationService
            self._preparation_service = PreparationService(self)
            publish_capability()

        def terminate(self):
            service = getattr(self, '_preparation_service', None)
            if service is not None:
                service.close()
            return super().terminate()

        def _retrieve_motion_gen_result(self, result, **kwargs):
            from continuous_trajectory import ACTIONS, prepare, validate_publication
            continuous = self.target_description in ACTIONS
            self._continuous_diagnostics = None
            if hasattr(result, '_continuous_validated'):
                from planning_deadline import commit
                from planning_receipt import begin_candidate, write_receipt
                request = getattr(self, '_trial_request', None)
                try:
                    if request is not None: begin_candidate(request)
                    accepted = commit(self,result)
                except Exception as error:
                    accepted = False
                    self.status = base.CuroboStatus.PLANNING_FAILED
                    self._hold_current_arm_command()
                    self._continuous_diagnostics = dict(accepted=False,reason=str(error))
                self._trial_ever_accepted = getattr(self,'_trial_ever_accepted',False) or accepted
                if request is not None:
                    try:
                        write_receipt(request,status=self.status,accepted=accepted,
                            ever_accepted=self._trial_ever_accepted,hold_only=not accepted,
                            diagnostics=dict(continuous=self._continuous_diagnostics,
                                path=getattr(self,'_book_lower_path_diagnostics',None),
                                planning_budget=getattr(self,'_planning_deadline_report',None)))
                    except Exception as error:
                        accepted=False
                        self.status=base.CuroboStatus.PLANNING_FAILED
                        self._hold_current_arm_command()
                        print('Validated trajectory receipt failed; holding: '+str(error),flush=True)
                return accepted
            if continuous and bool(result.success.item()):
                result = prepare(self, result)
            from j6_only import ACTION, check_path, moving_joint
            if moving_joint(self.target_description) is not None and bool(result.success.item()):
                try:
                    names = list(self.total_arm_joint_names)
                    for candidate in (result.get_interpolated_plan(), result.optimized_plan):
                        full = self._motion_gen.get_full_js(candidate).get_ordered_joint_state(names)
                        check_path(full.position.detach().cpu().numpy(), self.arm_joint_position, names, moving_joint(self.target_description))
                    from j6_only import validate_result_before_publish
                    validate_result_before_publish(self, result)
                    print('Single right wrist joint path verified; fixed joints, collision, limits and dynamics checked.', flush=True)
                except Exception as error:
                    print('Single right wrist joint path rejected: '+str(error), flush=True)
                    self.status = base.CuroboStatus.PLANNING_FAILED
                    self._hold_current_arm_command()
                    return False
            request = getattr(self, '_trial_request', None)
            if request is not None:
                from planning_receipt import begin_candidate
                begin_candidate(request)
            self._book_lower_path_diagnostics = None
            from lift_continuity import lift_start, MAX_JOINT_EXCURSION_DEG
            local_start = lift_start(self)
            self._lift_continuity_diagnostics = (dict(getattr(self, '_lift_continuity_diagnostics', None) or {}, policy='measured_joint_neighbourhood',
                limit_deg=MAX_JOINT_EXCURSION_DEG, start_joints_rad=local_start.tolist(),
                accepted=False, phase='solver_result', solver_status=str(result.status))
                if local_start is not None else None)
            accepted = original._retrieve_motion_gen_result(self, result, **kwargs)
            if continuous and accepted:
                try:
                    validate_publication(self, result)
                except Exception as error:
                    accepted = False
                    self._continuous_diagnostics = dict(self._continuous_diagnostics or {},
                        accepted=False, publication_error=f'{type(error).__name__}: {error}')
                    self.status = base.CuroboStatus.PLANNING_FAILED
                    self._hold_current_arm_command()
                    print('Continuous publication rejected: '+str(error), flush=True)
            if request is not None:
                # This hook runs before _retrieve_motion_gen_result returns to
                # the publisher. False means only measured hold was retained.
                self._trial_ever_accepted = getattr(self, '_trial_ever_accepted', False) or bool(accepted)
                try:
                    from planning_receipt import write_receipt
                    write_receipt(request, status=self.status, accepted=bool(accepted),
                                  ever_accepted=self._trial_ever_accepted,
                                  hold_only=(getattr(self, '_cmd_plan', 'unknown') is None
                                             and getattr(self.curobo_cmd, 'cmd_plan_size', -1) == 0
                                             and getattr(self.curobo_cmd, 'waypoints_size', -1) == 0),
                                  diagnostics=dict(path=self._book_lower_path_diagnostics,
                                      continuous=self._continuous_diagnostics,
                                      planning_budget=getattr(self,'_book_plan_failure',None) or getattr(self,'_planning_deadline_report',None),
                                      lift_continuity=getattr(self, '_lift_continuity_diagnostics', None)))
                except Exception as error:
                    print('Trial receipt unavailable; automatic retry disabled: '+str(error), flush=True)
            return accepted

        def _plan_cartesian_candidate(self, cu_js, position, orientation, plan_config):
            from continuous_trajectory import ACTIONS
            if self.target_description not in ACTIONS:
                return original._plan_cartesian_candidate(self,cu_js,position,orientation,plan_config)
            from planning_deadline import validated_result
            budget=self._book_plan_budget
            position,orientation=position.copy(),orientation.copy()
            def work(snapshot):
                config=budget.configure(plan_config)
                from descent_refinement import is_descent
                if is_descent(snapshot):
                    # RT time scaling is checked on the real controller spline.
                    # Avoid divergent secondary dt compression for descent.
                    config.enable_finetune_trajopt=False
                    snapshot._deadline_report['descent_policy']='GPU_MotionGen_then_local_GPU_IK_then_verified_RT_retime'
                result=snapshot._plan_cartesian_unbounded(cu_js,position,orientation,config)
                return validated_result(snapshot,result,original._retrieve_motion_gen_result)
            return budget.run('continuous_cartesian_full_pipeline',work)

        def _plan_cartesian_unbounded(self, cu_js, position, orientation, plan_config):
            self._lift_continuity_diagnostics = None
            return original._plan_cartesian_unbounded(self, cu_js, position, orientation, plan_config)

        def _plan_joint_candidate(self, start, goal):
            result=self._joint_candidate_unchecked(start,goal)
            if hasattr(self,'_continuous_planning_deadline'):
                from planning_deadline import validated_joint_result
                return validated_joint_result(self,result,original._retrieve_motion_gen_result)
            return result

        def _joint_candidate_unchecked(self, start, goal):
            from preparation_connection import ACTION as PREP_ACTION, candidate as preparation_candidate
            if str(self.target_description).split(':',1)[0] == PREP_ACTION:
                return preparation_candidate(self,start,goal)
            from j6_only import ACTION, moving_joint
            if str(self.target_description).split(':',1)[0] == 'right_trial_j7_only':
                from j7_trajectory import candidate
                return candidate(self, start, goal)
            if moving_joint(self.target_description) is not None:
                from j6_trajectory import candidate
                return candidate(self, start, goal)
            result = original._plan_joint_candidate(self, start, goal)
            if not bool(result.success.item()):
                try:
                    from v4_1_endpoint_diagnostics import diagnose_endpoint
                    report = dict(action=self.target_description, unix_s=time.time(),
                                  solver_status=str(result.status), source='CPU diagnostics; GPU decision authoritative')
                    for name, state in [('start', start), ('goal', goal)]:
                        report[name] = diagnose_endpoint(self._v31_direct.collision, state.joint_names,
                                                         state.position.detach().cpu().numpy(), max_contacts=20)
                    print('Dual-gripper endpoint diagnostics: ' + json.dumps(report), flush=True)
                    directory = Path(os.environ.get('DUAL_GRIPPER_DIAGNOSTICS',
                                     str(Path(__file__).parent/'runtime/endpoint_diagnostics')))
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f'{time.time_ns()}.json'
                    path.write_text(json.dumps(report, indent=2)+'\n')
                except Exception as error:
                    print(f'Dual-gripper diagnostics unavailable: {type(error).__name__}: {error}', flush=True)
            return result

    base.CuroboPlanner = DiagnosticPlanner
