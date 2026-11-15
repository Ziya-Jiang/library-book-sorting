"""J6-only contact-point search. Never substitute a fixed workspace Y for a book."""
import copy
import time
import numpy as np
from j6_geometry import align_goal, AlignmentRejected

CONTACT_CENTER_MARGIN_MM=20.0
MAX_CENTER_SHIFT_MM=80.0
Y_STEP_MM=10.0
# Read-only prediction budget, independent of the unchanged execution/RT lease.
SEARCH_SECONDS=45.0
PATH_SECONDS=10.0
ALIGN_SECONDS=8.0
MIN_ALIGN_SECONDS=2.0


def measured_contact_window(spine_xyz_mm, far_xyz_mm):
    """Use measured support on BOTH ends, not just a broad cover bounding box."""
    spans=[]
    for name,points in [('spine',spine_xyz_mm),('far_edge_1_2',far_xyz_mm)]:
        points=np.asarray(points,float)
        if (points.ndim!=2 or points.shape[1]!=3 or len(points)<30
                or not np.isfinite(points).all()):
            return dict(available=False,reason='insufficient_measured_'+name)
        spans.append(np.percentile(points[:,1],[5,95]).tolist())
    lo=max(s[0] for s in spans)+CONTACT_CENTER_MARGIN_MM
    hi=min(s[1] for s in spans)-CONTACT_CENTER_MARGIN_MM
    return dict(available=bool(lo<=hi),lower_y_mm=lo,upper_y_mm=hi,
                source='same_frame_spine_and_far_edge_1_2_base_Y',
                percentiles=[5,95],spine_y_span_mm=spans[0],far_edge_y_span_mm=spans[1],
                contact_center_margin_mm=CONTACT_CENTER_MARGIN_MM,
                limitation='observed_contact_center_range_not_complete_scene_collision_proof')


def contact_candidates(reference,batch_id,epoch,center_y_mm):
    if (reference.get('batch_id')!=batch_id or not batch_id or reference.get('epoch')!=epoch
            or not reference.get('frame_token')
            or reference.get('method')!='initial_left_wrist_spine_base_Z_median'
            or reference.get('far_edge',{}).get('reference')!='top_book_farthest_edge_1_2'):
        raise ValueError('自适应准备点缺少同轮同会话的书脊和1—2边证据')
    window=reference.get('contact_y_window',{})
    if not window.get('available'):
        return [],dict(window,reason=window.get('reason','no_measured_contact_window'))
    if window.get('source')!='same_frame_spine_and_far_edge_1_2_base_Y':
        raise ValueError('自适应准备点的测量范围来源无效')
    lo=float(window['lower_y_mm']);hi=float(window['upper_y_mm'])
    if not np.isfinite([lo,hi,center_y_mm]).all() or lo>hi:
        raise ValueError('自适应准备Y范围无效')
    lo=max(lo,center_y_mm-MAX_CENTER_SHIFT_MM);hi=min(hi,center_y_mm+MAX_CENTER_SHIFT_MM)
    if lo>hi:return [],dict(window,reason='contact_range_outside_allowed_center_shift')
    ys=[lo,hi]
    for distance in np.arange(Y_STEP_MM,MAX_CENTER_SHIFT_MM+.1,Y_STEP_MM):
        ys.extend([center_y_mm-distance,center_y_mm+distance])
    ys=sorted({float(y) for y in ys if lo<=y<=hi and abs(y-center_y_mm)>.01},
              key=lambda y:(abs(y-center_y_mm),y))
    return ys,dict(window,search_lower_y_mm=lo,search_upper_y_mm=hi,
                   max_center_shift_mm=MAX_CENTER_SHIFT_MM)


class PreparationRejected(ValueError):
    def __init__(self,detail):
        self.preparation_diagnostic=detail
        failures=detail.get('attempts',[])
        first=failures[0] if failures else {}
        super().__init__('Y→X→抬升→准备连接预检未通过；'
                         f'中心Y={detail["nominal_target_mm"][1]:.3f}mm，'
                         f'中心失败阶段={first.get("failed_stage","budget")}，'
                         f'原因={first.get("reason","search_budget_exhausted")}；'
                         f'已检查{len(failures)}个Y位置，耗时{detail["elapsed_s"]:.2f}s；'
                         '未执行Approach，逐项证据见preparation_failure.attempts')


def predicted_lift_state(model,proof,start,target,lift_mm):
    """Do not use another candidate's or an unvalidated path's terminal state."""
    if (not proof.get('accepted') or not proof.get('no_motion')
            or [p.get('stage') for p in proof.get('phases',[])]!=['approach_y','advance_x','lift_z']
            or not all(p.get('accepted') for p in proof['phases'])):
        raise ValueError('Incomplete Y/X/lift path proof')
    origin=np.asarray(proof.get('start_joints_rad'),float)
    goal=np.asarray(proof.get('target_xyz_m'),float)
    end=np.asarray(proof.get('end_joints_rad'),float)
    if (origin.shape!=(14,) or goal.shape!=(3,) or end.shape!=(14,)
            or not np.isfinite([origin,end]).all() or not np.isfinite(goal).all()
            or np.max(abs(origin-start))>1e-6 or np.max(abs(goal-target))>1e-9
            or not np.isclose(proof.get('lift_mm',float('nan')),lift_mm,rtol=0,atol=1e-6)):
        raise ValueError('Prediction start/target/lift identity mismatch')
    if np.any(end<model.bounds[0]) or np.any(end>model.bounds[1]) or np.max(abs(end[:7]-start[:7]))>1e-6:
        raise ValueError('Predicted lift endpoint changed left arm or joint limits')
    xyz,_=model.tcp(end)
    if np.max(abs(xyz[0,1]-target))>.003:
        raise ValueError('Predicted lift endpoint does not match requested TCP')
    return end.copy()


def next_contact_candidate(pending,center,attempts):
    """Order only: same-run axis trends never replace any candidate's checks.

    Use candidates that passed every non-axis geometry check. Extrapolate at
    most one grid step, so an uncertain trend cannot favor a distant endpoint
    merely because it is farther along the fitted line. Missing diagnostics
    retain the original nearest-center order.
    """
    observations=[]
    for attempt in attempts:
        if attempt.get('failed_stage') not in ('preparation_geometry','preparation_connection'):
            continue
        ratios=[]
        for detail in attempt.get('detail',{}).get('best_candidates',[]):
            checks=detail.get('checks',{})
            if not all(checks.get(k) is True for k in
                       ('position','tool_forward','joint_limits','held_left','endpoint_collision_free')):
                continue
            try:
                error=float(detail['axis_error_deg']);limit=float(detail['limits']['axis_deg'])
            except (KeyError,TypeError,ValueError):continue
            if np.isfinite([error,limit]).all() and error>=0 and limit>0:ratios.append(error/limit)
        if ratios:observations.append((float(attempt['y_mm']),min(ratios)))
    ranked=[]
    for y in pending:
        priority=dict(policy='nearest_center',axis_observations=len(observations))
        score=0.
        if len(observations)>=2:
            near=sorted(observations,key=lambda row:abs(row[0]-y))[:2]
            (a,ea),(b,eb)=near
            if abs(a-b)>.01:
                step=float(np.clip(y-a,-Y_STEP_MM,Y_STEP_MM))
                estimate=max(0.,ea+(ea-eb)/(a-b)*step)
                score=max(0.,estimate-1.)
                priority.update(policy='same_run_axis_trend_one_step',predicted_axis_ratio=estimate,
                                evidence_y_mm=[a,b],extrapolation_step_mm=step)
        ranked.append(((score,abs(y-center),y),y,priority))
    _,y,priority=min(ranked,key=lambda row:row[0])
    return y,priority


def select_preparation(model,joints,grippers,nominal_target_m,reference,batch_id,epoch,*,
                       predict_path,search_seconds=SEARCH_SECONDS):
    target=np.asarray(nominal_target_m,float)
    if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('准备点坐标无效')
    center=float(target[1]*1000)
    ys,window=contact_candidates(reference,batch_id,epoch,center)
    start=np.asarray(joints,float)
    if start.shape!=(14,) or not np.isfinite(start).all():raise ValueError('Invalid prediction start')
    xyz,_=model.tcp(start);lift_mm=float((target[2]-xyz[0,1,2])*1000)
    seconds=float(search_seconds)
    if not np.isfinite(seconds) or not 0<seconds<=SEARCH_SECONDS:raise ValueError('Invalid preparation chain budget')
    began=time.monotonic();deadline=began+seconds
    attempts=[]
    def solve(y,priority):
        remaining=deadline-time.monotonic()
        if remaining<=MIN_ALIGN_SECONDS:return None
        point=target.copy();point[1]=y*.001
        path_budget=min(PATH_SECONDS,remaining-MIN_ALIGN_SECONDS)
        item=dict(y_mm=y,status='rejected',path_budget_s=path_budget,candidate_priority=priority)
        attempts.append(item)
        phase_began=time.monotonic()
        proof=predict_path(point.copy(),path_budget)
        item.update(approach_lift_preflight=proof,path_elapsed_s=time.monotonic()-phase_began)
        if not proof.get('accepted'):
            item.update(failed_stage=proof.get('failed_stage','approach_lift_prediction'),
                        reason=proof.get('reason','path_rejected'))
            return None
        predicted=predicted_lift_state(model,proof,start,point,lift_mm)
        item['predicted_lift_joints_rad']=predicted.tolist()
        remaining=deadline-time.monotonic()
        if remaining<MIN_ALIGN_SECONDS:
            item.update(failed_stage='preparation_budget',reason='insufficient_reserved_alignment_time')
            return None
        alignment_budget=min(ALIGN_SECONDS,remaining)
        item['alignment_budget_s']=alignment_budget
        phase_began=time.monotonic()
        try:
            # Actual execution aligns at the arrived TCP, not the original reset.
            goal,geometry=align_goal(model,predicted,grippers,search_seconds=alignment_budget)
        except AlignmentRejected as error:
            best=error.summary.get('best_candidates',[])
            reasons=best[0].get('rejection_reasons',[]) if best else ['no_candidate']
            item.update(failed_stage='preparation_connection' if any('connection' in r for r in reasons) else 'preparation_geometry',
                        reason=','.join(reasons),detail=error.summary,
                        alignment_elapsed_s=time.monotonic()-phase_began)
            return None
        if time.monotonic()>deadline:
            item.update(failed_stage='preparation_budget',reason='late_preparation_result_discarded')
            return None
        # Diagnostic only. align_goal already passed every hard geometry/J7/
        # connection-path gate; do not spend more time chasing extra margin.
        quality=max(geometry['position_error_mm']/.25,geometry['axis_error_deg']/.5,
                    geometry['tool_forward_error_deg']/.5)
        item.update(status='accepted',quality_score=quality,alignment_elapsed_s=time.monotonic()-phase_began)
        detail=dict(nominal_target_mm=(target*1000).tolist(),selected_target_mm=(point*1000).tolist(),
                    center_offset_y_mm=float(y-center),grasp_center_y_mm=center,
                    contact_window=window,geometry=geometry,attempts=copy.deepcopy(attempts),
                    approach_lift_preflight=copy.deepcopy(proof),predicted_lift_joints_rad=predicted.tolist(),
                    preparation_start_source='validated_Y_X_lift_endpoint',
                    elapsed_s=time.monotonic()-began,sweep_considered=False,
                    budget_s=seconds,validation='Y_X_lift_then_GPU_preparation_connection_and_RT_checks',
                    selection_policy='first_fully_valid_candidate',
                    candidate_order_policy='same_run_axis_trend_then_nearest_center')
        return goal,detail
    result=solve(center,dict(policy='book_center_first'))
    if result is not None:return result
    pending=list(ys)
    while pending:
        if deadline-time.monotonic()<=MIN_ALIGN_SECONDS:break
        y,priority=next_contact_candidate(pending,center,attempts)
        pending.remove(y)
        result=solve(y,priority)
        if result is not None:return result
    raise PreparationRejected(dict(nominal_target_mm=(target*1000).tolist(),contact_window=window,
                                   attempts=attempts,elapsed_s=time.monotonic()-began,
                                   budget_s=seconds,budget_exhausted=deadline-time.monotonic()<=MIN_ALIGN_SECONDS))


def apply_preparation(report,goal,detail):
    """Only the push Approach changes; the measurement and grasp center stay intact."""
    center=float(report['measured']['center_mm'][1]);chosen=float(detail['selected_target_mm'][1])
    if (not np.isfinite([center,chosen]).all() or not np.isclose(center,detail['grasp_center_y_mm'])):
        raise ValueError('准备点与本轮原始书脊中心不一致')
    stages=copy.deepcopy(report['stages'])
    if len(stages)!=1 or stages[0].get('stage')!='approach_y' or stages[0].get('translation_axis')!='Y':
        raise ValueError('只允许替换初始拨书Approach的Y目标')
    stages[0]['tcp_m_wxyz'][1]=chosen*.001
    report.update(stages=stages,grasp_center_y_mm=center,preparation_y_mm=chosen,
                  offset_y_mm=chosen-float(report['right_tcp_mm'][1]),
                  preparation_target=dict(detail,target_joints_rad=np.asarray(goal).tolist()))


def grasp_y_after_push(previous_report,current_y_mm):
    if 'preparation_target' not in previous_report:return float(current_y_mm)
    center=float(previous_report['grasp_center_y_mm'])
    if not np.isfinite(center) or not np.isclose(center,previous_report['measured']['center_mm'][1]):
        raise ValueError('抓取中心不等于本轮实测书脊中心')
    return center
