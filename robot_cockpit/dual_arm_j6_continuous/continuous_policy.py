"""One request per Cartesian phase; no executing a chain of stop-to-stop pieces."""
import math
MAX_PHASE_MM=600.0

def phase_count(offset,minimum):
    if not math.isfinite(offset) or abs(offset)>MAX_PHASE_MM:
        raise ValueError(f'连续单轴动作超过{MAX_PHASE_MM:g}mm或目标无效；未拆段执行')
    return int(abs(offset)>minimum)

def run_phase(begin,end,axis,attempt,confirm_hold,records,planning_failed_status):
    """Keep failure identity, but never turn a rejected long path into motion pieces."""
    ok,failure=attempt(end)
    records.append(dict(policy='whole_phase_single_request',axis='XYZ'[axis],
                        start_mm=list(begin),target_mm=list(end),accepted=bool(ok),failure=failure))
    if not ok:
        raise RuntimeError('连续轨迹未确认；本阶段不拆段执行：'+str(failure))
    return 1
