"""Local, request-bound pre-publication evidence; never infer execution from BUSY."""
import copy
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

DIRECTORY = Path('/tmp') / f'dual-arm-plan-receipts-{os.getuid()}'
ACTIONS = ('book_plug_x', 'book_plug_y', 'book_plug_z')
PREFIX = ':trial_request:'


def process_stamp(pid):
    fields = (Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ValueError('Planner exited')
    return fields[19]


def new_request(pid):
    return dict(request_id=uuid.uuid4().hex, planner_pid=int(pid),
                planner_start=process_stamp(pid), issued_monotonic=time.monotonic())


def tagged(command, request):
    if command.description not in ACTIONS:
        return command
    result = copy.deepcopy(command)
    goal = np.asarray(command.arm_tool_command, float)
    if goal.shape != (2, 7) or not np.isfinite(goal).all():
        raise ValueError('Invalid request pose')
    request.update(action=command.description, target_poses=goal.tolist())
    result.description += PREFIX + request['request_id']
    return result


def decode(command):
    if command is None or PREFIX not in command.description:
        return command, None
    action, nonce = command.description.split(PREFIX, 1)
    if action not in ACTIONS or re.fullmatch('[0-9a-f]{32}', nonce) is None:
        raise ValueError('Invalid trial request identity')
    clean = copy.deepcopy(command)
    clean.description = action
    return clean, dict(request_id=nonce, action=action,
                      target_poses=np.asarray(command.arm_tool_command, float).tolist())


def write_receipt(request, *, status, accepted, ever_accepted, diagnostics, hold_only):
    DIRECTORY.mkdir(mode=0o700, exist_ok=True)
    report = dict(request, planner_pid=os.getpid(), planner_start=process_stamp(os.getpid()),
                  completed_monotonic=time.monotonic(), status=int(getattr(status, 'value', status)),
                  phase='candidate_accepted' if accepted else 'rejected_before_publication',
                  ever_accepted=bool(ever_accepted), hold_only=bool(hold_only), diagnostics=diagnostics)
    target = DIRECTORY/(request['request_id']+'.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, allow_nan=False)+'\n')
    temporary.replace(target)
    print('Trial planning receipt: '+json.dumps(report), flush=True)


def begin_candidate(request):
    # Invalidate older proof BEFORE another candidate can be accepted/published.
    # If removal fails, propagate the error rather than leave stale retry proof.
    (DIRECTORY/(request['request_id']+'.json')).unlink(missing_ok=True)


def verify_rejection(request, failure, failed_status, ik_failed_status=None, *, legacy_status_flag=False):
    """Fail closed on missing evidence, restart, stale goal, or any accepted candidate."""
    if not isinstance(failure, dict):
        return None
    status = failure.get('status')
    is_ik = ik_failed_status is not None and status == ik_failed_status
    # Only the old wait_for_motion result has this exact four-field shape.
    # Its flag also means robot_status != AVAILABLE, not necessarily publication.
    # Explicit tracking/hold/transport failures must never be reclassified.
    legacy_only = (legacy_status_flag and set(failure) == {
        'status', 'command_matched', 'execution_started', 'joint_command'})
    if ((status != failed_status and not is_ik) or failure.get('command_matched') is not True
            or failure.get('joint_command') is not False
            or (failure.get('execution_started') is True and not legacy_only)):
        return None
    try:
        report = json.loads((DIRECTORY/(request['request_id']+'.json')).read_text())
        for key in ('request_id', 'planner_pid', 'planner_start', 'action', 'target_poses'):
            if report[key] != request[key]:
                return None
        if process_stamp(request['planner_pid']) != request['planner_start']:
            return None
        now = time.monotonic()
        if not request['issued_monotonic'] <= report['completed_monotonic'] <= now:
            return None
        if now-report['completed_monotonic'] > 60.:
            return None
        if (report['phase'] != 'rejected_before_publication' or report['ever_accepted'] is not False
                or report['hold_only'] is not True
                or report['status'] != status):
            return None
        if is_ik:
            local = (report.get('diagnostics') or {}).get('lift_continuity') or {}
            if (report['action'] != 'book_plug_z'
                    or local.get('policy') != 'measured_joint_neighbourhood'
                    or local.get('phase') != 'solver_result'
                    or local.get('accepted') is not False
                    or local.get('solver_status') != 'MotionGenStatus.IK_FAIL'):
                return None
            report['retry_reason'] = 'local_lift_ik_failed_before_publication'
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None
