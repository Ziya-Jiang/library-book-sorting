"""Opt-in baseline timing; canonical actions retain all existing safety gates."""
MARKER = '|baseline_speed='
CONSTRAINT_MARKER = '|baseline_constraints='


def decode_constraints(description):
    """Explicit per-command baseline profile; never infer it for other entries."""
    if CONSTRAINT_MARKER not in description:
        return description, False
    action, tail = description.split(CONSTRAINT_MARKER, 1)
    expected = 'legacy_20261103'+MARKER
    if not action or not tail.startswith(expected) or CONSTRAINT_MARKER in tail:
        raise ValueError('Invalid baseline constraint profile')
    canonical=action+MARKER+tail[len(expected):]
    decode(canonical)  # Reject malformed speed/direct-servo use before dispatch.
    return canonical, True


def decode(description):
    if MARKER not in description:
        return description, None
    action, raw = description.rsplit(MARKER, 1)
    factor = float(raw)
    if not action or MARKER in action or not .4 <= factor <= .8:
        raise ValueError('Invalid baseline trajectory speed factor')
    if action in ('book_v3_1_stream', 'book_v3_xyz_servo', 'book_v3_1_hold'):
        raise ValueError('Baseline trajectory speed does not apply to direct servo')
    return action, factor


def configure(plan, factor):
    if factor is None:
        return plan
    if not .4 <= factor <= .8:
        raise ValueError('Invalid baseline trajectory speed factor')
    result = plan.clone()
    result.time_dilation_factor = factor
    return result
