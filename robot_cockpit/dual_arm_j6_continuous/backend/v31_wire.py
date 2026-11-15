"""Versioned Seldon V3.1 datagrams and bounded native command decoding."""
import json
import math
import hashlib
from collections import deque

PROTOCOL = 'dual-gripper-v1'
LEASE_S = .25
# Shared sender/receiver echo policy. Producer lease and RT freshness are separate.
ECHO_AGE_S = .30
MAX_PACKET = 60000


def encode(packet):
    raw = json.dumps(dict(packet, protocol=PROTOCOL), allow_nan=False,
                     separators=(',', ':')).encode('ascii')
    if len(raw) > MAX_PACKET:
        raise ValueError('V3.1 packet too large')
    return raw


def decode(raw):
    if len(raw) > MAX_PACKET:
        raise ValueError('V3.1 packet too large')
    packet = json.loads(raw)
    if not isinstance(packet, dict) or packet.get('protocol') != PROTOCOL:
        raise ValueError('V3.1 bridge protocol mismatch')
    return packet


def command(raw):
    if not isinstance(raw, str) or not raw.endswith('#') or len(raw) > MAX_PACKET:
        raise ValueError('Invalid native command')
    fields = iter(raw[:-1].split('#'))

    def take(integer=False):
        try:
            value = int(next(fields)) if integer else float(next(fields))
        except (StopIteration, ValueError) as error:
            raise ValueError('Malformed native command') from error
        if not math.isfinite(value):
            raise ValueError('Nonfinite native command')
        return value

    def arms():
        return [[take() for joint in range(7)] for arm in range(2)]

    duration, ratio = take(), take()
    if take(True) != 0 or take(True) != 0:
        raise ValueError('Only position control is supported')
    mode = take(True)
    if mode not in (0, 1, 2):
        raise ValueError('Unsupported panel mode')
    if [take(True), take(True), take(True)] != [2, 7, 7]:
        raise ValueError('Expected 2 x 7 arms')
    target = arms()
    count, interval = take(True), take()
    if not 0 <= count <= 64 or (mode == 1 and count < 2) or (mode != 1 and count != 0):
        raise ValueError('Invalid waypoint count')
    points = [arms() for point in range(count)]
    if not count and next(fields, None) not in ('', '0', '0.0'):
        raise ValueError('Invalid empty waypoint field')
    if [take(True), take(True), take(True)] != [2, 1, 1]:
        raise ValueError('Expected two independent 1-joint grippers')
    grippers = [[take() for joint in range(size)] for size in (1, 1)]
    if next(fields, None) is not None:
        raise ValueError('Unexpected command fields')
    if mode == 2:
        if duration != 0 or not 0 < ratio <= .3:
            raise ValueError('RAW requires zero target time and ratio <= 0.3')
    elif not .2 <= duration <= 30 or not 0 < ratio <= .1:
        raise ValueError('Invalid interpolated timing or speed ratio')
    if count and (not .05 <= interval <= 5 or count * interval > 30):
        raise ValueError('Invalid trajectory timing')
    if any(not 0 <= value <= 1 for row in grippers for value in row):
        raise ValueError('Invalid gripper target')
    return dict(mode=mode, duration=duration, ratio=ratio, target=target,
                points=points, dt=interval, grippers=grippers)


class CommandGate:
    def __init__(self, session):
        self.session = session
        self.sequence = -1
        self.session_tag = hashlib.sha256(session.encode()).hexdigest()[:12]
        self.discarded = 0
        self.last_attempt = None
        self.last_discard = None
        self.recent = deque(maxlen=8)
        self.last_discard_log = None
        self.expired_echoes = 0

    def _discard(self, reason, now):
        self.discarded += 1
        self.last_discard = dict(self.last_attempt, reason=reason,
                                policy='discard_without_execution_or_lease_renewal')
        self.recent.append(self.last_discard)
        if self.last_discard_log is None or now-self.last_discard_log >= 1.:
            print('V3.1 obsolete command discarded: '+json.dumps(
                dict(self.last_discard, discarded_count=self.discarded)), flush=True)
            self.last_discard_log = now
        return False

    def check_identity(self, packet, now):
        """Return False for obsolete same-session packets, without renewing a lease."""
        sequence = packet.get('sequence')
        echo = packet.get('echo')
        valid_sequence = type(sequence) is int and sequence >= 0
        valid_echo = type(echo) in (int, float) and math.isfinite(echo)
        self.last_attempt = dict(
            session_tag=self.session_tag, session_matches=packet.get('session') == self.session,
            received_sequence=sequence if valid_sequence else None,
            accepted_sequence=self.sequence, echo_age_s=now-echo if valid_echo else None,
            time_monotonic=now, kind='cartesian' if packet.get('kind') == 'cartesian' else 'native')
        if packet.get('session') != self.session:
            raise ValueError('Expired control session')
        if not valid_sequence:
            raise ValueError('Invalid command sequence; expected nonnegative integer')
        if not valid_echo:
            raise ValueError('Command feedback echo expired')
        if sequence <= self.sequence:
            return self._discard('duplicate_or_reordered', now)
        if now-echo < 0:
            raise ValueError('Command feedback echo expired')
        if now-echo > ECHO_AGE_S:
            # A delayed datagram is not a new command or a session stop.
            # Never decode its target, commit its sequence, or renew any lease.
            # The independent last-valid-arrival and loop deadlines still stop
            # a genuinely stalled stream, including an endless stream of old packets.
            self.expired_echoes += 1
            return self._discard('feedback_echo_expired', now)
        return True

    def commit(self, packet):
        self.sequence = packet['sequence']
        self.recent.append(dict(self.last_attempt, outcome='decoded'))

    def diagnostics(self):
        return dict(session_tag=self.session_tag, accepted_sequence=self.sequence,
                    discarded_count=self.discarded, last_attempt=self.last_attempt,
                    expired_echo_count=self.expired_echoes,
                    last_discard=self.last_discard, recent=list(self.recent))

    def accept(self, packet, now):
        if not self.check_identity(packet, now):
            return None
        result = command(packet.get('command'))
        self.commit(packet)
        return result
