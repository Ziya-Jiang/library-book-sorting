"""Join two host journals by wire digest; never subtract host clocks.

Usage: python analyze_packet_trace.py --pc PC_RUN_DIRECTORY --board BOARD_RUN_DIRECTORY
Both directories may be offline copies. Does not connect to the robot.
"""
import argparse
import json
from pathlib import Path


def load(directory):
    records = {}
    metadata = []
    skipped_lines = 0
    for path in sorted(Path(directory).glob('*.jsonl')):
        for text in path.read_text(errors='replace').splitlines():
            try:
                row = json.loads(text)
            except ValueError:
                skipped_lines += 1
                continue  # A live/abruptly stopped writer may leave a partial tail.
            if 'index' in row and 'side' in row:
                records[(row['side'], row['index'])] = row
            else:
                metadata.append(row)
    return sorted(records.values(), key=lambda r:r['index']), metadata, skipped_lines


def lane(sender, receiver, kind):
    tx = {r['wire_sha256']:r for r in sender
          if r['event'] == 'send_begin' and (r.get('kind') == 'state') == (kind == 'state')}
    rx = {r['wire_sha256']:r for r in receiver
          if r['event'] == 'receive' and r.get('arrival_monotonic_s') is not None}
    returns = {r['wire_sha256']:r for r in sender if r['event'] == 'send_return'}
    matched = sorted(((t, rx[d]) for d,t in tx.items() if d in rx),
                     key=lambda pair:pair[0]['send_started_monotonic_s'])
    comparisons = []
    for (a, ar), (b, br) in zip(matched, matched[1:]):
        if a.get('session_tag') != b.get('session_tag'): continue
        send_gap = (b['send_started_monotonic_s']-a['send_started_monotonic_s'])*1000
        receive_gap = (br['arrival_monotonic_s']-ar['arrival_monotonic_s'])*1000
        # Two within-host intervals: clock epoch offsets cancel. This is a
        # CHANGE in transport delay, not an absolute one-way latency estimate.
        comparisons.append(dict(previous_sequence=a.get('sequence'), sequence=b.get('sequence'),
            session_tag=b.get('session_tag'), send_gap_ms=send_gap,
            receive_gap_ms=receive_gap, transport_delay_increase_ms=receive_gap-send_gap,
            receiver_queue_ms=br.get('queue_ms'),
            send_syscall_ms=returns.get(b['wire_sha256'], {}).get('syscall_ms')))
    return dict(sent_records=len(tx), matched=len(matched),
        unmatched_sequences=[t.get('sequence') for d,t in tx.items() if d not in rx][-30:],
        largest_sender_gaps=sorted(comparisons,key=lambda x:x['send_gap_ms'],reverse=True)[:8],
        largest_transport_delay_increases=sorted(comparisons,
            key=lambda x:x['transport_delay_increase_ms'],reverse=True)[:8],
        largest_receiver_queues=sorted((r for _,r in matched),
            key=lambda x:x.get('queue_ms',0),reverse=True)[:5])


def analyze(pc, board):
    result = dict(schema=1, clock_policy='compare_intervals_within_each_host_only',
        limitations=[
            'send_return proves OS acceptance, not physical NIC transmission.',
            'Delay increases include sender/receiver network stacks and the link; they do not identify a router or cable.',
            'Unmatched packets may be outside capture coverage or missing diagnostics; do not equate them with packet loss.',
            'Different clock rates may affect long-interval comparisons; no absolute one-way latency is inferred.'])
    result['pc_to_board_commands'] = lane(pc, board, 'command')
    result['board_to_pc_feedback'] = lane(board, pc, 'state')
    result['faults'] = [r for r in pc+board if r['event'] == 'fault']
    result['logging_dropped_max'] = {side:max([r.get('dropped',0) for r in rows] or [0])
                                    for side,rows in [('pc',pc),('board',board)]}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pc',type=Path,required=True)
    parser.add_argument('--board',type=Path,required=True)
    args=parser.parse_args()
    pc, pm, ps = load(args.pc)
    board, bm, bs = load(args.board)
    result=analyze(pc,board)
    result['capture_metadata']=dict(pc=pm,board=bm,partial_lines=dict(pc=ps,board=bs))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
