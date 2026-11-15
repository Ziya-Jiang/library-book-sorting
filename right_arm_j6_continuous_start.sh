#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /mnt/ssd/robotarm/env/robotram_wx/bin/python -B "$ROOT/robot_cockpit/dual_arm_j6_continuous/start_entry.py" "$@"
