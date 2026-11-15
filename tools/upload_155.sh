#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${1:-erazer@10.30.3.155}"
# Upload only. Installation and camera startup are explicit documented commands.
rsync -a --exclude='.git' --exclude='.secrets' --exclude='__pycache__' \
  --exclude='/robot_cockpit' --exclude='/dual_v2_2_agent' --exclude='/dual_v2_2_control' \
  --exclude='/dual_v2_2_configuration' --exclude='/third_party' --exclude='/dist' \
  "$ROOT/" "$HOST:library-book-sorting-package/"
printf '%s\n' 'Uploaded to ~/library-book-sorting-package; no robot or service action performed.'
