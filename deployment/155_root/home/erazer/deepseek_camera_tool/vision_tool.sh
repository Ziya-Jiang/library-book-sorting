#!/usr/bin/env bash
# Vision Tool: canonical entry; keep the original service and API names compatible.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$HERE/camera_tool.sh" "$@"
