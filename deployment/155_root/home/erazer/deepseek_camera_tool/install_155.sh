#!/usr/bin/env bash
set -euo pipefail
# Run on155 in the uploaded source directory. Never touches robot/RT services.
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
python3 -m venv --without-pip --system-site-packages "$HERE/.venv"
#155 system pip is available, while ensurepip is not. Use its module inside this venv.
"$HERE/.venv/bin/python" -m pip install --no-cache-dir --disable-pip-version-check -r "$HERE/requirements.txt"
"$HERE/.venv/bin/python" -c 'import cv2,numpy,mcp,httpx; print("Camera tool dependencies ready")'
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/deepseek-camera-tool.service" <<EOF
[Unit]
Description=155 read-only DeepSeek Harness camera tools
[Service]
Type=simple
WorkingDirectory=$HERE
ExecStart=$HERE/.venv/bin/python $HERE/server.py
Restart=on-failure
RestartSec=3
Nice=5
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user start deepseek-camera-tool.service
systemctl --user is-active deepseek-camera-tool.service
