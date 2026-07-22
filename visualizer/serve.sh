#!/usr/bin/env bash
# Launch (or relaunch) the replay GUI server on this machine. Idempotent.
# Usage: bash serve.sh [port]     (default 8000)
set -e
PORT="${1:-8000}"
cd "$(dirname "$0")"

# kill a previous instance of this server (matches our cmdline only)
pkill -f "python3 server\.py" 2>/dev/null && sleep 0.3 || true

nohup python3 server.py "$PORT" > /tmp/replay_server.log 2>&1 &
sleep 0.5
if curl -sf "http://localhost:$PORT/api/runs" > /dev/null; then
  echo "replay server up on localhost:$PORT (log: /tmp/replay_server.log)"
else
  echo "server failed to start — see /tmp/replay_server.log" >&2
  exit 1
fi
