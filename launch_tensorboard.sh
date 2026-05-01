#!/usr/bin/env bash

# Launch TensorBoard to view training progress
# Usage: ./launch_tensorboard.sh [logdir] [port]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${1:-$SCRIPT_DIR/outputs}"
PORT="${2:-6006}"

if [[ ! -d "$LOGDIR" ]]; then
    mkdir -p "$LOGDIR"
    echo "📁 Created log directory: $LOGDIR"
fi

# Kill any existing tensorboard (exclude this script's own process tree)
pgrep -f bin/tensorboard | while read pid; do
    [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null
done
sleep 0.5

echo "TensorBoard is ready"
echo "   Log directory: $LOGDIR"
echo "   Port: $PORT"
echo "   URL: http://localhost:$PORT"

exec "$SCRIPT_DIR/.venv/bin/tensorboard" \
    --logdir="$LOGDIR" \
    --port="$PORT" \
    --bind_all \
    --load_fast=false
