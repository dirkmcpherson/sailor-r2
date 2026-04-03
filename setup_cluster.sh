#!/bin/bash
# Setup sailor-r2 on a fresh machine / cluster node.
#
# Prerequisites:
#   - Python 3.10+
#   - CUDA GPU
#   - Git SSH access to github.com/dirkmcpherson repos
#
# Layout after running this script:
#   workspace/
#     SAILOR/          <-- cloned if not present
#     sailor-r2/       <-- this repo (you should already be in it)
#       .venv/         <-- created by this script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== sailor-r2 cluster setup ==="
echo "Working directory: $(pwd)"
echo ""

# 1. Check/clone SAILOR as a sibling
SAILOR_DIR="${SAILOR_ROOT:-$(realpath ../SAILOR 2>/dev/null || echo "")}"
if [ -z "$SAILOR_DIR" ] || [ ! -d "$SAILOR_DIR" ]; then
    echo "SAILOR not found at ../SAILOR — cloning..."
    git clone git@github.com:dirkmcpherson/SAILOR.git ../SAILOR
    SAILOR_DIR="$(realpath ../SAILOR)"
else
    echo "Found SAILOR at: $SAILOR_DIR"
fi

# Verify datasets exist
if [ ! -f "$SAILOR_DIR/datasets/gymhil_datasets/pickcube/trajectory_hil_left.h5" ]; then
    echo "WARNING: GymHIL datasets not found at $SAILOR_DIR/datasets/gymhil_datasets/"
    echo "You'll need to copy them from another machine or download them."
fi

# 2. Create venv
if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
else
    echo "Venv already exists."
fi
source .venv/bin/activate
echo "Python: $(python --version) at $(which python)"

# 3. Install deps
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[envs]"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate:  source $(pwd)/.venv/bin/activate"
echo "To train:     MUJOCO_GL=egl PYOPENGL_PLATFORM=egl bash experiments/train_left_gymhil.sh"
echo ""
echo "If SAILOR is not at ../SAILOR, set SAILOR_ROOT:"
echo "  export SAILOR_ROOT=/path/to/SAILOR"
