#!/usr/bin/env python3
"""Entry point for sailor-r2 training.

Usage mirrors SAILOR's ``train_sailor.py`` but uses R2-Dreamer world model:

    python train.py --configs cfg_dp_mppi maniskill \\
        --task maniskill__pullcube --num_exp_trajs 50 --seed 0

    # Use standard DreamerV3 reconstruction instead of Barlow Twins:
    python train.py --configs cfg_dp_mppi maniskill \\
        --task rep_loss=dreamer
"""

import os
import sys

# Ensure SAILOR is importable
_sailor_root = os.environ.get(
    "SAILOR_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SAILOR")),
)
if os.path.isdir(_sailor_root) and _sailor_root not in sys.path:
    sys.path.insert(0, _sailor_root)

# Monkey-patch SAILOR's SAILORTrainer before it gets imported by train_sailor.py
import sailor.sailor_trainer as _st
from sailor_r2.sailor_trainer import SAILORTrainer
_st.SAILORTrainer = SAILORTrainer

# Now exec SAILOR's train_sailor.py which does argparse + calls train_eval()
_train_sailor_path = os.path.join(_sailor_root, "train_sailor.py")
if not os.path.exists(_train_sailor_path):
    print(f"ERROR: Could not find {_train_sailor_path}")
    print(f"Set SAILOR_ROOT env var to the SAILOR repo directory.")
    sys.exit(1)

# Run it as __main__ so the if __name__ == "__main__" block executes
exec(compile(open(_train_sailor_path).read(), _train_sailor_path, "exec"))
