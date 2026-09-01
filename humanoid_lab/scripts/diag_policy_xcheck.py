# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Cross-check probe: SAME checkpoint, SAME inputs -> compare
  (a) ActorCriticDH.act_inference (training-side network, as IsaacLab uses)
  (b) ExportedDH JIT (the deployment path sim2sim uses)
  (c) ActorCriticDH.act (stochastic, for reference)

Feeds three inputs: all-zeros obs, default-pose obs, and a mildly asymmetric
obs. Any (a)!=(b) mismatch means the export path corrupts the policy; any
crazy first-frame action in BOTH means the checkpoint itself behaves oddly on
zero-history obs (train-time reset distribution issue).

Usage: python humanoid_lab/scripts/diag_policy_xcheck.py --checkpoint auto
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--checkpoint", type=str, default="auto")
args_cli, _ = parser.parse_known_args()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid_lab.algo.actor_critic_dh import ActorCriticDH  # noqa: E402
from humanoid_lab.scripts.export_policy_lab import (  # noqa: E402
    POLICY_CFG, SHORT_FRAME_STACK, NUM_SINGLE_OBS, FRAME_STACK, find_checkpoint,
    load_exported_policy, ExportedDH,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def main(args):
    ckpt = find_checkpoint(args.checkpoint)
    print(f"[xcheck] checkpoint: {ckpt}")
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    ac = ActorCriticDH(SHORT_FRAME_STACK * NUM_SINGLE_OBS, NUM_SINGLE_OBS, 3 * 73, 12, **POLICY_CFG)
    ac.load_state_dict(loaded["model_state_dict"])
    ac = ac.eval()

    exported = load_exported_policy(ckpt)  # deepcopy'd cpu modules
    jit = torch.jit.script(exported)
    jit = jit.eval()

    total = FRAME_STACK * NUM_SINGLE_OBS  # 3102
    default_q = np.array([0.4, -0.4, 0.05, -0.05, -0.31, 0.31, 0.49, 0.49, -0.21, -0.21, 0.0, 0.0],
                         dtype=np.float32)

    def make_obs(kind):
        single = np.zeros(NUM_SINGLE_OBS, dtype=np.float32)
        if kind == "default_pose":
            single[5:17] = (default_q - default_q)  # q - default = 0 anyway
            single[0] = 0.0; single[1] = 1.0        # sin, cos of phase 0
        elif kind == "asym":
            single[5:17] = np.linspace(-0.1, 0.1, 12).astype(np.float32)
        return torch.tensor(np.tile(single, FRAME_STACK)[None, :])

    for kind in ["zeros", "default_pose", "asym"]:
        obs = make_obs(kind)
        with torch.no_grad():
            a_train = ac.act_inference(obs)
            a_jit = jit(obs)
            diff = (a_train - a_jit).abs().max().item()
        print(f"[xcheck] {kind:13s} train={np.round(a_train[0].numpy(), 3).tolist()}")
        print(f"[xcheck] {kind:13s} jit ={np.round(a_jit[0].numpy(), 3).tolist()}")
        print(f"[xcheck] {kind:13s} max|diff| = {diff:.2e}")

    out = os.path.join(_REPO_ROOT, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out, exist_ok=True)
    torch.save({"ckpt": ckpt}, os.path.join(out, "xcheck_done.pt"))


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[xcheck] FATAL: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
