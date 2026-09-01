# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Dump the EXACT first-policy-step observation vector the training env
produces right after reset (clean config, stand command), plus the action the
policy takes, and save to gm_play for offline comparison against the obs the
mujoco sim2sim builds at t=0 (which produces a diverging first action).

Usage: python humanoid_lab/scripts/diag_obs_dump.py --checkpoint auto
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default="auto")
parser.add_argument("--steps", type=int, default=3)
args_cli, _ = parser.parse_known_args()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid_lab import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_lab.envs import X1DHStandEnv, X1DHStandEnvCfg  # noqa: E402
from humanoid_lab.algo.actor_critic_dh import ActorCriticDH  # noqa: E402
from humanoid_lab.scripts.export_policy_lab import (  # noqa: E402
    POLICY_CFG, SHORT_FRAME_STACK, NUM_SINGLE_OBS, find_checkpoint,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def ensure_x1_usd(cache_dir):
    from isaaclab.sim.converters import UrdfConverter
    from humanoid_lab.envs.x1.x1_env_cfg import X1_URDF_CONVERTER_CFG
    usd = os.path.join(cache_dir, "x1.usd")
    if os.path.exists(usd):
        return usd
    os.makedirs(cache_dir, exist_ok=True)
    return UrdfConverter(X1_URDF_CONVERTER_CFG.replace(usd_dir=cache_dir, usd_file_name="x1.usd")).usd_path


def main(args):
    cfg = X1DHStandEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.noise.add_noise = False
    dr = cfg.domain_rand
    for flag in ("push_robots", "randomize_friction", "randomize_gains", "randomize_torque",
                 "randomize_coulomb_friction", "randomize_motor_offset", "randomize_joint_armature",
                 "randomize_base_mass", "randomize_com", "randomize_link_mass",
                 "add_dof_lag", "add_imu_lag"):
        setattr(dr, flag, False)
    cfg.commands.curriculum = False
    cfg.commands.gait = ["stand"] * 3

    usd_cache = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "urdf_cache")
    cfg.scene.robot.spawn.usd_path = ensure_x1_usd(usd_cache)

    env = X1DHStandEnv(cfg=cfg, render_mode=None)
    ckpt = find_checkpoint(args.checkpoint)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    ac = ActorCriticDH(SHORT_FRAME_STACK * NUM_SINGLE_OBS, NUM_SINGLE_OBS, 3 * 73, 12, **POLICY_CFG)
    ac.load_state_dict(loaded["model_state_dict"])
    ac = ac.eval().to(env.device)
    print(f"[obsdump] loaded {ckpt}")

    obs_dict, _ = env.reset()
    dump = {}
    for step in range(args.steps):
        obs = obs_dict["policy"].to(env.device)
        single = obs[0, :NUM_SINGLE_OBS].detach().cpu().numpy()
        print(f"[obsdump] step {step} single obs (47):")
        labels = ["sin", "cos", "cmd_vx", "cmd_vy", "cmd_wz"] + \
                 [f"q_{i}" for i in range(12)] + [f"dq_{i}" for i in range(12)] + \
                 [f"a_{i}" for i in range(12)] + ["w_x", "w_y", "w_z", "roll", "pitch", "yaw"]
        for k, (lab, val) in enumerate(zip(labels, single)):
            print(f"[obsdump]   [{k:2d}] {lab:8s} = {val:+.4f}")
        with torch.no_grad():
            action = ac.act_inference(obs)
        print(f"[obsdump] action: {np.round(action[0].cpu().numpy(), 3).tolist()}")
        dump[f"step{step}_obs_single"] = single.tolist()
        dump[f"step{step}_action"] = action[0].cpu().numpy().tolist()
        obs_dict, rew, term, trunc, extras = env.step(action)

    out = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out, exist_ok=True)
    torch.save(dump, os.path.join(out, "obs_dump.pt"))
    print(f"[obsdump] saved {out}/obs_dump.pt")


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[obsdump] FATAL: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
