# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Diagnostic: run the trained policy in IsaacLab with a CLEAN config
(1 env, no obs noise, no domain randomization) and log the base state +
joint trajectories, mirroring the mujoco sim2sim time series.

Purpose: same-checkpoint A/B comparison.
  - If this also collapses  -> the policy itself is brittle at clean eval
    (training-side issue), mujoco is innocent.
  - If this walks/stands    -> mujoco physics mismatch; compare per-joint
    trajectories to locate the divergence.

Usage:
    python humanoid_lab/scripts/diag_isaaclab_play.py --checkpoint auto --trial stand
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="x1_dh_stand")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--trial", type=str, default="stand", choices=["stand", "walk"])
parser.add_argument("--max_steps", type=int, default=2400)  # 24 s @ 100 Hz
parser.add_argument("--checkpoint", type=str, default="auto")
args_cli, _ = parser.parse_known_args()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch  # noqa: E402

from humanoid_lab import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_lab.envs import X1DHStandEnv, X1DHStandEnvCfg  # noqa: E402
from humanoid_lab.algo.actor_critic_dh import ActorCriticDH  # noqa: E402
from humanoid_lab.scripts.export_policy_lab import (  # noqa: E402
    POLICY_CFG, SHORT_FRAME_STACK, NUM_SINGLE_OBS, find_checkpoint,
)
from humanoid_lab.utils.torch_utils import get_euler_xyz_tensor  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def ensure_x1_usd(cache_dir: str) -> str:
    from isaaclab.sim.converters import UrdfConverter
    from humanoid_lab.envs.x1.x1_env_cfg import X1_URDF_CONVERTER_CFG

    usd_path = os.path.join(cache_dir, "x1.usd")
    if os.path.exists(usd_path):
        return usd_path
    os.makedirs(cache_dir, exist_ok=True)
    conv_cfg = X1_URDF_CONVERTER_CFG.replace(usd_dir=cache_dir, usd_file_name="x1.usd")
    converter = UrdfConverter(conv_cfg)
    return converter.usd_path


def main(args):
    cfg = X1DHStandEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # ---- clean evaluation config (nominal physics, no noise, no rand) ----
    cfg.noise.add_noise = False
    dr = cfg.domain_rand
    dr.push_robots = False
    dr.randomize_friction = False
    dr.randomize_gains = False
    dr.randomize_torque = False
    dr.randomize_coulomb_friction = False
    dr.randomize_motor_offset = False
    dr.randomize_joint_armature = False
    dr.randomize_base_mass = False
    dr.randomize_com = False
    dr.randomize_link_mass = False
    dr.add_dof_lag = False
    dr.add_imu_lag = False
    cfg.commands.curriculum = False
    if args.trial == "stand":
        cfg.commands.gait = ["stand", "stand", "stand"]
    else:
        cfg.commands.gait = ["walk_omnidirectional", "walk_omnidirectional", "walk_omnidirectional"]

    usd_cache = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "urdf_cache")
    cfg.scene.robot.spawn.usd_path = ensure_x1_usd(usd_cache)

    print(f"[diag] creating clean env (trial={args.trial}) ...")
    env = X1DHStandEnv(cfg=cfg, render_mode=None)
    print(f"[diag] dof order: {list(env.dof_names)}")

    ckpt = find_checkpoint(args.checkpoint)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    ac = ActorCriticDH(SHORT_FRAME_STACK * NUM_SINGLE_OBS, NUM_SINGLE_OBS, 3 * 73, 12, **POLICY_CFG)
    ac.load_state_dict(loaded["model_state_dict"])
    ac = ac.eval().to(env.device)
    print(f"[diag] loaded {ckpt} (iter={loaded.get('iter')})")

    obs_dict, _ = env.reset()
    log = {k: [] for k in ("t", "z", "roll", "pitch", "vx", "vy", "wz", "tau_max",
                           "foot_l", "foot_r", "cmd_vx", "q", "action", "terminated")}
    fallen_at = None

    for step in range(args.max_steps):
        obs = obs_dict["policy"].to(env.device)
        with torch.no_grad():
            action = ac.act_inference(obs)
        obs_dict, rew, terminated, truncated, extras = env.step(action)
        t = step * 0.01
        rs = env.root_states[0]
        eu = get_euler_xyz_tensor(env.base_quat)[0].cpu().numpy()
        contact = env.contact_forces[0, env.feet_indices, 2].cpu() > 5.0
        log["t"].append(t)
        log["z"].append(float(rs[2]))
        log["roll"].append(float(eu[0])); log["pitch"].append(float(eu[1]))
        log["vx"].append(float(rs[7])); log["vy"].append(float(rs[8])); log["wz"].append(float(rs[11]))
        log["tau_max"].append(float(env.torques[0].abs().max()))
        log["foot_l"].append(bool(contact[0])); log["foot_r"].append(bool(contact[1]))
        log["cmd_vx"].append(float(env.commands[0, 0]))
        log["q"].append(env.dof_pos[0].detach().cpu().numpy().tolist())
        log["action"].append(action[0].detach().cpu().numpy().tolist())
        log["terminated"].append(bool(terminated[0]))
        if bool(terminated[0]) and fallen_at is None:
            fallen_at = t
        if step % 10 == 0:
            print(f"[diag] {t:5.2f} z={rs[2]:.3f} roll={eu[0]:+.3f} pitch={eu[1]:+.3f} "
                  f"vx={rs[7]:+.2f} tau={float(env.torques[0].abs().max()):6.1f} "
                  f"fL={int(contact[0])} fR={int(contact[1])} cmd={float(env.commands[0,0]):+.2f}"
                  + ("  <<TERMINATED" if bool(terminated[0]) else ""))

    print(f"[diag] done. terminated_at={fallen_at} (None = survived all {args.max_steps*0.01:.0f}s)")
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"diag_isaaclab_{args.trial}.pt")
    torch.save({"log": log, "fallen_at": fallen_at, "dof_names": list(env.dof_names),
                "checkpoint": ckpt}, out)
    print(f"[diag] saved {out}")


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[diag] FATAL: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
