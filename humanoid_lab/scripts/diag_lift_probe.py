# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Lift-reward probe: clean IsaacLab rollout with a WALKING command that logs,
per policy step, the full lift-reward state machine:
    phase / sin_pos / stance_mask / contact(feet) / feet_z / feet_height /
    feet_air_time / air_time reward

Purpose: decide whether the persistently-low feet_air_time / feet_clearance
rewards are (a) a computation bug or (b) the policy physically never lifting
its feet (shuffling gait). With correct contact data:
  - swing phase + contact=1         -> physically dragging (training issue)
  - feet_z max << 0.03 in swing      -> no lift (training issue)
  - feet_z > 0.05 but reward 0       -> computation bug

Usage:
    python humanoid_lab/scripts/diag_lift_probe.py --checkpoint auto \
        --vx 0.8 --steps 600
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default="auto")
parser.add_argument("--vx", type=float, default=0.8)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=600)  # 6 s @ 100 Hz
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
    return UrdfConverter(conv_cfg).usd_path


def main(args):
    cfg = X1DHStandEnvCfg()
    cfg.scene.num_envs = args.num_envs
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
    cfg.commands.gait = ["walk_omnidirectional"] * 3

    usd_cache = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "urdf_cache")
    cfg.scene.robot.spawn.usd_path = ensure_x1_usd(usd_cache)

    print(f"[probe] clean env, cmd=({args.vx},{args.vy},{args.wz})")
    env = X1DHStandEnv(cfg=cfg, render_mode=None)

    ckpt = find_checkpoint(args.checkpoint)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    ac = ActorCriticDH(SHORT_FRAME_STACK * NUM_SINGLE_OBS, NUM_SINGLE_OBS, 3 * 73, 12, **POLICY_CFG)
    ac.load_state_dict(loaded["model_state_dict"])
    ac = ac.eval().to(env.device)
    print(f"[probe] loaded {ckpt} (iter={loaded.get('iter')})")

    obs_dict, _ = env.reset()
    # force a walking command on env 0 (override whatever the gait sampler chose)
    env.commands[0, 0] = args.vx
    env.commands[0, 1] = args.vy
    env.commands[0, 2] = args.wz

    stats = {"lift_max": 0.0, "swing_steps": 0, "swing_contact_steps": 0,
             "air_time_max": 0.0, "air_reward_total": 0.0}
    print("[probe]  t    phase  sin    st(L,R) ct(L,R) feetz(L,R)    hgt(L,R)   airT(L,R)")
    for step in range(args.steps):
        obs = obs_dict["policy"].to(env.device)
        with torch.no_grad():
            action = ac.act_inference(obs)
        # keep the command pinned (resample would overwrite it)
        env.commands[0, 0] = args.vx
        env.commands[0, 1] = args.vy
        env.commands[0, 2] = args.wz

        obs_dict, rew, terminated, truncated, extras = env.step(action)

        # recompute the lift state machine EXACTLY as the reward does
        contact = env.contact_forces[0, env.feet_indices, 2] > 5.0
        stance = env._get_stance_mask()[0]
        contact_filt = contact | (stance > 0) | env.last_contacts[0]
        first_contact = (env.feet_air_time[0] > 0.0) * contact_filt
        air_r = env.feet_air_time[0].clamp(0, 0.5) * first_contact

        feet_z = env.rigid_state[0, env.feet_indices, 2] - cfg.rewards.feet_to_ankle_distance
        phase = env._get_phase()[0].item()
        sin_pos = torch.sin(2 * torch.pi * env._get_phase())[0].item()

        for f in range(2):
            if stance[f] < 0.5:  # swing foot
                stats["swing_steps"] += 1
                if bool(contact[f]):
                    stats["swing_contact_steps"] += 1
                stats["lift_max"] = max(stats["lift_max"], float(feet_z[f]))
        stats["air_time_max"] = max(stats["air_time_max"], float(env.feet_air_time[0].max()))
        stats["air_reward_total"] += float(rew[0])

        if step % 5 == 0:
            print(f"[probe] {step*0.01:5.2f} {phase:5.2f} {sin_pos:+5.2f}  "
                  f"{int(bool(stance[0]))},{int(bool(stance[1]))}    "
                  f"{int(bool(contact[0]))},{int(bool(contact[1]))}   "
                  f"{feet_z[0]:+.3f},{feet_z[1]:+.3f}  "
                  f"{env.feet_height[0,0]:+.3f},{env.feet_height[0,1]:+.3f}  "
                  f"{env.feet_air_time[0,0]:.3f},{env.feet_air_time[0,1]:.3f}")

    print("\n[probe] ===== VERDICT DATA =====")
    print(f"[probe] swing steps: {stats['swing_steps']}, swing-with-contact: {stats['swing_contact_steps']} "
          f"({100*stats['swing_contact_steps']/max(1,stats['swing_steps']):.0f}% -> dragging if high)")
    print(f"[probe] max swing foot height: {stats['lift_max']*100:.1f} cm (need >3cm for clearance reward)")
    print(f"[probe] max air_time: {stats['air_time_max']:.3f} s (0 => never airborne)")
    print(f"[probe] total step reward: {stats['air_reward_total']:.2f}")

    out = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out, exist_ok=True)
    torch.save(stats, os.path.join(out, "lift_probe.pt"))
    print(f"[probe] saved {out}/lift_probe.pt")


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[probe] FATAL: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
