# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Open-loop physics A/B replay: the DEFINITIVE physics-parity test.

Phase 1: clean IsaacLab env (stand gait) rolls out the policy for N steps,
logging per policy step: newest obs frame, action, q(12), dq(12), root z,
roll/pitch, max |tau|.

Phase 2: mujoco (patched mjcf from sim2sim_eval) starts from the SAME initial
state and applies the LOGGED actions OPEN-LOOP (pure PD per substep). If the
state trajectories track the training trace, physics parity holds and any
sim2sim collapse is obs/policy-feedback related; if they diverge immediately,
the physics differs (solver/contact/etc).

Usage: python humanoid_lab/scripts/diag_physics_replay.py --checkpoint auto
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default="auto")
parser.add_argument("--steps", type=int, default=200)  # 2 s @ 100 Hz
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
from humanoid_lab.utils.torch_utils import get_euler_xyz_tensor  # noqa: E402

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
    # ---------------- Phase 1: IsaacLab clean rollout ----------------
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

    print("[replay] phase 1: IsaacLab clean rollout (stand)")
    env = X1DHStandEnv(cfg=cfg, render_mode=None)
    ckpt = find_checkpoint(args.checkpoint)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    ac = ActorCriticDH(SHORT_FRAME_STACK * NUM_SINGLE_OBS, NUM_SINGLE_OBS, 3 * 73, 12, **POLICY_CFG)
    ac.load_state_dict(loaded["model_state_dict"])
    ac = ac.eval().to(env.device)

    obs_dict, _ = env.reset()
    train_log = {k: [] for k in ("z", "roll", "pitch", "q", "dq", "action", "tau_max")}
    for step in range(args.steps):
        obs = obs_dict["policy"].to(env.device)
        with torch.no_grad():
            action = ac.act_inference(obs)
        obs_dict, rew, term, trunc, extras = env.step(action)
        rs = env.root_states[0]
        eu = get_euler_xyz_tensor(env.base_quat)[0].cpu().numpy()
        train_log["z"].append(float(rs[2]))
        train_log["roll"].append(float(eu[0])); train_log["pitch"].append(float(eu[1]))
        train_log["q"].append(env.dof_pos[0].detach().cpu().numpy().tolist())
        train_log["dq"].append(env.dof_vel[0].detach().cpu().numpy().tolist())
        train_log["action"].append(action[0].detach().cpu().numpy().tolist())
        train_log["tau_max"].append(float(env.torques[0].abs().max()))
        if bool(term[0]):
            print(f"[replay] training episode TERMINATED at step {step}")
            args.steps = step
            break
    # capture initial state right after a fresh reset (same distribution)
    obs_dict, _ = env.reset()
    init_q = env.dof_pos[0].detach().cpu().numpy().copy()
    init_z = float(env.root_states[0, 2])
    print(f"[replay] init state after reset: z={init_z:.4f} q0={np.round(init_q,3).tolist()[:4]}...")
    print(f"[replay] training trace: z {train_log['z'][0]:.3f} -> {train_log['z'][-1]:.3f}, "
          f"tau_max {max(train_log['tau_max']):.1f}, steps={args.steps}")
    simulation_app.close()

    # ---------------- Phase 2: mujoco open-loop replay ----------------
    print("[replay] phase 2: mujoco open-loop replay of the SAME actions")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--timeout", "60",
                               "--retries", "2", "mujoco", "-i",
                               "https://pypi.tuna.tsinghua.edu.cn/simple"])
        import mujoco

    from humanoid_lab.scripts.sim2sim_eval import (
        _patch_foot_soles, MJCF_PATH, MJCF_PATCHED_PATH, build_joint_maps,
        DEFAULT_DOF_POS, KPS, KDS, ACTION_SCALE,
    )
    _patch_foot_soles(MJCF_PATH)
    model = mujoco.MjModel.from_xml_path(MJCF_PATCHED_PATH)
    model.opt.timestep = 0.001
    for jn in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if nm and ("hip" in nm or "knee" in nm or "ankle" in nm):
            model.dof_damping[model.jnt_dofadr[jn]] = 0.0
    qpos_adr, dof_adr, act_adr, tau_lo, tau_hi = build_joint_maps(mujoco, model)

    data = mujoco.MjData(model)
    data.qpos[qpos_adr] = init_q
    data.qpos[2] = init_z
    mujoco.mj_step(model, data)

    mj_log = {k: [] for k in ("z", "roll", "pitch", "q", "tau_max")}
    from humanoid_lab.scripts.sim2sim_eval import quat_xyzw_to_euler
    for step in range(args.steps):
        a = np.array(train_log["action"][step], dtype=np.double)
        target_q = a * ACTION_SCALE
        for _ in range(10):
            q = data.qpos[qpos_adr].astype(np.double)
            dq = data.qvel[dof_adr].astype(np.double)
            tau = KPS * (target_q + DEFAULT_DOF_POS - q) - KDS * dq
            data.ctrl[act_adr] = np.clip(tau, tau_lo, tau_hi)
            mujoco.mj_step(model, data)
        quat_xyzw = data.sensor("body-orientation").data[[1, 2, 3, 0]].astype(np.double)
        eu = quat_xyzw_to_euler(quat_xyzw)
        mj_log["z"].append(float(data.qpos[2]))
        mj_log["roll"].append(float(eu[0])); mj_log["pitch"].append(float(eu[1]))
        mj_log["q"].append(data.qpos[qpos_adr].tolist())
        mj_log["tau_max"].append(float(np.max(np.abs(data.ctrl[act_adr]))))

    print("\n[replay] ===== OPEN-LOOP TRAJECTORY COMPARISON (train vs mujoco) =====")
    print(f"{'step':>5} {'z_tr':>6} {'z_mj':>6} {'dz':>7} {'roll_tr':>8} {'roll_mj':>8} {'pitch_tr':>8} {'pitch_mj':>8} {'dq_max':>7}")
    worst = 0.0
    for s in range(min(args.steps, 60)):
        dz = mj_log["z"][s] - train_log["z"][s]
        qtr = np.array(train_log["q"][s]); qmj = np.array(mj_log["q"][s])
        dq_max = float(np.max(np.abs(qmj - qtr)))
        worst = max(worst, abs(dz), dq_max)
        if s % 5 == 0 or s < 10:
            print(f"{s:5d} {train_log['z'][s]:6.3f} {mj_log['z'][s]:6.3f} {dz:+7.3f} "
                  f"{train_log['roll'][s]:+8.3f} {mj_log['roll'][s]:+8.3f} "
                  f"{train_log['pitch'][s]:+8.3f} {mj_log['pitch'][s]:+8.3f} {dq_max:7.3f}")
    print(f"\n[replay] worst |dz| or |dq| over {min(args.steps,60)} steps: {worst:.3f}")
    print("[replay] VERDICT: <0.05 => physics parity OK (collapse is obs/policy-side); "
          ">0.2 => physics mismatch")

    out = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out, exist_ok=True)
    torch.save({"train": train_log, "mujoco": mj_log, "init_q": init_q.tolist(),
                "init_z": init_z}, os.path.join(out, "physics_replay.pt"))
    print(f"[replay] saved {out}/physics_replay.pt")


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[replay] FATAL: {e}", file=sys.stderr)
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.exit(1)
