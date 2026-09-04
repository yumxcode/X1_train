# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Sim2sim collapse A/B diagnostic matrix (stand trial only, no video).

The v6 policy (and v4/v5 before it) collapses in mujoco at a consistent
1.5-1.8s with monotonically drifting pitch, while the same policy is stable
in IsaacLab. This script isolates WHERE the divergence lives by ablation:

  variants:
    base       : current pipeline as-is (mjcf patched, solref 0.005 1)
    stiff      : global solref timeconst 0.002 (harder contact)
    soft       : global solref timeconst 0.012 (softer contact)
    no_euler   : zero the euler-angle block in the policy observation
    no_angvel  : zero the angular-velocity block in the policy observation
    settle2s   : zero-action PD hold for the first 2s, then policy on
                 (isolates the landing transient vs closed-loop drift)

Each variant runs a 24s STAND trial (zero commands) and reports survival
time + final state. Fast (no rendering).

Usage:
    python humanoid_lab/scripts/diag_ab_matrix.py --checkpoint auto
"""

import argparse
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid_lab.scripts.export_policy_lab import find_checkpoint, load_exported_policy  # noqa: E402
from humanoid_lab.scripts.sim2sim_eval import (  # noqa: E402
    _ensure_mujoco,
    ACTION_SCALE,
    CLIP_ACTION,
    CLIP_OBS,
    CYCLE_TIME,
    DECIMATION,
    DEFAULT_DOF_POS,
    KDS,
    KPS,
    NUM_ACTIONS,
    NUM_COMMANDS,
    NUM_SINGLE_OBS,
    OBS_SCALE_ANG_VEL,
    OBS_SCALE_DOF_POS,
    OBS_SCALE_DOF_VEL,
    SIM_DT,
    STAND_COM_THRESHOLD,
    SW_SWITCH,
    build_model_mjcf_patched,
    build_joint_maps,
    quat_xyzw_to_euler,
    quat_xyzw_to_rot,
)

FRAME_STACK = 66


def run_variant(mujoco, model, policy, name, dur_s=24.0, zero_euler=False,
                zero_angvel=False, settle_s=0.0, omega_from_qvel=False,
                omega_tf=None, omega_ema_s=0.0, schedule=None):
    qpos_adr, dof_adr, act_adr, tau_lo, tau_hi = build_joint_maps(mujoco, model)
    data = mujoco.MjData(model)
    data.qpos[qpos_adr] = DEFAULT_DOF_POS
    data.qpos[2] = 0.70
    mujoco.mj_step(model, data)

    torso_ids = []
    for cands in (["x1-body", "base_link"], ["body_yaw", "lumber_yaw"],
                  ["body_roll", "lumber_roll"], ["body_pitch", "lumber_pitch"]):
        for nm in cands:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if i >= 0:
                torso_ids.append(i)
                break
    foot_ids = []
    for cands in (["left_ankle_roll_link", "left_ankle_roll"],
                  ["right_ankle_roll_link", "right_ankle"]):
        for nm in cands:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if i >= 0:
                foot_ids.append(i)
                break

    hist = [np.zeros([1, NUM_SINGLE_OBS]) for _ in range(FRAME_STACK)]
    action = np.zeros(NUM_ACTIONS)
    omega_filt = np.zeros(3)
    ema_alpha = None  # per-step EMA coefficient, set by variant
    rec = {"t": [], "z": [], "pitch": [], "roll": []}
    rec["vx"] = []; rec["x"] = []
    fall_t, fall_reason = None, None
    total = int(dur_s / SIM_DT)
    cnt = 0

    for step in range(total):
        t = step * SIM_DT
        q = data.qpos[qpos_adr].astype(np.double)
        dq = data.qvel[dof_adr].astype(np.double)
        quat_xyzw = data.sensor("body-orientation").data[[1, 2, 3, 0]].astype(np.double)
        rot = quat_xyzw_to_rot(quat_xyzw)
        v_base = rot.T @ data.qvel[:3].astype(np.double)
        omega = data.sensor("body-angular-velocity").data.astype(np.double)
        omega_qvel = rot.T @ data.qvel[3:6].astype(np.double)
        if omega_from_qvel:
            omega = omega_qvel
        if omega_tf is not None:
            omega = omega_tf(omega.copy())
        # IMU lowpass: EMA at 1 kHz with time constant omega_ema_s.
        # Real gyros have limited bandwidth; training obs came from PhysX
        # TGS whose velocity trajectories are inherently smoother than
        # mujoco Euler contact ringing.
        if omega_ema_s > 0:
            if ema_alpha is None:
                pass
            a = 1.0 - math.exp(-SIM_DT / omega_ema_s)
            omega_filt = omega_filt + a * (omega - omega_filt)
            omega = omega_filt.copy()
        else:
            omega_filt = omega.copy()
        eu = quat_xyzw_to_euler(quat_xyzw)
        eu[eu > math.pi] -= 2 * math.pi
        base_pos = data.qpos[:3].astype(np.double)

        if any(np.linalg.norm(data.cfrc_ext[i]) > 1.0 for i in torso_ids):
            fall_t, fall_reason = t, "torso_contact"
        elif abs(eu[0]) > 1.5:
            fall_t, fall_reason = t, "roll>1.5"
        elif abs(eu[1]) > 1.5:
            fall_t, fall_reason = t, "pitch>1.5"
        elif base_pos[2] < 0.40:
            fall_t, fall_reason = t, "z<0.40"

        # scheduled command (None = stand for the whole trial)
        cmd = (0.0, 0.0, 0.0)
        if schedule is not None:
            cmd = next((c for (t0, t1, c) in schedule if t0 <= t < t1), (0.0, 0.0, 0.0))
        vx_cmd, vy_cmd, wz_cmd = cmd

        if cnt % DECIMATION == 0:
            if SW_SWITCH:
                vel_norm = math.sqrt(vx_cmd**2 + vy_cmd**2 + wz_cmd**2)
                if vel_norm <= STAND_COM_THRESHOLD:
                    cnt = 0
            obs = np.zeros([1, NUM_SINGLE_OBS])
            ph = cnt * SIM_DT
            obs[0, 0] = math.sin(2 * math.pi * ph / CYCLE_TIME)
            obs[0, 1] = math.cos(2 * math.pi * ph / CYCLE_TIME)
            obs[0, 2] = vx_cmd * 2.0  # OBS_SCALE_LIN_VEL
            obs[0, 3] = vy_cmd * 2.0
            obs[0, 4] = wz_cmd * 1.0  # OBS_SCALE_ANG_VEL
            obs[0, NUM_COMMANDS:NUM_COMMANDS + NUM_ACTIONS] = (q - DEFAULT_DOF_POS) * OBS_SCALE_DOF_POS
            obs[0, NUM_COMMANDS + NUM_ACTIONS:NUM_COMMANDS + 2 * NUM_ACTIONS] = dq * OBS_SCALE_DOF_VEL
            obs[0, NUM_COMMANDS + 2 * NUM_ACTIONS:NUM_COMMANDS + 3 * NUM_ACTIONS] = action
            obs[0, NUM_COMMANDS + 3 * NUM_ACTIONS:NUM_COMMANDS + 3 * NUM_ACTIONS + 3] = (
                np.zeros(3) if zero_angvel else omega)
            obs[0, NUM_COMMANDS + 3 * NUM_ACTIONS + 3:NUM_COMMANDS + 3 * NUM_ACTIONS + 6] = (
                np.zeros(3) if zero_euler else eu)
            obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
            hist.append(obs)
            hist.pop(0)

            if t < settle_s:
                action = np.zeros(NUM_ACTIONS)
            else:
                inp = np.concatenate([h[0] for h in hist], axis=0)[None, :].astype(np.float32)
                with torch.no_grad():
                    action = policy(torch.from_numpy(inp))[0].numpy().astype(np.double)
                action = np.clip(action, -CLIP_ACTION, CLIP_ACTION)
        cnt += 1

        target_q = action * ACTION_SCALE
        tau = KPS * (DEFAULT_DOF_POS + target_q - q) - KDS * dq
        tau = np.clip(tau, tau_lo, tau_hi)
        data.ctrl[act_adr] = tau
        mujoco.mj_step(model, data)

        if step % 10 == 0:
            rec["t"].append(t); rec["z"].append(base_pos[2])
            rec["pitch"].append(eu[1]); rec["roll"].append(eu[0])
            rec["vx"].append(float(v_base[0])); rec["x"].append(float(base_pos[0]))
            if step % 100 == 0 and t < 1.6:
                d_om = omega_qvel - omega
                print(f"[diag][{name:10s}] t={t:4.2f} gyro={np.array2string(omega, precision=3)} "
                      f"qvel={np.array2string(omega_qvel, precision=3)} diff={np.array2string(d_om, precision=3)}")
        if fall_t is not None:
            break

    i_last = len(rec["t"]) - 1
    print(f"[diag][{name:10s}] survive={fall_t if fall_t else dur_s:6.2f}s "
          f"reason={fall_reason or '-':12s} final_z={rec['z'][i_last]:.3f} "
          f"pitch={rec['pitch'][i_last]:+.3f} roll={rec['roll'][i_last]:+.3f} "
          f"x={rec['x'][i_last]:+.2f}m vx_end={rec['vx'][i_last]:+.2f}")
    return fall_t or dur_s, fall_reason


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="auto")
    p.add_argument("--gl", type=str, default="egl")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ.setdefault("MUJOCO_GL", args.gl)
    mujoco = _ensure_mujoco(args.gl)
    global INT_IMPLICITFAST
    INT_IMPLICITFAST = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)

    ckpt = find_checkpoint(args.checkpoint)
    print(f"[diag] checkpoint: {ckpt}")
    exported = load_exported_policy(ckpt)
    policy = torch.jit.script(exported)
    policy.eval()

    model = build_model_mjcf_patched(mujoco)
    model.opt.timestep = SIM_DT

    variants = []
    # ---- solver-matrix mode: WALK trial + mujoco solver options ----------
    # The walk-onset collapse (t~2s+1s, both obs modes) is a gait-dynamics
    # divergence. PhysX TGS integrates implicitly; mujoco default Euler is
    # explicit and known to be less stable for contact-rich pendulum systems.
    # implicitfast + condim6 (rolling/torsional friction) are the standard
    # mujoco-side stability upgrades.
    walk_schedule = [(0.0, 2.0, (0.0, 0.0, 0.0)),
                     (2.0, 20.0, (1.0, 0.0, 0.0)),
                     (20.0, 24.0, (0.0, 0.0, 0.0))]

    def _with_opts(**opts):
        def applier(m):
            for k, v in opts.items():
                setattr(m.opt, k, v)
            return m
        return applier

    def _condim6(m):
        n = 0
        for g in range(m.ngeom):
            if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0:
                m.geom_condim[g] = 6
                n += 1
        print(f"[diag] condim6 on {n} collision geoms")
        return m

    def make(mujoco_, opt_fns):
        m = build_model_mjcf_patched(mujoco_)
        m.opt.timestep = SIM_DT
        for fn in opt_fns:
            m = fn(m)
        return m

    variants.append(("base_euler", dict(model_fns=[])))
    variants.append(("implicitfast", dict(
        model_fns=[lambda m: (setattr(m.opt, "integrator", INT_IMPLICITFAST), m)[1]])))
    variants.append(("condim6", dict(model_fns=[_condim6])))
    variants.append(("ifast_cd6", dict(
        model_fns=[_condim6, lambda m: (setattr(m.opt, "integrator", INT_IMPLICITFAST), m)[1]])))
    variants.append(("ts5e4_euler", dict(
        model_fns=[lambda m: (setattr(m.opt, "timestep", 0.0005), m)[1]])))
    for v in variants:
        v[1]["schedule"] = walk_schedule

    results = {}
    for name, kw in variants:
        m = make(mujoco, kw.pop("model_fns"))
        results[name] = run_variant(mujoco, m, policy, name, **kw)

    out = os.path.join(_REPO_ROOT, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out, exist_ok=True)
    torch.save({"ab_matrix": {k: v for k, v in results.items()}, "checkpoint": ckpt},
               os.path.join(out, "diag_ab_matrix.pt"))
    print("[diag] matrix complete -> gm_play/diag_ab_matrix.pt")


if __name__ == "__main__":
    main()
