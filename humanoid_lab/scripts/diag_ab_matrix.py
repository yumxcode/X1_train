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
                zero_angvel=False, settle_s=0.0, omega_from_qvel=False):
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
    rec = {"t": [], "z": [], "pitch": [], "roll": []}
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

        if cnt % DECIMATION == 0:
            if SW_SWITCH:
                cnt = 0  # stand command for the whole trial
            obs = np.zeros([1, NUM_SINGLE_OBS])
            ph = cnt * SIM_DT
            obs[0, 0] = math.sin(2 * math.pi * ph / CYCLE_TIME)
            obs[0, 1] = math.cos(2 * math.pi * ph / CYCLE_TIME)
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
            if step % 100 == 0 and t < 1.6:
                d_om = omega_qvel - omega
                print(f"[diag][{name:10s}] t={t:4.2f} gyro={np.array2string(omega, precision=3)} "
                      f"qvel={np.array2string(omega_qvel, precision=3)} diff={np.array2string(d_om, precision=3)}")
        if fall_t is not None:
            break

    i_last = len(rec["t"]) - 1
    print(f"[diag][{name:10s}] survive={fall_t if fall_t else dur_s:6.2f}s "
          f"reason={fall_reason or '-':12s} final_z={rec['z'][i_last]:.3f} "
          f"pitch={rec['pitch'][i_last]:+.3f} roll={rec['roll'][i_last]:+.3f}")
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

    ckpt = find_checkpoint(args.checkpoint)
    print(f"[diag] checkpoint: {ckpt}")
    exported = load_exported_policy(ckpt)
    policy = torch.jit.script(exported)
    policy.eval()

    model = build_model_mjcf_patched(mujoco)
    model.opt.timestep = SIM_DT

    variants = []
    variants.append(("base", dict()))
    variants.append(("omega_qvel", dict(omega_from_qvel=True)))
    variants.append(("no_angvel", dict(zero_angvel=True)))

    results = {}
    for name, kw in variants:
        if "solref" in kw:
            sr = kw.pop("solref")
            m2 = build_model_mjcf_patched(mujoco)
            m2.opt.timestep = SIM_DT
            n = 0
            for g in range(m2.ngeom):
                if m2.geom_type[g] == 0:  # mjGEOM_PLANE
                    m2.geom_solref[g] = [sr[0], sr[1]]
                    n += 1
            # robot collision geoms: contact pair with floor = any geom with contype
            for g in range(m2.ngeom):
                if m2.geom_contype[g] != 0 and m2.geom_type[g] != 0:
                    m2.geom_solref[g] = [sr[0], sr[1]]
                    n += 1
            print(f"[diag] {name}: solref={sr} applied to {n} geoms")
            results[name] = run_variant(mujoco, m2, policy, name, **kw)
        else:
            results[name] = run_variant(mujoco, model, policy, name, **kw)

    out = os.path.join(_REPO_ROOT, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(out, exist_ok=True)
    torch.save({"ab_matrix": {k: v for k, v in results.items()}, "checkpoint": ckpt},
               os.path.join(out, "diag_ab_matrix.pt"))
    print("[diag] matrix complete -> gm_play/diag_ab_matrix.pt")


if __name__ == "__main__":
    main()
