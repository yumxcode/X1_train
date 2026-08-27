# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Headless Mujoco sim2sim evaluation for the IsaacLab-trained X1 walking policy.

Ports ``humanoid/scripts/sim2sim.py`` to a fully headless, gradmotion-friendly
pipeline:
  * no pygame joystick / interactive viewer  -> scripted command schedule
  * offscreen rendering (MUJOCO_GL=egl)      -> mp4 video per trial
  * quantitative metrics                     -> strict PASS/FAIL verdict

Fixes vs the original script:
  * ``get_obs`` always-true body-name conditions (``if '5_link' or ... in b``)
    are replaced with exact body ids: root pose from ``qpos[:7]``, feet from
    the ``*_ankle_roll_link`` bodies.
  * torques are clipped to the per-joint actuator ctrlrange (the original flat
    500 Nm clip was looser than the hardware).

Usage:
    python humanoid_lab/scripts/sim2sim_eval.py --checkpoint auto --video 1

Exit codes: 0 = all strict criteria pass; 3 = metric FAIL; 4 = video FAIL.
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------------- #
# mujoco bootstrap (install if missing; choose GL backend)                      #
# --------------------------------------------------------------------------- #
def _ensure_mujoco(gl_backend: str):
    os.environ.setdefault("MUJOCO_GL", gl_backend)
    try:
        import mujoco  # noqa: F401

        return mujoco
    except ImportError:
        print("[sim2sim] mujoco not installed -> pip install mujoco")
        for extra in ([], ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]):
            cmd = [sys.executable, "-m", "pip", "install", "--timeout",
                   "60", "--retries", "2", "mujoco"] + extra
            print(f"[sim2sim] running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, timeout=420)
                import mujoco  # noqa: F401

                return mujoco
            except subprocess.SubprocessError as exc:
                print(f"[sim2sim] pip attempt failed: {exc}")
        raise RuntimeError("failed to install mujoco (pypi + tsinghua mirror)")


# --------------------------------------------------------------------------- #
# robot / task constants (must mirror humanoid_lab/envs/x1/x1_env_cfg.py)      #
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid_lab.scripts.export_policy_lab import find_checkpoint, load_exported_policy  # noqa: E402

MJCF_PATH = os.path.join(_REPO_ROOT, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")

NUM_ACTIONS = 12
FRAME_STACK = 66
NUM_SINGLE_OBS = 47
NUM_COMMANDS = 5

DEFAULT_DOF_POS = np.array(
    [0.4, 0.05, -0.31, 0.49, -0.21, 0.0, -0.4, -0.05, 0.31, 0.49, -0.21, 0.0],
    dtype=np.double,
)
KPS = np.array([30, 40, 35, 100, 35, 35] * 2, dtype=np.double)
KDS = np.array([3, 3.0, 4, 10, 0.5, 0.5] * 2, dtype=np.double)

ACTION_SCALE = 0.5
DECIMATION = 10
SIM_DT = 0.001
CYCLE_TIME = 0.7
STAND_COM_THRESHOLD = 0.05
SW_SWITCH = True

OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 1.0
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
CLIP_OBS = 100.0
CLIP_ACTION = 100.0

TORSO_BODIES = ["x1-body", "body_yaw", "body_roll", "body_pitch"]
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# ---- strict pass criteria (docs/PASS_CRITERIA.md, fixed before training) ---- #
CRITERIA = {
    "trial_duration_s": 24.0,
    "fall_base_height": 0.40,
    "tracking_vx_median_max": 0.25,
    "tracking_vy_median_max": 0.20,
    "tracking_wz_mean_max": 0.30,
    "base_height_mean_range": (0.52, 0.70),
    "base_height_min": 0.45,
    "torque_saturation_ratio_max": 0.005,
    "displacement_ratio_min": {"forward": 0.70, "max": 0.65},
    "video_min_duration_s": 20.0,
    "video_min_frames_per_s": 24.0,
    "video_min_travel_m": 2.0,
}

TRIALS = {
    "stand": [(0.0, 24.0, (0.0, 0.0, 0.0))],
    "forward": [(0.0, 2.0, (0.0, 0.0, 0.0)), (2.0, 20.0, (1.0, 0.0, 0.0)), (20.0, 24.0, (0.0, 0.0, 0.0))],
    "omni": [
        (0.0, 2.0, (0.0, 0.0, 0.0)),
        (2.0, 8.0, (0.5, 0.3, 0.0)),
        (8.0, 14.0, (0.5, -0.3, 0.0)),
        (14.0, 19.0, (0.0, 0.0, 0.5)),
        (19.0, 24.0, (0.3, 0.0, -0.4)),
    ],
    "max": [(0.0, 2.0, (0.0, 0.0, 0.0)), (2.0, 21.0, (1.2, 0.0, 0.0)), (21.0, 24.0, (0.0, 0.0, 0.0))],
}
INIT_HEIGHT = 0.70  # training parity: cfg.env.init_state_z (mjcf default 0.8 gives 2x impact energy)
GRACE_S = 1.0  # excluded from tracking stats after each command switch


# --------------------------------------------------------------------------- #
# math helpers (numpy-only, no scipy)                                          #
# --------------------------------------------------------------------------- #
def quat_xyzw_to_rot(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.double,
    )


def quat_xyzw_to_euler(quat):
    x, y, z, w = quat
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw])


# --------------------------------------------------------------------------- #
# video writer with encoder fallbacks                                          #
# --------------------------------------------------------------------------- #
class VideoWriter:
    def __init__(self, path, width, height, fps):
        self.path, self.w, self.h, self.fps = path, width, height, fps
        self.frames = []
        self.backend = None
        self._writer = None
        try:
            import imageio.v2 as iio

            self._iio = iio
            self._writer = iio.get_writer(
                path, fps=fps, codec="libx264", quality=8, macro_block_size=None
            )
            self.backend = "imageio"
        except Exception as exc:
            print(f"[sim2sim] imageio unavailable ({exc}); trying cv2")
            try:
                import cv2

                self._cv2 = cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
                self.backend = "cv2"
            except Exception as exc2:
                print(f"[sim2sim] cv2 unavailable ({exc2}); buffering frames for ffmpeg")
                self.backend = "buffer"

    def add(self, rgb):
        if self.backend == "imageio":
            self._writer.append_data(rgb)
        elif self.backend == "cv2":
            self._writer.write(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))
        else:
            self.frames.append(rgb)

    def close(self):
        if self.backend in ("imageio", "cv2"):
            self._writer.close()
        elif self.frames and self.backend == "buffer":
            import imageio.v2 as iio

            iio.mimsave(self.path, self.frames, fps=self.fps)
            self.frames = []


# --------------------------------------------------------------------------- #
# rollout                                                                      #
# --------------------------------------------------------------------------- #
def run_trial(mujoco, model, policy, trial_name, schedule, out_dir, make_video, width, height, fps):
    data = mujoco.MjData(model)
    data.qpos[-NUM_ACTIONS:] = DEFAULT_DOF_POS
    data.qpos[2] = INIT_HEIGHT  # match training spawn (init_state_z), not the mjcf 0.8
    mujoco.mj_step(model, data)

    torso_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in TORSO_BODIES]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
    ctrlrange = model.actuator_ctrlrange.copy()  # (12, 2)
    tau_lo, tau_hi = ctrlrange[:, 0], ctrlrange[:, 1]

    renderer = None
    video_writer = None
    cam = None
    if make_video:
        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            video_writer = VideoWriter(
                os.path.join(out_dir, f"sim2sim_{trial_name}.mp4"), width, height, fps
            )
        except Exception as exc:
            print(f"[sim2sim] WARNING offscreen rendering failed: {exc}")
            renderer = None
    hist_obs = [np.zeros([1, NUM_SINGLE_OBS], dtype=np.double) for _ in range(FRAME_STACK)]
    target_q = np.zeros(NUM_ACTIONS, dtype=np.double)
    action = np.zeros(NUM_ACTIONS, dtype=np.double)

    total_steps = int(CRITERIA["trial_duration_s"] / SIM_DT)
    render_every = max(1, round(1.0 / (fps * SIM_DT)))
    log_every = DECIMATION  # record metrics at policy rate (100 Hz)

    rec = {
        k: []
        for k in ("t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "base_z",
                  "roll", "pitch", "tau_max_abs", "tau_sat", "foot_l", "foot_r",
                  "torso_contact", "base_x", "base_y")
    }
    fall = None  # (t, reason)
    count_lowlevel = 0

    for step in range(total_steps):
        t = step * SIM_DT

        # --- scheduled command ------------------------------------------------
        cmd = next((c for (t0, t1, c) in schedule if t0 <= t < t1), (0.0, 0.0, 0.0))
        vx_cmd, vy_cmd, wz_cmd = cmd

        # --- observation ingredients ------------------------------------------
        q = data.qpos[-NUM_ACTIONS:].astype(np.double)
        dq = data.qvel[-NUM_ACTIONS:].astype(np.double)
        quat_xyzw = data.sensor("body-orientation").data[[1, 2, 3, 0]].astype(np.double)
        rot = quat_xyzw_to_rot(quat_xyzw)
        v_base = rot.T @ data.qvel[:3].astype(np.double)
        omega = data.sensor("body-angular-velocity").data.astype(np.double)
        eu_ang = quat_xyzw_to_euler(quat_xyzw)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        base_pos = data.qpos[:3].astype(np.double)

        foot_contact = [abs(data.cfrc_ext[i][2]) > 5.0 for i in foot_ids]
        torso_contact = any(
            np.linalg.norm(data.cfrc_ext[i]) > 1.0 for i in torso_ids
        )

        # --- fall detection (mirrors training termination) ---------------------
        if torso_contact:
            fall = (t, "torso_contact")
        elif abs(eu_ang[0]) > 1.5:
            fall = (t, "roll>1.5")
        elif abs(eu_ang[1]) > 1.5:
            fall = (t, "pitch>1.5")
        elif base_pos[2] < CRITERIA["fall_base_height"]:
            fall = (t, "base_height<0.40")

        # --- policy @100 Hz -----------------------------------------------------
        if count_lowlevel % DECIMATION == 0:
            vel_norm = math.sqrt(vx_cmd**2 + vy_cmd**2 + wz_cmd**2)
            if SW_SWITCH and vel_norm <= STAND_COM_THRESHOLD:
                count_lowlevel = 0

            obs = np.zeros([1, NUM_SINGLE_OBS], dtype=np.float32)
            phase_t = count_lowlevel * SIM_DT
            obs[0, 0] = math.sin(2 * math.pi * phase_t / CYCLE_TIME)
            obs[0, 1] = math.cos(2 * math.pi * phase_t / CYCLE_TIME)
            obs[0, 2] = vx_cmd * OBS_SCALE_LIN_VEL
            obs[0, 3] = vy_cmd * OBS_SCALE_LIN_VEL
            obs[0, 4] = wz_cmd * OBS_SCALE_ANG_VEL
            obs[0, NUM_COMMANDS:NUM_COMMANDS + NUM_ACTIONS] = (q - DEFAULT_DOF_POS) * OBS_SCALE_DOF_POS
            obs[0, NUM_COMMANDS + NUM_ACTIONS:NUM_COMMANDS + 2 * NUM_ACTIONS] = dq * OBS_SCALE_DOF_VEL
            obs[0, NUM_COMMANDS + 2 * NUM_ACTIONS:NUM_COMMANDS + 3 * NUM_ACTIONS] = action
            obs[0, NUM_COMMANDS + 3 * NUM_ACTIONS:NUM_COMMANDS + 3 * NUM_ACTIONS + 3] = omega
            obs[0, NUM_COMMANDS + 3 * NUM_ACTIONS + 3:NUM_COMMANDS + 3 * NUM_ACTIONS + 6] = eu_ang
            obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)

            hist_obs.append(obs)
            hist_obs.pop(0)

            policy_input = np.concatenate([h[0] for h in hist_obs], axis=0)[None, :].astype(np.float32)
            with torch.no_grad():
                action = policy(torch.from_numpy(policy_input))[0].numpy().astype(np.double)
            action = np.clip(action, -CLIP_ACTION, CLIP_ACTION)
            target_q = action * ACTION_SCALE

        # --- PD @1 kHz -----------------------------------------------------------
        tau = KPS * (target_q + DEFAULT_DOF_POS - q) - KDS * dq
        tau_cmd = np.clip(tau, tau_lo, tau_hi)
        data.ctrl[:] = tau_cmd
        mujoco.mj_step(model, data)
        count_lowlevel += 1

        # --- record @100 Hz -------------------------------------------------------
        if step % log_every == 0:
            rec["t"].append(t)
            rec["cmd_vx"].append(vx_cmd)
            rec["cmd_vy"].append(vy_cmd)
            rec["cmd_wz"].append(wz_cmd)
            rec["vx"].append(v_base[0])
            rec["vy"].append(v_base[1])
            rec["wz"].append(omega[2])
            rec["base_z"].append(base_pos[2])
            rec["roll"].append(eu_ang[0])
            rec["pitch"].append(eu_ang[1])
            rec["tau_max_abs"].append(float(np.max(np.abs(tau_cmd))))
            rec["tau_sat"].append(float(np.mean((np.abs(tau_cmd) >= 0.98 * np.maximum(np.abs(tau_lo), np.abs(tau_hi))))))
            rec["foot_l"].append(bool(foot_contact[0]))
            rec["foot_r"].append(bool(foot_contact[1]))
            rec["torso_contact"].append(bool(torso_contact))
            rec["base_x"].append(base_pos[0])
            rec["base_y"].append(base_pos[1])

        # --- render ----------------------------------------------------------------
        if renderer is not None and step % render_every == 0:
            if cam is None:
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, cam)
                cam.distance = 4.0
                cam.azimuth = 90.0
                cam.elevation = -10.0
            cam.lookat[:] = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
            renderer.update_scene(data, cam)
            video_writer.add(renderer.render())

        if fall:
            print(f"[sim2sim] FALL at t={fall[0]:.2f}s reason={fall[1]} ({trial_name})")
            break

    if video_writer is not None:
        video_writer.close()
    if renderer is not None:
        renderer.close()

    return {k: np.asarray(v) for k, v in rec.items()}, fall


# --------------------------------------------------------------------------- #
# metrics + verdict                                                            #
# --------------------------------------------------------------------------- #
def active_mask(rec, schedule, grace_s=GRACE_S):
    """Mask of policy steps inside a moving-command segment, minus grace period.

    A sample at time t counts as active iff the schedule segment containing t
    has a moving command AND the sample is at least ``grace_s`` into the segment.
    """
    t = np.asarray(rec["t"], dtype=float)
    keep = np.zeros(len(t), dtype=bool)
    for i, ti in enumerate(t):
        for (t0, t1, c) in schedule:
            if t0 - 1e-9 <= ti < t1:
                moving = abs(c[0]) + abs(c[1]) + abs(c[2]) > STAND_COM_THRESHOLD
                keep[i] = moving and (ti - t0) >= grace_s - 1e-9
                break
    return keep


def evaluate_trial(name, rec, fall, schedule):
    m = active_mask(rec, schedule)
    checks = {}
    stats = {}

    # S1 survival
    checks["S1_no_fall"] = fall is None and rec["t"][-1] >= CRITERIA["trial_duration_s"] - 0.02
    stats["survived_s"] = float(rec["t"][-1])

    # S2 tracking
    err_vx = np.abs(rec["vx"][m] - rec["cmd_vx"][m]) if m.any() else np.array([0.0])
    err_vy = np.abs(rec["vy"][m] - rec["cmd_vy"][m]) if m.any() else np.array([0.0])
    err_wz = np.abs(rec["wz"][m] - rec["cmd_wz"][m]) if m.any() else np.array([0.0])
    stats["err_vx_median"] = float(np.median(err_vx))
    stats["err_vy_median"] = float(np.median(err_vy))
    stats["err_wz_mean"] = float(np.mean(err_wz))
    checks["S2_tracking"] = (
        stats["err_vx_median"] <= CRITERIA["tracking_vx_median_max"]
        and stats["err_vy_median"] <= CRITERIA["tracking_vy_median_max"]
        and stats["err_wz_mean"] <= CRITERIA["tracking_wz_mean_max"]
    )

    # S3 base height
    stats["base_z_mean_active"] = float(np.mean(rec["base_z"][m])) if m.any() else 0.0
    stats["base_z_min"] = float(np.min(rec["base_z"]))
    lo, hi = CRITERIA["base_height_mean_range"]
    checks["S3_base_height"] = (
        lo <= stats["base_z_mean_active"] <= hi and stats["base_z_min"] > CRITERIA["base_height_min"]
    )

    # S4 torque safety
    stats["tau_p99"] = float(np.percentile(rec["tau_max_abs"], 99))
    stats["tau_max"] = float(np.max(rec["tau_max_abs"]))
    stats["tau_sat_ratio"] = float(np.mean(rec["tau_sat"]))
    checks["S4_torque"] = stats["tau_sat_ratio"] <= CRITERIA["torque_saturation_ratio_max"]

    # S5 displacement (forward walks only)
    if name in CRITERIA["displacement_ratio_min"]:
        integ = float(np.sum(rec["cmd_vx"][:-1] * np.diff(rec["t"])))  # \int vx_cmd dt
        disp = float(rec["base_x"][-1] - rec["base_x"][0])
        stats["cmd_integral_vx"] = integ
        stats["net_forward_disp"] = disp
        stats["disp_ratio"] = disp / integ if integ > 1e-6 else 0.0
        checks["S5_displacement"] = stats["disp_ratio"] >= CRITERIA["displacement_ratio_min"][name]
    else:
        stats["travel_xy"] = float(np.hypot(rec["base_x"][-1] - rec["base_x"][0], rec["base_y"][-1] - rec["base_y"][0]))

    return checks, stats


def evaluate_videos(out_dir, trials, travel):
    checks, stats = {}, {}
    for name in trials:
        path = os.path.join(out_dir, f"sim2sim_{name}.mp4")
        ok = os.path.isfile(path) and os.path.getsize(path) > 100_000
        dur = 0.0
        if ok:
            try:
                import imageio.v2 as iio

                rd = iio.get_reader(path)
                meta = rd.get_meta_data() or {}
                fps_v = float(meta.get("fps") or 0)
                n = 0
                try:
                    n = len(rd)
                except Exception:
                    n = 0
                dur = (n / fps_v) if (n and fps_v) else float(meta.get("duration") or 0)
                rd.close()
            except Exception as exc:
                print(f"[sim2sim] video probe failed for {name}: {exc}")
                dur = 0.0
        checks[f"S6_video_{name}"] = (
            ok
            and dur >= CRITERIA["video_min_duration_s"]
            and travel.get(name, 0.0) >= CRITERIA["video_min_travel_m"]
        )
        stats[f"video_{name}_dur_s"] = dur
    return checks, stats


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Headless X1 sim2sim evaluation")
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--gl", type=str, default="egl", choices=["egl", "osmesa", "glfw", "disabled"])
    parser.add_argument("--video", type=int, default=1)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--trials", type=str, default="forward,omni,max")
    parser.add_argument("--allow_fail", action="store_true",
                        help="exit 0 even on FAIL (pipeline smoke tests)")
    args = parser.parse_args()

    # live logs on gradmotion (SDK tails stdout)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    mujoco = _ensure_mujoco(args.gl)

    out_dir = args.out_dir or os.path.join(
        _REPO_ROOT, "logs", "x1_dh_stand", "sim2sim_eval", time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    ckpt = find_checkpoint(args.checkpoint)
    print(f"[sim2sim] checkpoint: {ckpt}")
    exported = load_exported_policy(ckpt)
    policy = torch.jit.script(exported)
    policy.eval()

    print(f"[sim2sim] mujoco model: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    model.opt.timestep = SIM_DT
    # enlarge the offscreen framebuffer (the shipped mjcf caps it at 640 px)
    model.vis.global_.offwidth = max(args.width, 640)
    model.vis.global_.offheight = max(args.height, 480)
    print(f"[sim2sim] offscreen framebuffer: {model.vis.global_.offwidth}x{model.vis.global_.offheight}")

    trial_names = [s.strip() for s in args.trials.split(",") if s.strip()]
    results, all_checks, travel, metrics_pack = {}, {}, {}, {}
    for name in trial_names:
        print(f"[sim2sim] === trial {name} ===")
        rec, fall = run_trial(
            mujoco, model, policy, name, TRIALS[name], out_dir,
            make_video=bool(args.video), width=args.width, height=args.height, fps=args.fps,
        )
        checks, stats = evaluate_trial(name, rec, fall, TRIALS[name])
        all_checks.update({f"{k}[{name}]": v for k, v in checks.items()})
        travel[name] = stats.get("net_forward_disp", stats.get("travel_xy", 0.0))
        results[name] = {"checks": checks, "stats": stats, "fall": fall}
        metrics_pack[name] = {k: (v[::10].tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else None)
                              for k, v in rec.items()}
        print(json.dumps(stats, indent=2))

    video_checks, video_stats = evaluate_videos(out_dir, trial_names, travel)
    all_checks.update(video_checks)

    verdict = {
        "checkpoint": ckpt,
        "criteria": CRITERIA,
        "trials": results,
        "video": video_stats,
        "checks": all_checks,
        "overall_pass": all(all_checks.values()),
    }
    verdict_path = os.path.join(out_dir, "verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=float)

    # pack metrics for gradmotion SDK upload (pt in gm_play/)
    gm_play_dir = os.path.join(_REPO_ROOT, "logs", "x1_dh_stand", "gm_play")
    os.makedirs(gm_play_dir, exist_ok=True)
    torch.save({"verdict": verdict, "metrics": metrics_pack},
               os.path.join(gm_play_dir, "sim2sim_verdict.pt"))

    print("\n================ SIM2SIM VERDICT ================")
    for k, v in all_checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print(f"  verdict: {verdict_path}")
    print("==================================================")

    if args.allow_fail:
        sys.exit(0)
    if not all(v for k, v in all_checks.items() if k.startswith("S6")):
        sys.exit(4)
    sys.exit(0 if verdict["overall_pass"] else 3)


if __name__ == "__main__":
    main()
