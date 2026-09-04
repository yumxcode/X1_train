# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Record the trained X1 policy playing inside IsaacLab (offscreen video).

Produces the IsaacLab-side counterpart of the mujoco sim2sim videos: same
checkpoint, same forward command schedule (0-2s stand, 2-20s vx=1.0, 20-24s
stand), clean eval config (no noise, no domain randomization). A TiledCamera
anchored to the robot follows the walk (official IsaacLab camera tutorial
convention). Frames are captured at 20 Hz and encoded to mp4 (SDK uploads
anything under logs/x1_dh_stand/).

Usage:
    python humanoid_lab/scripts/play_video_lab.py --checkpoint auto --trial walk
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--trial", type=str, default="walk", choices=["stand", "walk", "both"])
parser.add_argument("--max_steps", type=int, default=2400)  # 24 s @ 100 Hz
parser.add_argument("--checkpoint", type=str, default="auto")
parser.add_argument("--fps", type=int, default=20)
parser_cli, _ = parser.parse_known_args()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=parser_cli.headless, offscreen_render=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch  # noqa: E402

from humanoid_lab import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_lab.algo.actor_critic_dh import ActorCriticDH  # noqa: E402
from humanoid_lab.envs import X1DHStandEnv, X1DHStandEnvCfg  # noqa: E402
from humanoid_lab.scripts.export_policy_lab import (  # noqa: E402
    NUM_SINGLE_OBS, POLICY_CFG, SHORT_FRAME_STACK, find_checkpoint,
)
from humanoid_lab.scripts.sim2sim_eval import VideoWriter  # noqa: E402

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


# forward-trial command schedule (mirrors sim2sim TRIALS["forward"])
def cmd_at(t):
    if t < 2.0 or t >= 20.0:
        return 0.0
    return 1.0


def run_trial(env, ac, trial, out_dir, max_steps, fps):
    from isaaclab.sensors import TiledCamera  # noqa: F401  (type ref only)

    cam = env.scene["tiled_camera"]
    obs_dict, _ = env.reset()
    # freeze the gait scheduler: no in-episode resampling, we drive commands
    env.gait_time[:] = 10**9
    capture_every = max(1, round(100 / fps))  # policy 100 Hz -> fps

    vw = VideoWriter(os.path.join(out_dir, f"isaaclab_play_{trial}.mp4"), 848, 480, fps)
    z_last, vx_last = 0.0, 0.0
    t0 = time.time()
    for step in range(max_steps):
        t = step * 0.01
        vx_cmd = cmd_at(t) if trial == "walk" else 0.0
        env.commands[:, 0] = vx_cmd
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        obs = obs_dict["policy"].to(env.device)
        with torch.no_grad():
            action = ac.act_inference(obs)
        obs_dict, _, terminated, _, _ = env.step(action)
        rs = env.root_states[0]
        z_last, vx_last = float(rs[2]), float(rs[7])
        if step % 10 == 0:
            print(f"[play:{trial}] t={t:5.2f} z={z_last:.3f} vx={vx_last:+.2f} "
                  f"cmd={vx_cmd:+.2f}" + ("  <<TERMINATED" if bool(terminated[0]) else ""))
        if step % capture_every == 0:
            rgb = cam.data.output["rgb"]
            frame = rgb[0].detach().cpu().numpy()[:, :, :3]
            vw.add(frame)
        if bool(terminated[0]):
            print(f"[play:{trial}] TERMINATED at t={t:.2f}")
            break
    vw.close()
    print(f"[play:{trial}] video done: z_end={z_last:.3f} vx_end={vx_last:+.2f} "
          f"({max_steps * 0.01:.0f}s, wall {time.time() - t0:.1f}s)")


def main(args):
    from isaaclab.sensors import TiledCameraCfg

    cfg = X1DHStandEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # ---- clean evaluation config (same as diag_isaaclab_play) ----
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
    # ---- FIXED world camera at spawn (follow-cam attach mode hung the      #
    # RTX renderer init on this A10 image, 3 attempts: 005/016/020). A      #
    # camera on a FRESH prim under /World/Camera is spawned fresh (no       #
    # collision with the robot articulation). The walk trial starts at      #
    # origin and walks forward ~x, so a static camera at (3.5, 2.5, 1.2)    #
    # frames most of it.                                                    #
    from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg

    cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.5, 2.5, 1.2), rot=(0.36, 0.36, 0.609, -0.609), convention="ros",
        ),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 50.0)),
        width=848, height=480, update_period=0.05,
    )

    usd_cache = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "urdf_cache")
    cfg.scene.robot.spawn.usd_path = ensure_x1_usd(usd_cache)

    print("[play] creating clean env with tiled camera ...")
    env = X1DHStandEnv(cfg=cfg, render_mode=None)
    # NOTE: InteractiveScene has no __contains__; `in` falls back to iteration
    # (__getitem__(0) -> KeyError '0'). Access directly instead.
    try:
        env.scene["tiled_camera"]
    except KeyError as exc:
        raise RuntimeError(f"tiled_camera missing from scene -- cannot record ({exc})")

    ckpt = find_checkpoint(args.checkpoint)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    ac = ActorCriticDH(SHORT_FRAME_STACK * NUM_SINGLE_OBS, NUM_SINGLE_OBS, 3 * 73, 12, **POLICY_CFG)
    ac.load_state_dict(loaded["model_state_dict"])
    ac = ac.eval().to(env.device)
    print(f"[play] loaded {ckpt} (iter={loaded.get('iter')})")

    # Write videos OUTSIDE the SDK scan range first: the gradmotion SDK
    # uploads any *.mp4 it sees under logs/<exp>/ THE MOMENT the file is
    # detected -- grabbing half-written files (48-byte truncated uploads,
    # TASK_20260904_049). Only after both writers are closed do we copy the
    # finished files into the scanned directory.
    tmp_dir = os.path.join("/tmp", "x1_play_video", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(tmp_dir, exist_ok=True)
    trials = ["walk", "stand"] if args.trial == "both" else [args.trial]
    for trial in trials:
        run_trial(env, ac, trial, tmp_dir, args.max_steps, args.fps)

    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand",
                           "play_video", os.path.basename(tmp_dir))
    os.makedirs(out_dir, exist_ok=True)
    import shutil

    for trial in trials:
        src = os.path.join(tmp_dir, f"isaaclab_play_{trial}.mp4")
        shutil.copyfile(src, os.path.join(out_dir, f"isaaclab_play_{trial}.mp4"))
        print(f"[play] finalized {trial} video -> {out_dir}")
    print(f"[play] all videos under {out_dir}")


if __name__ == "__main__":
    try:
        main(parser_cli)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[play] FATAL: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
