# Copyright (c) 2024, AgiBot Inc. All rights reserved.
"""Train the X1 stand/walk policy on Isaac Sim / IsaacLab.

Usage:
    python humanoid_lab/scripts/train_lab.py --headless --num_envs 4096
    python humanoid_lab/scripts/train_lab.py --headless --resume --load_run -1
"""

import argparse
import os
import sys
from datetime import datetime

# parse CLI before AppLauncher (it must be the first Isaac Sim call)
parser = argparse.ArgumentParser(description="X1 training on IsaacLab")
parser.add_argument("--task", type=str, default="x1_dh_stand")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=5)
parser.add_argument("--max_iterations", type=int, default=-1)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--experiment_name", type=str, default="x1_dh_stand")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--resume_from_path", type=str, default="",
                    help="explicit checkpoint file (e.g. gradmotion-mounted model_1000_xxx.pt). "
                         "'auto' searches the repo root / /workspace for model_*.pt")
parser.add_argument("--load_run", type=str, default="-1")
parser.add_argument("--checkpoint", type=int, default=-1)
parser.add_argument("--rl_device", type=str, default="cuda:0")
args_cli, unknown = parser.parse_known_args()

# make the repo root (e.g. /workspace/isaaclab/X1_train) importable when the
# script is launched by path (gm-run <repo>/humanoid_lab/scripts/train_lab.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# --- Isaac imports must happen after the app is up ---
import torch  # noqa: E402

from humanoid_lab import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid_lab.algo import DHOnPolicyRunner  # noqa: E402
from humanoid_lab.envs import (  # noqa: E402
    VecEnvAdapter,
    X1DHStandCfgPPO,
    X1DHStandEnv,
    X1DHStandEnvCfg,
)
from humanoid_lab.scripts.utils import class_to_dict, get_load_path  # noqa: E402


def ensure_x1_usd(cache_dir: str) -> str:
    """Convert the X1 URDF to a USD file once, then return the cached path.

    The IsaacLab contact sensor (and the rest of the pipeline) is only
    exercised on pre-converted USD assets (UsdFileCfg), which is the
    officially supported flow. The conversion runs once per container.
    """
    from isaaclab.sim.converters import UrdfConverter  # noqa: E402

    from humanoid_lab.envs.x1.x1_env_cfg import X1_URDF_CONVERTER_CFG  # noqa: E402

    usd_path = os.path.join(cache_dir, "x1.usd")
    if os.path.exists(usd_path):
        print(f"[train_lab] using cached X1 USD: {usd_path}")
        return usd_path

    os.makedirs(cache_dir, exist_ok=True)
    conv_cfg = X1_URDF_CONVERTER_CFG.replace(usd_dir=cache_dir, usd_file_name="x1.usd")
    print(f"[train_lab] converting URDF -> USD in {cache_dir} ...")
    converter = UrdfConverter(conv_cfg)
    print(f"[train_lab] conversion done: {converter.usd_path}")
    return converter.usd_path


def _discover_checkpoint(hint: str) -> str:
    """Locate a checkpoint file: explicit path, glob, or 'auto' search."""
    import glob as _glob

    candidates = []
    if hint and hint not in ("auto", "-1", ""):
        if os.path.isfile(hint):
            return hint
        candidates = sorted(_glob.glob(hint))
        if not candidates:
            raise FileNotFoundError(f"resume checkpoint not found: {hint}")
    else:
        for pattern in (
            os.path.join(LEGGED_GYM_ROOT_DIR, "model_*.pt"),
            os.path.join(LEGGED_GYM_ROOT_DIR, "*.pt"),
            "/workspace/model_*.pt",
            "/workspace/*/model_*.pt",
        ):
            candidates.extend(_glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError("auto search found no mounted *.pt checkpoint")

    def _iter_of(p):
        try:
            return int(torch.load(p, map_location="cpu", weights_only=False).get("iter", -1))
        except Exception:
            return -1

    best = max(candidates, key=_iter_of)
    print(f"[train_lab] checkpoint candidates: {candidates} -> {best}")
    return best


def main(args):
    env_cfg = X1DHStandEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    torch.manual_seed(args.seed)

    # URDF -> USD (cached) and inject the path into the spawn config
    usd_cache_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "x1_dh_stand", "urdf_cache"
    )
    usd_path = ensure_x1_usd(usd_cache_dir)
    env_cfg.scene.robot.spawn.usd_path = usd_path

    print(f"[train_lab] creating env with {args.num_envs} envs ...")
    env = X1DHStandEnv(cfg=env_cfg, render_mode=None)
    venv = VecEnvAdapter(env, env_cfg)

    train_cfg = X1DHStandCfgPPO()
    if args.max_iterations > 0:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.run_name:
        train_cfg.runner.run_name = args.run_name
    if args.experiment_name:
        train_cfg.runner.experiment_name = args.experiment_name
    train_cfg.runner.resume = args.resume
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint

    all_cfg = {"runner_class_name": train_cfg.runner_class_name}
    all_cfg.update(class_to_dict(train_cfg))

    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported_data"
    )
    os.makedirs(log_root, exist_ok=True)
    resume_path = None
    if args.resume_from_path:
        src = _discover_checkpoint(args.resume_from_path)
        # copy the mounted checkpoint into a fresh run dir so that the whole run
        # (initial + continued checkpoints) lives in one place for the SDK scan
        log_dir = os.path.join(
            log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + train_cfg.runner.run_name
        )
        os.makedirs(log_dir, exist_ok=True)
        it = torch.load(src, map_location="cpu", weights_only=False).get("iter", 0)
        dst = os.path.join(log_dir, f"model_{it}.pt")
        import shutil

        shutil.copyfile(src, dst)
        resume_path = dst
        print(f"[train_lab] resuming from mounted checkpoint {src} (iter={it})")
    elif args.resume:
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run,
                                    checkpoint=train_cfg.runner.checkpoint)
        log_dir = os.path.dirname(resume_path)
        print(f"[train_lab] resuming from {resume_path}")
    else:
        log_dir = os.path.join(
            log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + train_cfg.runner.run_name
        )
        os.makedirs(log_dir, exist_ok=True)

    runner = DHOnPolicyRunner(venv, all_cfg, log_dir, device=args.rl_device)
    if resume_path is not None:
        runner.load(resume_path, load_optimizer=True)

    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=False,
    )


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"[train_lab] FATAL: {e}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
