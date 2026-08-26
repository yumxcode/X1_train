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
parser.add_argument("--load_run", type=str, default="-1")
parser.add_argument("--checkpoint", type=int, default=-1)
parser.add_argument("--rl_device", type=str, default="cuda:0")
args_cli, unknown = parser.parse_known_args()

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


def main(args):
    env_cfg = X1DHStandEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    torch.manual_seed(args.seed)

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
    if args.resume:
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
    if args.resume:
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
