# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""Export an IsaacLab-trained X1 checkpoint to a deployment JIT policy.

Standalone re-implementation of ``humanoid/scripts/export_policy_dh.py`` that
does NOT import ``humanoid.envs`` (which requires Isaac Gym). It only depends
on ``torch`` and the pure-torch networks in ``humanoid_lab.algo``, so it runs
on any image (IsaacSim image, plain pytorch image, ...).

The exported module takes the full long-history observation
``(1, frame_stack * num_single_obs)`` and returns the 12 joint position
targets (in action units) -- identical to the original deployment export.

Usage:
    python humanoid_lab/scripts/export_policy_lab.py \
        --checkpoint /path/to/model_8000.pt --out_dir logs/x1_dh_stand/exported_policies
"""

import argparse
import copy
import os
import sys
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from humanoid_lab.algo.actor_critic_dh import ActorCriticDH  # noqa: E402

# ---- architecture constants (must match X1DHStandCfgPPO in x1_env_cfg.py) ----
POLICY_CFG = dict(
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[768, 256, 128],
    state_estimator_hidden_dims=[256, 128, 64],
    kernel_size=[6, 4],
    filter_size=[32, 16],
    stride_size=[3, 2],
    lh_output_dim=64,
    in_channels=66,
)
FRAME_STACK = 66
SHORT_FRAME_STACK = 5
NUM_SINGLE_OBS = 47
NUM_ACTIONS = 12


class ExportedDH(torch.nn.Module):
    """Deployment wrapper: state estimator + long-history CNN + actor MLP."""

    def __init__(self, actor, long_history, state_estimator, num_short_obs, in_channels, num_proprio_obs):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.long_history = copy.deepcopy(long_history).cpu()
        self.state_estimator = copy.deepcopy(state_estimator).cpu()
        self.num_short_obs = num_short_obs
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs

    def forward(self, observations):
        short_history = observations[..., -self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed_long_history = self.long_history(
            observations.view(-1, self.in_channels, self.num_proprio_obs)
        )
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history), dim=-1)
        return self.actor(actor_obs)


def find_checkpoint(path_or_glob: str) -> str:
    """Resolve a checkpoint path; supports glob patterns and 'auto' search."""
    if path_or_glob and path_or_glob not in ("auto", "-1", ""):
        if os.path.isfile(path_or_glob):
            return path_or_glob
        import glob as _glob

        hits = sorted(_glob.glob(path_or_glob))
        if hits:
            return hits[-1]
        raise FileNotFoundError(f"checkpoint not found: {path_or_glob}")
    # auto: search the repo root and common mount locations
    import glob as _glob

    candidates = []
    for pattern in (
        os.path.join(_REPO_ROOT, "model_*.pt"),
        os.path.join(_REPO_ROOT, "*.pt"),
        "/workspace/model_*.pt",
        "/workspace/*/model_*.pt",
    ):
        candidates.extend(_glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError("auto search found no *.pt checkpoint")

    def iter_of(p):
        try:
            it = torch.load(p, map_location="cpu", weights_only=False).get("iter", -1)
            return int(it)
        except Exception:
            return -1

    return max(candidates, key=iter_of)


def load_exported_policy(checkpoint_path: str) -> ExportedDH:
    num_short_obs = SHORT_FRAME_STACK * NUM_SINGLE_OBS
    actor_critic = ActorCriticDH(
        num_short_obs,
        NUM_SINGLE_OBS,
        3 * 73,  # privileged obs dims (critic, unused at deployment)
        NUM_ACTIONS,
        **POLICY_CFG,
    )
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = loaded["model_state_dict"] if "model_state_dict" in loaded else loaded
    actor_critic.load_state_dict(state)
    actor_critic.eval()
    print(f"[export_policy_lab] loaded checkpoint: {checkpoint_path} (iter={loaded.get('iter', '?')})")
    return ExportedDH(
        actor_critic.actor,
        actor_critic.long_history,
        actor_critic.state_estimator,
        num_short_obs,
        POLICY_CFG["in_channels"],
        NUM_SINGLE_OBS,
    )


def main():
    parser = argparse.ArgumentParser(description="Export X1 policy to JIT")
    parser.add_argument("--checkpoint", type=str, default="auto",
                        help="checkpoint path/glob, or 'auto' to search mounted files")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_REPO_ROOT, "logs", "x1_dh_stand", "exported_policies"))
    args = parser.parse_args()

    ckpt = find_checkpoint(args.checkpoint)
    exported = load_exported_policy(ckpt)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "policy_dh.jit")
    scripted = torch.jit.script(exported)
    scripted.save(out_path)
    print(f"[export_policy_lab] exported JIT policy -> {out_path}")

    # smoke: forward pass shape check
    dummy = torch.zeros(1, FRAME_STACK * NUM_SINGLE_OBS)
    with torch.no_grad():
        out = scripted(dummy)
    assert out.shape == (1, NUM_ACTIONS), f"bad output shape {out.shape}"
    print(f"[export_policy_lab] smoke forward OK, action shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
