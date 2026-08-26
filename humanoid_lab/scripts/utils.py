# Copyright (c) 2024, AgiBot Inc. All rights reserved.
"""Small helpers for the IsaacLab training entry point (no isaacgym imports)."""

import os


def class_to_dict(cfg) -> dict:
    """Recursively convert a nested plain-class config into a dict."""
    out = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        val = getattr(cfg, key)
        if isinstance(val, (int, float, str, bool, list, tuple, dict, type(None))):
            out[key] = val
        elif hasattr(val, "__dict__") or isinstance(val, type):
            try:
                out[key] = class_to_dict(val)
            except Exception:
                out[key] = str(val)
    return out


def get_load_path(root, load_run=-1, checkpoint=-1) -> str:
    """Find the model path to resume from (mirrors humanoid.utils.helpers.get_load_path)."""
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception as exc:
        raise ValueError(f"No runs in this directory: {root}") from exc
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [f for f in os.listdir(load_run) if "model" in f]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    return os.path.join(load_run, model)
