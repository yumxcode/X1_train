#!/usr/bin/env python3
"""Evaluate strict training criteria T1-T6 from gradmotion task charts.

Usage:
    GM_API_KEY=... python3 tools/eval_train_criteria.py TASK_xxx [--window 100]

Reads Train/mean_reward, Train/mean_episode_length and the Episode/rew_* keys
via `gm task data get`, then prints a PASS/FAIL table per docs/PASS_CRITERIA.md.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

KEYS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode/rew_tracking_lin_vel",
    "Episode/rew_ref_joint_pos",
    "Episode/rew_collision",
]

CRITERIA = {
    "T1_mean_reward_ge": 200.0,
    "T2_mean_ep_len_ge": 2100.0,
    "T3_track_lin_ge": 1.20,
    "T4_ref_joint_ge": 1.40,
    "T5_collision_ge": -0.005,
    "T6_nocollapse_ratio": 0.9,
}


def gm_data_get(task_id, key):
    out = subprocess.run(
        ["gm", "task", "data", "get", "--task-id", task_id, "--data-key", key,
         "--sampling-mode", "accelerate", "--max-data-points", "10000"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"gm data get failed for {key}: {out.stderr[-300:]}")
    objs = []
    dec = json.JSONDecoder()
    s = out.stdout.strip()
    i = 0
    while i < len(s):
        o, j = dec.raw_decode(s, i)
        objs.append(o)
        i = j
        while i < len(s) and s[i] in " \n\r\t":
            i += 1
    for o in objs:
        d = o.get("data") if isinstance(o, dict) else None
        if isinstance(d, dict) and key in d:
            inner = d[key]
            vals = inner.get("value") or inner.get("values") or []
            return [float(v) for v in vals]
    raise RuntimeError(f"no data for key {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    series = {k: gm_data_get(args.task_id, k) for k in KEYS}
    n = len(series["Train/mean_reward"])
    print(f"task {args.task_id}: {n} iterations recorded")

    results = {}
    mr = series["Train/mean_reward"]
    last = mr[-args.window:]
    results["T1_mean_reward_ge_200"] = (statistics.mean(last), statistics.mean(last) >= CRITERIA["T1_mean_reward_ge"])
    el = series["Train/mean_episode_length"][-args.window:]
    results["T2_mean_ep_len_ge_2100"] = (statistics.mean(el), statistics.mean(el) >= CRITERIA["T2_mean_ep_len_ge"])
    tl = series["Episode/rew_tracking_lin_vel"][-args.window:]
    results["T3_track_lin_ge_1.20"] = (statistics.mean(tl), statistics.mean(tl) >= CRITERIA["T3_track_lin_ge"])
    rj = series["Episode/rew_ref_joint_pos"][-args.window:]
    results["T4_ref_joint_ge_1.40"] = (statistics.mean(rj), statistics.mean(rj) >= CRITERIA["T4_ref_joint_ge"])
    co = series["Episode/rew_collision"][-args.window:]
    results["T5_collision_ge_-0.005"] = (statistics.mean(co), statistics.mean(co) >= CRITERIA["T5_collision_ge"])
    mx = max(mr)
    results["T6_no_collapse_last100_ge_0.9x_max"] = (statistics.mean(last) / mx if mx else 0,
                                                     statistics.mean(last) >= CRITERIA["T6_nocollapse_ratio"] * mx)

    print("\n===== TRAINING CRITERIA (last-{} mean) =====".format(args.window))
    allpass = True
    for k, (val, ok) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k:42s} = {val:.4f}")
        allpass &= ok
    print(f"  OVERALL: {'PASS' if allpass else 'FAIL'}")
    sys.exit(0 if allpass else 3)


if __name__ == "__main__":
    main()
