# Copyright (c) 2024, AgiBot Inc. All rights reserved.
"""Environment probe for gradmotion IsaacLab tasks (one-shot, no Isaac import).

Answers, from inside the running task container:
  1. clone dir name / workspace layout (mainCodeUri base for startScript)
  2. where a resume checkpoint (checkPointFilePath) actually lands on disk
  3. python / torch / isaaclab / isaacsim versions + GPU
  4. writable paths for the SDK scan range (logs/{exp}/)

Usage: gm-run <repo>/humanoid_lab/scripts/probe_env.py
"""

import glob
import json
import os
import platform
import subprocess
import sys

OUT_DIR = os.path.join(os.getcwd(), "logs", "x1_dh_stand", "probe")
os.makedirs(OUT_DIR, exist_ok=True)

report = {}


def scan(label, path, depth=2):
    """Best-effort listing of a directory up to depth."""
    try:
        entries = []
        for root, dirs, files in os.walk(path):
            lvl = root[len(path):].count(os.sep)
            if lvl >= depth:
                dirs[:] = []
                continue
            for d in dirs:
                entries.append(os.path.join(root, d))
            for f in files:
                entries.append(os.path.join(root, f))
            if len(entries) > 400:
                break
        report[label] = entries[:400]
    except Exception as e:  # noqa: BLE001
        report[label] = f"ERR {e}"


print("=== probe_env ===")
report["cwd"] = os.getcwd()
report["argv"] = sys.argv
report["hostname"] = platform.node()
report["python"] = sys.version
report["uid"] = os.getuid()

scan("workspace_root", "/workspace", depth=3)
scan("root_dirs", "/", depth=1)

# mounted checkpoints anywhere plausible
report["ckpt_workspace"] = sorted(
    glob.glob("/workspace/**/model_*.pt", recursive=True)
)
report["ckpt_root_glob"] = sorted(glob.glob("/**/*.pt", recursive=False)) + sorted(
    glob.glob("/model_*.pt")
) + sorted(glob.glob("/personal/**/*.pt", recursive=True))
report["ckpt_cwd"] = sorted(glob.glob(os.path.join(os.getcwd(), "**", "*.pt"), recursive=True))
report["env_personal"] = os.environ.get("PERSONAL_DATA_PATH", None)
report["env_mount_hints"] = {
    k: v for k, v in os.environ.items()
    if any(s in k.upper() for s in ("CHECK", "MOUNT", "MODEL", "PERSONAL", "CODE", "REPO"))
}

try:
    import torch  # noqa: E402

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
except Exception as e:  # noqa: BLE001
    report["torch"] = f"ERR {e}"

for pkg in ("isaaclab", "isaacsim", "isaaclab_sim", "omni.isaac.lab"):
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True, timeout=60,
        )
        first = [l for l in out.stdout.splitlines() if l.startswith(("Name", "Version"))]
        if first:
            report.setdefault("pip", {})[pkg] = " ".join(first)
    except Exception as e:  # noqa: BLE001
        report.setdefault("pip", {})[pkg] = f"ERR {e}"

out_path = os.path.join(OUT_DIR, "probe_report.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=1, default=str)
print(f"[probe_env] report written: {out_path}")
print(json.dumps({k: (v if not isinstance(v, list) else v[:12]) for k, v in report.items()}, indent=1, default=str))
