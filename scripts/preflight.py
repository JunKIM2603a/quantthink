#!/usr/bin/env python3
"""Collect local diagnostics; never download a model or certify research gates."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "accelerate", "vllm", "datasets", "safetensors", "numpy")
TORCH_PROBE = r'''
import json
import torch
out = {"version": torch.__version__, "cuda_build": torch.version.cuda,
       "cuda_available": torch.cuda.is_available(), "devices": []}
if out["cuda_available"]:
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            p = torch.cuda.get_device_properties(i)
            out["devices"].append({"index": i, "name": p.name,
                "memory_total_bytes": p.total_memory,
                "compute_capability": [p.major, p.minor],
                "bf16_supported": bool(torch.cuda.is_bf16_supported())})
print(json.dumps(out))
'''


def run_command(args: list[str], timeout: int = 15) -> dict[str, Any]:
    """Use argv, not a shell; return diagnostic failures rather than hiding them."""
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, check=False)
        # Do not publish stderr, which may contain private paths or environment values.
        return {"returncode": p.returncode, "stdout": p.stdout.strip(),
                "stderr_present": bool(p.stderr.strip())}
    except FileNotFoundError:
        return {"returncode": None, "error": "command_not_found"}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "error": "timeout"}
    except OSError as exc:
        return {"returncode": None, "error": type(exc).__name__}


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) != 4:
            raise ValueError("Unexpected nvidia-smi column count")
        rows.append({"index": int(row[0]), "name": row[1].strip(),
                     "memory_total_mib": float(row[2]), "driver_version": row[3].strip()})
    return rows


def assess_torch(probe: dict[str, Any], min_gpus: int) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if probe.get("status") != "ok":
        problems.append("torch_probe_not_successful")
    if probe.get("cuda_available") is not True:
        problems.append("torch_cuda_unavailable")
    if len(probe.get("devices", [])) < min_gpus:
        problems.append(f"fewer_than_{min_gpus}_visible_cuda_devices")
    return not problems, problems


def collect(skip_torch: bool = False, min_gpus: int = 2) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    smi = run_command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                       "--format=csv,noheader,nounits"])
    gpu_info: dict[str, Any] = {"status": "unavailable", "devices": []}
    if smi.get("returncode") == 0:
        try:
            gpu_info = {"status": "ok", "devices": parse_gpu_csv(smi["stdout"])}
        except (ValueError, csv.Error):
            gpu_info["status"] = "parse_error"
    else:
        gpu_info["error"] = smi.get("error", "nonzero_exit")
    torch_info: dict[str, Any] = {"status": "skipped" if skip_torch else "not_installed"}
    if not skip_torch and packages["torch"] is not None:
        probe = run_command([sys.executable, "-c", TORCH_PROBE], timeout=45)
        if probe.get("returncode") == 0:
            try:
                parsed = json.loads(probe["stdout"])
                if not isinstance(parsed, dict):
                    raise ValueError("Expected object")
                torch_info = {**parsed, "status": "ok"}
            except (ValueError, TypeError):
                torch_info = {"status": "invalid_json"}
        else:
            torch_info = {"status": "probe_failed", "error": probe.get("error", "nonzero_exit")}
    sha = run_command(["git", "rev-parse", "HEAD"])
    worktree = run_command(["git", "status", "--porcelain"])
    ready, problems = assess_torch(torch_info, min_gpus)
    try:
        ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        ram_bytes = None
    return {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "os": platform.system(),
        "os_release": platform.release(), "machine": platform.machine(),
        "cpu_logical_count": os.cpu_count(), "ram_total_bytes": ram_bytes,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "git_commit": sha.get("stdout") if sha.get("returncode") == 0 else None,
        "git_dirty": bool(worktree.get("stdout")) if worktree.get("returncode") == 0 else None,
        "packages": packages, "nvidia_smi": gpu_info, "torch_probe": torch_info,
        "diagnostic_status": "CUDA_VISIBLE" if ready else "ENVIRONMENT_NOT_READY",
        "problems": problems,
        "scope": "Local environment inspection only; no model loaded and no research gate passed."
    }


def write_report(path: Path, report: dict[str, Any], overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/preflight.json")
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_gpus < 1:
        parser.error("--min-gpus must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; choose another filename or use --overwrite")
    report = collect(args.skip_torch, args.min_gpus)
    try:
        write_report(args.output, report, args.overwrite)
    except OSError as exc:
        print(f"Could not write report: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"diagnostic_status": report["diagnostic_status"],
                      "problems": report["problems"], "scope": report["scope"]}, indent=2))
    return 2 if args.require_cuda and report["diagnostic_status"] != "CUDA_VISIBLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
