"""R0 실행기 후보. 기본값은 승인 상태 점검이며 --execute 전에 연구 기록을 검사합니다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

from runtime_assets import ROOT, canonical_hash, load_contracts, package_versions, write_new
from runtime_contracts import generate_r0, load_prepared_calibration, require_gates, unmet_gates
from prepare_runtime_assets import load_tokenizer
from qwen2_awq_adapter import load_upstream, run_awq


def repository_state():
    def git(*args):
        value = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
        if value.returncode:
            raise ValueError("R0는 커밋된 Git 작업 트리에서 실행해야 합니다.")
        return value.stdout.strip()
    state = {"commit": git("rev-parse", "HEAD"), "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    if state["tracked_dirty"]:
        raise ValueError("추적 파일에 커밋되지 않은 변경이 있습니다.")
    return state


def run(output, assets, upstream_dir, device, candidate, refs, policy, status):
    require_gates(status)  # 네트워크·가중치 로딩보다 먼저 검사합니다.
    repo = repository_state()
    versions = package_versions(["torch", "transformers", "huggingface-hub", "numpy"])
    if (versions["torch"] != policy["runtime"]["torch_observed"]
            or versions["transformers"] != policy["runtime"]["transformers"]
            or versions["huggingface-hub"] != "0.36.2"):
        raise ValueError("검토한 런타임 버전과 다릅니다.")
    blocks, prepared = load_prepared_calibration(assets, candidate, refs, policy)
    upstream = load_upstream(upstream_dir, policy)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "R0_STARTED",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "repository": repo, "packages": versions, "python": platform.python_version(),
              "candidate_sha256": canonical_hash(candidate), "policy_sha256": canonical_hash(policy),
              "preparation_report_sha256": canonical_hash(prepared),
              "research_status_sha256": canonical_hash(status),
              "scope": "두 합성 프롬프트의 BF16/AWQ R0. H1·MATH-500·장문맥 재현 결과 아님.",
              "attempts": [], "completed_awq_layers": 0}
    def checkpoint():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "run.json")
    checkpoint()
    try:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise ValueError("CUDA 장치가 필요합니다.")
        tokenizer, tokenizer_evidence = load_tokenizer(refs)
        asset = next(x for x in refs["assets"] if x["role"] == "bf16_reference")
        model = AutoModelForCausalLM.from_pretrained(
            asset["repo_id"], revision=asset["revision"], token=False, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        report.update(model_reference=asset, tokenizer=tokenizer_evidence)
        for item in generate_r0(model, tokenizer, candidate, arm="bf16", device=device):
            report["attempts"].append(item)
            checkpoint()
            if item["finish_reason"] in ("error", "unexpected_stop"):
                raise ValueError("BF16 R0에 실행·종료 오류가 있어 AWQ 탐색을 시작하지 않습니다.")
        def on_layer(row):
            write_new(output / f"awq_layer_{row['layer']:03d}.json", row)
            report["completed_awq_layers"] += 1
            checkpoint()
        awq = run_awq(model, blocks, upstream, device=device, on_layer=on_layer)
        report["awq_status"] = awq["status"]
        report["awq_layer_diagnostics"] = [
            {key: row[key] for key in ("layer", "scale_only_output_difference", "canonicalized_output_difference")}
            for row in awq["layers"]]
        for item in generate_r0(model, tokenizer, candidate, arm="awq_w3", device=device):
            report["attempts"].append(item)
            checkpoint()
        if any(x["finish_reason"] in ("error", "unexpected_stop") for x in report["attempts"]):
            raise ValueError("AWQ R0에 실행·종료 오류가 있습니다.")
        report["status"] = "R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW"
        report["model_ready"] = False
        checkpoint()
        return report
    except Exception as exc:
        report.update(status="R0_ENGINEERING_FAILURE", error_type=type(exc).__name__, error=str(exc))
        checkpoint()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        candidate, refs, policy = load_contracts()
        status = json.loads((ROOT / "docs/research_status.json").read_text())
        if not args.execute:
            print(json.dumps({"status": "R0_INSPECTION_ONLY", "unmet_gates": unmet_gates(status),
                              "model_loaded": False, "network_used": False}, ensure_ascii=False, indent=2))
            return 0
        require_gates(status)
        if args.assets_dir is None or args.upstream_dir is None or args.output_dir is None:
            parser.error("실행에는 자산·원본 코드·출력 디렉터리가 필요합니다.")
        result = run(args.output_dir, args.assets_dir, args.upstream_dir, args.device,
                     candidate, refs, policy, status)
        print(result["status"])
        return 0
    except Exception as exc:
        parser.exit(2, f"R0 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

