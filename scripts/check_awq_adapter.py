"""무작위 소형 Qwen2로 원본 AWQ 함수 연결을 점검합니다. 사전학습 모델·데이터 다운로드 없음."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from runtime_assets import canonical_hash, implementation_fingerprint, load_contracts, package_versions, write_new
from qwen2_awq_adapter import load_upstream, run_awq


def run_check(upstream_dir, device, policy):
    versions = package_versions(["torch", "transformers", "numpy", "tqdm"])
    if versions["transformers"] != policy["runtime"]["transformers"]:
        raise ValueError("Transformers 4.51.3 환경에서 실행하세요.")
    if versions["torch"] != policy["runtime"]["torch_observed"]:
        raise ValueError("검토한 PyTorch 2.7.1+cu118 환경과 다릅니다. 변경 환경은 먼저 검토해야 합니다.")
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise ValueError("지정한 CUDA 장치를 사용할 수 없습니다.")
    upstream = load_upstream(upstream_dir, policy)
    cfg = policy["synthetic_check"]
    torch.manual_seed(cfg["seed"])
    config = Qwen2Config(vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"],
                        intermediate_size=cfg["intermediate_size"],
                        num_hidden_layers=cfg["num_hidden_layers"],
                        num_attention_heads=cfg["num_attention_heads"],
                        num_key_value_heads=cfg["num_key_value_heads"],
                        max_position_embeddings=cfg["sequence_length"],
                        attention_dropout=0.0, use_cache=False, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    blocks = torch.randint(0, cfg["vocab_size"], (1, cfg["sequence_length"])).tolist()
    started = time.monotonic()
    report = run_awq(model, blocks, upstream, device=device,
                     expected_shape=(1, cfg["sequence_length"]))
    limit = cfg["max_relative_rms"]
    diagnostics = [row[key]["relative_rms"] for row in report["layers"]
                   for key in ("scale_only_output_difference", "canonicalized_output_difference")]
    if not diagnostics or any(x is None or x > limit for x in diagnostics):
        raise ValueError("합성 모델의 스케일·역변환 출력 오차가 사전 명시한 점검 한도를 넘었습니다.")
    if not any(p["rms_error"] > 0 for row in report["layers"] for p in row["canonical_weight_error"]):
        raise ValueError("W3 변환 후 가중치 변화가 관찰되지 않았습니다.")
    return {"schema_version": 1, "status": "SYNTHETIC_AWQ_ADAPTER_PASS_NOT_MODEL_READY",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_sha256": canonical_hash(policy), "packages": versions,
            "implementation_sha256": implementation_fingerprint([
                "check_awq_adapter.py", "qwen2_awq_adapter.py", "runtime_assets.py", "awq_weight_space.py"]),
            "device": device, "elapsed_seconds": time.monotonic() - started,
            "pretrained_weights": False, "benchmark_data": False,
            "synthetic_contract": cfg, "source_files": upstream["source_files"], "awq": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("출력 파일이 이미 있습니다. 덮어쓰지 않습니다.")
    try:
        _, _, policy = load_contracts()
        report = run_check(args.upstream_dir, args.device, policy)
        write_new(args.output, report)
        print(json.dumps({"status": report["status"], "packages": report["packages"],
                          "policy_sha256": report["policy_sha256"], "device": report["device"],
                          "implementation_sha256": report["implementation_sha256"],
                          "elapsed_seconds": report["elapsed_seconds"],
                          "layer_diagnostics": [{key: row[key] for key in
                              ("layer", "scale_only_output_difference", "canonicalized_output_difference")}
                              for row in report["awq"]["layers"]]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        parser.exit(2, f"합성 어댑터 점검 실패: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
