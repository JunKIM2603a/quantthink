"""종료 직전 고정 prefix 6회 진단 후보. 기본은 읽기 전용 계획, 별도 승인 전 GPU 실행 차단."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash
from fixed_prefix_diagnostic_contracts import CONFIG as BASE_CONFIG, compare_logits
from run_fixed_prefix_diagnostic import IMPLEMENTATION as BASE_IMPLEMENTATION, verify_sources

CONFIG = ROOT / "configs/termination_prefix_v01.json"
IMPLEMENTATION = ["run_termination_prefix_diagnostic.py", "audit_fixed_prefix_logits.py", *BASE_IMPLEMENTATION]


def require_authorization(status, cfg):
    scope = status.get("termination_prefix_diagnostic", {})
    if (status.get("research_approval") != "APPROVED"
            or status.get("novelty_gate") != "SCOPED_CONTRIBUTION_ACCEPTED"
            or status.get("protocol_gate") != "ACCEPTED"
            or scope.get("authorization") != "APPROVED"
            or scope.get("config_sha256") != canonical_hash(cfg)
            or scope.get("remaining_forward_calls") != 6):
        raise ValueError("종료 prefix 추가6회는 별도 승인 전입니다. 완료한 기존12회 승인을 재사용하지 않습니다.")
    hashes = {name: file_hash(ROOT / "scripts" / name) for name in IMPLEMENTATION}
    if hashes != scope.get("implementation_sha256"):
        raise ValueError("새 승인에 연결된 코드 해시와 다릅니다.")
    return hashes


def read_inputs(evidence, old_run_path, audit_path, cfg, base):
    for path, key in [(evidence, "evidence_packet_sha256"),
                      (old_run_path, "source_fixed_prefix_run_sha256"),
                      (audit_path, "source_logits_audit_sha256")]:
        if file_hash(path) != cfg[key]:
            raise ValueError(f"검토된 근거 바이트 해시 불일치: {key}")
    if canonical_hash(base) != cfg["base_fixed_prefix_config_sha256"]:
        raise ValueError("기존 고정 runtime 설정 참조 불일치")
    expected = {"states": ["B", "B_repeat", "Q"], "comparisons": [["B", "B_repeat"], ["B", "Q"]],
                "maximum_forward_calls": 6, "maximum_forward_input_tokens": 14970,
                "maximum_sequence_length": 2495, "logit_positions_per_input": 129,
                "new_generation_tokens": 0, "maximum_raw_logit_bytes": 470393856,
                "use_cache": False, "effective_sliding_window": None,
                "calibration_search": False, "network_download": False, "automatic_retry": False,
                "temperature": 0.6, "top_p": 0.95}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("고정6회 범위·예산·필터 설정이 변경됐습니다.")
    packet = json.loads(Path(evidence).read_text(encoding="utf-8"))
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    old = json.loads(Path(old_run_path).read_text(encoding="utf-8"))
    if (audit["status"] != "SAVED_LOGITS_AUDITED_FLOAT64_FILTER_DIAGNOSTICS"
            or audit["source_run_sha256"] != cfg["source_fixed_prefix_run_sha256"]
            or not audit["t1_recalculation"]["all_match"]
            or old["config_sha256"] != canonical_hash(base)):
        raise ValueError("앞선12회와 완료한 CPU 감사의 연결 불일치")
    source = packet["cached_model_config"]["configuration"]
    effective = copy.deepcopy(source)
    effective["sliding_window"] = None
    if (canonical_hash(source) != base["model_configuration_sha256"]
            or canonical_hash(effective) != base["effective_source_configuration_sha256"]):
        raise ValueError("원본/진단용 모델 설정 불일치")
    refs = cfg["inputs"]
    if [(r["prefix_id"], r["problem_id"], r["arm"], r["seed"]) for r in refs] != [
            ("D2-03_B_pre_eos", "D2-03", "bf16", 42),
            ("D2-03_Q_same_length", "D2-03", "awq_w3_replay", 42)]:
        raise ValueError("BF16 종료 직전·같은 길이 AWQ 경로 두 개가 필요합니다.")
    inputs = []
    for ref in refs:
        rows = [r for r in packet["run_snapshot"]["attempts"]
                if all(r[k] == ref[k] for k in ("problem_id", "arm", "seed"))]
        if len(rows) != 1 or canonical_hash(rows[0]) != ref["attempt_sha256"]:
            raise ValueError("기존 응답의 누락·중복·변경")
        row = rows[0]
        ids = row["input_ids"] + row["generated_ids"][:2424]
        positions = list(range(2366, 2495))
        if (len(row["input_ids"]) != ref["input_token_count"]
                or ref["input_token_count"] != 71 or ref["generated_prefix_tokens"] != 2424
                or len(ids) != ref["sequence_length"] or ref["sequence_length"] != 2495
                or canonical_hash(ids) != ref["sequence_ids_sha256"]
                or positions != ref["logit_positions"]
                or any(type(x) is not int or not 0 <= x < base["vocab_size"] for x in ids)
                or base["eos_token_id"] in row["generated_ids"][:2424]):
            raise ValueError("EOS를 미리 포함했거나 고정 prefix/위치가 변경됐습니다.")
        eos_next = row["generated_ids"][2424] == base["eos_token_id"]
        if eos_next != ref["source_next_token_is_eos"] or eos_next != (ref["arm"] == "bf16"):
            raise ValueError("종료 anchor의 원본 다음 토큰 불일치")
        if ref["arm"] == "bf16" and (len(row["generated_ids"]) != 2425
                or row["finish_reason"] != "eos" or row["right_censored"]):
            raise ValueError("원본 BF16 자연 EOS anchor 불일치")
        if ref["arm"] == "awq_w3_replay" and (len(row["generated_ids"]) != 4096
                or row["finish_reason"] != "length" or not row["right_censored"]):
            raise ValueError("원본 AWQ 검열 경로 불일치")
        inputs.append({**ref, "ids": ids})
    return inputs, source, effective


def execute(args, cfg, base, status, inputs, source, effective):
    implementation = require_authorization(status, cfg)  # heavy import·출력 생성 전에 검사
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("기존 종료 진단 폴더를 보존합니다. 자동 재시도/재개 없음.")
    from run_difficulty_pilot import checkpoint, parameter_hash, repo_state
    from runtime_assets import load_contracts, package_versions, verify_upstream
    from replay_r0_awq import replay, verify_r0_files
    repository = repo_state()
    versions = package_versions(["torch", "transformers", "huggingface-hub", "numpy"])
    if versions != base["packages"]:
        raise ValueError("고정 패키지와 다릅니다. 자동 설치/업데이트 없음.")
    candidate, refs, policy = load_contracts()
    r0_review = json.loads((ROOT / base["r0_review"]).read_text())
    old, r0_files = verify_r0_files(args.r0_dir, r0_review)
    if old["candidate_sha256"] != canonical_hash(candidate) or old["policy_sha256"] != canonical_hash(policy):
        raise ValueError("R0 후보/정책 참조 불일치")
    upstream_files = verify_upstream(args.upstream_dir, policy)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config
    from huggingface_hub import hf_hub_download
    from qwen2_awq_adapter import load_upstream
    from audit_fixed_prefix_logits import compare_filtered
    installed = verify_sources(base)
    asset = next(x for x in refs["assets"] if x["role"] == "bf16_reference")
    if asset != base["model_reference"]:
        raise ValueError("고정 모델 revision 불일치")
    path = hf_hub_download(asset["repo_id"], "config.json", revision=asset["revision"],
                           token=False, local_files_only=True)
    if canonical_hash(json.loads(Path(path).read_text())) != canonical_hash(source):
        raise ValueError("기존 캐시 모델 설정 불일치")
    device = torch.device(base["device"])
    if not torch.cuda.is_available():
        raise ValueError("기존 CUDA 환경이 필요합니다.")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("기존 BF16 환경이 필요합니다.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "TERMINATION_PREFIX_STARTED", "model_ready": False,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "repository": repository,
              "config_sha256": canonical_hash(cfg), "research_status_sha256": canonical_hash(status),
              "implementation_sha256": implementation, "packages": versions, "inputs": cfg["inputs"],
              "source_fixed_prefix_run_sha256": cfg["source_fixed_prefix_run_sha256"],
              "source_logits_audit_sha256": cfg["source_logits_audit_sha256"],
              "evidence_packet_sha256": cfg["evidence_packet_sha256"],
              "source_configuration_sha256": canonical_hash(source),
              "effective_source_configuration_sha256": canonical_hash(effective),
              "installed_sources": installed, "r0_input_files": r0_files,
              "upstream_source_files": upstream_files, "parameter_sha256": {},
              "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                          "dtype": "bfloat16", "attention_implementation": "sdpa", "use_cache": False,
                          "sliding_window": None, "tf32_matmul": False, "tf32_cudnn": False,
                          "float32_matmul_precision": "highest", "deterministic_algorithms": False},
              "filtered_analysis": {"temperature": .6, "top_p": .95, "dtype": "float64",
                                    "tie_order": "ascending_token_id", "actual_gpu_sampling_replay": False},
              "new_generation_tokens": 0, "calibration_search": False, "forward_records": [],
              "comparisons": [], "functional_equivalence_verdict": "NOT_CERTIFIED",
              "actual_generation_cache_or_mask_impact_tested": False}
    save = lambda: checkpoint(args.output_dir / "run.json", report)
    save()

    def measure(model, state):
        model.cpu()
        sha = parameter_hash(model)
        expected = base["parameter_sha256"]["awq_w3_replay" if state == "Q" else "bf16"]
        if sha != expected:
            raise ValueError(f"기존 파라미터 해시와 다름: {state}. 해당 forward 전에 중단.")
        report["parameter_sha256"][state] = sha
        model.to(device).eval()
        for row in inputs:
            if len(report["forward_records"]) >= 6:
                raise ValueError("6회 forward 상한 초과")
            print(f"종료 prefix {len(report['forward_records']) + 1}/6: {state} {row['prefix_id']}", flush=True)
            ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
            pos = torch.arange(ids.shape[1], device=device)
            keep = torch.tensor(row["logit_positions"], dtype=torch.long, device=device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos[None, :],
                            cache_position=pos, use_cache=False, output_attentions=False,
                            output_hidden_states=False, logits_to_keep=keep, return_dict=True)
            torch.cuda.synchronize(device)
            if out.past_key_values is not None:
                raise ValueError("예상하지 않은 cache 반환")
            array = out.logits[0].float().cpu().numpy().copy()
            if array.shape != (129, base["vocab_size"]) or not np.isfinite(array).all():
                raise ValueError("logit shape/유한성 불일치")
            filename = f"logits_{state}_{row['prefix_id']}.npy"
            path = args.output_dir / filename
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            report["forward_records"].append({"state": state, "prefix_id": row["prefix_id"],
                "input_ids_sha256": row["sequence_ids_sha256"], "sequence_length": len(row["ids"]),
                "logit_positions": row["logit_positions"], "file": filename, "file_sha256": file_hash(path),
                "shape": list(array.shape), "stored_dtype": str(array.dtype),
                "elapsed_seconds_including_save": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            save()
            del out, array, ids, pos, keep

    def compare_pair(a, b):
        for inp in inputs:
            def load(state):
                rec = next(r for r in report["forward_records"]
                           if (r["state"], r["prefix_id"]) == (state, inp["prefix_id"]))
                path = args.output_dir / rec["file"]
                if file_hash(path) != rec["file_sha256"]:
                    raise ValueError("저장 직후 logit 해시 변경")
                return np.load(path, mmap_mode="r", allow_pickle=False)
            x, y = load(a), load(b)
            t1 = compare_logits(x, y, inp["logit_positions"], base["eos_token_id"])
            filtered = [{"position": pos, **compare_filtered(x[i], y[i], base["eos_token_id"])}
                        for i, pos in enumerate(inp["logit_positions"])]
            report["comparisons"].append({"reference": a, "observed": b, "prefix_id": inp["prefix_id"],
                "t1": t1, "filtered_per_position": filtered,
                "anchor_eos_retained_reference": filtered[-1]["reference"]["eos_retained"],
                "anchor_eos_retained_observed": filtered[-1]["observed"]["eos_retained"]})
            save()

    try:
        model = AutoModelForCausalLM.from_pretrained(asset["repo_id"], revision=asset["revision"],
            token=False, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            config=Qwen2Config.from_dict(copy.deepcopy(effective)), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").eval()
        if (model.__class__.__name__ != "Qwen2ForCausalLM" or len(model.model.layers) != 28
                or model.config.sliding_window is not None or model.config.use_sliding_window is not False
                or model.config._attn_implementation != "sdpa"
                or any(p.dtype != torch.bfloat16 for p in model.parameters())):
            raise ValueError("고정 모델·BF16·SDPA·문맥 정책 불일치")
        report["resolved_model_configuration_sha256"] = canonical_hash(model.config.to_dict())
        prior = json.loads(args.source_run.read_text())
        if report["resolved_model_configuration_sha256"] != prior["resolved_model_configuration_sha256"]:
            raise ValueError("앞선 진단과 실제 model.config 해시가 다릅니다.")
        measure(model, "B")
        measure(model, "B_repeat")
        compare_pair("B", "B_repeat")
        if not all(c["t1"]["exact_equal"] for c in report["comparisons"]):
            report["status"] = "TERMINATION_PREFIX_BASELINE_REPEAT_DIFFERENCE"
            save()
            return report
        verify_r0_files(args.r0_dir, r0_review)
        upstream = load_upstream(args.upstream_dir, policy)
        report["awq_replay"] = replay(model, args.r0_dir, upstream, device,
            on_layer=lambda i: print(f"저장 recipe 재적용 {i + 1}/28", flush=True))
        measure(model, "Q")
        compare_pair("B", "Q")
        if len(report["forward_records"]) != 6 or len(report["comparisons"]) != 4:
            raise ValueError("6회·4비교 수집 누락")
        report["status"] = "TERMINATION_PREFIX_COLLECTED_PENDING_REVIEW"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return report
    except BaseException as exc:
        report.update(status="TERMINATION_PREFIX_INTERRUPTED" if isinstance(exc, KeyboardInterrupt)
                      else "TERMINATION_PREFIX_FAILED", error_type=type(exc).__name__, error=str(exc)[:1000])
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "results/local/difficulty_v01/review_evidence_v01.json")
    parser.add_argument("--source-run", type=Path, default=ROOT / "results/local/fixed_prefix_v01/run.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/local/fixed_prefix_v01/logits_audit_v01.json")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "results/local/r0_v03")
    parser.add_argument("--upstream-dir", type=Path, default=ROOT / "results/local/awq_reference_v03")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local/termination_prefix_v01")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="기존 파일만 읽는 계획 조회; 기본 동작")
    mode.add_argument("--execute", action="store_true", help="이 추가6회 범위의 별도 승인 뒤에만 GPU 실행")
    args = parser.parse_args(argv)
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        status = json.loads((ROOT / "docs/research_status.json").read_text(encoding="utf-8"))
        if args.execute:
            require_authorization(status, cfg)
        inputs, source, effective = read_inputs(args.evidence, args.source_run, args.audit, cfg, base)
        if not args.execute:
            print(json.dumps({"status": "TERMINATION_PREFIX_PLAN_ONLY_NO_MODEL_EXECUTION",
                              "authorization": status.get("termination_prefix_diagnostic", {}).get("authorization", "PENDING"),
                              "config_sha256": canonical_hash(cfg),
                              "inputs": [{k: v for k, v in r.items() if k not in {"ids", "logit_positions"}}
                                         for r in inputs],
                              **{k: cfg[k] for k in ("states", "maximum_forward_calls", "maximum_forward_input_tokens",
                                  "maximum_sequence_length", "new_generation_tokens", "maximum_raw_logit_bytes")}},
                             ensure_ascii=False, indent=2))
            return 0
        result = execute(args, cfg, base, status, inputs, source, effective)
        print(result["status"])
        print(f"검토할 파일: {args.output_dir / 'run.json'}")
        return 0
    except Exception as exc:
        parser.exit(2, f"종료 prefix 진단 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
