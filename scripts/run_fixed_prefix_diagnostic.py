"""고정 prefix 좌표·logit 진단 후보. 기본은 계획 조회, 승인 기록과 --execute가 있어야 GPU 실행."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash
from fixed_prefix_diagnostic_contracts import (
    CONFIG, PAIRS, compare_logits, difference_stats, read_inputs, require_authorization,
)

IMPLEMENTATION = [
    "run_fixed_prefix_diagnostic.py", "fixed_prefix_diagnostic_contracts.py",
    "replay_r0_awq.py", "qwen2_awq_adapter.py", "awq_weight_space.py",
    "runtime_assets.py", "run_difficulty_pilot.py", "difficulty_pilot_contracts.py",
    "runtime_contracts.py", "reproduction_contracts.py",
]


def verify_sources(cfg):
    from transformers import modeling_attn_mask_utils
    from transformers.integrations import sdpa_attention
    from transformers.models.qwen2 import configuration_qwen2, modeling_qwen2
    proofs = []
    for mod in (configuration_qwen2, modeling_qwen2, modeling_attn_mask_utils, sdpa_attention):
        path = Path(mod.__file__)
        raw = path.read_bytes()
        proof = {"file": path.name, "git_blob_sha1": hashlib.sha1(
            f"blob {len(raw)}\0".encode() + raw).hexdigest(), "sha256": hashlib.sha256(raw).hexdigest()}
        if proof not in cfg["transformers_source_files"]:
            raise ValueError(f"검토한 설치 코드와 다릅니다: {path.name}")
        proofs.append(proof)
    return proofs


def scale_only(model, folder, upstream, device):
    import torch
    from replay_r0_awq import validate_recipe
    model.cpu()
    with torch.no_grad(), torch.cuda.device(device):
        for i, layer in enumerate(model.model.layers):
            row = json.loads((folder / f"awq_layer_{i:03d}.json").read_text())
            validate_recipe(row, i, {n: list(p.shape) for n, p in layer.named_parameters()})
            layer.to(device)
            scales = [(s["previous"], tuple(s["following"]),
                       torch.tensor(s["scales"], dtype=torch.bfloat16, device=device)) for s in row["scales"]]
            upstream["apply_scale"](layer, scales)
            if any(not torch.isfinite(p).all().item() for p in layer.parameters()):
                raise ValueError("S 상태 파라미터가 비유한 값입니다.")
            layer.cpu()
            del scales
            print(f"S 저장 scale 적용 {i + 1}/28", flush=True)
    torch.cuda.empty_cache()


def canonicalize(model, folder, original_aux, device):
    import torch
    from awq_weight_space import apply_scale_ledger
    from qwen2_awq_adapter import parameter_arrays
    model.cpu()
    rows = []
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            layer.to(device)
            ledger = json.loads((folder / f"awq_layer_{i:03d}.json").read_text())["scales"]
            restored = apply_scale_ledger(parameter_arrays(layer), ledger, inverse=True)
            for name, param in layer.named_parameters():
                param.copy_(torch.as_tensor(restored[name], dtype=param.dtype, device=device))
                if not torch.isfinite(param).all().item():
                    raise ValueError("C 상태 BF16 재적용 값이 비유한 값입니다.")
                if name in original_aux[i]:
                    base = original_aux[i][name].float().numpy()
                    applied = param.detach().float().cpu().numpy()
                    rows.append({"layer": i, "parameter": name,
                                 "inverse_float64_vs_B": difference_stats(base, restored[name]),
                                 "reapplied_bf16_vs_B": difference_stats(base, applied),
                                 "recast_only": difference_stats(restored[name], applied)})
            del restored
            layer.cpu()
            print(f"C 좌표 복원 {i + 1}/28", flush=True)
    torch.cuda.empty_cache()
    if len(rows) != 140:
        raise ValueError("norm/bias 진단 140개가 필요합니다.")
    return rows


def linear_hash(model):
    import torch
    from replay_r0_awq import LINEARS
    h = hashlib.sha256()
    for i, layer in enumerate(model.model.layers):
        for name in LINEARS:
            p = layer.get_submodule(name).weight.detach().cpu().contiguous()
            h.update(json.dumps([i, name, str(p.dtype), list(p.shape)]).encode() + b"\0")
            h.update(memoryview(p.view(torch.uint8).numpy()).cast("B"))
    return h.hexdigest()


def execute(args, cfg, status, inputs, source, effective):
    # 이 gate는 torch import·GPU 접근·출력 폴더 생성보다 먼저 확인합니다.
    require_authorization(status, cfg)
    from run_difficulty_pilot import checkpoint, parameter_hash, repo_state
    from runtime_assets import load_contracts, package_versions, verify_upstream
    from replay_r0_awq import LINEARS, replay, validate_recipe, verify_r0_files
    repository = repo_state()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("기존 진단 폴더를 보존합니다. 자동 재실행·재개하지 않습니다.")
    implementation = {n: file_hash(ROOT / "scripts" / n) for n in IMPLEMENTATION}
    if implementation != status["fixed_prefix_diagnostic"].get("implementation_sha256"):
        raise ValueError("승인에 연결된 실행 코드 해시와 다릅니다.")
    versions = package_versions(["torch", "transformers", "huggingface-hub", "numpy"])
    if versions != cfg["packages"]:
        raise ValueError(f"고정 환경과 다릅니다. 자동 설치/업데이트 없음: {versions}")
    candidate, refs, policy = load_contracts()
    r0_review = json.loads((ROOT / cfg["r0_review"]).read_text())
    old, r0_files = verify_r0_files(args.r0_dir, r0_review)
    if old["candidate_sha256"] != canonical_hash(candidate) or old["policy_sha256"] != canonical_hash(policy):
        raise ValueError("R0 후보/정책 불일치")
    upstream_files = verify_upstream(args.upstream_dir, policy)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, Qwen2Config
    from qwen2_awq_adapter import load_upstream
    installed_sources = verify_sources(cfg)
    asset = next(x for x in refs["assets"] if x["role"] == "bf16_reference")
    if asset != cfg["model_reference"]:
        raise ValueError("고정 모델 revision 불일치")
    config_file = hf_hub_download(asset["repo_id"], "config.json", revision=asset["revision"],
                                   token=False, local_files_only=True)
    if canonical_hash(json.loads(Path(config_file).read_text())) != canonical_hash(source):
        raise ValueError("기존 모델 캐시의 설정이 첨부와 다릅니다.")
    device = torch.device(cfg["device"])
    if not torch.cuda.is_available():
        raise ValueError("기존 CUDA 환경이 필요합니다.")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 지원이 필요합니다.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "FIXED_PREFIX_DIAGNOSTIC_STARTED", "model_ready": False,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "repository": repository,
              "config_sha256": canonical_hash(cfg), "research_status_sha256": canonical_hash(status),
              "implementation_sha256": implementation, "packages": versions,
              "evidence_packet_sha256": cfg["evidence_packet_sha256"],
              "mask_report_sha256": cfg["mask_report_sha256"],
              "source_configuration_sha256": canonical_hash(source),
              "effective_source_configuration_sha256": canonical_hash(effective),
              "config_change": {"sliding_window": {"original": 4096, "effective": None}},
              "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                          "dtype": "bfloat16", "attention_implementation": "sdpa", "use_cache": False,
                          "tf32_matmul": False, "tf32_cudnn": False, "float32_matmul_precision": "highest",
                          "deterministic_algorithms": False},
              "installed_sources": installed_sources, "r0_input_files": r0_files,
              "upstream_source_files": upstream_files, "inputs": cfg["inputs"],
              "parameter_sha256": {}, "forward_records": [], "comparisons": [],
              "new_generation_tokens": 0, "calibration_search": False,
              "actual_generation_mask_impact_tested": False,
              "functional_equivalence_verdict": "NOT_CERTIFIED"}
    save = lambda: checkpoint(args.output_dir / "run.json", report)
    save()

    def load_baseline():
        config = Qwen2Config.from_dict(copy.deepcopy(effective))
        model = AutoModelForCausalLM.from_pretrained(
            asset["repo_id"], revision=asset["revision"], token=False, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if (model.__class__.__name__ != "Qwen2ForCausalLM" or len(model.model.layers) != 28
                or model.config.sliding_window is not None or model.config.use_sliding_window is not False
                or model.config._attn_implementation != "sdpa"
                or any(p.dtype != torch.bfloat16 for p in model.parameters())):
            raise ValueError("고정 모델·dtype·문맥 정책 불일치")
        if parameter_hash(model) != cfg["parameter_sha256"]["bf16"]:
            raise ValueError("기존 난도 시험 BF16 파라미터 해시 불일치")
        report.setdefault("resolved_model_configuration_sha256", canonical_hash(model.config.to_dict()))
        if report["resolved_model_configuration_sha256"] != canonical_hash(model.config.to_dict()):
            raise ValueError("B 재로딩 설정 불일치")
        return model.eval()

    def measure(model, state):
        model.cpu()
        report["parameter_sha256"][state] = parameter_hash(model)
        if state in {"B", "B_repeat"} and report["parameter_sha256"][state] != cfg["parameter_sha256"]["bf16"]:
            raise ValueError("BF16 반복 중 파라미터 해시 변경")
        if state == "Q" and report["parameter_sha256"][state] != cfg["parameter_sha256"]["awq_w3_replay"]:
            raise ValueError("기존 난도 시험 AWQ 파라미터 해시 불일치. Q forward 전 중단.")
        model.to(device).eval()
        save()
        for row in inputs:
            if len(report["forward_records"]) >= cfg["maximum_forward_calls"]:
                raise ValueError("forward 예산 초과")
            print(f"forward {len(report['forward_records']) + 1}/12: {state} {row['problem_id']}", flush=True)
            ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
            positions = torch.arange(ids.shape[1], device=device)
            keep = torch.tensor(row["logit_positions"], dtype=torch.long, device=device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            with torch.inference_mode():
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               position_ids=positions[None, :], cache_position=positions,
                               use_cache=False, output_attentions=False, output_hidden_states=False,
                               logits_to_keep=keep, return_dict=True)
            torch.cuda.synchronize(device)
            if result.past_key_values is not None:
                raise ValueError("예상하지 않은 cache 반환")
            array = result.logits[0].float().cpu().numpy().copy()
            if array.shape != (129, cfg["vocab_size"]) or not np.isfinite(array).all():
                raise ValueError("logit 출력 shape·유한성 오류")
            path = args.output_dir / f"logits_{state}_{row['problem_id']}.npy"
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            report["forward_records"].append({"state": state, "problem_id": row["problem_id"],
                "input_ids_sha256": row["sequence_ids_sha256"], "sequence_length": row["sequence_length"],
                "logit_positions": row["logit_positions"], "file": path.name, "file_sha256": file_hash(path),
                "shape": list(array.shape), "stored_dtype": str(array.dtype),
                "elapsed_seconds_including_save": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            del result, array, ids, positions, keep
            save()

    def compare_pairs(pairs):
        for left, right in pairs:
            for row in inputs:
                a = np.load(args.output_dir / f"logits_{left}_{row['problem_id']}.npy", mmap_mode="r", allow_pickle=False)
                b = np.load(args.output_dir / f"logits_{right}_{row['problem_id']}.npy", mmap_mode="r", allow_pickle=False)
                result = compare_logits(a, b, row["logit_positions"], cfg["eos_token_id"])
                report["comparisons"].append({"reference": left, "observed": right,
                                               "problem_id": row["problem_id"], **result})
                del a, b
        save()

    try:
        print("고정 BF16 캐시 로딩. 새 토큰 생성·토크나이저·calibration 탐색 없음.", flush=True)
        model = load_baseline()
        original_aux = []
        for i, layer in enumerate(model.model.layers):
            row = json.loads((args.r0_dir / f"awq_layer_{i:03d}.json").read_text())
            validate_recipe(row, i, {n: list(p.shape) for n, p in layer.named_parameters()})
            original_aux.append({n: p.detach().cpu().clone() for n, p in layer.named_parameters()
                                 if n not in {x + ".weight" for x in LINEARS}})
        upstream = load_upstream(args.upstream_dir, policy)
        measure(model, "B")
        measure(model, "B_repeat")
        compare_pairs(PAIRS[:1])
        if not all(c["exact_equal"] for c in report["comparisons"]):
            report["status"] = "BASELINE_REPEAT_DIFFERENCE_REQUIRES_REVIEW"
            save()
            return report
        scale_only(model, args.r0_dir, upstream, device)
        measure(model, "S")
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_baseline()
        verify_r0_files(args.r0_dir, r0_review)
        report["awq_replay"] = replay(model, args.r0_dir, upstream, device,
            on_layer=lambda i: print(f"Q 저장 recipe 재적용 {i + 1}/28", flush=True))
        measure(model, "Q")
        verify_r0_files(args.r0_dir, r0_review)
        report["auxiliary_coordinates"] = canonicalize(model, args.r0_dir, original_aux, device)
        measure(model, "C")
        model.cpu()
        before = linear_hash(model)
        with torch.no_grad():
            for i, layer in enumerate(model.model.layers):
                for name, value in original_aux[i].items():
                    target = layer.get_parameter(name)
                    target.copy_(value)
                    if not torch.equal(target, value):
                        raise ValueError("Cw norm/bias 복원 불일치")
        after = linear_hash(model)
        if before != after:
            raise ValueError("Cw에서 C의 선형 weight가 변경됐습니다.")
        report["C_to_Cw_linear_sha256"] = {"before": before, "after": after}
        measure(model, "Cw")
        compare_pairs(PAIRS[1:])
        if len(report["forward_records"]) != 12 or len(report["comparisons"]) != 10:
            raise ValueError("진단 결과 누락")
        report["status"] = "FIXED_PREFIX_DIAGNOSTICS_COLLECTED_PENDING_REVIEW"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return report
    except BaseException as exc:
        report.update(status="FIXED_PREFIX_DIAGNOSTIC_INTERRUPTED" if isinstance(exc, KeyboardInterrupt)
                      else "FIXED_PREFIX_DIAGNOSTIC_FAILED", error_type=type(exc).__name__, error=str(exc)[:1000])
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "results/local/difficulty_v01/review_evidence_v01.json")
    parser.add_argument("--mask-report", type=Path, default=ROOT / "results/local/difficulty_v01/sdpa_mask_review_v01.json")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "results/local/r0_v03")
    parser.add_argument("--upstream-dir", type=Path, default=ROOT / "results/local/awq_reference_v03")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local/fixed_prefix_v01")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="기존 파일 해시·입력·예산만 확인; 기본 동작")
    mode.add_argument("--execute", action="store_true", help="이 새 범위의 승인 기록이 있을 때만 GPU forward")
    args = parser.parse_args(argv)
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        status = json.loads((ROOT / "docs/research_status.json").read_text(encoding="utf-8"))
        inputs, source, effective = read_inputs(args.evidence, args.mask_report, cfg)
        if not args.execute:
            preview = [{"problem_id": r["problem_id"], "source_arm": r["arm"], "seed": r["seed"],
                        "sequence_length": r["sequence_length"], "sequence_ids_sha256": r["sequence_ids_sha256"],
                        "logit_position_first": r["logit_positions"][0],
                        "logit_position_last": r["logit_positions"][-1],
                        "logit_position_count": len(r["logit_positions"])} for r in inputs]
            print(json.dumps({"status": "FIXED_PREFIX_PLAN_ONLY_NO_MODEL_EXECUTION",
                "authorization": status.get("fixed_prefix_diagnostic", {}).get("authorization", "PENDING"),
                "config_sha256": canonical_hash(cfg), "inputs": preview,
                **{k: cfg[k] for k in ("states", "comparisons", "maximum_forward_calls",
                    "maximum_forward_input_tokens", "new_generation_tokens", "maximum_sequence_length",
                    "maximum_raw_logit_bytes", "acceptance_ko")}}, ensure_ascii=False, indent=2))
            return 0
        result = execute(args, cfg, status, inputs, source, effective)
        print(result["status"])
        print(f"검토할 파일: {args.output_dir / 'run.json'} (logit 배열은 로컬 보존)")
        return 0
    except Exception as exc:
        parser.exit(2, f"고정 prefix 진단 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
