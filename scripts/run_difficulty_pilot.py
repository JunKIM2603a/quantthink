"""자체 난도 20문제 BF16/AWQ 재구성 탐색. --execute 없이는 모델·토크나이저를 로드하지 않습니다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash, load_suite, prompt_for, summarize
from difficulty_pilot_resume import (
    ProgressReporter, attempt_key, backup_before_resume, check_parameter_hash, load_resume,
    mark_interrupted, output_lock, resume_plan, verify_compatibility,
)
from replay_r0_awq import replay, validate_recipe, verify_r0_files


def require_scope(status, cfg):
    if (status.get("novelty_gate") != "SCOPED_CONTRIBUTION_ACCEPTED"
            or status.get("protocol_gate") != "ACCEPTED"
            or status.get("research_approval") != "APPROVED"):
        raise ValueError("연구 기본 gate 기록을 확인하세요.")
    scope = status.get("difficulty_pilot", {})
    if (scope.get("authorization") != "USER_REQUESTED_BOUNDED_DIFFICULTY_PILOT"
            or scope.get("config_sha256") != canonical_hash(cfg)):
        raise ValueError("현재 요청에 따른 난도 시험 범위/설정 해시 기록이 없습니다.")


def repo_state():
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise ValueError("커밋된 저장소 작업 트리에서 실행하세요.")
        return result.stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("추적 파일에 커밋되지 않은 변경이 있습니다.")
    return {"commit": git("rev-parse", "HEAD"), "tracked_dirty": False}


def checkpoint(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parameter_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":")).encode() + b"\0")
        digest.update(memoryview(cpu.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def cached_tokenizer(refs):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    asset = next(x for x in refs["assets"] if x["role"] == "bf16_reference")
    for name, expected in refs["model_configuration_sha256"].items():
        path = hf_hub_download(asset["repo_id"], name, revision=asset["revision"],
                               token=False, local_files_only=True)
        if canonical_hash(json.loads(Path(path).read_text())) != expected:
            raise ValueError("기존 캐시의 모델 설정 해시가 다릅니다.")
    tokenizer = AutoTokenizer.from_pretrained(asset["repo_id"], revision=asset["revision"], token=False,
                                             local_files_only=True, trust_remote_code=False)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (151646, 151643, 151643):
        raise ValueError("검토한 특수 토큰과 다릅니다.")
    return tokenizer, asset


def execute(args, cfg, suite, status):
    require_scope(status, cfg)
    repo = repo_state()
    if args.resume:
        previous, plan = load_resume(args.output_dir / "run.json", cfg, suite)
        print(f"재개 확인: 완료 {plan['completed']}/80 보존, 남은 {plan['remaining']}개", flush=True)
        if not plan["remaining"]:
            print("모든 응답이 저장되어 있습니다. 새 생성 없이 종료합니다.", flush=True)
            return previous
    elif args.output_dir.exists():
        raise ValueError("출력 폴더가 이미 있습니다. 결과를 삭제하지 말고 --execute --resume으로 이어가세요.")
    from runtime_assets import load_contracts, package_versions, verify_upstream
    candidate, refs, policy = load_contracts()
    versions = package_versions(["torch", "transformers", "huggingface-hub", "numpy"])
    if versions != {"torch": "2.7.1+cu118", "transformers": "4.51.3", "huggingface-hub": "0.36.2", "numpy": "1.26.4"}:
        raise ValueError(f"R0와 동일한 검토 환경이 필요합니다: {versions}")
    review = json.loads((ROOT / cfg["awq_source"]).read_text())
    old, proofs = verify_r0_files(args.r0_dir, review)
    if old["candidate_sha256"] != canonical_hash(candidate) or old["policy_sha256"] != canonical_hash(policy):
        raise ValueError("R0의 후보/정책과 현재 고정 참조가 다릅니다.")
    source_proof = verify_upstream(args.upstream_dir, policy)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from runtime_contracts import prompt_ids
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("CUDA 장치가 필요합니다.")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("선택 장치에서 BF16을 지원하지 않습니다.")
    # 모든 입력 길이를 먼저 확인합니다. 토크나이저 점검 스크립트·준비 자료를 다시 실행하지 않습니다.
    tokenizer, asset = cached_tokenizer(refs)
    inputs = {p["id"]: prompt_ids(tokenizer, prompt_for(suite, p)) for p in suite["problems"]}
    if any(len(ids) > cfg["max_prompt_tokens"] or len(ids) + cfg["max_new_tokens"] > cfg["max_total_context_tokens"] for ids in inputs.values()):
        raise ValueError("입력/전체 문맥 상한 초과. 입력을 잘라 실행하지 않습니다.")
    print("기존 캐시에서 입력을 확인합니다. 데이터 준비·calibration 탐색은 실행하지 않습니다.", flush=True)
    report = {"schema_version": 1, "status": "DIFFICULTY_PILOT_STARTED", "model_ready": False,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "repository": repo,
              "config_sha256": canonical_hash(cfg), "suite_sha256": canonical_hash(suite),
              "research_status_sha256": canonical_hash(status), "packages": versions,
              "python": platform.python_version(), "model_reference": asset,
              "r0_input_files": proofs, "upstream_source_files": source_proof,
              "implementation_sha256": {n: file_hash(ROOT / "scripts" / n) for n in
                                          ["run_difficulty_pilot.py", "difficulty_pilot_resume.py", "difficulty_pilot_contracts.py", "replay_r0_awq.py",
                                           "qwen2_awq_adapter.py", "runtime_assets.py", "runtime_contracts.py", "reproduction_contracts.py"]},
              "generation_config": {**cfg["decoding"], "max_new_tokens": cfg["max_new_tokens"], "use_cache": True, "batch_size": 1},
              "gpu": {"device": str(device), "name": torch.cuda.get_device_name(device),
                      "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory},
              "runtime_flags": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "float32_matmul_precision": torch.get_float32_matmul_precision()},
              "attempts": [], "completed_replay_layers": 0, "parameter_sha256": {},
              "awq_recipe_search_performed": False}
    with output_lock(args.output_dir, resume=args.resume):
        return execute_locked(args, cfg, suite, report, inputs, tokenizer, policy, review, device)


def execute_locked(args, cfg, suite, current, inputs, tokenizer, policy, review, device):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList
    from runtime_contracts import generated_record
    from qwen2_awq_adapter import load_upstream

    if args.resume:
        # lock 획득 전에 다른 실행이 진척됐을 가능성까지 반영해 다시 읽습니다.
        report, plan = load_resume(args.output_dir / "run.json", cfg, suite)
        verify_compatibility(report, current, inputs)
        if not plan["remaining"]:
            return report
        backup = backup_before_resume(args.output_dir / "run.json")
    else:
        report, backup = current, None
        plan = resume_plan(report, cfg, suite)
    segment = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
               "repository": current["repository"], "implementation_sha256": current["implementation_sha256"],
               "research_status_sha256": current["research_status_sha256"],
               "completed_at_start": plan["completed"], "completed_attempts_sha256": plan["completed_attempts_sha256"],
               "previous_status": report["status"], "resume_backup": backup,
               "previous_active_attempt": report.pop("active_attempt", None),
               "partial_token_continuation": False,
               "progress_monitor": "15초 간격 표시; 매 토큰 종료 조건은 항상 False",
               "timing_note": "이 실행 구간은 진행 표시 비용을 포함. 이전 코드의 생성 시간과 엄밀한 성능 비교를 하지 않음."}
    report.setdefault("execution_segments", []).append(segment)
    report["status"] = "DIFFICULTY_PILOT_RESUMED" if args.resume else "DIFFICULTY_PILOT_STARTED"
    for key in ("error_type", "error"):
        if key in report:
            segment["previous_" + key] = report.pop(key)
    completed = set(plan["completed_keys"])
    asset = current["model_reference"]
    save = lambda: checkpoint(args.output_dir / "run.json", report)
    save()
    try:
        print("고정 BF16 모델을 캐시에서 로딩합니다. SDPA 경고 자체는 중단 신호가 아닙니다.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            asset["repo_id"], revision=asset["revision"], token=False, trust_remote_code=False,
            use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if cfg["max_total_context_tokens"] > model.config.max_position_embeddings:
            raise ValueError("모델 설정의 전체 문맥 상한 초과")
        if model.__class__.__name__ != "Qwen2ForCausalLM" or len(model.model.layers) != 28:
            raise ValueError("고정 28계층 Qwen2ForCausalLM이 필요합니다.")
        upstream = load_upstream(args.upstream_dir, policy)
        # BF16 생성 40회를 소비하기 전에 28개 recipe의 dtype·shape·경로를 먼저 확인합니다.
        for index, layer in enumerate(model.model.layers):
            recipe = json.loads((args.r0_dir / f"awq_layer_{index:03d}.json").read_text())
            validate_recipe(recipe, index, {n: list(p.shape) for n, p in layer.named_parameters()})
        print("입력 20개와 저장 recipe 28개 확인 완료. 새 calibration 탐색 없음.", flush=True)
        generation = GenerationConfig(do_sample=True, num_beams=1, **cfg["decoding"],
                                      bos_token_id=151646, eos_token_id=151643, pad_token_id=151643,
                                      min_new_tokens=0, forced_eos_token_id=None, stop_strings=None,
                                      max_new_tokens=cfg["max_new_tokens"])
        resolved = generation.to_dict()
        if report.get("resolved_generation_config", resolved) != resolved:
            raise ValueError("재개 전후 해석된 GenerationConfig가 다릅니다.")
        report["resolved_generation_config"] = resolved
        print("BF16 가중치 해시를 계산·대조합니다.", flush=True)
        check_parameter_hash(report, "bf16", parameter_hash(model))
        save()
        for arm in cfg["arms"]:
            if all((p["id"], arm, seed) in completed for p in suite["problems"] for seed in cfg["seeds"]):
                print(f"{arm}: 저장된 40개 응답을 재사용합니다. 생성 생략.", flush=True)
                continue
            if arm == "awq_w3_replay":
                report["completed_replay_layers"] = 0
                def on_layer(index):
                    report["completed_replay_layers"] = index + 1
                    save()
                    print(f"AWQ 저장 recipe 재적용 {index + 1}/28", flush=True)
                # 생성 도중 원본 파일이 바뀌었는지도 다시 대조합니다. 모델 추론 재실행은 아닙니다.
                verify_r0_files(args.r0_dir, review)
                report["awq_replay"] = replay(model, args.r0_dir, upstream, device, on_layer=on_layer)
            model.cpu()
            if arm == "awq_w3_replay":
                print("AWQ 재구성 가중치 해시를 계산·대조합니다.", flush=True)
                check_parameter_hash(report, arm, parameter_hash(model))
            print(f"{arm}: GPU 이동 후 미완료 응답을 시작합니다.", flush=True)
            model.to(device).eval()
            save()
            for problem in suite["problems"]:
                for seed in cfg["seeds"]:
                    if (problem["id"], arm, seed) in completed:
                        continue
                    ids = inputs[problem["id"]]
                    row = {"problem_id": problem["id"], "level": problem["level"], "arm": arm, "seed": seed,
                           "execution_segment_index": len(report["execution_segments"]) - 1,
                           "input_ids": ids, "input_ids_sha256": canonical_hash(ids), "input_token_count": len(ids),
                           "prompt_sha256": canonical_hash(prompt_for(suite, problem))}
                    label = f"{len(report['attempts']) + 1}/80 {arm} {problem['id']} seed={seed}"
                    report["active_attempt"] = {"problem_id": problem["id"], "arm": arm, "seed": seed,
                                                "started_at_utc": datetime.now(timezone.utc).isoformat()}
                    save()
                    print(f"시작 {label} (최대 {cfg['max_new_tokens']}토큰)", flush=True)
                    try:
                        tensor = torch.tensor([ids], device=device, dtype=torch.long)
                        torch.manual_seed(seed)
                        row["rng_before"] = {"cpu": hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest(),
                                             "cuda": hashlib.sha256(bytes(torch.cuda.get_rng_state(device).tolist())).hexdigest()}
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        started = time.monotonic()
                        reporter = ProgressReporter(label, cfg["max_new_tokens"])
                        class ProgressOnly(StoppingCriteria):
                            def __call__(self, input_ids, scores, **kwargs):
                                reporter.tick(input_ids.shape[-1] - len(ids))
                                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                        with torch.inference_mode():
                            output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                                    generation_config=generation, use_cache=True,
                                                    stopping_criteria=StoppingCriteriaList([ProgressOnly()]))
                        torch.cuda.synchronize(device)
                        generated = output[0, len(ids):].detach().cpu().tolist()
                        row.update(generated_record(generated, tokenizer, cfg["max_new_tokens"]),
                                   elapsed_seconds=time.monotonic() - started,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                        if row["finish_reason"] not in {"eos", "length"}:
                            raise ValueError("예상하지 않은 생성 종료")
                        del tensor, output
                    except Exception as exc:
                        row.update(finish_reason="error", error_type=type(exc).__name__, error=str(exc)[:1000])
                        report["attempts"].append(row)
                        save()
                        raise
                    report["attempts"].append(row)
                    completed.add(attempt_key(row))
                    report.pop("active_attempt", None)
                    save()
                    print(f"{len(report['attempts'])}/80 {arm} {problem['id']} seed={seed} "
                          f"{row['finish_reason']} tokens={len(generated)}", flush=True)
        report["status"] = "DIFFICULTY_PILOT_EXECUTED_PENDING_REVIEW"
        save()
    except KeyboardInterrupt:
        mark_interrupted(report)
        save()
        raise
    except Exception as exc:
        report.update(status="DIFFICULTY_PILOT_STOPPED_NO_AUTOMATIC_RETRY", error_type=type(exc).__name__, error=str(exc)[:1000])
        save()
        raise
    finally:
        segment.update(completed_at_end=len(report["attempts"]), status=report["status"],
                       ended_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        checkpoint(args.output_dir / "summary.json", summarize(report, cfg, suite))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="완료 응답을 보존하고 기존 결과 폴더에서 미완료 응답만 실행")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "results/local/r0_v03")
    parser.add_argument("--upstream-dir", type=Path, default=ROOT / "results/local/awq_reference_v03")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local/difficulty_v01")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        cfg, suite = load_suite()
        status = json.loads((ROOT / "docs/research_status.json").read_text())
        require_scope(status, cfg)
        if not args.execute:
            if args.resume:
                _, plan = load_resume(args.output_dir / "run.json", cfg, suite)
                print(json.dumps({"status": "DIFFICULTY_RESUME_INSPECTION_ONLY",
                                  "completed": plan["completed"], "remaining": plan["remaining"],
                                  "next_attempt": plan["pending_keys"][:1], "model_loaded": False}, ensure_ascii=False, indent=2))
                return 0
            print(json.dumps({"status": "DIFFICULTY_PLAN_INSPECTION_ONLY", "problems": 20,
                              "attempts": 80, "max_new_tokens": cfg["max_new_tokens"],
                              "maximum_generation_tokens": cfg["maximum_generation_tokens"],
                              "config_sha256": canonical_hash(cfg), "model_loaded": False,
                              "network_used": False, "local_r0_files_verified": False}, ensure_ascii=False, indent=2))
            return 0
        result = execute(args, cfg, suite, status)
        print(result["status"])
        return 0
    except KeyboardInterrupt:
        print("사용자 중단: 저장된 완료 응답은 보존했습니다. --execute --resume으로 미완료 응답부터 이어갈 수 있습니다.", flush=True)
        return 130
    except Exception as exc:
        parser.exit(2, f"난도 시험 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
