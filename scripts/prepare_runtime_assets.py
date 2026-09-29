"""고정 원본 코드·개발 데이터·calibration 토큰을 준비하는 명령. 기본값은 오프라인 점검입니다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from runtime_assets import (
    ROOT, canonical_hash, development_bundle, download_sources, fetch_upstream,
    file_digests, implementation_fingerprint, iter_jsonl, load_contracts, package_versions, plan_sources,
    question_hash, read_development, select_calibration, write_jsonl, write_new,
)


def inspection(candidate, policy):
    return {"status": "OFFLINE_PREPARATION_INSPECTION",
            "candidate_sha256": canonical_hash(candidate),
            "policy_sha256": canonical_hash(policy),
            "packages": package_versions(["numpy", "torch", "transformers", "huggingface-hub",
                                           "pyarrow", "zstandard", "tqdm"]),
            "source_paths": policy["source_paths"],
            "network_used": False, "model_weights_loaded": False}


def load_tokenizer(refs):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    versions = package_versions(["transformers"])
    if versions["transformers"] != "4.51.3":
        raise ValueError("토크나이저 점검과 같은 Transformers 4.51.3이 필요합니다.")
    asset = next(x for x in refs["assets"] if x["role"] == "bf16_reference")
    for name, expected in refs["model_configuration_sha256"].items():
        path = hf_hub_download(asset["repo_id"], name, revision=asset["revision"],
                               token=False, endpoint="https://huggingface.co")
        if canonical_hash(json.loads(Path(path).read_text())) != expected:
            raise ValueError("모델 설정 해시가 검토한 스냅샷과 다릅니다.")
    tokenizer = AutoTokenizer.from_pretrained(asset["repo_id"], revision=asset["revision"],
                                             token=False, trust_remote_code=False)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (151646, 151643, 151643):
        raise ValueError("토크나이저 특수 토큰이 사용자 검토 기록과 다릅니다.")
    return tokenizer, {"repo_id": asset["repo_id"], "revision": asset["revision"],
                       "transformers": versions["transformers"], "add_special_tokens": False}


def prepare_data(output, candidate, refs, policy, *, include_calibration):
    if os.environ.get("HF_ENDPOINT", "https://huggingface.co") != "https://huggingface.co":
        raise ValueError("출처 검증에는 공식 Hugging Face endpoint를 사용해야 합니다.")
    deps = package_versions(["huggingface-hub", "pyarrow", "zstandard", "transformers"])
    if deps["huggingface-hub"] != "0.36.2" or deps["pyarrow"] != "20.0.0":
        raise ValueError("준비 의존성 huggingface-hub 0.36.2와 pyarrow 20.0.0을 확인하세요.")
    if include_calibration and (deps["zstandard"] != "0.23.0" or deps["transformers"] != "4.51.3"):
        raise ValueError("calibration에는 zstandard 0.23.0과 Transformers 4.51.3이 필요합니다.")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint="https://huggingface.co", token=False)
    roles = ["development", "confirmation"] + (["awq_calibration"] if include_calibration else [])
    plans = plan_sources(api, refs, policy, roles)
    print(f"원본 파일 다운로드 계획: {sum(x['size'] for x in plans):,} bytes", flush=True)
    paths, proofs = download_sources(plans, hf_hub_download)
    development = read_development(paths["development"])
    # 원본 JSONL에는 정답도 있지만, 보존·선택하는 값은 problem 하나입니다.
    confirmation = [{"problem": row["problem"]} for _, row in iter_jsonl(paths["confirmation"])]
    manifest, inputs, gold = development_bundle(development, confirmation, candidate, policy)
    manifest.update(status="SOURCE_FILES_VERIFIED_CANDIDATE_NOT_ACCEPTED",
                    source_files=[x for x in proofs if x["role"] != "awq_calibration"],
                    inspection_refs_sha256=canonical_hash(refs))
    artifacts = {"development_manifest.json": manifest}
    if include_calibration:
        tokenizer, tokenizer_evidence = load_tokenizer(refs)
        exclusions = {question_hash(row["question"]) for row in development}
        exclusions.update(question_hash(row["problem"]) for row in confirmation)
        rows = iter_jsonl(paths["awq_calibration"], compressed=True,
                          max_rows=policy["calibration"]["max_source_rows"])
        blocks, calibration = select_calibration(
            rows, lambda text: tokenizer.encode(text, add_special_tokens=False),
            exclusions, policy)
        calibration.update(status="SOURCE_FILES_VERIFIED_CANDIDATE_NOT_ACCEPTED",
                           source_file=next(x for x in proofs if x["role"] == "awq_calibration"),
                           tokenizer=tokenizer_evidence, candidate_sha256=canonical_hash(candidate),
                           preparation_policy_sha256=canonical_hash(policy),
                           inspection_refs_sha256=canonical_hash(refs))
        artifacts["calibration_manifest.json"] = calibration
        artifacts["calibration_tokens.json"] = {"blocks": blocks}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "r1_inputs.jsonl", inputs)
    write_jsonl(output / "r1_gold.jsonl", gold)
    for name, obj in artifacts.items():
        write_new(output / name, obj)
    files = {p.name: file_digests(p) for p in sorted(output.iterdir()) if p.is_file()}
    report = {"schema_version": 1, "status": "PREPARED_CANDIDATE_NOT_RUN_APPROVAL",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_sha256": canonical_hash(candidate), "policy_sha256": canonical_hash(policy),
              "inspection_refs_sha256": canonical_hash(refs), "source_files": proofs,
              "packages": deps, "include_calibration": include_calibration,
              "implementation_sha256": implementation_fingerprint([
                  "prepare_runtime_assets.py", "runtime_assets.py", "build_development_manifest.py", "evaluate_gsm8k.py"]),
              "artifacts": files, "model_weights_loaded": False, "generation_run": False,
              "confirmation_use": "원본 바이트 해시 검사 후 problem 필드의 중복 제외만 수행; 정답 평가·생성 없음"}
    write_new(output / "preparation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inspect", "upstream", "development", "all"], default="inspect")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        candidate, refs, policy = load_contracts()
        if args.mode == "inspect":
            print(json.dumps(inspection(candidate, policy), ensure_ascii=False, indent=2))
            return 0
        if not args.online or args.output_dir is None:
            parser.error("다운로드 작업에는 --online과 --output-dir을 명시하세요.")
        if args.mode == "upstream":
            proof = fetch_upstream(args.output_dir, policy)
            print(f"고정 AWQ 원본 파일 {len(proof)}개 해시 일치. 코드는 아직 실행하지 않았습니다.")
            return 0
        if args.output_dir.exists():
            parser.error("출력 디렉터리가 이미 있습니다. 새 경로를 지정하세요.")
        report = prepare_data(args.output_dir, candidate, refs, policy, include_calibration=args.mode == "all")
        print(report["status"])
        return 0
    except Exception as exc:
        parser.exit(2, f"준비 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
