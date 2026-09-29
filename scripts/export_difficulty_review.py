"""기존 난도 시험의 원본 기록을 비공개 검토용으로 묶습니다. 새 추론·다운로드 없음."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash, load_suite, summarize


def read_snapshot(path):
    data = Path(path).read_bytes()
    return json.loads(data), {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def resume_evidence(report, folder):
    """기존 백업과 최종 기록의 앞부분을 대조하며 원본 파일은 수정하지 않습니다."""
    evidence = []
    for index, segment in enumerate(report.get("execution_segments", [])):
        ref = segment.get("resume_backup")
        if not ref:
            continue
        digest = ref.get("sha256", "")
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or ref.get("file") != f"run.before_resume.{digest}.json"):
            raise ValueError("재개 백업 이름/해시 형식 불일치")
        path = folder / ref["file"]
        item = {"execution_segment_index": index, "reference": ref}
        if not path.is_file():
            evidence.append({**item, "status": "BACKUP_NOT_FOUND_NOT_VERIFIED"})
            continue
        previous, proof = read_snapshot(path)
        if proof["sha256"] != digest:
            raise ValueError("재개 백업 바이트 해시 불일치")
        old_rows = previous["attempts"]
        if (segment.get("completed_at_start") != len(old_rows)
                or segment.get("completed_attempts_sha256") != canonical_hash(old_rows)
                or old_rows != report["attempts"][:len(old_rows)]):
            raise ValueError("재개 전후 완료 응답이 다릅니다.")
        evidence.append({**item, "status": "BACKUP_HASH_AND_COMPLETED_PREFIX_MATCH",
                         "source": proof, "preserved_attempts": len(old_rows),
                         "backup_snapshot": previous})
    return evidence


def cached_model_config(report):
    """이미 있는 고정 config.json만 읽습니다. 모델·토크나이저는 만들지 않습니다."""
    refs = json.loads((ROOT / "configs/asset_inspection_refs.json").read_text(encoding="utf-8"))
    asset = next(a for a in refs["assets"] if a["role"] == "bf16_reference")
    recorded = report.get("model_reference", {})
    if any(recorded.get(k) != asset[k] for k in ("repo_id", "revision")):
        raise ValueError("실행 모델과 고정 설정 참조가 다릅니다.")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(asset["repo_id"], "config.json", revision=asset["revision"],
                               token=False, local_files_only=True)
    except (ImportError, OSError) as exc:
        # 캐시 부재 때문에 전체 기존 결과 검토를 막거나 다운로드로 전환하지 않습니다.
        return {"status": "CACHED_CONFIG_UNAVAILABLE", "error_type": type(exc).__name__}
    config, proof = read_snapshot(path)
    digest = canonical_hash(config)
    if digest != refs["model_configuration_sha256"]["config.json"]:
        raise ValueError("캐시 모델 설정 해시가 고정 참조와 다릅니다.")
    return {"status": "PINNED_CACHED_CONFIG_MATCH", "source": proof,
            "canonical_sha256": digest, "configuration": config,
            "scope": "저장된 config.json. 실행 당시 해석된 attention mask나 모델 객체의 검증은 아님."}


def build_packet(run_path, summary_path, *, include_cached_config=False):
    cfg, suite = load_suite()
    report, run_proof = read_snapshot(run_path)
    summary, summary_proof = read_snapshot(summary_path)
    observed = summarize(report, cfg, suite)
    if observed != summary:
        raise ValueError("기존 run.json에서 읽은 결과와 제출용 summary.json이 다릅니다.")
    if observed["observed_attempts"] != 80 or observed["missing_attempts"]:
        raise ValueError("이 검토 묶음은 저장된 80개 응답 전체를 대상으로 합니다.")
    backups = resume_evidence(report, Path(run_path).parent)
    config = cached_model_config(report) if include_cached_config else {"status": "NOT_REQUESTED"}
    if (file_hash(run_path) != run_proof["sha256"]
            or file_hash(summary_path) != summary_proof["sha256"]):
        raise ValueError("읽는 동안 원본 파일이 바뀌었습니다. 실행이 끝난 뒤 확인하세요.")
    return {"schema_version": 1, "status": "EXISTING_DIFFICULTY_EVIDENCE_NOT_CERTIFICATION",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "model_ready": False,
            "new_model_or_tokenizer_execution": False, "network_download": False,
            "exporter_sha256": file_hash(Path(__file__)),
            "source_files": {"run.json": run_proof, "summary.json": summary_proof},
            "config_sha256": canonical_hash(cfg), "suite_sha256": canonical_hash(suite),
            "run_summary_match": True, "run_snapshot": report, "summary_snapshot": summary,
            "resume_backup_evidence": backups, "cached_model_config": config,
            "sharing": "전체 생성문·ID·실행 기록 포함. 사용자와의 비공개 검토용이며 공개 저장소에 커밋하지 않음.",
            "limitations": "기존 JSON의 대조·복사이며 실모델 파라미터나 전체 logit의 새 검증이 아님."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-cached-model-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("출력 파일이 이미 있습니다. 기존 검토 파일을 첨부하세요.")
        packet = build_packet(args.run, args.summary,
                              include_cached_config=args.include_cached_model_config)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(packet, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        print("저장80개와 요약 일치. 새 생성·토크나이저 실행·다운로드 없음.")
        print(f"기존 캐시 설정: {packet['cached_model_config']['status']}")
        for item in packet["resume_backup_evidence"]:
            print(f"재개 백업: {item['status']}, 보존 응답={item.get('preserved_attempts', '미확인')}")
        print(f"비공개 첨부 파일: {args.output}")
        return 0
    except Exception as exc:
        parser.exit(2, f"검토 파일 작성 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
