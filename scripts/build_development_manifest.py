"""로컬 문제 파일에서 R1 후보 목록을 만듭니다. 다운로드·추론·정답 공개는 하지 않습니다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "gsm8k-question-split-v0.2-candidate"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalized_question(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("문제는 비어 있지 않은 문자열이어야 합니다.")
    return " ".join(value.split())


def question_hash(value):
    return hashlib.sha256(normalized_question(value).encode()).hexdigest()


def problem_id(question):
    return "gsm8k/train/" + question_hash(question)


def build_manifest(development, confirmation, *, n=100, seed=20260929,
                   expected_confirmation_rows=500):
    """순서 불변 선택. confirmation에는 문제 해시만 사용하며 정답은 참조하지 않습니다."""
    if type(n) is not int or n < 1 or type(seed) is not int:
        raise ValueError("선택 수와 시드를 확인하세요.")
    if type(expected_confirmation_rows) is not int or expected_confirmation_rows < 1:
        raise ValueError("확증 데이터의 예상 행 수는 양의 정수여야 합니다.")
    if len(confirmation) != expected_confirmation_rows:
        raise ValueError("확증 문제 행 수가 예상과 다릅니다. 부분 파일을 사용하지 마세요.")
    excluded_hashes = set()
    for row in confirmation:
        if not isinstance(row, dict):
            raise ValueError("확증 데이터 행은 JSON 객체여야 합니다.")
        excluded_hashes.add(question_hash(row.get("problem")))
    if len(excluded_hashes) != expected_confirmation_rows:
        raise ValueError("확증 문제에 중복이 있습니다. 원본을 검토하세요.")

    unique = {}
    duplicate_rows = 0
    for row in development:
        if not isinstance(row, dict):
            raise ValueError("개발 데이터 행은 JSON 객체여야 합니다.")
        q = normalized_question(row.get("question"))
        qhash = question_hash(q)
        if qhash in unique:
            # An unlikely hash collision must not silently drop a question.
            if unique[qhash] != q:
                raise ValueError("문제 해시 충돌입니다.")
            duplicate_rows += 1
        else:
            unique[qhash] = q
    eligible = [(h, q) for h, q in unique.items() if h not in excluded_hashes]
    if len(eligible) < n:
        raise ValueError("중복·확증 문제 제외 후 개발 문제가 부족합니다.")
    eligible.sort(key=lambda x: (hashlib.sha256(f"{seed}\n{x[1]}".encode()).hexdigest(), x[0]))
    selected = [{"id": problem_id(q), "question_sha256": h} for h, q in eligible[:n]]
    return {
        "schema_version": 1, "algorithm_version": VERSION,
        "status": "LOCAL_CANDIDATE_NOT_SOURCE_AUTHENTICATED",
        "selection_seed": seed, "requested_count": n,
        "normalization": "공백 정규화만 적용; 대소문자·Unicode·의미 중복은 통합하지 않음",
        "counts": {"development_rows": len(development), "duplicate_development_rows": duplicate_rows,
                   "confirmation_rows": len(confirmation),
                   "excluded_confirmation_overlap": len(unique.keys() & excluded_hashes),
                   "eligible_unique": len(eligible), "selected": len(selected),
                   "remaining_unassigned": len(eligible) - n},
        "confirmation_question_set_sha256": canonical_hash(sorted(excluded_hashes)),
        "selected_ids_sha256": canonical_hash([r["id"] for r in selected]),
        "selected": selected,
        "limitations": [
            "입력 파일 내용만 검사했습니다. 명시한 원격 revision에서 받은 파일인지 인증하지 않습니다.",
            "질문 텍스트 중복 제외는 의미 중복 또는 사전학습 오염 부재를 보장하지 않습니다.",
            "남은 개발 문제는 아직 KL 매칭·검증에 배정하지 않았습니다.",
        ],
    }


def read_jsonl(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True, help="question 필드가 있는 GSM8K train JSONL")
    parser.add_argument("--confirmation", type=Path, required=True, help="problem 필드가 있는 MATH-500 문제 JSONL")
    parser.add_argument("--refs", type=Path, default=ROOT / "configs/asset_inspection_refs.json")
    parser.add_argument("--candidate", type=Path, default=ROOT / "configs/reproduction_candidate.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("출력 파일이 이미 있습니다. 덮어쓰지 않습니다.")
    try:
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        refs = json.loads(args.refs.read_text(encoding="utf-8"))
        if canonical_hash(candidate) != refs["manifest_canonical_sha256"]:
            raise ValueError("후보 설정과 참조 해시가 다릅니다.")
        dev, dhash = read_jsonl(args.development)
        confirm, chash = read_jsonl(args.confirmation)
        result = build_manifest(dev, confirm, n=candidate["r1"]["n_problems"],
                                seed=candidate["r1"]["selection_seed"])
        result["input_file_sha256"] = {"development": dhash, "confirmation": chash}
        result["declared_sources_not_authenticated"] = [
            {k: asset[k] for k in ("role", "repo_id", "repo_type", "revision")}
            for asset in refs["assets"] if asset["role"] in ("development", "confirmation")
        ]
        result["manifest_canonical_sha256"] = canonical_hash(candidate)
        result["inspection_refs_sha256"] = canonical_hash(refs)
        write_new(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"목록 생성 실패: {type(exc).__name__}: {exc}")
    print(f"후보 목록 생성 완료: {result['counts']['selected']}개; 원본 출처 인증은 별도입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
