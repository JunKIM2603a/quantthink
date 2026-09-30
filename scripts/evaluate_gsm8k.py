"""GSM8K 개발 실험용 보수적 수치 평가기 후보. MATH-500 기호 정답·overthinking 판정은 미지원."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re

from build_development_manifest import canonical_hash, read_jsonl, write_new

VERSION = "gsm8k-final-region-v0.2-candidate"
NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?\Z")


def numeric_value(text):
    """제한된 숫자 문법만 허용. eval, 근사 비교, 임의 단위 삭제를 하지 않습니다."""
    if not isinstance(text, str) or len(text) > 256:
        raise ValueError("수치 표현이 유효하지 않습니다.")
    s = text.strip().replace("−", "-")
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    frac = re.fullmatch(r"([+-]?)\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", s)
    if frac:
        sign, numerator, denominator = frac.groups()
        a, b = plain_number(numerator), plain_number(denominator)
        if not b:
            raise ValueError("0으로 나눌 수 없습니다.")
        return (-1 if sign == "-" else 1) * a / b
    if s.count("/") == 1:
        numerator, denominator = s.split("/")
        a, b = plain_number(numerator), plain_number(denominator)
        if not b:
            raise ValueError("0으로 나눌 수 없습니다.")
        return a / b
    return plain_number(s)


def plain_number(s):
    s = s.strip()
    if not NUMBER.fullmatch(s):
        raise ValueError("지원하지 않는 수치 표기입니다.")
    if sum(c.isdigit() for c in s) > 100:
        raise ValueError("숫자 자릿수가 너무 많습니다.")
    if "e" in s.lower() and abs(int(s.lower().split("e")[1])) > 100:
        raise ValueError("지수 범위를 벗어났습니다.")
    return Fraction(s.replace(",", ""))


def gold_value(answer):
    if not isinstance(answer, str) or answer.count("####") != 1:
        raise ValueError("GSM8K 원본 정답의 #### 구분자가 필요합니다.")
    return numeric_value(answer.split("####", 1)[1])


def boxed_values(text):
    values, pos = [], 0
    for _ in range(32):
        start = text.find(r"\boxed", pos)
        if start < 0:
            return values
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text) or text[cursor] != "{":
            raise ValueError("boxed 괄호가 없습니다.")
        begin, depth = cursor + 1, 1
        cursor += 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("boxed 괄호가 닫히지 않았습니다.")
        values.append(numeric_value(text[begin:cursor - 1]))
        pos = cursor
    raise ValueError("boxed 표현이 지나치게 많습니다.")


def parse_final_answer(generated_text):
    if not isinstance(generated_text, str) or len(generated_text) > 2_000_000:
        return {"parse_status": "INVALID_TEXT", "value": None}
    # 입력 템플릿에 시작 태그가 있습니다. 디코딩할 때 종료 태그를 보존해야 합니다.
    if generated_text.count("</think>") != 1 or "<think>" in generated_text:
        return {"parse_status": "NO_UNAMBIGUOUS_FINAL_REGION", "value": None}
    final = generated_text.split("</think>", 1)[1].strip()
    if not final:
        return {"parse_status": "EMPTY_FINAL_REGION", "value": None}
    try:
        values = boxed_values(final)
        # 명시적인 정답 행과 boxed 정답이 서로 다른지도 검사합니다.
        marked = []
        for line in final.splitlines():
            match = re.fullmatch(r"\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+|:\s*)(.*?)\s*",
                                 line, flags=re.I)
            if match and r"\boxed" not in match[1]:
                value = match[1].strip()
                if value.endswith("."):
                    value = value[:-1]
                marked.append(numeric_value(value))
        values.extend(marked)
        if not values:
            # 최종 영역 전체가 숫자인 경우는 허용하며, 임의의 마지막 숫자를 추출하지 않습니다.
            values = [numeric_value(final)]
        if len(set(values)) != 1:
            return {"parse_status": "AMBIGUOUS_FINAL_ANSWER", "value": None}
        return {"parse_status": "PARSED", "value": str(values[0])}
    except (ValueError, ZeroDivisionError, OverflowError):
        return {"parse_status": "UNPARSEABLE_FINAL_ANSWER", "value": None}


def evaluate_grid(gold_rows, generations, *, arms=("bf16", "awq_w3"), seeds=(42, 43)):
    """모든 계획된 문제×arm×seed를 보존합니다. 누락·실행 오류는 실험 불완전으로 처리."""
    if not arms or not seeds or len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
        raise ValueError("arm·seed 목록을 중복 없이 지정하세요.")
    if any(type(s) is not int for s in seeds) or any(not isinstance(a, str) or not a for a in arms):
        raise ValueError("arm·seed 형식이 유효하지 않습니다.")
    gold = {}
    for row in gold_rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in gold:
            raise ValueError("정답 ID가 비어 있거나 중복됐습니다.")
        gold[key] = gold_value(row["answer"])
    if not gold:
        raise ValueError("정답 목록이 없습니다.")
    observed = {}
    for row in generations:
        if type(row.get("seed")) is not int:
            raise ValueError("생성 seed 형식이 유효하지 않습니다.")
        key = (row["id"], row["arm"], row["seed"])
        if key in observed or key[0] not in gold or key[1] not in arms or key[2] not in seeds:
            raise ValueError("생성 ID/arm/seed가 계획과 다르거나 중복됐습니다.")
        reason = row.get("finish_reason")
        if reason not in ("eos", "length", "error", "unexpected_stop"):
            raise ValueError("finish_reason을 명시해야 합니다.")
        if reason in ("eos", "length") and not isinstance(row.get("generated_text"), str):
            raise ValueError("완료된 생성에는 generated_text 문자열이 필요합니다.")
        observed[key] = row
    items = []
    for key_id in sorted(gold):
        for arm in arms:
            for seed in seeds:
                source = observed.get((key_id, arm, seed))
                item = {"id": key_id, "arm": arm, "seed": seed, "correct": None}
                if source is None:
                    item.update(record_status="MISSING", parse_status="NOT_SCORED")
                elif source["finish_reason"] in ("error", "unexpected_stop"):
                    item.update(record_status="ENGINEERING_FAILURE", parse_status="NOT_SCORED",
                                finish_reason=source["finish_reason"])
                else:
                    parsed = parse_final_answer(source["generated_text"])
                    item.update(record_status="COMPLETED", finish_reason=source["finish_reason"],
                                **parsed)
                    item["correct"] = (parsed["value"] is not None
                                       and Fraction(parsed["value"]) == gold[key_id])
                items.append(item)
    summary = {}
    for arm in arms:
        rows = [x for x in items if x["arm"] == arm]
        done = [x for x in rows if x["record_status"] == "COMPLETED"]
        correct = sum(x["correct"] is True for x in rows)
        full = len(done) == len(rows)
        summary[arm] = {
            "scheduled": len(rows), "completed": len(done),
            "missing": sum(x["record_status"] == "MISSING" for x in rows),
            "engineering_failure": sum(x["record_status"] == "ENGINEERING_FAILURE" for x in rows),
            "correct": correct,
            "primary_accuracy": correct / len(rows) if full else None,
            "completed_accuracy_descriptive": correct / len(done) if done else None,
            "scheduled_correct_fraction_lower_bound": correct / len(rows),
            "parse_failures": sum(x["parse_status"] != "PARSED" for x in done),
            "budget_stops": sum(x.get("finish_reason") == "length" for x in done),
        }
    return {
        "schema_version": 1, "evaluator_version": VERSION,
        "status": "COMPLETE_GRID_CANDIDATE_SCORING" if all(
            x["record_status"] == "COMPLETED" for x in items) else "INCOMPLETE_GRID_NO_PRIMARY_COMPARISON",
        "scope": "GSM8K 수치 평가기 후보; parse 실패는 별도 감사 대상, overthinking 자동 판정 아님",
        "requires_parse_audit": any(x["parse_failures"] for x in summary.values()),
        "main_hypothesis_ready": False,
        "summary": summary, "items": items,
    }


def validate_planned_ids(gold, manifest, candidate):
    """정답 파일의 누락이 계획된 문제 수를 줄이지 않도록 후보 목록과 대조합니다."""
    if manifest["manifest_canonical_sha256"] != canonical_hash(candidate):
        raise ValueError("개발 목록과 평가 후보 설정의 해시가 다릅니다.")
    selected = manifest["selected"]
    ids = [row["id"] for row in selected]
    if (len(ids) != candidate["r1"]["n_problems"] or len(set(ids)) != len(ids)
            or manifest["selected_ids_sha256"] != canonical_hash(ids)):
        raise ValueError("개발 목록의 ID 수·중복·해시를 확인하세요.")
    for row in selected:
        if (not re.fullmatch(r"[0-9a-f]{64}", row["question_sha256"])
                or row["id"] != "gsm8k/train/" + row["question_sha256"]):
            raise ValueError("개발 목록의 문제 해시와 ID가 다릅니다.")
    gold_ids = [row["id"] for row in gold]
    if len(gold_ids) != len(set(gold_ids)) or set(gold_ids) != set(ids):
        raise ValueError("정답 ID 목록이 계획된 개발 문제 전체와 일치하지 않습니다.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True, help="id, answer 필드가 있는 정답 JSONL")
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--development-manifest", type=Path, required=True,
                   help="build_development_manifest.py로 만든 개발 문제 후보 목록")
    p.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reproduction_candidate.json")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("출력 파일이 이미 있습니다.")
    try:
        gold, ghash = read_jsonl(a.gold)
        generations, rhash = read_jsonl(a.generations)
        config = json.loads(a.candidate.read_text(encoding="utf-8"))
        manifest = json.loads(a.development_manifest.read_text(encoding="utf-8"))
        validate_planned_ids(gold, manifest, config)
        result = evaluate_grid(gold, generations, arms=config["r1"]["arms"],
                               seeds=config["r1"]["sampling_seeds"])
        result["input_file_sha256"] = {"gold": ghash, "generations": rhash}
        result["manifest_canonical_sha256"] = canonical_hash(config)
        result["development_manifest_canonical_sha256"] = canonical_hash(manifest)
        result["requires_source_provenance_review"] = True
        write_new(a.output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(f"평가 실패: {type(exc).__name__}: {exc}")
    print(result["status"])
    return 0 if result["status"] == "COMPLETE_GRID_CANDIDATE_SCORING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
