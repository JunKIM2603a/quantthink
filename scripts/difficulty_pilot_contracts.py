"""난도 시험의 고정 입력·최종 답·검열 집계. 표준 라이브러리만 사용합니다."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import statistics

from reproduction_contracts import termination_record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/difficulty_pilot_v01.json"
ARMS = ("bf16", "awq_w3_replay")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rational(text):
    """정수/정확한 십진수/분수만 허용. 수식 실행이나 일반 수학 동치 추정 없음."""
    if not isinstance(text, str) or len(text) > 128:
        raise ValueError("지원 범위 밖 답안")
    text = re.sub(r"\s+", "", text.replace("−", "-"))
    frac = re.fullmatch(r"([+-]?)\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", text)
    if frac:
        sign, numerator, denominator = frac.groups()
        return (-1 if sign == "-" else 1) * Fraction(int(numerator), int(denominator))
    if not re.fullmatch(r"[+-]?(?:\d+/[+-]?\d+|\d+(?:\.\d+)?|\.\d+)", text):
        raise ValueError("지원 범위 밖 답안")
    if "/" in text:
        a, b = text.split("/")
        return Fraction(int(a), int(b))
    return Fraction(text)


def load_suite(config_path=CONFIG):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / cfg["suite_path"]).read_text(encoding="utf-8"))
    if canonical_hash(suite) != cfg["suite_sha256"]:
        raise ValueError("동결된 문제 해시와 다릅니다.")
    problems = suite["problems"]
    if (len(problems) != 20 or Counter(p["level"] for p in problems) != {k: 4 for k in range(1, 6)}
            or len({p["id"] for p in problems}) != 20
            or len({p["question_en"] for p in problems}) != 20):
        raise ValueError("5단계 × 4개의 고유 문제 계약과 다릅니다.")
    for p in problems:
        rational(p["answer"])
        if not p["id"].startswith(f"D{p['level']}-"):
            raise ValueError("문제 ID와 난도 불일치")
    if (cfg["arms"] != list(ARMS) or cfg["seeds"] != [42, 43]
            or cfg["max_new_tokens"] != 4096 or cfg["max_prompt_tokens"] != 768
            or cfg["max_total_context_tokens"] != 4864
            or cfg["level_pass_min_stable_problems"] != 3
            or cfg["expected_attempts"] != 80 or cfg["maximum_generation_tokens"] != 327680
            or cfg["decoding"] != {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}):
        raise ValueError("v01의 고정 실행·집계 범위를 변경할 수 없습니다.")
    return cfg, suite


def prompt_for(suite, problem):
    # 답, 풀이, 난도, ID를 모델에 전달하지 않습니다.
    return problem["question_en"] + "\n\n" + suite["instruction"]


def expected_keys(cfg, suite):
    return {(p["id"], arm, seed) for p in suite["problems"]
            for arm in cfg["arms"] for seed in cfg["seeds"]}


def final_answer(text):
    """think 닫힘 뒤 단 하나의 boxed 숫자만 파싱. 모호함은 수동 검토로 남깁니다."""
    if text.count("</think>") != 1:
        return {"parse_status": "THINK_BOUNDARY_MISSING_OR_MULTIPLE", "answer": None}
    final = text.split("</think>", 1)[1]
    if "<think>" in final or final.count("\\boxed") != 1:
        return {"parse_status": "FINAL_BOX_MISSING_OR_MULTIPLE", "answer": None}
    match = re.search(r"\\boxed\s*\{", final)
    if not match:
        return {"parse_status": "FINAL_BOX_MALFORMED", "answer": None}
    start, depth, end = match.end(), 1, None
    for index in range(start, len(final)):
        depth += (final[index] == "{") - (final[index] == "}")
        if depth == 0:
            end = index
            break
    if end is None:
        return {"parse_status": "FINAL_BOX_MALFORMED", "answer": None}
    # 박스 뒤 답을 번복한 문장이나 다른 답이 있으면 자동 채점하지 않습니다.
    tail = final[end + 1:].strip().replace("\\]", "").replace("\\)", "")
    if tail.strip(" .$*\n\r\t"):
        return {"parse_status": "TEXT_AFTER_FINAL_BOX", "answer": None}
    try:
        value = rational(final[start:end])
    except (ValueError, ZeroDivisionError):
        return {"parse_status": "UNSUPPORTED_FINAL_VALUE", "answer": None}
    return {"parse_status": "PARSED", "answer": str(value)}


def score_attempt(row, problem, cfg):
    if row.get("finish_reason") == "error":
        return {"outcome": "ERROR", "strict_success": False, "observed_final_correct": None,
                "parse_status": "NOT_EVALUATED", "right_censored": None,
                "finish_reason": "error", "generated_tokens": None, "reasoning_tokens": None}
    term = termination_record(row["generated_ids"], {151643}, cfg["max_new_tokens"])
    if any(row.get(k) != v for k, v in term.items()):
        raise ValueError("생성 ID와 저장된 종료 필드가 다릅니다.")
    closes = [i for i, t in enumerate(row["generated_ids"]) if t == 151649]
    if len(closes) != row["generated_text"].count("</think>"):
        raise ValueError("생성 ID와 디코딩된 think 경계 수가 다릅니다.")
    parsed = final_answer(row["generated_text"])
    correct = (rational(parsed["answer"]) == rational(problem["answer"])) if parsed["answer"] else None
    if term["finish_reason"] == "length":
        outcome = "CENSORED"
    elif term["finish_reason"] != "eos":
        outcome = "ERROR"
    elif correct is None:
        outcome = "UNSCORABLE"
    else:
        outcome = "CORRECT" if correct else "INCORRECT"
    return {**parsed, "outcome": outcome, "observed_final_correct": correct,
            "strict_success": outcome == "CORRECT", "right_censored": term["right_censored"],
            "finish_reason": term["finish_reason"],
            "generated_tokens": len(row["generated_ids"]),
            "reasoning_tokens": closes[0] if len(closes) == 1 else None}


def summarize(report, cfg, suite):
    if report["config_sha256"] != canonical_hash(cfg) or report["suite_sha256"] != canonical_hash(suite):
        raise ValueError("실행과 평가의 문제/계획 해시가 다릅니다.")
    expected = expected_keys(cfg, suite)
    by_id = {p["id"]: p for p in suite["problems"]}
    rows, scores = {}, {}
    for row in report["attempts"]:
        key = (row["problem_id"], row["arm"], row["seed"])
        if key not in expected or key in rows:
            raise ValueError("계획 밖 또는 중복된 생성 기록입니다.")
        if row.get("finish_reason") != "error":
            if (canonical_hash(row["input_ids"]) != row["input_ids_sha256"]
                    or len(row["input_ids"]) != row["input_token_count"]
                    or row["prompt_sha256"] != canonical_hash(prompt_for(suite, by_id[key[0]]))):
                raise ValueError("입력/프롬프트 해시 불일치")
        rows[key] = row
        scores[key] = score_attempt(row, by_id[key[0]], cfg)
    for p in suite["problems"]:
        related = [r for k, r in rows.items() if k[0] == p["id"] and r.get("finish_reason") != "error"]
        if len({r["input_ids_sha256"] for r in related}) > 1:
            raise ValueError("동일 문제의 조건/시드 간 입력 ID가 다릅니다.")
    levels = []
    for level in range(1, 6):
        ids = [p["id"] for p in suite["problems"] if p["level"] == level]
        for arm in cfg["arms"]:
            keys = [(pid, arm, seed) for pid in ids for seed in cfg["seeds"]]
            present = [scores[k] for k in keys if k in scores]
            count = Counter(s["outcome"] for s in present)
            stable = [pid for pid in ids if all(scores.get((pid, arm, seed), {}).get("strict_success", False)
                                               for seed in cfg["seeds"])]
            complete = len(present) == len(keys) and all(s["outcome"] in {"CORRECT", "INCORRECT"} for s in present)
            verdict = ("PASS" if len(stable) >= cfg["level_pass_min_stable_problems"] else "FAIL") if complete else "INCONCLUSIVE"
            lengths = [s["generated_tokens"] for s in present if s["generated_tokens"] is not None]
            levels.append({"level": level, "arm": arm, "verdict": verdict, "expected_attempts": len(keys),
                           "observed_attempts": len(present), "missing_attempts": len(keys) - len(present),
                           "outcomes": dict(count), "stable_problems": len(stable), "stable_problem_ids": stable,
                           "strict_success_rate_over_planned": count["CORRECT"] / len(keys),
                           "eos_rate_over_planned": sum(s["finish_reason"] == "eos" for s in present) / len(keys),
                           "censored_rate_over_planned": count["CENSORED"] / len(keys),
                           "observed_final_correct_in_censored": sum(s["outcome"] == "CENSORED" and s["observed_final_correct"] is True for s in present),
                           "median_observed_tokens_including_censored": statistics.median(lengths) if lengths else None,
                           "natural_length_comparison_eligible": len(present) == len(keys) and all(s["finish_reason"] == "eos" for s in present)})
    pairs = []
    for p in suite["problems"]:
        for seed in cfg["seeds"]:
            b, a = (scores.get((p["id"], arm, seed)) for arm in cfg["arms"])
            eligible = bool(b and a and all(s["outcome"] in {"CORRECT", "INCORRECT"} for s in (b, a)))
            pairs.append({"problem_id": p["id"], "level": p["level"], "seed": seed,
                          "paired_final_comparison_eligible": eligible,
                          "bf16_correct_awq_incorrect": bool(eligible and b["strict_success"] and not a["strict_success"]),
                          "bf16_incorrect_awq_correct": bool(eligible and a["strict_success"] and not b["strict_success"]),
                          "both_incorrect": bool(eligible and not b["strict_success"] and not a["strict_success"]),
                          "both_correct": bool(eligible and b["strict_success"] and a["strict_success"]),
                          "natural_total_length_delta_awq_minus_bf16": a["generated_tokens"] - b["generated_tokens"] if b and a and b["finish_reason"] == a["finish_reason"] == "eos" else None,
                          "think_segment_delta_awq_minus_bf16": a["reasoning_tokens"] - b["reasoning_tokens"] if b and a and b["reasoning_tokens"] is not None and a["reasoning_tokens"] is not None else None})
    boundaries = {}
    for arm in cfg["arms"]:
        found = 0
        for row in (r for r in levels if r["arm"] == arm):
            if row["verdict"] != "PASS":
                break
            found = row["level"]
        passed = [r["level"] for r in levels if r["arm"] == arm and r["verdict"] == "PASS"]
        boundaries[arm] = {"highest_contiguous_pass_from_D1": found or None,
                           "all_pass_levels": passed,
                           "nonmonotonic_or_gapped_passes": any(x > found for x in passed)}
    throughput = {}
    for arm in cfg["arms"]:
        measured = [r for k, r in rows.items() if k[1] == arm and r.get("finish_reason") in {"eos", "length"}
                    and r.get("elapsed_seconds", 0) > 0]
        seconds = sum(r["elapsed_seconds"] for r in measured)
        tokens = sum(len(r["generated_ids"]) for r in measured)
        throughput[arm] = {"measured_attempts": len(measured), "elapsed_generation_seconds": seconds,
                           "generated_tokens_including_eos": tokens, "tokens_per_generation_second": tokens / seconds if seconds else None,
                           "scope": "batch=1 생성 호출 시간; 로딩/recipe 재적용/파일 저장 제외. native low-bit 성능 아님."}
    return {"status": "EXPLORATORY_DIFFICULTY_SUMMARY_NOT_CERTIFICATION", "model_ready": False,
            "raw_generations_included": False, "expected_attempts": len(expected),
            "observed_attempts": len(rows), "missing_attempts": len(expected - set(rows)),
            "levels": levels, "boundaries": boundaries, "pairs": pairs, "throughput": throughput,
            "manual_review_required": [dict(problem_id=k[0], arm=k[1], seed=k[2],
                                             outcome=s["outcome"], parse_status=s["parse_status"])
                                       for k, s in scores.items() if s["outcome"] not in {"CORRECT", "INCORRECT"}],
            "per_attempt_scores": [dict(problem_id=k[0], arm=k[1], seed=k[2], **s) for k, s in scores.items()],
            "interpretation": "자체 20문제의 기술적 탐색. 임시 난도 경계·소표본이며 동등성, MATH 재현, 정답 포기 또는 H1/H2의 판정이 아님."}
