"""기존 R0 JSON만 읽어 좌표 오차·생성 ID의 종료 계약을 요약합니다.

모델/토크나이저 로딩, 네트워크, 추론, 파일 쓰기를 수행하지 않습니다.
출력은 기능적 동등성 인증이 아니며 원본 파일과 별도로 검토해야 합니다.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

EXPECTED_LAYERS = 28
BUDGET = 128
EOS = 151643
LINEARS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
DIAGNOSTICS = ("scale_only_output_difference", "canonicalized_output_difference")


def reject_constant(value):
    raise ValueError(f"비유한 JSON 상수: {value}")


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw, parse_constant=reject_constant), hashlib.sha256(raw).hexdigest()


def number(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"유한한 음이 아닌 수가 필요합니다: {label}")
    return value


def error_rows(rows, layer, *, relative):
    result = []
    names = set()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"계층 {layer}: 파라미터 오차 목록 누락")
    for row in rows:
        name, count = row["parameter"], row["elements"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"계층 {layer}: 파라미터 이름 누락/중복")
        names.add(name)
        if type(count) is not int or count < 1:
            raise ValueError(f"계층 {layer}: 파라미터 원소 수 오류")
        item = {"layer": layer, "parameter": name, "elements": count,
                "rms_error": number(row["rms_error"], name)}
        if relative:
            item["relative_rms_error"] = number(
                row["relative_rms_error"], name, nullable=True)
        result.append(item)
    return result


def group_errors(rows, *, relative):
    groups = {}
    for row in rows:
        groups.setdefault(row["parameter"], []).append(row)
    summaries = []
    for name, items in sorted(groups.items()):
        count = sum(x["elements"] for x in items)
        # 텐서별 RMS의 단순 평균은 전체 원소 RMS가 아닙니다.
        pooled = math.hypot(*(math.sqrt(x["elements"] / count) * x["rms_error"]
                              for x in items))
        worst = max(items, key=lambda x: x["rms_error"])
        summary = {"parameter": name, "records": len(items), "elements": count,
                   "element_weighted_rms_error": pooled,
                   "min_rms_error": min(x["rms_error"] for x in items),
                   "max_rms_error": worst["rms_error"], "max_rms_layer": worst["layer"],
                   "nonzero_layers": [x["layer"] for x in items if x["rms_error"] > 0]}
        if relative:
            valid = [x for x in items if x["relative_rms_error"] is not None]
            peak = max(valid, key=lambda x: x["relative_rms_error"]) if valid else None
            summary.update(max_relative_rms_error=peak["relative_rms_error"] if peak else None,
                           max_relative_rms_layer=peak["layer"] if peak else None,
                           undefined_relative_layers=[x["layer"] for x in items
                                                      if x["relative_rms_error"] is None])
        summaries.append(summary)
    return summaries


def termination_summary(row, issues):
    label = f"{row.get('arm')}/{row.get('fixture_id')}"
    result = {k: row.get(k) for k in ("arm", "fixture_id", "seed", "input_token_count")}
    ids = row.get("generated_ids")
    if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
        issues.append(f"{label}: 생성 ID 누락/형식 오류")
        return result
    eos_positions = [i for i, token in enumerate(ids) if token == EOS]
    final_eos = ids[-1] == EOS
    if len(ids) > BUDGET or any(i < len(ids) - 1 for i in eos_positions):
        issues.append(f"{label}: 예산 초과 또는 EOS 이후 토큰 존재")
    recomputed = {
        "generated_token_count_including_eos": len(ids),
        "finish_reason": "eos" if final_eos else "length" if len(ids) == BUDGET else "unexpected_stop",
        "budget_reached": len(ids) == BUDGET,
        "right_censored": len(ids) == BUDGET and not final_eos,
    }
    mismatch = [k for k, v in recomputed.items() if k not in row or type(row[k]) is not type(v) or row[k] != v]
    if mismatch:
        issues.append(f"{label}: 종료 필드와 ID 재집계 불일치: {','.join(mismatch)}")
    if recomputed["finish_reason"] == "unexpected_stop":
        issues.append(f"{label}: EOS도 예산 종료도 아님")
    inputs = row.get("input_ids")
    if (not isinstance(inputs, list) or not inputs
            or any(type(t) is not int or t < 0 for t in inputs)
            or len(inputs) != row.get("input_token_count")):
        issues.append(f"{label}: 입력 ID/길이 기록 오류")
    result.update(recomputed, recorded_fields_match=not mismatch,
                  eos_positions_zero_based=eos_positions,
                  think_close_positions_zero_based=[i for i, t in enumerate(ids) if t == 151649])
    return result


def summarize(folder):
    run, run_sha = read_json(folder / "run.json")
    paths = sorted(folder.glob("awq_layer_*.json"))
    issues, layers, weights, auxiliary = [], [], [], []
    sources = [{"file": "run.json", "sha256": run_sha}]
    diagnostics = {key: [] for key in DIAGNOSTICS}
    expected_weights = {name + ".weight" for name in LINEARS}
    for path in paths:
        row, sha = read_json(path)
        sources.append({"file": path.name, "sha256": sha})
        index = row["layer"]
        if type(index) is not int or index < 0:
            raise ValueError(f"계층 번호 형식 오류: {path.name}")
        layers.append(index)
        if path.name != f"awq_layer_{index:03d}.json":
            issues.append(f"{path.name}: 내부 계층 번호와 파일명이 다름")
        if sorted(row["quantized_linears"]) != sorted(LINEARS):
            issues.append(f"계층 {index}: 양자화 선형 계층 범위 불일치")
        w = error_rows(row["canonical_weight_error"], index, relative=True)
        a = error_rows(row["canonical_auxiliary_error"], index, relative=False)
        if {x["parameter"] for x in w} != expected_weights:
            issues.append(f"계층 {index}: 가중치 오차 대상 불일치")
        if any(x["parameter"] in expected_weights for x in a):
            issues.append(f"계층 {index}: 가중치가 보조 잔차 목록에 중복")
        if not {"input_layernorm.weight", "post_attention_layernorm.weight"}.issubset(
                {x["parameter"] for x in a}):
            issues.append(f"계층 {index}: norm 잔차 누락")
        weights.extend(w)
        auxiliary.extend(a)
        for key in DIAGNOSTICS:
            metric = row[key]
            diagnostics[key].append({"layer": index, **{
                k: number(metric[k], f"{index}/{key}/{k}", nullable=k == "relative_rms")
                for k in ("rms", "max_abs", "relative_rms")}})
    missing = sorted(set(range(EXPECTED_LAYERS)) - set(layers))
    unexpected = sorted(set(layers) - set(range(EXPECTED_LAYERS)))
    duplicates = sorted(k for k, n in Counter(layers).items() if n != 1)
    if missing or unexpected or duplicates or run.get("completed_awq_layers") != EXPECTED_LAYERS:
        issues.append("계층 0~27의 유일한 파일 집합 또는 완료 계층 수 불일치")
    by_layer_aux = [{x["parameter"] for x in auxiliary if x["layer"] == i} for i in layers]
    if by_layer_aux and any(names != by_layer_aux[0] for names in by_layer_aux):
        issues.append("계층별 norm/bias 잔차 대상 목록이 다름")
    attempts = run["attempts"]
    identities = [(x.get("arm"), x.get("fixture_id"), x.get("seed")) for x in attempts]
    expected = {(arm, fixture, 42) for arm in ("bf16", "awq_w3") for fixture in (0, 1)}
    if len(identities) != 4 or set(identities) != expected:
        issues.append("BF16/AWQ 각 2개 fixture·seed 42 기록 불일치")
    generation = [termination_summary(row, issues) for row in attempts]
    for fixture in (0, 1):
        pair = [x for x in attempts if x.get("fixture_id") == fixture]
        if len(pair) == 2 and pair[0].get("input_ids") != pair[1].get("input_ids"):
            issues.append(f"fixture {fixture}: BF16/AWQ 입력 ID가 다름")
    if run.get("model_ready") is not False:
        issues.append("R0 model_ready=false 기록과 다름")
    if run.get("status") != "R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW":
        issues.append("R0 실행 완료 보고 상태와 다름")
    if run.get("awq_status") != "AWQ_TRANSFORM_APPLIED_NOT_EQUIVALENCE_CERTIFIED":
        issues.append("AWQ 변환 완료 보고 상태와 다름")
    peaks = {}
    for key, rows in diagnostics.items():
        valid = [x for x in rows if x["relative_rms"] is not None]
        peaks[key] = {
            "max_relative_rms_record": max(valid, key=lambda x: x["relative_rms"]) if valid else None,
            "max_abs_record": max(rows, key=lambda x: x["max_abs"]) if rows else None,
            "undefined_relative_layers": [x["layer"] for x in rows if x["relative_rms"] is None],
        }
    return {
        "status": "READ_ONLY_R0_SUMMARY_NOT_CERTIFICATION",
        "source_run": {k: run.get(k) for k in (
            "status", "repository", "completed_awq_layers", "awq_status", "model_ready")},
        "source_files": sources,
        "layer_coverage": {"expected": EXPECTED_LAYERS, "files": len(paths),
                           "missing": missing, "unexpected": unexpected, "duplicates": duplicates},
        "structural_issues": issues,
        "output_difference_peaks": peaks,
        "canonical_weight_error": group_errors(weights, relative=True),
        "canonical_auxiliary_error": group_errors(auxiliary, relative=False),
        "generation_id_review": generation,
        "interpretation_limits": [
            "저장된 집계를 요약한 결과이며 가중치·활성값으로 오차를 다시 계산하지 않음",
            "가중치·norm/bias 오차는 float64 역변환 배열 기준이며 BF16 재적용 전의 값",
            "계층 출력 차이는 BF16 재적용 모델에서 측정된 기존 기록",
            "보조 파라미터의 참조 RMS·최대 절대 잔차·위치는 기존 JSON에 없음",
            "전체 모델 logit·실모델 동등성·정확도·정답 포기·길이 증가를 판정하지 않음",
            "빈 structural_issues 목록은 기능적 동등성 PASS가 아님",
            "생성 텍스트·입출력 ID·scales/clips 원시 배열을 출력하지 않음",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("results/local/r0_v03"))
    args = parser.parse_args()
    try:
        result = summarize(args.result_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result["structural_issues"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"기존 R0 JSON 읽기 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
