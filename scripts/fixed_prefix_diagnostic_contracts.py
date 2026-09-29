"""고정 prefix 진단의 근거·예산 계약과 수치 요약. 모델 import 없음."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash

CONFIG = ROOT / "configs/fixed_prefix_diagnostic_v01.json"
STATES = ["B", "B_repeat", "S", "Q", "C", "Cw"]
PAIRS = [["B", "B_repeat"], ["B", "S"], ["B", "Q"], ["Q", "C"], ["C", "Cw"]]


def read_inputs(evidence, mask_report, cfg):
    """첨부와 동일한 기존 파일에서만 입력을 복사. 토큰 재생성·토크나이저 없음."""
    for path, key in ((evidence, "evidence_packet_sha256"), (mask_report, "mask_report_sha256")):
        if file_hash(path) != cfg[key]:
            raise ValueError(f"검토한 근거와 바이트 해시 불일치: {key}")
    packet = json.loads(Path(evidence).read_text(encoding="utf-8"))
    mask = json.loads(Path(mask_report).read_text(encoding="utf-8"))
    if mask["status"] != "DISABLED_FLAG_WINDOW_MASK_OBSERVED":
        raise ValueError("검토한 CPU 마스크 결과가 아닙니다.")
    source = packet["cached_model_config"]["configuration"]
    effective = copy.deepcopy(source)
    effective["sliding_window"] = None
    if (canonical_hash(source) != cfg["model_configuration_sha256"]
            or canonical_hash(effective) != cfg["effective_source_configuration_sha256"]
            or source["use_sliding_window"] is not False or source["sliding_window"] != 4096):
        raise ValueError("원본/진단용 모델 설정 불일치")
    if (cfg["states"] != STATES or cfg["comparisons"] != PAIRS
            or cfg["maximum_forward_calls"] != 12 or cfg["new_generation_tokens"] != 0
            or cfg["use_cache"] is not False or cfg["effective_sliding_window"] is not None
            or cfg["calibration_search"] or cfg["network_download"] or cfg["automatic_retry"]):
        raise ValueError("고정 진단 범위가 변경됐습니다.")
    if [(r["problem_id"], r["arm"], r["seed"]) for r in cfg["inputs"]] != [
            ("D2-03", "bf16", 42), ("D3-01", "bf16", 42)]:
        raise ValueError("고정 탐색 입력 불일치")
    inputs = []
    for ref in cfg["inputs"]:
        selected = [a for a in packet["run_snapshot"]["attempts"] if
                    all(a[k] == ref[k] for k in ("problem_id", "arm", "seed"))]
        if len(selected) != 1 or canonical_hash(selected[0]) != ref["attempt_sha256"]:
            raise ValueError("고정 응답 누락·중복·변경")
        row = selected[0]
        ids = row["input_ids"] + row["generated_ids"][:2048]
        positions = [len(row["input_ids"]) - 1 + k for k in range(0, 2049, 16)]
        if (len(ids) != ref["sequence_length"] or len(row["input_ids"]) != ref["input_token_count"]
                or ref["generated_prefix_tokens"] != 2048 or ref["logit_positions"] != positions
                or canonical_hash(ids) != ref["sequence_ids_sha256"]
                or any(type(x) is not int or not 0 <= x < cfg["vocab_size"] for x in ids)
                or cfg["eos_token_id"] in row["generated_ids"][:2048]):
            raise ValueError("고정 prefix/위치/토큰 계약 불일치")
        inputs.append({**ref, "ids": ids})
    if (sum(x["sequence_length"] for x in inputs) * len(STATES) != cfg["maximum_forward_input_tokens"]
            or max(x["sequence_length"] for x in inputs) != cfg["maximum_sequence_length"]
            or cfg["maximum_sequence_length"] >= 4096):
        raise ValueError("고정 forward 예산·문맥 상한 불일치")
    return inputs, source, effective


def require_authorization(status, cfg):
    scope = status.get("fixed_prefix_diagnostic", {})
    if (status.get("research_approval") != "APPROVED"
            or status.get("novelty_gate") != "SCOPED_CONTRIBUTION_ACCEPTED"
            or status.get("protocol_gate") != "ACCEPTED"
            or scope.get("authorization") != "APPROVED"
            or scope.get("config_sha256") != canonical_hash(cfg)):
        raise ValueError("새 고정 prefix GPU 진단은 아직 실행 승인 전입니다. 기존 R0/난도 시험 승인을 재사용하지 않습니다.")


def difference_stats(reference, observed):
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("비교 배열의 shape·유한성 오류")
    d = b - a
    rms, base = float(np.sqrt(np.mean(d * d))), float(np.sqrt(np.mean(a * a)))
    index = np.unravel_index(int(np.abs(d).argmax()), d.shape)
    return {"elements": int(a.size), "rms": rms, "reference_rms": base,
            "relative_rms": rms / base if base else None,
            "max_abs": float(np.abs(d[index])), "max_abs_index": [int(x) for x in index],
            "exact_equal": bool(np.array_equal(a, b))}


def compare_logits(reference, observed, positions, eos):
    """전체 vocabulary의 T=1 분포. 입력 logits는 BF16→FP32, 계산은 float64."""
    import numpy as np
    a, b = np.asarray(reference), np.asarray(observed)
    if (a.ndim != 2 or a.shape != b.shape or a.shape[0] != len(positions)
            or not 0 <= eos < a.shape[1] or a.shape[1] < 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("logit shape·위치·유한성 오류")
    rows = []
    for i, position in enumerate(positions):
        x, y = a[i].astype(np.float64), b[i].astype(np.float64)
        stats = difference_stats(x, y)
        def log_softmax(z):
            shifted = z - z.max()
            return shifted - np.log(np.exp(shifted).sum())
        lx, ly = log_softmax(x), log_softmax(y)
        px, py = np.exp(lx), np.exp(ly)
        first, second = np.partition(x, -2)[-2:]
        top_a, top_b = int(x.argmax()), int(y.argmax())
        margin = float(second - first)
        # 이 부등식은 구현 일관성 검사. 동등성의 임의 허용 한도가 아닙니다.
        bound = margin > 2 * stats["max_abs"]
        if bound and top_a != top_b:
            raise ValueError("argmax margin 경계와 관측이 모순됩니다.")
        rows.append({"position": position, **stats,
                     "reference_top1": top_a, "observed_top1": top_b,
                     "top1_changed": top_a != top_b, "reference_top1_top2_margin": margin,
                     "margin_guarantees_same_top1": bound,
                     "kl_reference_to_observed": float(np.sum(px * (lx - ly))),
                     "total_variation": float(0.5 * np.abs(px - py).sum()),
                     "reference_eos_log_probability": float(lx[eos]),
                     "observed_eos_log_probability": float(ly[eos]),
                     "eos_log_probability_delta": float(ly[eos] - lx[eos]),
                     "eos_probability_delta": float(py[eos] - px[eos])})
    exact = all(r["exact_equal"] for r in rows)
    flips = sum(r["top1_changed"] for r in rows)
    label = ("EXACT_LOGITS_ON_TESTED_POSITIONS" if exact else
             "TOP1_CHANGED_ON_FIXED_PREFIX" if flips else "LOGITS_CHANGED_TOP1_UNCHANGED")
    return {"status": label, "positions": len(rows), "exact_equal": exact,
            "top1_changed_count": flips,
            "rms": float(np.sqrt(np.mean([r["rms"] ** 2 for r in rows]))),
            "max_abs": max(r["max_abs"] for r in rows),
            "mean_kl": float(np.mean([r["kl_reference_to_observed"] for r in rows])),
            "max_total_variation": max(r["total_variation"] for r in rows),
            "max_abs_eos_log_probability_delta": max(abs(r["eos_log_probability_delta"]) for r in rows),
            "per_position": rows, "model_equivalence_certified": False}
