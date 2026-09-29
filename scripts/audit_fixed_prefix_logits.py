"""저장된 fixed_prefix_v01 logits만 읽는 CPU 감사. 모델·생성·GPU·네트워크 호출 없음."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash
from fixed_prefix_diagnostic_contracts import CONFIG, PAIRS, STATES, compare_logits

REVIEW = ROOT / "configs/fixed_prefix_result_review_v01.json"
# 같은 저장 배열에서 JSON 수치를 재계산할 때만 사용. 모델 동등성 허용오차가 아님.
RECALC_RTOL, RECALC_ATOL = 1e-10, 1e-12
TEMPERATURE, TOP_P = 0.6, 0.95


def check_equal(expected, actual, path="comparison"):
    """범주/구조는 정확히, 부동소수점 집계는 명시한 재계산 오차 내에서 대조."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"집계 필드 불일치: {path}")
        for key in expected:
            check_equal(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"집계 길이 불일치: {path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            check_equal(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not math.isfinite(expected)
                or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=RECALC_RTOL, abs_tol=RECALC_ATOL)):
            raise ValueError(f"수치 재계산 불일치: {path}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"집계 값 불일치: {path}")


def validate_run(path, review, cfg):
    if file_hash(path) != review["source_run_sha256"]:
        raise ValueError("검토한 fixed_prefix_v01/run.json과 바이트 해시가 다릅니다.")
    run = json.loads(path.read_text(encoding="utf-8"))
    if (run["status"] != "FIXED_PREFIX_DIAGNOSTICS_COLLECTED_PENDING_REVIEW"
            or run["config_sha256"] != canonical_hash(cfg)
            or run["config_sha256"] != review["config_sha256"]
            or run["repository"]["commit"] != review["execution_commit"]
            or run["research_status_sha256"] != review["execution_research_status_sha256"]
            or run["inputs"] != cfg["inputs"]):
        raise ValueError("수신 결과·실행 당시 설정/상태·입력 참조가 일치하지 않습니다.")
    for name, sha in run["implementation_sha256"].items():
        if file_hash(ROOT / "scripts" / name) != sha:
            raise ValueError(f"수집 당시 구현이 변경됐습니다: {name}")
    expected = [(state, r["problem_id"]) for state in STATES for r in cfg["inputs"]]
    if [(r["state"], r["problem_id"]) for r in run["forward_records"]] != expected:
        raise ValueError("12개 원시 배열의 순서·개수가 다릅니다.")
    expected_pairs = [(a, b, r["problem_id"]) for a, b in PAIRS for r in cfg["inputs"]]
    if [(r["reference"], r["observed"], r["problem_id"]) for r in run["comparisons"]] != expected_pairs:
        raise ValueError("10개 비교의 순서·개수가 다릅니다.")
    for rec in run["forward_records"]:
        inp = next(r for r in cfg["inputs"] if r["problem_id"] == rec["problem_id"])
        if (rec["file"] != f"logits_{rec['state']}_{rec['problem_id']}.npy"
                or rec["shape"] != [len(inp["logit_positions"]), cfg["vocab_size"]]
                or rec["logit_positions"] != inp["logit_positions"]
                or rec["stored_dtype"] != "float32"):
            raise ValueError("원시 배열의 파일명·shape·위치·dtype 계약 불일치")
    return run


def load_logits(folder, rec):
    path = folder / rec["file"]
    if path.name != rec["file"] or path.is_symlink():
        raise ValueError("같은 폴더의 일반 .npy 파일만 읽습니다.")
    if file_hash(path) != rec["file_sha256"]:
        raise ValueError(f"원시 logit 바이트 해시 불일치: {path.name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (list(array.shape) != rec["shape"] or array.dtype != np.dtype("float32")
            or array.ndim != 2 or not np.isfinite(array).all()):
        raise ValueError(f"원시 logit shape·dtype·유한성 불일치: {path.name}")
    return array


def filtered_distribution(logits, eos, temperature=TEMPERATURE, top_p=TOP_P):
    """float64 수학적 재구성. GPU FP32 softmax/정렬/실제 RNG 재현이 아님.

    오름차순 확률 누적합 <= 1-top_p를 제거하고 최소1개 유지한다.
    동점은 token ID 오름차순으로 정렬하며 잘리는 동점 집합을 별도로 기록한다.
    """
    x = np.asarray(logits, dtype=np.float64)
    if (x.ndim != 1 or x.size < 2 or not np.isfinite(x).all()
            or not 0 <= eos < x.size or not temperature > 0 or not 0 < top_p <= 1):
        raise ValueError("분포 진단 입력·온도·top-p 오류")
    shifted = (x - x.max()) / temperature
    weights = np.exp(shifted)
    probability = weights / weights.sum()
    log_probability = shifted - np.log(weights.sum())
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(probability[order], dtype=np.float64)
    keep_sorted = cumulative > (1 - top_p)
    keep_sorted[-1] = True
    keep = np.zeros(x.size, dtype=bool)
    keep[order[keep_sorted]] = True
    mass = float(probability[keep].sum())
    filtered = np.where(keep, probability / mass, 0.0)
    boundary = float(x[keep].min())
    tied = x == boundary
    tied_count, tied_kept = int(tied.sum()), int((tied & keep).sum())
    info = {
        "retained_count": int(keep.sum()), "retained_full_probability_mass": mass,
        "boundary_logit": boundary, "boundary_tie_count": tied_count,
        "boundary_tie_kept": tied_kept, "split_boundary_tie": tied_kept != tied_count,
        "minimum_cumulative_distance_to_cutoff": float(np.abs(cumulative - (1 - top_p)).min()),
        "eos_rank_best": int((x > x[eos]).sum()) + 1,
        "eos_rank_worst": int((x >= x[eos]).sum()),
        "eos_full_log_probability": float(log_probability[eos]),
        "eos_full_probability": float(probability[eos]),
        "eos_retained": bool(keep[eos]), "eos_filtered_probability": float(filtered[eos]),
        "eos_boundary_tie_sensitive": bool(tied[eos] and tied_kept != tied_count),
    }
    return filtered, info


def compare_filtered(reference, observed, eos):
    p, a = filtered_distribution(reference, eos)
    q, b = filtered_distribution(observed, eos)
    p_support, q_support = p > 0, q > 0
    missing = p_support & ~q_support
    reverse_missing = q_support & ~p_support
    middle = (p + q) / 2
    js = float(0.5 * (np.sum(p[p_support] * np.log(p[p_support] / middle[p_support]))
                      + np.sum(q[q_support] * np.log(q[q_support] / middle[q_support]))))
    singular = bool(missing.any())
    kl = None if singular else float(np.sum(p[p_support] * np.log(p[p_support] / q[p_support])))
    x, y = np.asarray(reference), np.asarray(observed)
    top_x, top_y = int(x.argmax()), int(y.argmax())
    sorted_x, sorted_y = np.partition(x, -2)[-2:], np.partition(y, -2)[-2:]
    return {
        "reference": a, "observed": b,
        "total_variation": float(np.abs(p - q).sum() / 2), "jensen_shannon_nats": js,
        "kl_reference_to_observed": kl, "kl_is_infinite_due_to_support": singular,
        "reference_support_missing_in_observed": int(missing.sum()),
        "observed_support_missing_in_reference": int(reverse_missing.sum()),
        "reference_mass_outside_observed_support": float(p[missing].sum()),
        "observed_mass_outside_reference_support": float(q[reverse_missing].sum()),
        "reference_argmax": top_x, "observed_argmax": top_y,
        "reference_margin": float(sorted_x[1] - sorted_x[0]),
        "observed_margin": float(sorted_y[1] - sorted_y[0]),
        "reference_maximizer_count": int((x == x.max()).sum()),
        "observed_maximizer_count": int((y == y.max()).sum()),
        "observed_argmax_is_reference_maximizer": bool(x[top_y] == x.max()),
        "reference_argmax_is_observed_maximizer": bool(y[top_x] == y.max()),
    }


def audit_arrays(run, folder, cfg, progress=print):
    arrays = {}
    for i, rec in enumerate(run["forward_records"], 1):
        arrays[rec["state"], rec["problem_id"]] = load_logits(folder, rec)
        progress(f"원시 배열 해시·shape 확인 {i}/12")
    results = []
    for i, old in enumerate(run["comparisons"], 1):
        a, b, problem = old["reference"], old["observed"], old["problem_id"]
        x, y = arrays[a, problem], arrays[b, problem]
        positions = next(r["logit_positions"] for r in cfg["inputs"] if r["problem_id"] == problem)
        computed = compare_logits(x, y, positions, cfg["eos_token_id"])
        expected = {k: v for k, v in old.items() if k not in {"reference", "observed", "problem_id"}}
        check_equal(expected, computed, f"{a}/{b}/{problem}")
        rows = [{"position": pos, **compare_filtered(x[j], y[j], cfg["eos_token_id"])}
                for j, pos in enumerate(positions)]
        result = {
            "reference": a, "observed": b, "problem_id": problem,
            "t1_recomputed_matches_report": True,
            "t1_summary": {k: v for k, v in computed.items() if k != "per_position"},
            "filtered_float64_summary": {
                "positions": len(rows),
                "mean_total_variation": float(np.mean([r["total_variation"] for r in rows])),
                "max_total_variation": max(r["total_variation"] for r in rows),
                "mean_jensen_shannon_nats": float(np.mean([r["jensen_shannon_nats"] for r in rows])),
                "infinite_kl_positions": sum(r["kl_is_infinite_due_to_support"] for r in rows),
                "reference_eos_retained_positions": sum(r["reference"]["eos_retained"] for r in rows),
                "observed_eos_retained_positions": sum(r["observed"]["eos_retained"] for r in rows),
                "boundary_tie_sensitive_positions": sum(r["reference"]["split_boundary_tie"]
                                                        or r["observed"]["split_boundary_tie"] for r in rows),
            },
            "per_position": rows,
        }
        results.append(result)
        progress(f"T=1 재계산·T=0.6/top-p=0.95 탐색 {i}/10: {a}/{b} {problem}")
    # 실행 중 원본 파일이 바뀌었으면 완료 보고서를 만들지 않습니다.
    for rec in run["forward_records"]:
        if file_hash(folder / rec["file"]) != rec["file_sha256"]:
            raise ValueError("감사 중 원시 logit 파일이 변경됐습니다.")
    return results


def write_report(path, report):
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/local/fixed_prefix_v01/run.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local/fixed_prefix_v01/logits_audit_v01.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("기존 감사 결과를 보존합니다. 파일을 지우거나 자동 반복하지 마세요.")
    if not args.output.parent.is_dir():
        raise FileNotFoundError("기존 결과 폴더가 필요합니다. 모델 실행으로 만들지 마세요.")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    run = validate_run(args.run, review, cfg)
    script_sha = file_hash(Path(__file__))
    comparisons = audit_arrays(run, args.run.parent, cfg, progress=lambda s: print(s, flush=True))
    if file_hash(args.run) != review["source_run_sha256"]:
        raise ValueError("감사 중 run.json이 변경됐습니다.")
    report = {
        "schema_version": 1, "status": "SAVED_LOGITS_AUDITED_FLOAT64_FILTER_DIAGNOSTICS",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_sha256": file_hash(args.run), "review_reference_sha256": file_hash(REVIEW),
        "script_sha256": script_sha, "implementation_sha256": run["implementation_sha256"],
        "numpy_version": np.__version__, "raw_logit_files_verified": 12,
        "raw_logit_hashes": [{k: r[k] for k in ("file", "file_sha256", "shape", "stored_dtype")}
                             for r in run["forward_records"]],
        "t1_recalculation": {"comparisons": 10, "all_match": True,
                             "rtol": RECALC_RTOL, "atol": RECALC_ATOL,
                             "is_functional_equivalence_tolerance": False},
        "filtered_analysis": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": 0,
                              "repetition_penalty": 1.0, "min_tokens_to_keep": 1,
                              "dtype": "float64", "tie_order": "ascending_token_id",
                              "is_original_gpu_sampling_replay": False,
                              "scope_ko": "기존 고정 prefix logits의 수학적 필터 진단. GPU FP32 연산·불안정 동점 정렬·RNG·cache·실제 생성 경로의 재현 아님."},
        "comparisons": comparisons, "new_forward_calls": 0, "new_generation_tokens": 0,
        "gpu_or_network_used": False, "original_files_modified": False,
        "model_ready": False, "functional_equivalence_verdict": "NOT_CERTIFIED",
    }
    write_report(args.output, report)
    print(f"저장 logits 감사 완료: {args.output.name}. GPU/새 생성 0, 기능적 동등성 미인증.")


if __name__ == "__main__":
    main()
