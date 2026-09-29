"""AWQ 스케일 변환을 검사하는 NumPy 도구. 양자화기·모델 실행기가 아닙니다."""
from __future__ import annotations

import numpy as np


def apply_scale_ledger(parameters, ledger, *, inverse=False):
    """지원 범위: norm→linears, 같은 채널 수의 linear→linear. 항상 복사본 반환.

    역변환은 항목을 역순으로 적용합니다. clipping/rounding은 되돌리지 않습니다.
    실모델 BF16 오차 허용치나 함수 동등성을 이 함수의 성공으로 판정하지 않습니다.
    """
    result = {}
    for name, value in parameters.items():
        arr = np.asarray(value)
        if arr.dtype.kind != "f" or arr.ndim not in (1, 2) or not np.isfinite(arr).all():
            raise ValueError(f"유한한 실수 벡터·행렬이 필요합니다: {name}")
        result[name] = arr.astype(np.float64, copy=True)
    entries = list(ledger)
    for entry in reversed(entries) if inverse else entries:
        scales = np.asarray(entry["scales"], dtype=np.float64)
        if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("스케일은 유한한 양수 벡터여야 합니다.")
        previous, following = entry["previous"], entry["following"]
        if not isinstance(following, list) or not following or len(set(following)) != len(following):
            raise ValueError("뒤쪽 계층 목록은 중복 없이 명시해야 합니다.")
        if previous in following:
            raise ValueError("동일 계층을 변환 양쪽에 쓸 수 없습니다.")
        weight = result[previous + ".weight"]
        for name in following:
            w = result[name + ".weight"]
            if w.ndim != 2 or w.shape[1] != len(scales):
                raise ValueError("뒤쪽 선형 계층의 입력 채널 수가 다릅니다.")
        if entry["kind"] == "norm_linears":
            if weight.shape != scales.shape:
                raise ValueError("정규화 계층 채널 수가 다릅니다.")
            divisor = scales
        elif entry["kind"] == "linear_linear":
            if len(following) != 1 or weight.ndim != 2 or weight.shape[0] != len(scales):
                raise ValueError("현재 도구는 전체 출력 채널이 일치하는 선형 쌍만 지원합니다.")
            divisor = scales[:, None]
        else:
            raise ValueError("지원하지 않는 스케일 변환입니다.")
        result[previous + ".weight"] = weight * divisor if inverse else weight / divisor
        bias_key = previous + ".bias"
        if bias_key in result:
            if result[bias_key].shape != scales.shape:
                raise ValueError("앞쪽 bias 채널 수가 다릅니다.")
            result[bias_key] = result[bias_key] * scales if inverse else result[bias_key] / scales
        for name in following:
            key = name + ".weight"
            result[key] = result[key] / scales[None, :] if inverse else result[key] * scales[None, :]
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("변환 중 비유한 값이 발생했습니다.")
    return result


def canonical_error(reference, transformed_quantized, ledger, *, weight_names):
    """동일 BF16 기준 좌표로 복원한 뒤 선택된 weight들의 차이를 계산합니다."""
    if not weight_names or len(set(weight_names)) != len(weight_names):
        raise ValueError("비교할 weight 이름을 중복 없이 명시해야 합니다.")
    if set(reference) != set(transformed_quantized):
        raise ValueError("파라미터 키가 다릅니다.")
    canonical = apply_scale_ledger(transformed_quantized, ledger, inverse=True)
    rows = []
    for name in weight_names:
        if not name.endswith(".weight"):
            raise ValueError("weight 목록에 bias 또는 다른 항목이 포함됐습니다.")
        a = np.asarray(reference[name], dtype=np.float64)
        b = canonical[name]
        if a.shape != b.shape or not np.isfinite(a).all():
            raise ValueError("참조 shape 또는 수치가 유효하지 않습니다.")
        delta = b - a
        rms = float(np.sqrt(np.mean(delta * delta)))
        ref_rms = float(np.sqrt(np.mean(a * a)))
        if not np.isfinite(rms) or not np.isfinite(ref_rms):
            raise ValueError("오차 통계가 유한하지 않습니다.")
        rows.append({"parameter": name, "elements": int(a.size), "rms_error": rms,
                     "relative_rms_error": rms / ref_rms if ref_rms > 0 else None})
    return {"scope": "좌표 변환 수치 검사; 실모델·양자화기 검증 결과 아님", "parameters": rows}
