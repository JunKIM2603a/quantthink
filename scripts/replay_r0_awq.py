"""기존 R0 파일만 읽어 저장된 scale/clip을 재적용합니다. 탐색 함수는 호출하지 않습니다."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct

from difficulty_pilot_contracts import ROOT, file_hash

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def verify_r0_files(folder, review):
    folder = Path(folder)
    expected = {"run.json"} | {f"awq_layer_{i:03d}.json" for i in range(28)}
    records = review["source_files"]
    if len(records) != 29 or {r["file"] for r in records} != expected:
        raise ValueError("검토된 R0 입력 참조에 29개 파일이 필요합니다.")
    if {p.name for p in folder.glob("awq_layer_*.json")} != expected - {"run.json"}:
        raise ValueError("R0 계층 파일 범위가 0~27과 다릅니다.")
    for row in records:
        path = folder / row["file"]
        if path.is_symlink() or file_hash(path) != row["sha256"]:
            raise ValueError(f"검토 참조와 R0 파일 SHA-256 불일치: {row['file']}")
    run = json.loads((folder / "run.json").read_text())
    if (run["repository"]["commit"] != review["source_run_commit"]
            or run["status"] != "R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW"
            or run["completed_awq_layers"] != 28):
        raise ValueError("R0 실행 근거가 검토 참조와 다릅니다.")
    return run, records


def _bf16_value(value, *, positive):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError("scale/clip의 유한성 또는 부호 오류")
    # R0의 BF16 입력·가중치에서 나온 scale/clip이라는 재구성 가정을 검사합니다.
    try:
        encoded = struct.pack(">f", value)
    except OverflowError as exc:
        raise ValueError("BF16 범위 밖 recipe 값") from exc
    if struct.unpack(">f", encoded)[0] != value or encoded[-2:] != b"\0\0":
        raise ValueError("저장 recipe 값은 BF16로 정확히 복원되지 않습니다. dtype을 추정해 진행하지 않습니다.")


def validate_recipe(row, index, shapes):
    """torch 없이도 경로·shape·저장값 dtype의 재구성 가능성을 확인합니다."""
    if row["layer"] != index or row["quantized_linears"] != list(LINEARS):
        raise ValueError("R0 계층 ID 또는 선형 계층 범위 불일치")
    paths = [("norm_linears", "input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])]
    if shapes["self_attn.v_proj.weight"] == shapes["self_attn.o_proj.weight"]:
        paths.append(("linear_linear", "self_attn.v_proj", ["self_attn.o_proj"]))
    paths += [("norm_linears", "post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
              ("linear_linear", "mlp.up_proj", ["mlp.down_proj"])]
    if [(r["kind"], r["previous"], r["following"]) for r in row["scales"]] != paths:
        raise ValueError("스케일 순서·경로가 고정 Qwen2와 다릅니다.")
    for scale in row["scales"]:
        values = scale["scales"]
        if len(values) != shapes[scale["previous"] + ".weight"][0]:
            raise ValueError("스케일 이전 모듈 shape 오류")
        for name in scale["following"]:
            if shapes[name + ".weight"][1] != len(values):
                raise ValueError("스케일 다음 모듈 shape 오류")
        for value in values:
            _bf16_value(value, positive=True)
    clip_names = set(LINEARS) - {"self_attn.q_proj", "self_attn.k_proj"}
    if len(row["clips"]) != 5 or {c["parameter"] for c in row["clips"]} != clip_names:
        raise ValueError("clipping 대상이 5개 계층과 다릅니다.")
    for clip in row["clips"]:
        out_features, in_features = shapes[clip["parameter"] + ".weight"]
        if in_features % 128 or clip["shape"] != [out_features, in_features // 128, 1]:
            raise ValueError("clipping g128 shape 불일치")
        values = clip["max_values"]
        if len(values) != out_features:
            raise ValueError("clipping 출력 채널 수 불일치")
        for channel in values:
            if len(channel) != in_features // 128:
                raise ValueError("clipping 그룹 수 불일치")
            for group in channel:
                if len(group) != 1:
                    raise ValueError("clipping 마지막 차원 오류")
                _bf16_value(group[0], positive=False)


def replay(model, folder, upstream, device, *, on_layer=None):
    import torch
    if model.__class__.__name__ != "Qwen2ForCausalLM" or len(model.model.layers) != 28:
        raise ValueError("고정 28계층 Qwen2ForCausalLM이 필요합니다.")
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("모든 모델 파라미터가 BF16이어야 합니다.")
    # 원본에서 읽어 검증한 순서를 고정하며 search_scale/search_clip은 쓰지 않습니다.
    model.cpu().eval()
    with torch.no_grad(), torch.cuda.device(device):
        for index, layer in enumerate(model.model.layers):
            row = json.loads((Path(folder) / f"awq_layer_{index:03d}.json").read_text())
            shapes = {n: list(p.shape) for n, p in layer.named_parameters()}
            validate_recipe(row, index, shapes)
            if {n for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)} != set(LINEARS):
                raise ValueError("선형 계층 범위 불일치")
            layer.to(device)
            scales = [(s["previous"], tuple(s["following"]),
                       torch.tensor(s["scales"], device=device, dtype=torch.bfloat16)) for s in row["scales"]]
            clips = [(c["parameter"], torch.tensor(c["max_values"], device=device, dtype=torch.bfloat16))
                     for c in row["clips"]]
            upstream["apply_scale"](layer, scales)
            upstream["apply_clip"](layer, clips)
            for name in LINEARS:
                linear = layer.get_submodule(name)
                weight = upstream["quantize"](linear.weight.detach(), n_bit=3,
                                              zero_point=True, q_group_size=128)
                linear.weight.copy_(weight)
            if any(not torch.isfinite(p).all().item() for p in layer.parameters()):
                raise ValueError("재구성 파라미터가 비유한 값입니다.")
            layer.cpu()
            del scales, clips, weight
            torch.cuda.empty_cache()
            if on_layer:
                on_layer(index)
    return {"status": "REPLAY_FROM_SAVED_RECIPES_NOT_ORIGINAL_CHECKPOINT_CERTIFIED",
            "layers": 28, "search_performed": False, "native_low_bit_kernel": False,
            "recipe_dtype": "bfloat16_exact_roundtrip_checked",
            "original_checkpoint_hash_available": False,
            "interpretation": "저장 scale/clip의 적용 경로. 원 R0의 전체 파라미터 해시가 없어 원 체크포인트와 바이트 동등성을 증명하지 않음."}
