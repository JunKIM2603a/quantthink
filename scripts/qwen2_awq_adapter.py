"""검토한 AWQ 함수에 Qwen2 입력·토큰 블록·좌표 진단을 연결합니다. CUDA 전용 후보입니다."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np

from awq_weight_space import apply_scale_ledger, canonical_error
from runtime_assets import package_versions, verify_upstream

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
QCONFIG = {"zero_point": True, "q_group_size": 128}


def load_upstream(folder, policy):
    proof = verify_upstream(Path(folder), policy)
    prefix = "_quantthink_awq_" + policy["upstream"]["revision"]
    if prefix not in sys.modules:
        package = types.ModuleType(prefix)
        package.__path__ = [str(Path(folder).resolve() / "methods/awq")]
        sys.modules[prefix] = package
    scaling = importlib.import_module(prefix + ".auto_scale")
    clipping = importlib.import_module(prefix + ".auto_clip")
    quantizer = importlib.import_module(prefix + ".quantizer")
    return {"search_scale": scaling.auto_scale_block, "apply_scale": scaling.apply_scale,
            "search_clip": clipping.auto_clip_block, "apply_clip": clipping.apply_clip,
            "quantize": quantizer.pseudo_quantize_tensor, "source_files": proof}


def move_tree(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(x, device) for x in value)
    if isinstance(value, list):
        return [move_tree(x, device) for x in value]
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("지원하지 않는 계층 인자입니다. cache 객체를 전달하지 마세요.")


def validate_shape(blocks, *, expected_shape, vocab_size):
    if (not isinstance(blocks, list) or len(blocks) != expected_shape[0]
            or any(not isinstance(row, list) or len(row) != expected_shape[1] for row in blocks)):
        raise ValueError("calibration 토큰 블록 shape가 계획과 다릅니다.")
    if any(type(x) is not int or x < 0 or x >= vocab_size for row in blocks for x in row):
        raise ValueError("calibration 토큰이 vocabulary 범위를 벗어났습니다.")


def scale_ledger(layer, scales):
    import torch
    records = []
    expected_norms = {"input_layernorm", "post_attention_layernorm"}
    for previous, following, values in scales:
        previous_module = layer.get_submodule(previous)
        values = values.detach().float().cpu().numpy()
        if isinstance(previous_module, torch.nn.Linear):
            kind = "linear_linear"
            if previous not in ("self_attn.v_proj", "mlp.up_proj"):
                raise ValueError("검토하지 않은 선형 스케일 경로입니다.")
        elif previous in expected_norms:
            kind = "norm_linears"
        else:
            raise ValueError("검토하지 않은 정규화 스케일 경로입니다.")
        if any(name not in LINEARS for name in following):
            raise ValueError("검토하지 않은 스케일 대상입니다.")
        records.append({"kind": kind, "previous": previous,
                        "following": list(following), "scales": values.tolist()})
    expected_paths = [
        ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ]
    if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
        expected_paths.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    expected_paths.extend([
        ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ("mlp.up_proj", ["mlp.down_proj"]),
    ])
    if [(x["previous"], x["following"]) for x in records] != expected_paths:
        raise ValueError("원본 Qwen2의 스케일 경로·순서와 다릅니다.")
    # 순방향·역방향 구현이 지원하는 shape와 scale도 검사합니다.
    apply_scale_ledger(parameter_arrays(layer), records)
    return records


def parameter_arrays(layer):
    return {name: param.detach().float().cpu().numpy().copy()
            for name, param in layer.named_parameters()}


def output_difference(reference, observed):
    import torch
    delta = (reference.float() - observed.float()).double()
    if not torch.isfinite(delta).all():
        raise ValueError("계층 출력이 비유한 값입니다.")
    rms = float(delta.square().mean().sqrt().item())
    base = float(reference.double().square().mean().sqrt().item())
    return {"max_abs": float(delta.abs().max().item()), "rms": rms,
            "relative_rms": rms / base if base else None,
            "acceptance": "MEASURED_NOT_AUTOMATICALLY_CERTIFIED"}


def capture_first_input(model, tokens, device):
    import torch
    class CapturedInput(Exception):
        pass
    captured = {}
    def hook(module, args, kwargs):
        if len(args) != 1 or not torch.is_tensor(args[0]):
            raise ValueError("예상한 Qwen2 첫 계층 인자가 아닙니다.")
        if kwargs.get("past_key_value") is not None:
            raise ValueError("AWQ 탐색에는 KV cache를 사용하지 않습니다.")
        if not isinstance(kwargs.get("position_embeddings"), tuple):
            raise ValueError("Transformers 4.51.3 position_embeddings가 없습니다.")
        captured["hidden"] = args[0].detach().cpu()
        captured["kwargs"] = move_tree(kwargs, "cpu")
        raise CapturedInput
    handle = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        try:
            with torch.no_grad():
                model(input_ids=tokens.to(device), attention_mask=torch.ones_like(tokens, device=device),
                      use_cache=False)
        except CapturedInput:
            pass
    finally:
        handle.remove()
    if not captured:
        raise ValueError("Qwen2 첫 계층 입력을 포착하지 못했습니다.")
    return captured["hidden"], captured["kwargs"]


def run_awq(model, blocks, upstream, *, device="cuda:0", expected_shape=(128, 512),
            on_layer=None):
    """모델을 수정합니다. 실패 시 부분 변환 모델을 재사용하지 마세요.

    반환 상태는 변환·진단 실행 완료만 의미하며 실모델 동등성 승인과 다릅니다.
    """
    import torch
    if package_versions(["transformers"])["transformers"] != "4.51.3":
        raise ValueError("현재 어댑터는 Transformers 4.51.3만 대상으로 합니다.")
    if model.__class__.__name__ != "Qwen2ForCausalLM" or model.config.model_type != "qwen2":
        raise ValueError("검토한 Qwen2ForCausalLM만 지원합니다.")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("원본 AWQ 탐색은 CUDA 장치가 필요합니다.")
    validate_shape(blocks, expected_shape=expected_shape, vocab_size=model.config.vocab_size)
    if expected_shape[0] * expected_shape[1] < 512:
        raise ValueError("원본 clipping 탐색에는 512개 이상의 calibration 토큰이 필요합니다.")
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("모든 모델 파라미터가 BF16이어야 합니다.")
    model.eval()
    model.cpu()
    layers = model.model.layers
    for layer in layers:
        names = {n for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        if names != set(LINEARS):
            raise ValueError("Qwen2 선형 계층 범위가 검토한 7개와 다릅니다.")
        for name in LINEARS:
            weight = layer.get_submodule(name).weight
            if weight.shape[1] % 128 or weight.shape[0] % 64:
                raise ValueError("원본 g128·clipping 출력 채널 조건을 만족하지 않습니다.")
    cache_before = model.config.use_cache
    model.config.use_cache = False
    results = []
    # 원본 함수의 .cuda() 호출도 지정된 worker 장치를 사용하도록 합니다.
    with torch.cuda.device(device):
        try:
            model.model.embed_tokens.to(device)
            model.model.rotary_emb.to(device)
            layers[0].to(device)
            tokens = torch.tensor(blocks, dtype=torch.long)
            hidden, kwargs_cpu = capture_first_input(model, tokens, device)
            layers[0].cpu()
            model.model.embed_tokens.cpu()
            model.model.rotary_emb.cpu()
            for index, layer in enumerate(layers):
                layer.to(device)
                original = parameter_arrays(layer)
                inputs = hidden.to(device)
                kwargs = move_tree(kwargs_cpu, device)
                features, handles = {}, []
                for name in LINEARS:
                    def hook(module, args, output, name=name):
                        if name in features:
                            raise ValueError("동일 선형 계층이 예기치 않게 여러 번 호출됐습니다.")
                        features[name] = args[0].detach().cpu()
                    handles.append(layer.get_submodule(name).register_forward_hook(hook))
                try:
                    with torch.no_grad():
                        reference_output = layer(inputs, **kwargs)[0]
                finally:
                    for handle in handles:
                        handle.remove()
                if set(features) != set(LINEARS):
                    raise ValueError("calibration 계층 입력이 누락됐습니다.")
                # 원본 탐색과 같은 참조 활성값 전파. 양자화 출력을 다음 탐색에 섞지 않습니다.
                hidden = reference_output.detach().cpu()
                with torch.no_grad():
                    scales = upstream["search_scale"](layer, dict(kwargs), w_bit=3,
                                                       q_config=dict(QCONFIG), input_feat=features)
                    ledger = scale_ledger(layer, scales)
                    upstream["apply_scale"](layer, scales, input_feat_dict=features)
                    scaled_output = layer(inputs, **kwargs)[0]
                    scale_diagnostic = output_difference(reference_output, scaled_output)
                    del scaled_output
                    clips = upstream["search_clip"](layer, w_bit=3, q_config=dict(QCONFIG), input_feat=features)
                    expected_clips = set(LINEARS) - {"self_attn.q_proj", "self_attn.k_proj"}
                    if {name for name, _ in clips} != expected_clips or len(clips) != len(expected_clips):
                        raise ValueError("원본 clipping 대상이 검토한 범위와 다릅니다.")
                    upstream["apply_clip"](layer, clips)
                    layer.to(device)
                    for name in LINEARS:
                        linear = layer.get_submodule(name)
                        quantized = upstream["quantize"](linear.weight.detach(), n_bit=3, **QCONFIG)
                        if not torch.isfinite(quantized).all():
                            raise ValueError("양자화 가중치가 비유한 값입니다.")
                        linear.weight.copy_(quantized)
                    transformed = parameter_arrays(layer)
                    canonical = apply_scale_ledger(transformed, ledger, inverse=True)
                    transformed_output = layer(inputs, **kwargs)[0]
                    try:
                        for name, param in layer.named_parameters():
                            param.copy_(torch.as_tensor(canonical[name], device=device, dtype=param.dtype))
                        canonical_output = layer(inputs, **kwargs)[0]
                        inverse_diagnostic = output_difference(transformed_output, canonical_output)
                    finally:
                        for name, param in layer.named_parameters():
                            param.copy_(torch.as_tensor(transformed[name], device=device, dtype=param.dtype))
                errors = canonical_error(original, transformed, ledger,
                                         weight_names=[x + ".weight" for x in LINEARS])
                auxiliary = []
                for name, value in original.items():
                    if name not in {x + ".weight" for x in LINEARS}:
                        delta = canonical[name] - value
                        auxiliary.append({"parameter": name, "elements": int(value.size),
                                          "rms_error": float(np.sqrt(np.mean(delta * delta))),
                                          "scope": "양자화 대상 밖 norm·bias의 좌표 복원 잔차"})
                result = {"layer": index, "scales": ledger,
                          "clips": [{"parameter": n, "shape": list(v.shape),
                                     "max_values": v.detach().float().cpu().tolist()} for n, v in clips],
                          "scale_only_output_difference": scale_diagnostic,
                          "canonicalized_output_difference": inverse_diagnostic,
                          "canonical_weight_error": errors["parameters"],
                          "canonical_auxiliary_error": auxiliary,
                          "quantized_linears": list(LINEARS)}
                results.append(result)
                if on_layer:
                    on_layer(result)
                layer.cpu()
                del original, transformed, canonical, features, inputs, kwargs
                del reference_output, transformed_output, canonical_output, scales, clips
                torch.cuda.empty_cache()
        finally:
            model.config.use_cache = cache_before
            model.cpu()
    return {"status": "AWQ_TRANSFORM_APPLIED_NOT_EQUIVALENCE_CERTIFIED",
            "weight_bits": 3, "group_size": 128, "zero_point": True,
            "activation_dtype": "bfloat16", "native_low_bit_kernel": False,
            "calibration_shape": list(expected_shape), "layers": results}
