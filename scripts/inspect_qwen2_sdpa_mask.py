"""고정 Qwen2의 문맥 경계 마스크를 CPU에서 확인합니다. 모델·생성·GPU·다운로드 없음."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash

POSITIONS = (4094, 4095, 4096, 4204)
SOURCE_BLOBS = {
    "configuration_qwen2.py": "2e82f1976f3922f3620415f4eace6c6e046243f8",
    "modeling_qwen2.py": "16a7316e2d0e56eafe301a7f2d8693d6cc6c73ec",
    "modeling_attn_mask_utils.py": "dfdd976f0156139024c6f7788870a71e4a41754d",
    "sdpa_attention.py": "9c924c048ad52929a2d0f890a22295d5a56ef505",
}


def read_plan(evidence):
    review = json.loads((ROOT / "configs/difficulty_evidence_review_v01.json").read_text(encoding="utf-8"))
    if file_hash(evidence) != review["evidence_packet_sha256"]:
        raise ValueError("검토한 review_evidence_v01.json과 바이트 해시가 다릅니다.")
    packet = json.loads(evidence.read_text(encoding="utf-8"))
    config = packet["cached_model_config"]["configuration"]
    refs = json.loads((ROOT / "configs/asset_inspection_refs.json").read_text(encoding="utf-8"))
    if canonical_hash(config) != refs["model_configuration_sha256"]["config.json"]:
        raise ValueError("고정 모델 설정 해시 불일치")
    if config.get("use_sliding_window") is not False or config.get("sliding_window") != 4096:
        raise ValueError("검토한 use_sliding_window=false, sliding_window=4096 설정과 다릅니다.")
    return config, {"status": "CPU_MASK_INSPECTION_PLAN", "device": "cpu",
                    "query_length": 1, "cache_class": "DynamicCache", "dtype": "bfloat16",
                    "zero_based_query_positions": list(POSITIONS),
                    "conditions": ["as_recorded", "in_memory_sliding_window_none_control"],
                    "evidence_packet_sha256": review["evidence_packet_sha256"],
                    "model_configuration_sha256": canonical_hash(config),
                    "new_model_or_tokenizer_execution": False, "new_generation_tokens": 0,
                    "gpu_execution": False, "network_download": False,
                    "original_files_or_generation_config_modified": False}


def inspect_masks(config, plan):
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    if versions != {"torch": "2.7.1+cu118", "transformers": "4.51.3"}:
        raise ValueError(f"검토한 기존 버전에서 실행하세요. 설치/업데이트는 하지 않습니다: {versions}")
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers import modeling_attn_mask_utils
    from transformers.integrations import sdpa_attention
    from transformers.models.qwen2 import configuration_qwen2, modeling_qwen2

    source_proofs = []
    for module in (configuration_qwen2, modeling_qwen2, modeling_attn_mask_utils, sdpa_attention):
        path = Path(module.__file__)
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != SOURCE_BLOBS[path.name]:
            raise ValueError(f"고정 upstream과 설치된 코드가 다릅니다: {path.name}, {blob}")
        source_proofs.append({"file": path.name, "git_blob_sha1": blob,
                              "sha256": hashlib.sha256(data).hexdigest()})

    records = []
    # 모델 객체를 만들지 않고 실제 마스크 메서드에 필요한 config/training/함수만 제공합니다.
    for condition in plan["conditions"]:
        cfg = configuration_qwen2.Qwen2Config.from_dict(copy.deepcopy(config))
        cfg._attn_implementation = "sdpa"
        if condition == "in_memory_sliding_window_none_control":
            cfg.sliding_window = None
        context = SimpleNamespace(config=cfg, training=False,
            _prepare_4d_causal_attention_mask_with_cache_position=
                modeling_qwen2.Qwen2Model._prepare_4d_causal_attention_mask_with_cache_position)
        for position in POSITIONS:
            cache = DynamicCache()
            kv = torch.zeros((1, 1, position, 1), dtype=torch.bfloat16, device="cpu")
            cache.update(kv, kv, 0)
            hidden = torch.zeros((1, 1, 1), dtype=torch.bfloat16, device="cpu")
            attention = torch.ones((1, position + 1), dtype=torch.long, device="cpu")
            mask = modeling_qwen2.Qwen2Model._update_causal_mask(
                context, attention, hidden, torch.tensor([position], device="cpu"), cache, False)
            if mask is None:
                blocked = []
            else:
                blocked = torch.nonzero(mask[0, 0, 0] != 0, as_tuple=False).flatten().tolist()
            records.append({"condition": condition, "query_position": position,
                            "key_value_length": position + 1,
                            "use_sliding_window": cfg.use_sliding_window,
                            "sliding_window": cfg.sliding_window,
                            "mask_kind": "implicit_sdpa_causal" if mask is None else "explicit_4d",
                            "blocked_key_count": len(blocked),
                            "first_blocked_key": blocked[0] if blocked else None,
                            "last_blocked_key": blocked[-1] if blocked else None})
    recorded = [r for r in records if r["condition"] == "as_recorded"]
    control = [r for r in records if r["condition"] != "as_recorded"]
    observed = (all(r["blocked_key_count"] == max(0, r["query_position"] - 4096 + 1)
                    for r in recorded)
                and any(r["blocked_key_count"] > 0 for r in recorded)
                and all(r["blocked_key_count"] == 0 for r in control))
    return {**plan, "status": "DISABLED_FLAG_WINDOW_MASK_OBSERVED" if observed else "MASK_RESULT_REQUIRES_REVIEW",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "packages": versions,
            "script_sha256": file_hash(Path(__file__)),
            "source_files": source_proofs, "records": records,
            "model_ready": False, "pilot_results_changed": False,
            "scope": "실제 설치 코드의 마스크 함수만 작은 CPU 텐서로 호출. 모델 forward, SDPA 연산, GPU kernel, 실제 생성 영향은 검증하지 않음."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="PyTorch/Transformers import 없이 계획만 표시")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config, plan = read_plan(args.evidence)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if args.output is None:
            raise ValueError("CPU 검사 결과를 보존할 --output을 지정하세요.")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("결과가 이미 있습니다. 기존 파일을 첨부하세요.")
        result = inspect_masks(config, plan)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(result["status"])
        for row in result["records"]:
            print(f"{row['condition']}: 위치={row['query_position']}, 차단된 이전키={row['blocked_key_count']}")
        print(f"CPU 마스크 확인만 완료. 첨부할 파일: {args.output}")
        return 0
    except Exception as exc:
        parser.exit(2, f"마스크 확인 중단: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
