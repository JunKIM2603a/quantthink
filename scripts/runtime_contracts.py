"""R0 공통 입력·종료·자산 연결 계약. 이 모듈을 import하면 모델을 실행하지 않습니다."""
from __future__ import annotations

import json
from pathlib import Path
import time

from audit_tokenizer_contract import PROMPTS
from reproduction_contracts import termination_record
from runtime_assets import canonical_hash, file_digests

REQUIRED_GATES = {"novelty_gate": "SCOPED_CONTRIBUTION_ACCEPTED",
                  "protocol_gate": "ACCEPTED", "research_approval": "APPROVED"}


def unmet_gates(status):
    return {key: {"required": expected, "observed": status.get(key)}
            for key, expected in REQUIRED_GATES.items() if status.get(key) != expected}


def require_gates(status):
    if unmet_gates(status):
        raise ValueError("기여 범위 판정·프로토콜 수락·연구 실행 승인 기록이 필요합니다.")


def load_prepared_calibration(folder, candidate, refs, policy):
    report = json.loads((folder / "preparation_report.json").read_text())
    if (report["status"] != "PREPARED_CANDIDATE_NOT_RUN_APPROVAL"
            or not report["include_calibration"]
            or report["candidate_sha256"] != canonical_hash(candidate)
            or report["policy_sha256"] != canonical_hash(policy)
            or report["inspection_refs_sha256"] != canonical_hash(refs)):
        raise ValueError("calibration 준비 보고서의 설정·참조가 다릅니다.")
    expected_names = {"r1_inputs.jsonl", "r1_gold.jsonl", "development_manifest.json",
                      "calibration_manifest.json", "calibration_tokens.json"}
    if set(report["artifacts"]) != expected_names:
        raise ValueError("필수 준비 파일 목록이 다릅니다.")
    for name, expected in report["artifacts"].items():
        if file_digests(folder / name) != expected:
            raise ValueError("준비 후 파일 바이트가 변경됐습니다.")
    calibration = json.loads((folder / "calibration_manifest.json").read_text())
    blocks = json.loads((folder / "calibration_tokens.json").read_text())["blocks"]
    if canonical_hash(blocks) != calibration["token_blocks_sha256"]:
        raise ValueError("calibration 토큰 해시가 다릅니다.")
    if calibration["candidate_sha256"] != canonical_hash(candidate):
        raise ValueError("calibration 후보 설정이 다릅니다.")
    return blocks, report


def prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=True, add_generation_prompt=True)
    if (ids.count(151646) != 1 or 151643 in ids or not ids or ids[-2:] != [151648, 198]):
        raise ValueError("입력 토큰이 검토한 프롬프트 계약과 다릅니다.")
    return ids


def generated_record(ids, tokenizer, budget):
    record = termination_record(ids, {151643}, budget)
    # 응답은 batch=1. PAD와 같은 EOS를 포함한 토큰 수를 먼저 기록한 뒤 디코딩용 끝 EOS만 뺍니다.
    text_ids = ids[:-1] if ids[-1] == 151643 else ids
    record["generated_ids"] = ids
    record["generated_text"] = tokenizer.decode(text_ids, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False)
    return record


def generate_r0(model, tokenizer, candidate, *, arm, device, seed=42):
    import torch
    from transformers import GenerationConfig
    cfg = candidate["r1"]
    cap = candidate["r0"]["max_new_tokens"]
    generation = GenerationConfig(
        do_sample=True, num_beams=1, temperature=cfg["temperature"], top_p=cfg["top_p"],
        top_k=cfg["top_k"], repetition_penalty=cfg["repetition_penalty"],
        bos_token_id=151646, eos_token_id=151643, pad_token_id=151643,
        min_new_tokens=0, forced_eos_token_id=None, stop_strings=None, max_new_tokens=cap)
    model.to(device).eval()
    for index, prompt in enumerate(PROMPTS):
        row = {"fixture_id": index, "arm": arm, "seed": seed}
        try:
            ids = prompt_ids(tokenizer, prompt)
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            torch.manual_seed(seed)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        generation_config=generation, use_cache=True)
            torch.cuda.synchronize(device)
            generated = output[0, len(ids):].detach().cpu().tolist()
            row.update(generated_record(generated, tokenizer, cap),
                       input_ids=ids, input_token_count=len(ids),
                       input_ids_sha256=canonical_hash(ids),
                       elapsed_seconds=time.monotonic() - started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        except Exception as exc:
            row.update(finish_reason="error", error_type=type(exc).__name__, error=str(exc))
        yield row

