#!/usr/bin/env python3
"""Inspect pinned configs and tokenize synthetic text on CPU; never load model weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json')
SHA40 = re.compile(r'[0-9a-f]{40}\Z')
SHA64 = re.compile(r'[0-9a-f]{64}\Z')
PROMPTS = (
    'Compute 17 + 25. Put the final answer in \\boxed{}.',
    'A fictional box holds 3 rows of 4 blue counters. Compute the total number of counters. Put the final answer in \\boxed{}.',
)
SCOPE = ('CPU tokenizer/config inspection only. No model weights, dataset rows, '
         'generation, CUDA, AWQ, long-context validation, approval or H1/H2 testing.')


class ContractError(ValueError):
    """A sanitized, researcher-readable contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def object_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'Expected a JSON object')
    return value


def find_report(directory: Path, refs: dict) -> Path:
    """Select only the specifically reviewed report, not whichever file is newest."""
    matches = []
    for path in sorted(directory.glob('asset_revisions*.json')):
        try:
            report = object_file(path)
        except (OSError, ValueError):
            continue
        if (report.get('recorded_at_utc') == refs['source_recorded_at_utc'] and
                report.get('manifest_canonical_sha256') == refs['manifest_canonical_sha256']):
            matches.append(path)
    require(len(matches) == 1, 'Expected exactly one reviewed asset_revisions report; use --metadata to select it explicitly')
    return matches[0]



def review_inputs(report: dict, manifest: dict, refs: dict) -> dict[str, dict]:
    """Check internal integrity against recorded references, not authenticity of an unsigned report."""
    digest = canonical_hash(manifest)
    require(digest == refs['manifest_canonical_sha256'] == report['manifest_canonical_sha256'],
            'Manifest changed or report/inspection references do not match; review explicitly')
    require(report.get('status') == 'METADATA_RESOLVED_NOT_RUN_READY' and report.get('errors') == [],
            'Metadata report is incomplete')
    expected = {a['role']: a for a in refs['assets']}
    rows = report['assets']
    require(len(rows) == len(expected) and len({a['role'] for a in rows}) == len(rows),
            'Wrong asset count or duplicate roles')
    require({a['role'] for a in rows} == set(expected), 'Asset roles differ')
    require([{k: a[k] for k in ('role', 'repo_id', 'repo_type')} for a in refs['assets']] == manifest['assets'],
            'Inspection references describe another manifest')
    for row in rows:
        reference = expected[row['role']]
        require(all(row.get(k) == reference[k] for k in ('repo_id', 'repo_type', 'revision')), 'Asset reference mismatch')
        require(isinstance(row['revision'], str) and bool(SHA40.fullmatch(row['revision'])), 'Invalid revision')
        require(row.get('status') == 'REVISION_RESOLVED', 'An asset was not resolved')
        require(row.get('private') is False and row.get('gated') is False, 'Public ungated assets only')
    model = next(a for a in rows if a['role'] == 'bf16_reference')
    config_entries = model['configuration_files']
    configs = {}
    for name in FILES:
        entry = config_entries[name]
        expected_hash = refs['model_configuration_sha256'][name]
        require(bool(SHA64.fullmatch(expected_hash)), 'Invalid configuration hash')
        require(canonical_hash(entry['content']) == entry['canonical_sha256'] == expected_hash,
                f'Configuration hash mismatch: {name}')
        expected_url = f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}"
        require(entry['source_url'] == expected_url, 'Configuration URL is not the recorded immutable source')
        configs[name] = entry['content']
    cfg, generation, tok = (configs[n] for n in FILES)
    require(cfg.get('model_type') == 'qwen2', 'Unexpected model architecture')
    require(cfg['eos_token_id'] == generation['eos_token_id'], 'Source EOS IDs differ')
    r1 = manifest['r1']
    require(r1['max_prompt_tokens'] + r1['max_new_tokens'] <= r1['max_total_context_tokens'], 'Inconsistent total budget')
    require(r1['max_total_context_tokens'] <= cfg['max_position_embeddings'], 'Proposed total exceeds model config capacity')
    require(r1['top_k'] == 0, 'This probe implements the existing top-k-disabled candidate only')
    require(isinstance(tok.get('chat_template'), str) and bool(tok['chat_template']), 'Missing chat template')
    return configs


def inspect_tokenizer(tok: Any, configs: dict, manifest: dict, generation_class: Any) -> dict:
    """Execute actual tokenizer methods (or test doubles), never model generation."""
    cfg, source_generation, source_tok = (configs[n] for n in FILES)
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    require(tok.is_fast, 'Expected a fast tokenizer')
    require(bos == source_generation['bos_token_id'], 'Tokenizer BOS differs from generation configuration')
    require(eos == source_generation['eos_token_id'] and pad == eos, 'EOS/PAD IDs differ from this contract')
    require(tok.chat_template == source_tok['chat_template'], 'Loaded chat template differs from recorded source')
    require(tok.model_max_length == source_tok['model_max_length'], 'Loaded tokenizer limit differs from source')
    require(tok.encode(tok.bos_token, add_special_tokens=False) == [bos], 'BOS does not encode as one expected token')
    require(tok.encode(tok.eos_token, add_special_tokens=False) == [eos], 'EOS does not encode as one expected token')
    think_close = tok.encode('</think>', add_special_tokens=False)
    require(bool(think_close) and eos not in think_close, 'Thinking boundary collides with EOS')
    records, rendered_texts, expected_ids = [], [], []
    for prompt in PROMPTS:
        chat = [{'role': 'user', 'content': prompt}]
        rendered = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        expected_text = tok.bos_token + '<｜User｜>' + prompt + '<｜Assistant｜><think>\n'
        require(rendered == expected_text, 'Single-user fixture renders unexpectedly')
        direct = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, truncation=False)
        encoded = tok(rendered, add_special_tokens=False, truncation=False, return_attention_mask=True)
        ids, mask = encoded['input_ids'], encoded['attention_mask']
        require(direct == ids, 'Direct template tokens differ from explicit no-extra-specials tokenization')
        require(bool(ids) and ids[0] == bos and ids.count(bos) == 1 and eos not in ids, 'BOS duplicated or prompt contains EOS')
        require(mask == [1] * len(ids), 'Unexpected unpadded attention mask')
        require(max(ids) < cfg['vocab_size'], 'Prompt contains an ID outside the model embedding range')
        defaults = tok(rendered, add_special_tokens=True, truncation=False)['input_ids']
        records.append({'fixture_id': len(records), 'input_ids': ids, 'input_token_count': len(ids),
                        'rendered_prompt_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
                        'bos_count_if_special_tokens_added_again': defaults.count(bos)})
        rendered_texts.append(rendered)
        expected_ids.append(ids)
    # CPU lists only. Preserve masks produced by the tokenizer; do not infer masks from EOS equality.
    old_side = tok.padding_side
    try:
        tok.padding_side = 'left'
        batch = tok(rendered_texts, add_special_tokens=False, truncation=False, padding=True, return_attention_mask=True)
    finally:
        tok.padding_side = old_side
    require(len(batch['input_ids']) == len(expected_ids) == len(batch['attention_mask']), 'Wrong batch size')
    require(len({len(row) for row in batch['input_ids']}) == 1, 'Padded batch is not rectangular')
    for ids, mask, expected in zip(batch['input_ids'], batch['attention_mask'], expected_ids):
        require(len(ids) == len(mask) and all(m in (0, 1) for m in mask), 'Invalid padding mask')
        require([x for x, m in zip(ids, mask) if m] == expected, 'Padding changed active prompt tokens')
        require(all(x == pad for x, m in zip(ids, mask) if not m), 'Unexpected pad token')
        require(mask == sorted(mask), 'Expected left padding')
    inherited = generation_class.from_dict(source_generation)
    r1 = manifest['r1']
    values = dict(do_sample=True, num_beams=1, temperature=r1['temperature'], top_p=r1['top_p'],
                  top_k=r1['top_k'], repetition_penalty=r1['repetition_penalty'],
                  bos_token_id=bos, eos_token_id=eos, pad_token_id=pad,
                  min_new_tokens=0, forced_eos_token_id=None, stop_strings=None,
                  max_new_tokens=manifest['r0']['max_new_tokens'])
    explicit = generation_class(**values)
    require(explicit.top_k == 0 and explicit.eos_token_id == eos, 'Explicit generation overrides were not applied')
    return {
        'tokenizer_class': type(tok).__name__, 'bos_token_id': bos, 'eos_token_id': eos, 'pad_token_id': pad,
        'source_model_bos_token_id': cfg['bos_token_id'], 'source_generation_bos_token_id': source_generation['bos_token_id'],
        'source_tokenizer_limit': source_tok['model_max_length'], 'model_config_capacity': cfg['max_position_embeddings'],
        'candidate_total_context': r1['max_total_context_tokens'], 'long_context_validated': False,
        'closing_think_ids_not_eos': think_close, 'inherited_top_k': inherited.top_k,
        'explicit_r0_generation_config_candidate_not_executed': values,
        'prompt_fixtures': records, 'left_padding_mask_check': 'PASS',
        'warnings': ['Source BOS discrepancy retained; use verified tokenizer prompt IDs, not model-config BOS substitution',
                     'Tokenizer length and model capacity are different metadata; long-context runtime is not certified',
                     'Prompt already includes opening think tag; it is not a generated token',
                     'Thinking-close is not whole-response EOS; preserve the final-answer region'],
    }


def write_new(path: Path, result: dict) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, help='Existing report; default finds the uniquely matching reviewed timestamp/hash under results/local')
    p.add_argument('--manifest', type=Path, default=ROOT / 'configs/reproduction_candidate.json')
    p.add_argument('--refs', type=Path, default=ROOT / 'configs/asset_inspection_refs.json')
    p.add_argument('--online', action='store_true', help='Download only pinned tokenizer/config assets; no weights or dataset rows')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = a.output or ROOT / f'results/local/tokenizer_contract_{stamp}.json'
    if path.exists():
        p.error('Output exists; choose a new path')
    result = {'schema_version': 1, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE}
    try:
        manifest, refs = object_file(a.manifest), object_file(a.refs)
        report_path = a.metadata or find_report(ROOT / 'results/local', refs)
        report = object_file(report_path)
        configs = review_inputs(report, manifest, refs)
        model = next(x for x in refs['assets'] if x['role'] == 'bf16_reference')
        result.update(manifest_canonical_sha256=canonical_hash(manifest), inspection_refs_sha256=canonical_hash(refs),
                      model_revision=model['revision'], metadata_integrity='PASS')
        if not a.online:
            result['status'] = 'METADATA_REVIEW_PASS_TOKENIZER_NOT_RUN'
        else:
            versions = {name: metadata.version(name) for name in refs['tokenizer_probe_versions']}
            require(versions == refs['tokenizer_probe_versions'], 'Install the declared tokenizer-probe package versions')
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
            # Disable framework imports for this process; CPU tokenization does not require PyTorch/TensorFlow/Flax.
            for key in ('USE_TORCH', 'USE_TF', 'USE_FLAX'):
                os.environ[key] = '0'
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer, GenerationConfig
            for name in FILES:
                local = hf_hub_download(repo_id=model['repo_id'], filename=name, revision=model['revision'], token=False)
                require(canonical_hash(object_file(Path(local))) == refs['model_configuration_sha256'][name],
                        f'Pinned download differs from reviewed configuration: {name}')
            tok = AutoTokenizer.from_pretrained(model['repo_id'], revision=model['revision'], use_fast=True,
                                               trust_remote_code=False, token=False)
            result['checks'] = inspect_tokenizer(tok, configs, manifest, GenerationConfig)
            versions.update({name: metadata.version(name) for name in ('tokenizers', 'huggingface-hub', 'numpy')})
            result['packages'] = versions
            result['status'] = 'TOKENIZER_CONTRACT_PASS_NOT_MODEL_READY'
    except Exception as exc:
        # Keep private filesystem paths/tokens out of the report; retain useful contract failures only.
        result['status'] = 'TOKENIZER_AUDIT_INCOMPLETE'
        result['error_type'] = type(exc).__name__
        if isinstance(exc, ContractError):
            result['contract_error'] = str(exc)
    try:
        write_new(path, result)
    except OSError as exc:
        print(f'Cannot write audit report: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Saved: {path}')
    return 2 if result['status'] == 'TOKENIZER_AUDIT_INCOMPLETE' else 0


if __name__ == '__main__':
    raise SystemExit(main())
