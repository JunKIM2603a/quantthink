"""기존20문제의32K 전체 문맥 재시험. 기본은 계획 조회이며 --execute에서만 실행."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import types

from long_generation_contracts import (
    ROOT, canonical_hash, file_hash, load_plan, require_scope, validate_resume,
    reserve_attempt, summarize_long, verify_resume_runtime, validate_mask_trace,
)
from difficulty_pilot_contracts import prompt_for
from difficulty_pilot_resume import output_lock, backup_before_resume, ProgressReporter, attempt_key


def execute(args, cfg, suite, runtime, status):
    hashes = require_scope(status, cfg)
    if args.output_dir.is_symlink():
        raise ValueError('출력 폴더는 심볼릭 링크일 수 없습니다.')
    if args.resume:
        previous = json.loads((args.output_dir / 'run.json').read_text())
        if not validate_resume(previous, cfg, suite)['remaining']:
            print('80개 완료 기록 보존. 모델을 로드하지 않고 종료합니다.', flush=True)
            return previous
    elif args.output_dir.exists():
        raise FileExistsError('결과 폴더가 있습니다. 완료분 재개는 --execute --resume을 사용하세요.')
    if file_hash(args.source_run) != cfg['source_run_sha256']:
        raise ValueError('검토한 기존20문제 원본 run.json과 다릅니다.')
    original = json.loads(args.source_run.read_text())
    from run_difficulty_pilot import repo_state, parameter_hash, checkpoint, cached_tokenizer
    from runtime_assets import load_contracts, package_versions, verify_upstream
    from replay_r0_awq import verify_r0_files, validate_recipe, replay
    repository = repo_state()
    packages = package_versions(['torch', 'transformers', 'huggingface-hub', 'numpy'])
    if packages != runtime['packages']:
        raise ValueError('고정 패키지 불일치. 자동 업데이트하지 않습니다.')
    candidate, refs, policy = load_contracts()
    review = json.loads((ROOT / cfg['awq_source']).read_text())
    r0, proofs = verify_r0_files(args.r0_dir, review)
    if r0['candidate_sha256'] != canonical_hash(candidate) or r0['policy_sha256'] != canonical_hash(policy):
        raise ValueError('기존 R0 후보/정책 불일치')
    upstream_proof = verify_upstream(args.upstream_dir, policy)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config, GenerationConfig, StoppingCriteria, StoppingCriteriaList
    from huggingface_hub import hf_hub_download
    from runtime_contracts import prompt_ids, generated_record
    from qwen2_awq_adapter import load_upstream
    from run_fixed_prefix_diagnostic import verify_sources
    sources = verify_sources(runtime)
    tokenizer, asset = cached_tokenizer(refs)
    if asset != runtime['model_reference']:
        raise ValueError('고정 model revision 불일치')
    inputs = {p['id']: prompt_ids(tokenizer, prompt_for(suite, p)) for p in suite['problems']}
    for row in original['attempts']:
        if row['input_ids'] != inputs[row['problem_id']]:
            raise ValueError('기존 문제의 실제 입력 ID가 달라짐')
    if any(len(ids) > cfg['max_prompt_tokens'] for ids in inputs.values()):
        raise ValueError('입력 상한 초과. 자동 자르기 없음.')
    source_path = Path(hf_hub_download(asset['repo_id'], 'config.json', revision=asset['revision'],
                                      token=False, local_files_only=True))
    source = json.loads(source_path.read_text())
    effective = copy.deepcopy(source)
    effective['sliding_window'] = None
    if (canonical_hash(source) != runtime['model_configuration_sha256']
            or canonical_hash(effective) != runtime['effective_source_configuration_sha256']
            or cfg['max_total_context_tokens'] > effective['max_position_embeddings']):
        raise ValueError('원본/유효 모델 설정 또는 전체 문맥 상한 불일치')
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise ValueError('기존 CUDA 환경이 필요합니다.')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 지원이 필요합니다.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(False)
    generation = GenerationConfig(do_sample=True, num_beams=1, **cfg['decoding'],
        bos_token_id=151646, eos_token_id=151643, pad_token_id=151643, min_new_tokens=0,
        forced_eos_token_id=None, stop_strings=None, max_new_tokens=cfg['max_new_tokens'])
    expected_generation = copy.deepcopy(original['resolved_generation_config'])
    expected_generation['max_new_tokens'] = cfg['max_new_tokens']
    if generation.to_dict() != expected_generation:
        raise ValueError('출력 상한 외 GenerationConfig가 기존 시험과 다릅니다.')
    current = {'schema_version': 1, 'status': 'LONG_GENERATION_STARTED', 'model_ready': False,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'repository': repository,
        'config_sha256': canonical_hash(cfg), 'suite_sha256': canonical_hash(suite),
        'research_status_sha256': canonical_hash(status), 'implementation_sha256': hashes,
        'source_run_sha256': cfg['source_run_sha256'], 'packages': packages, 'python': platform.python_version(),
        'model_reference': asset, 'r0_input_files': proofs, 'upstream_source_files': upstream_proof,
        'source_configuration_sha256': canonical_hash(source),
        'effective_source_configuration_sha256': canonical_hash(effective), 'installed_sources': sources,
        'generation_config': {**cfg['decoding'], 'max_new_tokens': cfg['max_new_tokens'], 'use_cache': True, 'batch_size': 1},
        'resolved_generation_config': generation.to_dict(),
        'gpu': {'device': str(device), 'name': torch.cuda.get_device_name(device),
                'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory},
        'runtime_flags': {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                          'deterministic_algorithms': False, 'float32_matmul_precision': 'highest'},
        'runtime_contract': {'dtype': 'bfloat16', 'attention': 'sdpa', 'use_cache': True,
                             'sliding_window': None, 'use_sliding_window': False, 'cache_class': 'DynamicCache'},
        'parameter_sha256': {}, 'attempts': [], 'generation_calls_started': 0,
        'execution_segments': [], 'calibration_search': False, 'old_results_overwritten': False,
        'mask_trace_scope_ko': '실제 생성의 고정 Qwen2 마스크 함수 반환·cache 길이 표본. CUDA SDPA kernel 내부 관찰이나 수치 동등성 검증은 아님.'}
    with output_lock(args.output_dir, resume=args.resume):
        if args.resume:
            report = json.loads((args.output_dir / 'run.json').read_text())
            plan = validate_resume(report, cfg, suite)
            verify_resume_runtime(report, current, inputs)
            backup = backup_before_resume(args.output_dir / 'run.json')
        else:
            report, backup = current, None
            plan = validate_resume(report, cfg, suite)
        segment = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'repository': repository,
                   'research_status_sha256': canonical_hash(status), 'completed_at_start': plan['completed'],
                   'completed_attempts_sha256': plan['completed_attempts_sha256'], 'backup': backup}
        report['execution_segments'].append(segment)
        report['status'] = 'LONG_GENERATION_RUNNING'
        save = lambda: checkpoint(args.output_dir / 'run.json', report)
        save()
        completed = set(plan['completed_keys'])
        initial_count = len(report['attempts'])
        try:
            print('기존 모델·recipe 사용. 전체 문맥(window=None), 생성 상한32768. 15초마다 진행 표시.', flush=True)
            model = AutoModelForCausalLM.from_pretrained(asset['repo_id'], revision=asset['revision'], token=False,
                trust_remote_code=False, use_safetensors=True, local_files_only=True,
                config=Qwen2Config.from_dict(copy.deepcopy(effective)), torch_dtype=torch.bfloat16,
                attn_implementation='sdpa').eval()
            if (model.__class__.__name__ != 'Qwen2ForCausalLM' or len(model.model.layers) != 28
                    or model.config.sliding_window is not None or model.config.use_sliding_window is not False
                    or model.config._attn_implementation != 'sdpa'
                    or any(p.dtype != torch.bfloat16 for p in model.parameters())):
                raise ValueError('고정 모델 구조/runtime 불일치')
            report['resolved_model_configuration_sha256'] = canonical_hash(model.config.to_dict())
            if report['resolved_model_configuration_sha256'] != 'd325732e5ce492cfc9be12a953c69f553380beab02735ce0eb386491b8897934':
                raise ValueError('검토한 전체 문맥 모델 설정 해시와 다름')
            upstream = load_upstream(args.upstream_dir, policy)
            for i, layer in enumerate(model.model.layers):
                rec = json.loads((args.r0_dir / f'awq_layer_{i:03d}.json').read_text())
                validate_recipe(rec, i, {n: list(p.shape) for n, p in layer.named_parameters()})
            original_mask = model.model._update_causal_mask
            trace = []
            first_decode = None

            def traced_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
                mask = original_mask(attention_mask, input_tensor, cache_position, past_key_values, output_attentions)
                past = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
                qlen = input_tensor.shape[1]
                if not trace or past == first_decode or past in cfg['mask_trace_positions']:
                    position = int(cache_position[0].item())
                    last = int(cache_position[-1].item())
                    layers = [int(x.shape[-2]) for x in getattr(past_key_values, 'key_cache', [])]
                    if past and (len(layers) != 28 or any(x != position for x in layers)):
                        raise ValueError('생성 cache 계층별 길이 불일치')
                    blocked = 0 if mask is None else int((mask[0, 0, -1, :last + 1] < 0).sum().item())
                    item = {'query_start_position': position, 'query_length': int(qlen), 'past_seen_tokens': past,
                            'cache_class': type(past_key_values).__name__, 'cached_layer_lengths': layers,
                            'mask_kind': 'implicit_sdpa' if mask is None else 'explicit_4d',
                            'masked_past_or_current_keys': blocked}
                    validate_mask_trace(item)
                    trace.append(item)
                return mask

            model.model._update_causal_mask = types.MethodType(traced_mask, model.model)
            for arm in cfg['arms']:
                if all((p['id'], arm, seed) in completed for p in suite['problems'] for seed in cfg['seeds']):
                    continue
                if len(report['attempts']) - initial_count >= args.max_attempts:
                    report['status'] = 'LONG_GENERATION_PAUSED_BETWEEN_ATTEMPTS'
                    return report
                if arm == 'awq_w3_replay':
                    verify_r0_files(args.r0_dir, review)
                    report['awq_replay'] = replay(model, args.r0_dir, upstream, device,
                        on_layer=lambda i: print(f'AWQ 저장 recipe {i + 1}/28', flush=True))
                model.cpu()
                digest = parameter_hash(model)
                if digest != cfg['parameter_sha256'][arm]:
                    raise ValueError(f'{arm} 파라미터 해시가 기존 모델과 다름')
                report['parameter_sha256'][arm] = digest
                model.to(device).eval()
                save()
                for problem in suite['problems']:
                    for seed in cfg['seeds']:
                        key = (problem['id'], arm, seed)
                        if key in completed:
                            continue
                        if len(report['attempts']) - initial_count >= args.max_attempts:
                            report['status'] = 'LONG_GENERATION_PAUSED_BETWEEN_ATTEMPTS'
                            return report
                        reserve_attempt(report, key, cfg)
                        save()
                        ids = inputs[problem['id']]
                        row = {'problem_id': problem['id'], 'arm': arm, 'seed': seed, 'level': problem['level'],
                               'input_ids': ids, 'input_ids_sha256': canonical_hash(ids), 'input_token_count': len(ids),
                               'prompt_sha256': canonical_hash(prompt_for(suite, problem)),
                               'execution_segment_index': len(report['execution_segments']) - 1}
                        tensor = torch.tensor([ids], device=device, dtype=torch.long)
                        torch.manual_seed(seed)
                        row['rng_before'] = {'cpu': hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest(),
                                             'cuda': hashlib.sha256(bytes(torch.cuda.get_rng_state(device).tolist())).hexdigest()}
                        label = f"{len(report['attempts']) + 1}/80 {arm} {problem['id']} seed={seed}"
                        print(f'시작 {label}', flush=True)
                        progress = ProgressReporter(label, cfg['max_new_tokens'])
                        class ProgressOnly(StoppingCriteria):
                            def __call__(self, input_ids, scores, **kwargs):
                                progress.tick(input_ids.shape[-1] - len(ids))
                                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                        trace = []
                        first_decode = len(ids)
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        started = time.monotonic()
                        with torch.inference_mode():
                            output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                generation_config=generation, use_cache=True,
                                stopping_criteria=StoppingCriteriaList([ProgressOnly()]))
                        torch.cuda.synchronize(device)
                        generated = output[0, len(ids):].detach().cpu().tolist()
                        row.update(generated_record(generated, tokenizer, cfg['max_new_tokens']),
                                   elapsed_seconds=time.monotonic() - started,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                   generation_mask_trace=copy.deepcopy(trace))
                        if row['finish_reason'] not in {'eos', 'length'} or not trace:
                            raise ValueError('생성 종료/마스크 기록 오류')
                        for marker in cfg['mask_trace_positions']:
                            if len(ids) <= marker <= len(ids) + len(generated) - 2:
                                if marker not in {x['query_start_position'] for x in trace}:
                                    raise ValueError('도달한 문맥 경계의 mask 기록 누락')
                        report['attempts'].append(row)
                        completed.add(key)
                        report.pop('active_attempt')
                        save()
                        checkpoint(args.output_dir / 'summary.json', summarize_long(report, cfg, suite))
                        print(f"완료 {label}: {row['finish_reason']}, {len(generated)}토큰", flush=True)
                        del output, tensor
            report['status'] = 'LONG_GENERATION_EXECUTED_PENDING_REVIEW'
            return report
        except BaseException as exc:
            report.update(status='LONG_GENERATION_INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'LONG_GENERATION_FAILED',
                          error_type=type(exc).__name__, error=str(exc)[:1000])
            raise
        finally:
            segment.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), completed_at_end=len(report['attempts']),
                           status=report['status'])
            save()
            checkpoint(args.output_dir / 'summary.json', summarize_long(report, cfg, suite))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--plan', action='store_true')
    parser.add_argument('--resume', action='store_true', help='응답 사이에서 멈춘 완료 기록만 재개')
    parser.add_argument('--max-attempts', type=int, default=80, help='이번 호출에서 완료할 응답 수. 1~80, 응답 사이에서 멈춤')
    parser.add_argument('--source-run', type=Path, default=ROOT / 'results/local/difficulty_v01/run.json')
    parser.add_argument('--r0-dir', type=Path, default=ROOT / 'results/local/r0_v03')
    parser.add_argument('--upstream-dir', type=Path, default=ROOT / 'results/local/awq_reference_v03')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/local/long_generation_v01')
    args = parser.parse_args(argv)
    try:
        cfg, suite, runtime = load_plan()
        if not 1 <= args.max_attempts <= 80:
            raise ValueError('--max-attempts는1~80이어야 합니다.')
        status = json.loads((ROOT / 'docs/research_status.json').read_text())
        if not args.execute:
            print(json.dumps({'status': 'LONG_GENERATION_PLAN_ONLY', 'config_sha256': canonical_hash(cfg),
                'authorization': status.get('long_generation', {}).get('authorization', 'PENDING'),
                'expected_attempts': 80, 'max_new_tokens': cfg['max_new_tokens'],
                'maximum_generation_tokens': cfg['maximum_generation_tokens'],
                'budget_thresholds': cfg['budget_thresholds'], 'sliding_window': None, 'use_cache': True,
                'model_or_tokenizer_loaded': False}, ensure_ascii=False, indent=2))
            return 0
        result = execute(args, cfg, suite, runtime, status)
        print(result['status'])
        print(f"검토할 파일: {args.output_dir / 'run.json'} 및 summary.json")
        return 0
    except Exception as exc:
        parser.exit(2, f'32K 재시험 중단: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
