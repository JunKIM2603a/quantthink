"""보고된 57완료/58시작 SSH 단절 한 건만 복구. 원본·소비 호출을 보존한다."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from difficulty_pilot_contracts import ROOT, canonical_hash
from difficulty_pilot_resume import attempt_key, resume_plan

POLICY = ROOT / 'configs/long_generation_ssh_recovery_v01.json'


def load_policy():
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    expected = {'recovery_id': 'long_generation_ssh_recovery_v01',
        'expected_status': 'LONG_GENERATION_RUNNING', 'expected_completed': 57,
        'expected_calls_started': 58, 'additional_generation_calls': 1,
        'maximum_generation_calls_including_abandoned': 81,
        'maximum_generation_tokens_including_abandoned': 2654208,
        'abandoned_call_token_upper_bound': 32768, 'expected_final_completed_attempts': 80,
        'remaining_completed_attempts_to_produce': 23, 'record_actual_interrupted_tokens_as': None,
        'restore_partial_rng_or_cache': False, 'automatic_retry': False, 'completed_attempts_repeated': False,
        'expected_active_attempt': {'problem_id': 'D3-01', 'arm': 'awq_w3_replay',
                                    'seed': 43, 'reserved_maximum_tokens': 32768}}
    if any(policy.get(k) != v for k, v in expected.items()):
        raise ValueError('보고된 SSH 단절 한 건의 복구 범위가 변경됐습니다.')
    source, target = policy['source_implementation_sha256'], policy['target_implementation_sha256']
    changed = {'run_long_generation.py', 'long_generation_contracts.py', 'long_generation_recovery.py'}
    if (len(source) != 13 or set(target) != set(source) | {'long_generation_recovery.py'}
            or set(policy['required_changes_only']) != changed
            or any(target[k] != v for k, v in source.items() if k not in changed)):
        raise ValueError('복구 이외의 기존 실행 코드 변경은 허용하지 않습니다.')
    return policy


def require_recovery_scope(status, hashes):
    policy = load_policy()
    scope = status.get('long_generation_ssh_recovery', {})
    if (scope.get('authorization') != 'USER_REQUESTED_SINGLE_SSH_RECOVERY'
            or scope.get('policy_sha256') != canonical_hash(policy)
            or hashes != policy['target_implementation_sha256']):
        raise ValueError('사용자 재시작 요청과 일치하는 제한 복구 기록/코드가 필요합니다.')


def read_report(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('재개 보고서는 심볼릭 링크일 수 없습니다.')
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_source(report, cfg, suite, policy):
    plan = resume_plan(report, cfg, suite)
    if (report.get('status') != policy['expected_status']
            or report.get('error') is not None or report.get('error_type') is not None
            or report.get('recovery_ledger') is not None
            or plan['completed'] != policy['expected_completed']
            or report.get('generation_calls_started') != policy['expected_calls_started']
            or report.get('active_attempt') != policy['expected_active_attempt']
            or plan['pending_keys'][0] != attempt_key(policy['expected_active_attempt'])
            or report.get('repository') != {'commit': policy['source_commit'], 'tracked_dirty': False}
            or report.get('implementation_sha256') != policy['source_implementation_sha256']
            or report.get('parameter_sha256') != cfg['parameter_sha256']
            or report.get('config_sha256') != policy['source_config_sha256']
            or canonical_hash(cfg) != policy['source_config_sha256']):
        raise ValueError('보고된 57완료/58시작·D3-01/AWQ/seed43 원본과 다릅니다. 자동 복구하지 않습니다.')
    return plan


def abandoned_attempt(policy):
    return {**policy['expected_active_attempt'], 'generation_call_index': 58,
            'actual_generated_tokens': None, 'charged_token_upper_bound': 32768,
            'partial_generated_ids_saved': False, 'rng_or_cache_restored': False,
            'reason_ko': '사용자 보고 SSH 단절 후 미완료 호출. 실제 종료 원인/소비 토큰 수는 미확인.'}


def recovery_allowance(report, cfg):
    ledger = report.get('recovery_ledger')
    if ledger is None:
        return 0
    policy = load_policy()
    source_sha = ledger.get('source_report_sha256', '')
    expected = {'recovery_id': policy['recovery_id'], 'policy_sha256': canonical_hash(policy),
        'source_status': policy['expected_status'], 'completed_before': 57,
        'source_generation_calls_started': 58, 'additional_authorized_calls': 1,
        'maximum_generation_calls': 81, 'abandoned_attempts': [abandoned_attempt(policy)],
        'source_implementation_sha256': policy['source_implementation_sha256'],
        'source_repository': {'commit': policy['source_commit'], 'tracked_dirty': False},
        'source_backup': {'file': f'run.before_resume.{source_sha}.json', 'sha256': source_sha}}
    if (len(source_sha) != 64 or any(c not in '0123456789abcdef' for c in source_sha)
            or any(ledger.get(k) != v for k, v in expected.items())
            or not ledger.get('recorded_at_utc')
            or len(report['attempts']) < 57
            or ledger.get('completed_prefix_sha256') != canonical_hash(report['attempts'][:57])
            or report.get('implementation_sha256') != policy['source_implementation_sha256']
            or report.get('repository') != expected['source_repository']
            or report.get('config_sha256') != policy['source_config_sha256']
            or canonical_hash(cfg) != policy['source_config_sha256']):
        raise ValueError('복구 기록·원본 참조·보존한 57개 응답이 일치하지 않습니다.')
    return 1


def prepare_recovery(previous, cfg, suite, source_sha):
    """메모리 복사본만 준비한다. 원본 백업 검증 전에는 디스크를 수정하지 않는다."""
    if previous.get('recovery_ledger') is not None:
        recovery_allowance(previous, cfg)
        if previous.get('active_attempt') or previous.get('status') in {
                'LONG_GENERATION_FAILED', 'LONG_GENERATION_INTERRUPTED'}:
            raise ValueError('이미 한 건을 복구했습니다. 추가 중단/오류는 자동 재시도하지 않습니다.')
        return copy.deepcopy(previous)
    policy = load_policy()
    plan = validate_source(previous, cfg, suite, policy)
    report = copy.deepcopy(previous)
    report['recovery_ledger'] = {
        'recovery_id': policy['recovery_id'], 'policy_sha256': canonical_hash(policy),
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_report_sha256': source_sha, 'source_status': previous['status'],
        'source_repository': copy.deepcopy(previous['repository']),
        'source_implementation_sha256': copy.deepcopy(previous['implementation_sha256']),
        'completed_before': plan['completed'], 'completed_prefix_sha256': plan['completed_attempts_sha256'],
        'source_generation_calls_started': previous['generation_calls_started'],
        'abandoned_attempts': [abandoned_attempt(policy)], 'additional_authorized_calls': 1,
        'maximum_generation_calls': 81,
        'source_backup': {'file': f'run.before_resume.{source_sha}.json', 'sha256': source_sha}}
    report.pop('active_attempt')
    report['status'] = 'LONG_GENERATION_RECOVERY_READY'
    recovery_allowance(report, cfg)
    return report


def verify_backup(report, folder, cfg, suite):
    if not recovery_allowance(report, cfg):
        return
    ledger = report['recovery_ledger']
    source, digest = read_report(Path(folder) / ledger['source_backup']['file'])
    if digest != ledger['source_report_sha256']:
        raise ValueError('복구 전 원본 백업 해시가 다릅니다.')
    validate_source(source, cfg, suite, load_policy())
    if source['attempts'] != report['attempts'][:57]:
        raise ValueError('완료한 57개 응답이 복구 전 원본과 달라졌습니다.')


def verify_code_migration(previous, current):
    policy = load_policy()
    if (not previous.get('recovery_ledger')
            or previous['implementation_sha256'] != policy['source_implementation_sha256']
            or current['implementation_sha256'] != policy['target_implementation_sha256']
            or previous['recovery_ledger'].get('policy_sha256') != canonical_hash(policy)):
        raise ValueError('검토한 복구 코드로의 변경만 허용합니다.')


def resume_candidate(path, cfg, suite, status, hashes, *, recover):
    """호출자가 출력 폴더 잠금을 잡은 상태에서 읽고 검증한다."""
    previous, source_sha = read_report(path)
    if recover or previous.get('recovery_ledger') is not None:
        require_recovery_scope(status, hashes)
    report = prepare_recovery(previous, cfg, suite, source_sha) if recover else previous
    if previous.get('recovery_ledger') is not None:
        verify_backup(report, Path(path).parent, cfg, suite)
    return report
