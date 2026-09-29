"""32K 재시험의 고정 범위·검열 집계·완료분 재개. 모델 의존성 없음."""
import json
from pathlib import Path
import statistics

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash, load_suite, summarize
from difficulty_pilot_resume import resume_plan
from run_fixed_prefix_diagnostic import IMPLEMENTATION as BASE_IMPLEMENTATION

CONFIG = ROOT / 'configs/long_generation_v01.json'
IMPLEMENTATION = ['run_long_generation.py', 'long_generation_contracts.py',
                  'difficulty_pilot_resume.py', *BASE_IMPLEMENTATION]


def load_plan(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    old, suite = load_suite()
    runtime = json.loads((ROOT / cfg['runtime_reference_path']).read_text())
    if canonical_hash(old) != cfg['base_config_sha256'] or canonical_hash(runtime) != cfg['runtime_reference_sha256']:
        raise ValueError('고정 원본 설정 또는 runtime 참조 해시 불일치')
    expected = dict(arms=old['arms'], seeds=[42, 43], suite_path=old['suite_path'],
                    suite_sha256=canonical_hash(suite), decoding=old['decoding'],
                    max_new_tokens=32768, max_prompt_tokens=768, max_total_context_tokens=33536,
                    maximum_generation_tokens=2621440, expected_attempts=80, maximum_generation_calls=80,
                    budget_thresholds=[4096, 8192, 16384, 32768], effective_sliding_window=None,
                    use_sliding_window=False, use_cache=True, expected_cache_class='DynamicCache',
                    low_censoring_rate_target_per_arm=.05, automatic_retry=False, calibration_search=False,
                    automatic_budget_increase=False, math500_used=False, h1_h2_test=False,
                    unfinished_attempt_automatic_retry=False, old_results_overwritten=False,
                    parameter_sha256=runtime['parameter_sha256'], level_pass_min_stable_problems=3)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('32K 고정 범위·디코딩·판정 계약 변경')
    return cfg, suite, runtime


def require_scope(status, cfg):
    scope = status.get('long_generation', {})
    if (status.get('research_approval') != 'APPROVED'
            or status.get('novelty_gate') != 'SCOPED_CONTRIBUTION_ACCEPTED'
            or status.get('protocol_gate') != 'ACCEPTED'
            or scope.get('authorization') != 'USER_REQUESTED_BOUNDED_LONG_GENERATION'
            or scope.get('config_sha256') != canonical_hash(cfg)):
        raise ValueError('사용자가 요청한 32K 재시험 범위의 실행 기록이 필요합니다.')
    hashes = {n: file_hash(ROOT / 'scripts' / n) for n in IMPLEMENTATION}
    if hashes != scope.get('implementation_sha256'):
        raise ValueError('기록된 실행 코드 해시와 다릅니다.')
    return hashes


def validate_resume(report, cfg, suite):
    plan = resume_plan(report, cfg, suite)
    if report.get('active_attempt') or report.get('status') in {'LONG_GENERATION_FAILED', 'LONG_GENERATION_INTERRUPTED'}:
        raise ValueError('미완료/오류 응답이 있습니다. 사용한 계산을 숨겨 재시도하지 말고 보고서부터 검토하세요.')
    if report.get('generation_calls_started') != plan['completed']:
        raise ValueError('시작한 생성 횟수와 완료 기록 불일치')
    if report['generation_calls_started'] > cfg['maximum_generation_calls']:
        raise ValueError('80회 생성 예산 초과')
    if report.get('config_sha256') != canonical_hash(cfg):
        raise ValueError('재개 설정 변경')
    return plan


def reserve_attempt(report, key, cfg):
    if report.get('active_attempt'):
        raise ValueError('이미 진행 중이거나 중단된 응답이 있습니다.')
    calls = report['generation_calls_started']
    if calls >= cfg['maximum_generation_calls']:
        raise ValueError('80회 생성 예산 소진')
    report['generation_calls_started'] = calls + 1
    report['active_attempt'] = {'problem_id': key[0], 'arm': key[1], 'seed': key[2],
                                'reserved_maximum_tokens': cfg['max_new_tokens']}


def budget_profiles(report, cfg):
    """한 경로의 여러 상한을 읽음. 각 상한마다 재생성하거나 검열을 오답으로 바꾸지 않음."""
    result = []
    rows = report['attempts']
    for cap in cfg['budget_thresholds']:
        for arm in cfg['arms']:
            group = [r for r in rows if r['arm'] == arm]
            if any(r['finish_reason'] not in {'eos', 'length'} for r in group):
                raise ValueError('예산별 집계에 실행 오류가 포함됨')
            if any(r['finish_reason'] == 'length' and len(r['generated_ids']) < cap for r in group):
                raise ValueError('상한보다 먼저 잘린 자료로 이후 종료율을 계산할 수 없습니다.')
            eos = sum(r['finish_reason'] == 'eos' and len(r['generated_ids']) <= cap for r in group)
            lengths = [min(len(r['generated_ids']), cap) for r in group]
            censored = len(group) - eos
            result.append({'budget': cap, 'arm': arm, 'expected_attempts': 40, 'observed_attempts': len(group),
                           'eos_by_budget': eos, 'censored_at_budget': censored,
                           'eos_rate_over_observed': eos / len(group) if group else None,
                           'censored_rate_over_observed': censored / len(group) if group else None,
                           'mean_min_length_budget_over_observed': statistics.mean(lengths) if lengths else None,
                           'all_40_observed': len(group) == 40,
                           'all_observed_natural_lengths_known': bool(group) and censored == 0,
                           'low_censoring_target_met': len(group) == 40 and censored / 40 <= .05})
    return result


def summarize_long(report, cfg, suite):
    summary = summarize(report, cfg, suite)
    summary['status'] = 'LONG_GENERATION_EXPLORATORY_SUMMARY_NOT_CERTIFICATION'
    summary['budget_profiles'] = budget_profiles(report, cfg)
    summary['generation_calls_started'] = report['generation_calls_started']
    summary['reserved_maximum_generation_tokens'] = report['generation_calls_started'] * cfg['max_new_tokens']
    summary['actual_completed_generation_tokens'] = sum(len(r['generated_ids']) for r in report['attempts'])
    summary['model_ready'] = False
    summary['comparison_to_v01_ko'] = ('출력 상한뿐 아니라 sliding_window=None의 전체 문맥 조건을 명시했다. '
        '과거 v01과의 차이를 상한만의 인과 효과로 해석하지 않는다. 새 실행 내부의4K/8K/16K/32K 비교는 동일 경로를 사용한다.')
    summary['interpretation'] = ('기존20문제 재시험인 탐색 자료. 종료·정답·상한 내 길이를 구분하고 검열을 오답으로 치환하지 않는다. '
        '5% 검열 목표는 운영 기준이며 검열이 남으면 전체 자연 길이 평균·H1은 확정하지 않는다.')
    return summary


def verify_resume_runtime(previous, current, inputs):
    from difficulty_pilot_resume import verify_compatibility
    verify_compatibility(previous, current, inputs)
    for key in ('implementation_sha256', 'source_run_sha256', 'source_configuration_sha256',
                'effective_source_configuration_sha256', 'installed_sources', 'runtime_contract'):
        if previous.get(key) != current.get(key):
            raise ValueError(f'32K 재개 조건 변경: {key}')


def validate_mask_trace(row):
    """생성 중 관측한 캐시 길이·마스크. None은 고정 SDPA 코드의 implicit 경로다."""
    if row['cache_class'] != 'DynamicCache':
        raise ValueError('예상하지 않은 생성 cache')
    if row['past_seen_tokens'] != row['query_start_position']:
        raise ValueError('생성 cache에 과거 문맥이 누락됨')
    if row['query_length'] == 1 and row['masked_past_or_current_keys'] != 0:
        raise ValueError('전체 문맥 조건에서 과거/현재 키가 차단됨')
