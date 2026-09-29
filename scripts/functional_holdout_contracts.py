"""새 고정16문맥의 입력·예산·수락 규칙. 표준 라이브러리만 사용."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from difficulty_pilot_contracts import ROOT, canonical_hash, file_hash
from run_fixed_prefix_diagnostic import IMPLEMENTATION as BASE_IMPLEMENTATION

CONFIG = ROOT / 'configs/functional_holdout_v01.json'
IMPLEMENTATION = ['run_functional_holdout.py', 'functional_holdout_contracts.py',
                  'audit_fixed_prefix_logits.py', *BASE_IMPLEMENTATION]
STATES = ['B', 'B_repeat', 'S', 'Q', 'C', 'Cw']
PAIRS = [['B','B_repeat'], ['B','S'], ['B','Q'], ['Q','C'], ['C','Cw']]
BOUNDS = {'mean_filtered_tv_max': .01, 'maximum_filtered_tv_max': .05,
          'final_anchor_eos_support_mismatches_max': 0}


def load_plan(config_path=CONFIG):
    cfg=json.loads(Path(config_path).read_text())
    suite=json.loads((ROOT/cfg['suite_path']).read_text())
    base=json.loads((ROOT/cfg['base_config_path']).read_text())
    if canonical_hash(suite)!=cfg['suite_sha256'] or canonical_hash(base)!=cfg['base_config_sha256']:
        raise ValueError('고정 문맥 또는 기존 runtime 설정 해시 불일치')
    expected={'states':STATES,'comparisons':PAIRS,'acceptance_comparisons':[['B','S'],['Q','C'],['C','Cw']],
              'number_of_inputs':16,'positions_per_input':32,'maximum_sequence_length':768,
              'maximum_forward_calls':96,'maximum_forward_input_tokens':73728,
              'maximum_raw_logit_bytes':1866989568,'new_generation_tokens':0,
              'temperature':.6,'top_p':.95,'use_cache':False,'effective_sliding_window':None,
              'position_rule':'last_32_consecutive_full_causal_context','seed':42,'sampling':False,
              'acceptance':BOUNDS,'calibration_search':False,'network_download':False,
              'automatic_retry':False,'math500_used':False,'r1_used':False,'new_pile_used':False,'h1_h2_test':False}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('고정 패널·범위·판정 변경')
    rows=suite['cases']
    if ([r['id'] for r in rows]!=[f'FH-{i:02d}' for i in range(1,17)]
            or Counter(r['family'] for r in rows)!={k:4 for k in ['algebra','counting','probability','number_theory']}
            or Counter(r['phase'] for r in rows)!={k:4 for k in ['setup','calculation','verification','final']}
            or len({r['question_en'] for r in rows})!=16):raise ValueError('16개 입력 구성 불일치')
    for row in rows:
        text=row['assistant_prefix_en']
        if not text or text.count('</think>')!=(row['phase']=='final'):
            raise ValueError('풀이 단계·완성 답 경계 불일치')
    return cfg,suite,base


def require_authorization(status,cfg):
    scope=status.get('functional_holdout',{})
    if (status.get('research_approval')!='APPROVED'
            or status.get('novelty_gate')!='SCOPED_CONTRIBUTION_ACCEPTED'
            or status.get('protocol_gate')!='ACCEPTED'
            or scope.get('authorization')!='APPROVED'
            or scope.get('remaining_forward_calls')!=96
            or scope.get('config_sha256')!=canonical_hash(cfg)):
        raise ValueError('새16문맥·최대96회 범위는 별도 승인 전이거나 잔여 예산이 없습니다.')
    hashes={n:file_hash(ROOT/'scripts'/n) for n in IMPLEMENTATION}
    if hashes!=scope.get('implementation_sha256'):raise ValueError('승인된 구현 해시와 다름')
    return hashes


def build_inputs(suite,tokenizer,cfg,base):
    from runtime_contracts import prompt_ids
    rows=[]
    for case in suite['cases']:
        prompt=case['question_en']+'\n\n'+suite['instruction']
        pids=prompt_ids(tokenizer,prompt)
        prefix=tokenizer.encode(case['assistant_prefix_en'],add_special_tokens=False)
        ids=pids+prefix
        if (len(prefix)<32 or len(ids)>768 or base['eos_token_id'] in ids
                or any(type(x) is not int or not 0<=x<base['vocab_size'] for x in ids)):
            raise ValueError(f"{case['id']}: 짧은 prefix·문맥 상한·EOS·토큰 범위 오류. 입력을 자르지 않습니다.")
        rows.append({'case_id':case['id'],'family':case['family'],'phase':case['phase'],
                     'ids':ids,'sequence_ids_sha256':canonical_hash(ids),'sequence_length':len(ids),
                     'prompt_token_count':len(pids),'prefill_token_count':len(prefix),
                     'logit_positions':list(range(len(ids)-32,len(ids)))})
    if len(rows)!=16 or 6*sum(r['sequence_length'] for r in rows)>73728:
        raise ValueError('전체 forward 입력 예산 초과')
    return rows


def classify(comparisons,suite,cfg):
    expected=[(a,b,r['id']) for a,b in PAIRS for r in suite['cases']]
    keys=[(c['reference'],c['observed'],c['case_id']) for c in comparisons]
    if keys!=expected:raise ValueError('80개 비교의 순서·중복·누락')
    for c in comparisons:
        if len(c['filtered_per_position'])!=32 or c['t1']['positions']!=32:
            raise ValueError('위치 개수 불일치')
        for row in c['filtered_per_position']:
            tv=row['total_variation']
            if not math.isfinite(tv) or not 0<=tv<=1+1e-12:raise ValueError('TV 유한성·범위 오류')
    if not all(c['t1']['exact_equal'] for c in comparisons if c['observed']=='B_repeat'):
        return {'status':'REPEATABILITY_DIFFERENCE','model_ready':False}
    results=[]
    final_ids={r['id'] for r in suite['cases'] if r['phase']=='final'}
    for left,right in cfg['acceptance_comparisons']:
        group=[c for c in comparisons if [c['reference'],c['observed']]==[left,right]]
        means=[statistics.mean(r['total_variation'] for r in c['filtered_per_position']) for c in group]
        mean=statistics.mean(means)
        maximum=max(r['total_variation'] for c in group for r in c['filtered_per_position'])
        mismatch=sum(c['filtered_per_position'][-1]['reference']['eos_retained']!=
                     c['filtered_per_position'][-1]['observed']['eos_retained'] for c in group if c['case_id'] in final_ids)
        limits=cfg['acceptance']
        reasons=[]
        if mean>limits['mean_filtered_tv_max']:reasons.append('MEAN_TV_EXCEEDS_BOUND')
        if maximum>limits['maximum_filtered_tv_max']:reasons.append('POSITION_TV_EXCEEDS_BOUND')
        if mismatch>limits['final_anchor_eos_support_mismatches_max']:reasons.append('FINAL_EOS_SUPPORT_DIFFERS')
        results.append({'reference':left,'observed':right,'mean_case_tv':mean,'maximum_position_tv':maximum,
                        'final_anchor_eos_support_mismatches':mismatch,'violations':reasons,
                        'within_declared_bounds':not reasons})
    return {'status':'WITHIN_ONE_STEP_BOUNDS_ON_FIXED_PANEL' if all(r['within_declared_bounds'] for r in results)
                    else 'COORDINATE_EFFECT_EXCEEDS_ONE_STEP_BOUND',
            'comparisons':results,'model_ready':False,'population_equivalence_test':False,
            'sequence_equivalence_certified':False,'h1_h2_tested':False}
