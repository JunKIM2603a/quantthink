"""새16문맥의 좌표 변환 점검 후보. 기본 --plan은 모델/토크나이저를 로드하지 않음."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time

from functional_holdout_contracts import (
    ROOT, PAIRS, canonical_hash, file_hash, load_plan, require_authorization, build_inputs, classify,
)


def execute(args,cfg,suite,base,status):
    implementation=require_authorization(status,cfg)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError('기존 결과를 보존합니다. 자동 재개/재시도 없음.')
    if file_hash(args.source_run)!=cfg['source_fixed_prefix_run_sha256']:
        raise ValueError('검토한 기존12회 보고와 다릅니다.')
    old=json.loads(args.source_run.read_text())
    if old['config_sha256']!=canonical_hash(base):raise ValueError('기존 runtime 참조 불일치')
    from run_difficulty_pilot import repo_state,parameter_hash,checkpoint,cached_tokenizer
    from runtime_assets import load_contracts,package_versions,verify_upstream
    from replay_r0_awq import verify_r0_files,validate_recipe,LINEARS,replay
    repository=repo_state()
    versions=package_versions(['torch','transformers','huggingface-hub','numpy'])
    if versions!=base['packages']:raise ValueError('고정 패키지 불일치. 자동 설치/업데이트 없음.')
    candidate,refs,policy=load_contracts()
    r0_review=json.loads((ROOT/base['r0_review']).read_text())
    r0,r0_files=verify_r0_files(args.r0_dir,r0_review)
    if r0['candidate_sha256']!=canonical_hash(candidate) or r0['policy_sha256']!=canonical_hash(policy):
        raise ValueError('기존 R0 후보/정책 불일치')
    upstream_files=verify_upstream(args.upstream_dir,policy)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM,Qwen2Config
    from qwen2_awq_adapter import load_upstream
    from run_fixed_prefix_diagnostic import verify_sources,scale_only,canonicalize,linear_hash
    from fixed_prefix_diagnostic_contracts import compare_logits
    from audit_fixed_prefix_logits import compare_filtered
    sources=verify_sources(base)
    tokenizer,asset=cached_tokenizer(refs)
    if asset!=base['model_reference']:raise ValueError('고정 모델 revision 불일치')
    token_files={}
    for name in ('tokenizer.json','tokenizer_config.json'):
        p=Path(hf_hub_download(asset['repo_id'],name,revision=asset['revision'],token=False,local_files_only=True))
        token_files[name]=file_hash(p)
    inputs=build_inputs(suite,tokenizer,cfg,base)
    config_path=Path(hf_hub_download(asset['repo_id'],'config.json',revision=asset['revision'],
                                   token=False,local_files_only=True))
    source=json.loads(config_path.read_text())
    effective=copy.deepcopy(source);effective['sliding_window']=None
    if (canonical_hash(source)!=base['model_configuration_sha256']
            or canonical_hash(effective)!=base['effective_source_configuration_sha256']):
        raise ValueError('원본/진단용 설정 불일치')
    device=torch.device(base['device'])
    if not torch.cuda.is_available():raise ValueError('기존 CUDA 환경이 필요합니다.')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():raise ValueError('BF16 지원이 필요합니다.')
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(False)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'status':'FUNCTIONAL_HOLDOUT_STARTED','model_ready':False,
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'repository':repository,
            'config_sha256':canonical_hash(cfg),'suite_sha256':canonical_hash(suite),
            'research_status_sha256':canonical_hash(status),'implementation_sha256':implementation,
            'source_fixed_prefix_run_sha256':cfg['source_fixed_prefix_run_sha256'],
            'packages':versions,'model_reference':asset,'tokenizer_file_sha256':token_files,
            'tokenizer_used_for_new_16_inputs_only':True,
            'inputs':inputs,'source_configuration_sha256':canonical_hash(source),
            'effective_source_configuration_sha256':canonical_hash(effective),
            'installed_sources':sources,'r0_input_files':r0_files,'upstream_source_files':upstream_files,
            'runtime':{'device':str(device),'gpu':torch.cuda.get_device_name(device),'dtype':'bfloat16',
                       'attention_implementation':'sdpa','use_cache':False,'sliding_window':None,
                       'tf32_matmul':False,'tf32_cudnn':False,'float32_matmul_precision':'highest',
                       'deterministic_algorithms':False,'seed':cfg['seed'],'sampling':False},
            'new_generation_tokens':0,'calibration_search':False,'forward_records':[],
            'parameter_sha256':{},'comparisons':[],'acceptance_bounds':cfg['acceptance'],
            'filtered_analysis':{'temperature':.6,'top_p':.95,'dtype':'float64',
                                 'tie_order':'ascending_token_id','actual_gpu_sampling_replay':False},
            'population_equivalence_test':False,'sequence_equivalence_certified':False}
    save=lambda:checkpoint(args.output_dir/'run.json',report)
    save()

    def load_b():
        model=AutoModelForCausalLM.from_pretrained(asset['repo_id'],revision=asset['revision'],token=False,
            local_files_only=True,trust_remote_code=False,use_safetensors=True,
            config=Qwen2Config.from_dict(copy.deepcopy(effective)),torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').eval()
        if (model.__class__.__name__!='Qwen2ForCausalLM' or len(model.model.layers)!=28
                or model.config.sliding_window is not None or model.config.use_sliding_window is not False
                or model.config._attn_implementation!='sdpa'
                or any(p.dtype!=torch.bfloat16 for p in model.parameters())
                or canonical_hash(model.config.to_dict())!=old['resolved_model_configuration_sha256']):
            raise ValueError('고정 모델 구조/runtime 설정 불일치')
        report['resolved_model_configuration_sha256']=canonical_hash(model.config.to_dict())
        return model

    def measure(model,state):
        model.cpu()
        digest=parameter_hash(model)
        if digest!=old['parameter_sha256'][state]:raise ValueError(f'{state} 파라미터 해시 불일치; forward 전 중단')
        report['parameter_sha256'][state]=digest
        model.to(device).eval()
        for row in inputs:
            if len(report['forward_records'])>=96:raise ValueError('96회 상한 초과')
            used=sum(x['sequence_length'] for x in report['forward_records'])
            if used+row['sequence_length']>73728:raise ValueError('처리입력 상한 초과')
            print(f"forward {len(report['forward_records'])+1}/96: {state} {row['case_id']}",flush=True)
            ids=torch.tensor([row['ids']],dtype=torch.long,device=device)
            pos=torch.arange(ids.shape[1],device=device)
            keep=torch.tensor(row['logit_positions'],dtype=torch.long,device=device)
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            start=time.monotonic()
            with torch.inference_mode():
                out=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=pos[None,:],
                    cache_position=pos,use_cache=False,output_attentions=False,output_hidden_states=False,
                    logits_to_keep=keep,return_dict=True)
            torch.cuda.synchronize(device)
            if out.past_key_values is not None:raise ValueError('예상하지 않은 cache 반환')
            arr=out.logits[0].float().cpu().numpy().copy()
            if arr.shape!=(32,base['vocab_size']) or not np.isfinite(arr).all():raise ValueError('logit shape/유한성 오류')
            path=args.output_dir/f"logits_{state}_{row['case_id']}.npy"
            with path.open('xb') as f:np.save(f,arr,allow_pickle=False)
            report['forward_records'].append({'state':state,'case_id':row['case_id'],
                'input_ids_sha256':row['sequence_ids_sha256'],'sequence_length':row['sequence_length'],
                'logit_positions':row['logit_positions'],'file':path.name,'file_sha256':file_hash(path),
                'shape':list(arr.shape),'stored_dtype':str(arr.dtype),
                'elapsed_seconds_including_save':time.monotonic()-start,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)})
            del out,arr,ids,pos,keep
            save()

    def compare(pairs):
        for left,right in pairs:
            for row in inputs:
                def read(which):
                    rec=next(r for r in report['forward_records'] if (r['state'],r['case_id'])==(which,row['case_id']))
                    p=args.output_dir/rec['file']
                    if file_hash(p)!=rec['file_sha256']:raise ValueError('저장 원시 logit 해시 변경')
                    return np.load(p,mmap_mode='r',allow_pickle=False)
                a,b=read(left),read(right)
                t1=compare_logits(a,b,row['logit_positions'],base['eos_token_id'])
                filtered=[{'position':p,**compare_filtered(a[i],b[i],base['eos_token_id'])}
                          for i,p in enumerate(row['logit_positions'])]
                report['comparisons'].append({'reference':left,'observed':right,'case_id':row['case_id'],
                                              't1':t1,'filtered_per_position':filtered})
                del a,b
                save()
                print(f"비교 {len(report['comparisons'])}/80: {left}/{right} {row['case_id']}",flush=True)

    try:
        if shutil.disk_usage(args.output_dir).free < cfg['maximum_raw_logit_bytes'] + 32*1024*1024:
            raise ValueError('원시 logit과 보고서를 보존할 디스크 여유 공간이 부족합니다.')
        model=load_b()
        original_aux=[]
        for i,layer in enumerate(model.model.layers):
            rec=json.loads((args.r0_dir/f'awq_layer_{i:03d}.json').read_text())
            validate_recipe(rec,i,{n:list(p.shape) for n,p in layer.named_parameters()})
            original_aux.append({n:p.detach().cpu().clone() for n,p in layer.named_parameters()
                                 if n not in {x+'.weight' for x in LINEARS}})
        upstream=load_upstream(args.upstream_dir,policy)
        measure(model,'B');measure(model,'B_repeat')
        compare(PAIRS[:1])
        if not all(c['t1']['exact_equal'] for c in report['comparisons']):
            report['status']='FUNCTIONAL_HOLDOUT_BASELINE_REPEAT_DIFFERENCE'
            report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
            return report
        scale_only(model,args.r0_dir,upstream,device);measure(model,'S')
        model.cpu();del model;gc.collect();torch.cuda.empty_cache()
        model=load_b()
        verify_r0_files(args.r0_dir,r0_review)
        report['awq_replay']=replay(model,args.r0_dir,upstream,device,
                                   on_layer=lambda i:print(f'Q recipe {i+1}/28',flush=True))
        measure(model,'Q')
        verify_r0_files(args.r0_dir,r0_review)
        report['auxiliary_coordinates']=canonicalize(model,args.r0_dir,original_aux,device)
        measure(model,'C')
        model.cpu();before=linear_hash(model)
        with torch.no_grad():
            for layer,aux in zip(model.model.layers,original_aux):
                for name,value in aux.items():
                    target=layer.get_parameter(name);target.copy_(value)
                    if not torch.equal(target,value):raise ValueError('Cw 보조 파라미터 복원 오류')
        after=linear_hash(model)
        if before!=after:raise ValueError('Cw에서 선형 weight가 바뀜')
        report['C_to_Cw_linear_sha256']={'before':before,'after':after}
        measure(model,'Cw')
        model.cpu();del model;gc.collect();torch.cuda.empty_cache()
        compare(PAIRS[1:])
        if len(report['forward_records'])!=96:raise ValueError('forward 수집 누락')
        report['panel_verdict']=classify(report['comparisons'],suite,cfg)
        report['status']='FUNCTIONAL_HOLDOUT_COLLECTED_PENDING_REVIEW'
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
        return report
    except BaseException as exc:
        report.update(status='FUNCTIONAL_HOLDOUT_INTERRUPTED' if isinstance(exc,KeyboardInterrupt)
                      else 'FUNCTIONAL_HOLDOUT_FAILED',error_type=type(exc).__name__,error=str(exc)[:1000])
        save();raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=ROOT/'results/local/fixed_prefix_v01/run.json')
    parser.add_argument('--r0-dir',type=Path,default=ROOT/'results/local/r0_v03')
    parser.add_argument('--upstream-dir',type=Path,default=ROOT/'results/local/awq_reference_v03')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/local/functional_holdout_v01')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--plan',action='store_true')
    group.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    try:
        cfg,suite,base=load_plan()
        status=json.loads((ROOT/'docs/research_status.json').read_text())
        if not args.execute:
            print(json.dumps({'status':'FUNCTIONAL_HOLDOUT_PLAN_ONLY','authorization':status.get('functional_holdout',{}).get('authorization','PENDING'),
                'config_sha256':canonical_hash(cfg),'suite_sha256':canonical_hash(suite),'cases':len(suite['cases']),
                **{k:cfg[k] for k in ['maximum_forward_calls','maximum_forward_input_tokens','maximum_sequence_length',
                                     'maximum_raw_logit_bytes','new_generation_tokens','acceptance']},
                'actual_token_lengths':'NOT_TOKENIZED_YET','model_or_tokenizer_loaded':False},ensure_ascii=False,indent=2))
            return 0
        require_authorization(status,cfg)
        report=execute(args,cfg,suite,base,status)
        print(report['status']);print(f"검토할 파일: {args.output_dir/'run.json'}")
        return 0
    except Exception as exc:
        parser.exit(2,f'새16문맥 진단 중단: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':raise SystemExit(main())
