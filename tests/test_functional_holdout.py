"""새16문맥 계약/판정만 검사. 기존 모델·토크나이저·AWQ 검사를 호출하지 않음."""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import functional_holdout_contracts as contract


class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):return list(range(40,100))


class PanelTest(unittest.TestCase):
    def setUp(self):self.cfg,self.suite,self.base=contract.load_plan()

    def data(self):
        return [{'reference':a,'observed':b,'case_id':case['id'],
                 't1':{'positions':32,'exact_equal':b=='B_repeat'},
                 'filtered_per_position':[{'position':k,'total_variation':0.,
                     'reference':{'eos_retained':False},'observed':{'eos_retained':False}} for k in range(32)]}
                for a,b in contract.PAIRS for case in self.suite['cases']]

    def test_independent_answer_calculations_and_no_old_prompt_reuse(self):
        expected=[Fraction(4*6+3*10,7),Fraction(250*80*110,10000),Fraction(1)/(Fraction(1,6)+Fraction(1,9)),
                  Fraction(20+7,5-2),math.comb(5,2)*4,math.comb(7,3)-math.comb(3,1)*math.comb(4,2),
                  math.factorial(7)//4,math.comb(7,2),Fraction(3*2,math.comb(5,2)),
                  Fraction(1,2)/(Fraction(1,2)+Fraction(1,4)),Fraction(3*3,3*3+3*3),Fraction(1,8),
                  pow(7,202,10),min(n for n in range(1,36) if n%5==2 and n%7==3),math.gcd(252,198),100//5+100//25]
        self.assertEqual([Fraction(c['reference_answer']) for c in self.suite['cases']],expected)
        old=json.loads((ROOT/'fixtures/difficulty_ladder_v01.json').read_text())
        normalize=lambda x:' '.join(x.lower().split())
        self.assertFalse({normalize(x['question_en']) for x in old['problems']} &
                         {normalize(x['question_en']) for x in self.suite['cases']})

    def test_input_position_and_eos_or_length_guard(self):
        original=copy.deepcopy(self.suite)
        with patch('runtime_contracts.prompt_ids',return_value=[1,2,3]):
            rows=contract.build_inputs(self.suite,FakeTokenizer(),self.cfg,self.base)
            self.assertEqual(rows[0]['logit_positions'],list(range(31,63)))
            self.assertEqual(sum(x['sequence_length'] for x in rows)*6,6048)
            with patch.object(FakeTokenizer,'encode',return_value=[42]*800):
                with self.assertRaises(ValueError):contract.build_inputs(self.suite,FakeTokenizer(),self.cfg,self.base)
            with patch.object(FakeTokenizer,'encode',return_value=[42]*31):
                with self.assertRaises(ValueError):contract.build_inputs(self.suite,FakeTokenizer(),self.cfg,self.base)
            with patch.object(FakeTokenizer,'encode',return_value=[42]*40+[151643]):
                with self.assertRaises(ValueError):contract.build_inputs(self.suite,FakeTokenizer(),self.cfg,self.base)
        self.assertEqual(self.suite,original)

    def test_frozen_scope_rejects_budget_threshold_and_suite_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json'
            for key,value in [('maximum_forward_calls',97),('temperature',1),('use_cache',True),
                              ('suite_sha256','0'*64),('acceptance',{'mean_filtered_tv_max':.02})]:
                c=copy.deepcopy(self.cfg);c[key]=value;p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):contract.load_plan(p)

    def test_exact_bound_and_quantization_comparison_are_distinct(self):
        data=self.data()
        for c in data:
            if c['observed']=='Q':
                for row in c['filtered_per_position']:row['total_variation']=1.
            elif c['observed']!='B_repeat':
                for row in c['filtered_per_position']:row['total_variation']=.01
        out=contract.classify(data,self.suite,self.cfg)
        self.assertEqual(out['status'],'WITHIN_ONE_STEP_BOUNDS_ON_FIXED_PANEL')
        self.assertFalse(out['model_ready']);self.assertFalse(out['sequence_equivalence_certified'])

    def test_local_outlier_mean_and_final_eos_are_separate_failure_modes(self):
        cases=[('local','POSITION_TV_EXCEEDS_BOUND'),('mean','MEAN_TV_EXCEEDS_BOUND'),('eos','FINAL_EOS_SUPPORT_DIFFERS')]
        for mode,reason in cases:
            data=self.data();group=[c for c in data if c['observed']=='S']
            if mode=='local':group[0]['filtered_per_position'][0]['total_variation']=.050001
            if mode=='mean':
                for c in group:
                    for row in c['filtered_per_position']:row['total_variation']=.010001
            if mode=='eos':group[3]['filtered_per_position'][-1]['observed']['eos_retained']=True
            out=contract.classify(data,self.suite,self.cfg)
            self.assertEqual(out['status'],'COORDINATE_EFFECT_EXCEEDS_ONE_STEP_BOUND')
            self.assertIn(reason,out['comparisons'][0]['violations'])

    def test_missing_duplicate_nonfinite_and_repeat_disagreement(self):
        data=self.data()
        with self.assertRaises(ValueError):contract.classify(data[:-1],self.suite,self.cfg)
        data[-1]=copy.deepcopy(data[-2])
        with self.assertRaises(ValueError):contract.classify(data,self.suite,self.cfg)
        data=self.data();data[0]['filtered_per_position'][0]['total_variation']=float('nan')
        with self.assertRaises(ValueError):contract.classify(data,self.suite,self.cfg)
        data=self.data();data[0]['t1']['exact_equal']=False
        self.assertEqual(contract.classify(data,self.suite,self.cfg)['status'],'REPEATABILITY_DIFFERENCE')

    def test_own_authorization_code_and_remaining_budget(self):
        s={'research_approval':'APPROVED','novelty_gate':'SCOPED_CONTRIBUTION_ACCEPTED','protocol_gate':'ACCEPTED',
           'termination_prefix_diagnostic':{'authorization':'APPROVED','remaining_forward_calls':6}}
        with self.assertRaises(ValueError):contract.require_authorization(s,self.cfg)
        scope={'authorization':'APPROVED','remaining_forward_calls':96,'config_sha256':contract.canonical_hash(self.cfg),
               'implementation_sha256':{n:contract.file_hash(ROOT/'scripts'/n) for n in contract.IMPLEMENTATION}}
        s['functional_holdout']=scope
        self.assertEqual(len(contract.require_authorization(s,self.cfg)),13)
        for key,val in [('remaining_forward_calls',0),('config_sha256','0'*64),('implementation_sha256',{})]:
            changed=copy.deepcopy(s);changed['functional_holdout'][key]=val
            with self.assertRaises(ValueError):contract.require_authorization(changed,self.cfg)

    def test_cli_plan_and_unapproved_execute_have_no_heavy_import_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'absent'
            code="""import json,pathlib,runpy,sys
class Block:
 def find_spec(self,fullname,*args,**kwargs):
  if fullname.split('.')[0] in {'torch','numpy','transformers','huggingface_hub'}:raise RuntimeError('HEAVY_IMPORT')
sys.meta_path.insert(0,Block())
sys.path.insert(0,'scripts')
original_read=pathlib.Path.read_text
def read(path,*a,**k):
 if path.resolve()==pathlib.Path('docs/research_status.json').resolve():
  return json.dumps({'research_approval':'APPROVED','novelty_gate':'SCOPED_CONTRIBUTION_ACCEPTED','protocol_gate':'ACCEPTED','functional_holdout':{'authorization':'PENDING','remaining_forward_calls':0}})
 return original_read(path,*a,**k)
pathlib.Path.read_text=read
sys.argv=['run_functional_holdout.py']+sys.argv[1:]
runpy.run_path('scripts/run_functional_holdout.py',run_name='__main__')
"""
            for mode,want in [('--plan',0),('--execute',2)]:
                p=subprocess.run([sys.executable,'-c',code,mode,'--output-dir',str(output),
                                  '--source-run',str(Path(d)/'missing.json')],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(p.returncode,want,p.stderr)
                self.assertNotIn('HEAVY_IMPORT',p.stderr)
                if mode=='--execute':self.assertIn('별도 승인',p.stderr)
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
