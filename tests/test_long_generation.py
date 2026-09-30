"""32K 새 범위의 검열·예산·재개·마스크 계약 검사. GPU/기존 시험을 실행하지 않음."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import long_generation_contracts as c
from reproduction_contracts import termination_record
from difficulty_pilot_contracts import prompt_for


class LongGenerationTest(unittest.TestCase):
    def setUp(self):
        self.cfg, self.suite, self.runtime = c.load_plan()

    def row(self, index=0, arm='bf16', seed=42, size=4096, eos=True):
        problem = self.suite['problems'][index]
        tokens = [10] * (size - (2 if eos else 1)) + [151649] + ([151643] if eos else [])
        ids = [151646, 151648, 198]
        return {'problem_id': problem['id'], 'level': problem['level'], 'arm': arm, 'seed': seed,
                'input_ids': ids, 'input_ids_sha256': c.canonical_hash(ids), 'input_token_count': len(ids),
                'prompt_sha256': c.canonical_hash(prompt_for(self.suite, problem)),
                'generated_ids': tokens, 'generated_text': 'analysis</think>\\boxed{' + problem['answer'] + '}',
                **termination_record(tokens, {151643}, self.cfg['max_new_tokens'])}

    def report(self, rows=None):
        rows = rows or []
        return {'config_sha256': c.canonical_hash(self.cfg), 'suite_sha256': c.canonical_hash(self.suite),
                'attempts': rows, 'parameter_sha256': self.cfg['parameter_sha256'].copy(),
                'generation_calls_started': len(rows), 'status': 'LONG_GENERATION_PAUSED_BETWEEN_ATTEMPTS'}

    def test_scope_and_frozen_original(self):
        self.assertEqual(self.cfg['maximum_generation_tokens'], 80 * 32768)
        old = json.loads((ROOT / self.cfg['base_config_path']).read_text())
        self.assertEqual(old['max_new_tokens'], 4096)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            for key, value in [('max_new_tokens', 65536), ('effective_sliding_window', 4096),
                               ('maximum_generation_calls', 81), ('seeds', [42]), ('use_cache', False)]:
                changed = copy.deepcopy(self.cfg)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    c.load_plan(path)

    def test_eos_at_cutoff_and_capped_lengths(self):
        report = self.report([self.row(size=4096), self.row(index=1, size=4097),
                              self.row(index=2, size=32768, eos=False)])
        profiles = c.budget_profiles(report, self.cfg)
        at4 = next(x for x in profiles if x['arm'] == 'bf16' and x['budget'] == 4096)
        self.assertEqual(at4['eos_by_budget'], 1)
        self.assertEqual(at4['censored_at_budget'], 2)
        self.assertEqual(at4['mean_min_length_budget_over_observed'], 4096)
        at32 = next(x for x in profiles if x['arm'] == 'bf16' and x['budget'] == 32768)
        self.assertEqual(at32['eos_by_budget'], 2)
        self.assertAlmostEqual(at32['mean_min_length_budget_over_observed'], (4096 + 4097 + 32768) / 3)
        self.assertFalse(at32['all_observed_natural_lengths_known'])
        self.assertFalse(at32['low_censoring_target_met'])

    def test_censoring_is_not_incorrect_or_success(self):
        report = self.report([self.row(size=32768, eos=False)])
        summary = c.summarize_long(report, self.cfg, self.suite)
        score = summary['per_attempt_scores'][0]
        self.assertEqual(score['outcome'], 'CENSORED')
        self.assertFalse(score['strict_success'])
        self.assertFalse(summary['model_ready'])
        self.assertEqual(summary['levels'][0]['verdict'], 'INCONCLUSIVE')

    def test_low_censoring_does_not_claim_full_natural_lengths(self):
        rows = [self.row(index=i // 2, seed=42 + i % 2, size=32768 if i < 2 else 50, eos=i >= 2)
                for i in range(40)]
        at32 = c.budget_profiles(self.report(rows), self.cfg)[-2]
        self.assertEqual(at32['censored_rate_over_observed'], .05)
        self.assertTrue(at32['low_censoring_target_met'])
        self.assertFalse(at32['all_observed_natural_lengths_known'])

    def test_insufficient_observation_is_rejected(self):
        bad = self.row(size=4096, eos=False)
        bad['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            c.budget_profiles(self.report([bad]), self.cfg)

    def test_resume_preserves_completed_and_rejects_active_or_budget_mismatch(self):
        report = self.report([self.row(size=50)])
        snapshot = copy.deepcopy(report)
        plan = c.validate_resume(report, self.cfg, self.suite)
        self.assertEqual((plan['completed'], plan['remaining']), (1, 79))
        self.assertEqual(plan['pending_keys'][0], (self.suite['problems'][0]['id'], 'bf16', 43))
        self.assertEqual(report, snapshot)
        c.reserve_attempt(report, plan['pending_keys'][0], self.cfg)
        self.assertEqual(report['generation_calls_started'], 2)
        with self.assertRaises(ValueError):
            c.validate_resume(report, self.cfg, self.suite)
        with self.assertRaises(ValueError):
            c.reserve_attempt(report, plan['pending_keys'][0], self.cfg)
        spent = self.report()
        spent['generation_calls_started'] = 80
        with self.assertRaises(ValueError):
            c.reserve_attempt(spent, plan['pending_keys'][0], self.cfg)

    def test_mask_trace_rejects_window_or_cache_loss(self):
        row = {'cache_class': 'DynamicCache', 'past_seen_tokens': 4096, 'query_start_position': 4096,
               'query_length': 1, 'masked_past_or_current_keys': 0}
        c.validate_mask_trace(row)
        for key, value in [('cache_class', 'SlidingWindowCache'), ('past_seen_tokens', 4095),
                           ('masked_past_or_current_keys', 1)]:
            with self.assertRaises(ValueError):
                c.validate_mask_trace({**row, key: value})

    def test_own_request_record_and_code_hashes(self):
        status = {'research_approval': 'APPROVED', 'novelty_gate': 'SCOPED_CONTRIBUTION_ACCEPTED',
                  'protocol_gate': 'ACCEPTED', 'functional_holdout': {'authorization': 'APPROVED'}}
        with self.assertRaises(ValueError):
            c.require_scope(status, self.cfg)
        status['long_generation'] = {'authorization': 'USER_REQUESTED_BOUNDED_LONG_GENERATION',
            'config_sha256': c.canonical_hash(self.cfg),
            'implementation_sha256': {n: c.file_hash(ROOT / 'scripts' / n) for n in c.IMPLEMENTATION}}
        self.assertEqual(c.require_scope(status, self.cfg), status['long_generation']['implementation_sha256'])
        status['long_generation']['implementation_sha256'] = {}
        with self.assertRaises(ValueError):
            c.require_scope(status, self.cfg)

    def test_cli_plan_and_unapproved_execution_do_not_import_models(self):
        code = """import json,pathlib,runpy,sys
class Block:
 def find_spec(self,fullname,*a,**k):
  if fullname.split('.')[0] in {'torch','numpy','transformers','huggingface_hub'}:raise RuntimeError('HEAVY_IMPORT')
sys.meta_path.insert(0,Block())
sys.path.insert(0,'scripts')
old=pathlib.Path.read_text
def read(p,*a,**k):
 if p.resolve()==pathlib.Path('docs/research_status.json').resolve():return '{}'
 return old(p,*a,**k)
pathlib.Path.read_text=read
sys.argv=['run_long_generation.py']+sys.argv[1:]
runpy.run_path('scripts/run_long_generation.py',run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'absent'
            for flag, expected in [('--plan', 0), ('--execute', 2)]:
                proc = subprocess.run([sys.executable, '-c', code, flag, '--output-dir', str(target)],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(proc.returncode, expected, proc.stderr)
                self.assertNotIn('HEAVY_IMPORT', proc.stderr)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
