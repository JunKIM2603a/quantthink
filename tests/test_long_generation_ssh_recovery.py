"""SSH 복구의 저장·호출 회계·잠금 검사. 합성 JSON만 사용하며 GPU를 실행하지 않는다."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import long_generation_contracts as c
import long_generation_recovery as recovery
import run_long_generation as runner
from difficulty_pilot_contracts import prompt_for
from difficulty_pilot_resume import backup_before_resume, output_lock, planned_order
from reproduction_contracts import termination_record


class SSHRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.cfg, self.suite, self.runtime = c.load_plan()
        self.policy = recovery.load_policy()
        self.hashes = {n: c.file_hash(ROOT / 'scripts' / n) for n in c.IMPLEMENTATION}
        self.status = {'research_approval': 'APPROVED', 'novelty_gate': 'SCOPED_CONTRIBUTION_ACCEPTED',
            'protocol_gate': 'ACCEPTED', 'long_generation': {
                'authorization': 'USER_REQUESTED_BOUNDED_LONG_GENERATION',
                'config_sha256': c.canonical_hash(self.cfg), 'implementation_sha256': self.hashes},
            'long_generation_ssh_recovery': {'authorization': 'USER_REQUESTED_SINGLE_SSH_RECOVERY',
                'policy_sha256': c.canonical_hash(self.policy)}}
        self.order = planned_order(self.cfg, self.suite)

    def row(self, key):
        problem = next(p for p in self.suite['problems'] if p['id'] == key[0])
        ids, generated = [151646, 151648, 198], [10, 151649, 151643]
        return {'problem_id': key[0], 'arm': key[1], 'seed': key[2], 'level': problem['level'],
            'input_ids': ids, 'input_ids_sha256': c.canonical_hash(ids), 'input_token_count': len(ids),
            'prompt_sha256': c.canonical_hash(prompt_for(self.suite, problem)),
            'generated_ids': generated, 'generated_text': 'analysis</think>\\boxed{' + problem['answer'] + '}',
            **termination_record(generated, {151643}, self.cfg['max_new_tokens'])}

    def source(self):
        return {'config_sha256': c.canonical_hash(self.cfg), 'suite_sha256': c.canonical_hash(self.suite),
            'attempts': [self.row(key) for key in self.order[:57]],
            'parameter_sha256': copy.deepcopy(self.cfg['parameter_sha256']),
            'generation_calls_started': 58, 'status': 'LONG_GENERATION_RUNNING',
            'active_attempt': copy.deepcopy(self.policy['expected_active_attempt']),
            'repository': {'commit': self.policy['source_commit'], 'tracked_dirty': False},
            'implementation_sha256': copy.deepcopy(self.policy['source_implementation_sha256']),
            'execution_segments': [], 'error': None}

    def candidate(self):
        return recovery.prepare_recovery(self.source(), self.cfg, self.suite, 'a' * 64)

    def test_preserve_57_and_charge_interrupted_call(self):
        source = self.source()
        snapshot = copy.deepcopy(source)
        report = recovery.prepare_recovery(source, self.cfg, self.suite, 'a' * 64)
        self.assertEqual(source, snapshot)
        self.assertEqual(report['attempts'], source['attempts'])
        self.assertEqual(report['generation_calls_started'], 58)
        self.assertNotIn('active_attempt', report)
        plan = c.validate_resume(report, self.cfg, self.suite)
        self.assertEqual((plan['completed'], plan['remaining']), (57, 23))
        self.assertEqual(plan['pending_keys'][0], ('D3-01', 'awq_w3_replay', 43))
        abandoned = report['recovery_ledger']['abandoned_attempts'][0]
        self.assertIsNone(abandoned['actual_generated_tokens'])
        self.assertEqual(abandoned['charged_token_upper_bound'], 32768)

    def test_finish_23_once_with_81_physical_calls_and_80_results(self):
        report = self.candidate()
        for key in self.order[57:]:
            c.reserve_attempt(report, key, self.cfg)
            with self.assertRaises(ValueError):
                c.reserve_attempt(report, key, self.cfg)
            report['attempts'].append(self.row(key))
            report.pop('active_attempt')
        self.assertEqual(c.validate_resume(report, self.cfg, self.suite)['remaining'], 0)
        self.assertEqual(report['generation_calls_started'], 81)
        with self.assertRaises(ValueError):
            c.reserve_attempt(report, self.order[-1], self.cfg)
        summary = c.summarize_long(report, self.cfg, self.suite)
        self.assertEqual(summary['abandoned_generation_calls'], 1)
        self.assertEqual(summary['maximum_generation_tokens_including_abandoned'], 2654208)
        self.assertEqual(summary['actual_completed_generation_tokens'], 240)
        self.assertIsNone(summary['abandoned_actual_generated_tokens'])
        self.assertFalse(summary['total_actual_generation_tokens_known'])
        self.assertFalse(summary['model_ready'])

    def test_idempotent_flag_but_second_interruption_rejected(self):
        report = self.candidate()
        self.assertEqual(recovery.prepare_recovery(report, self.cfg, self.suite, 'b' * 64), report)
        c.reserve_attempt(report, self.order[57], self.cfg)
        with self.assertRaises(ValueError):
            recovery.prepare_recovery(report, self.cfg, self.suite, 'b' * 64)
        with self.assertRaises(ValueError):
            c.validate_resume(report, self.cfg, self.suite)
        report.pop('active_attempt')
        report['status'] = 'LONG_GENERATION_FAILED'
        with self.assertRaises(ValueError):
            recovery.prepare_recovery(report, self.cfg, self.suite, 'b' * 64)

    def test_wrong_source_rejected_without_mutation(self):
        changes = [('status', 'LONG_GENERATION_FAILED'), ('error', 'OOM'), ('error_type', 'RuntimeError'),
            ('generation_calls_started', 57), ('attempts', self.source()['attempts'][:56]),
            ('active_attempt', {**self.policy['expected_active_attempt'], 'seed': 42}),
            ('repository', {'commit': '0' * 40, 'tracked_dirty': False}),
            ('implementation_sha256', self.hashes), ('config_sha256', '0' * 64),
            ('parameter_sha256', {'bf16': '0' * 64, 'awq_w3_replay': '1' * 64})]
        for key, value in changes:
            with self.subTest(key=key):
                report = self.source()
                report[key] = value
                before = copy.deepcopy(report)
                with self.assertRaises(ValueError):
                    recovery.prepare_recovery(report, self.cfg, self.suite, 'a' * 64)
                self.assertEqual(report, before)

    def test_plain_resume_still_rejects_interrupted_and_no_extra_allowance(self):
        with self.assertRaises(ValueError):
            c.validate_resume(self.source(), self.cfg, self.suite)
        report = self.source()
        report.pop('active_attempt')
        report['attempts'] = [self.row(k) for k in self.order]
        report['generation_calls_started'] = 80
        self.assertEqual(recovery.recovery_allowance(report, self.cfg), 0)
        with self.assertRaises(ValueError):
            c.reserve_attempt(report, self.order[-1], self.cfg)

    def test_tampered_ledger_prefix_or_counter_is_rejected(self):
        for field, value in [('additional_authorized_calls', 2), ('maximum_generation_calls', 82),
                             ('source_report_sha256', '../escape'), ('policy_sha256', '0' * 64)]:
            report = self.candidate()
            report['recovery_ledger'][field] = value
            with self.assertRaises(ValueError):
                c.validate_resume(report, self.cfg, self.suite)
        report = self.candidate()
        report['attempts'][0]['generated_text'] += ' changed'
        with self.assertRaises(ValueError):
            c.validate_resume(report, self.cfg, self.suite)
        report = self.candidate()
        report['generation_calls_started'] = 57
        with self.assertRaises(ValueError):
            c.validate_resume(report, self.cfg, self.suite)

    def test_exact_original_backup_and_readonly_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'run.json'
            raw = (json.dumps(self.source(), ensure_ascii=False, indent=3) + '\n').encode()
            path.write_bytes(raw)
            with output_lock(folder, resume=True):
                report = recovery.resume_candidate(path, self.cfg, self.suite, self.status,
                                                    self.hashes, recover=True)
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(list(folder.glob('run.before_resume.*')), [])
                with self.assertRaises(FileNotFoundError):
                    recovery.verify_backup(report, folder, self.cfg, self.suite)
                backup = backup_before_resume(path)
                recovery.verify_backup(report, folder, self.cfg, self.suite)
                self.assertEqual(backup, report['recovery_ledger']['source_backup'])
                backup_path = folder / backup['file']
                self.assertEqual(backup_path.read_bytes(), raw)
                self.assertEqual(backup['sha256'], hashlib.sha256(raw).hexdigest())
                backup_path.write_bytes(raw + b' ')
                with self.assertRaises(ValueError):
                    recovery.verify_backup(report, folder, self.cfg, self.suite)
                backup_path.unlink()
                backup_path.symlink_to(path)
                with self.assertRaises(ValueError):
                    recovery.verify_backup(report, folder, self.cfg, self.suite)

    def test_runtime_migration_requires_exact_source_target_and_conditions(self):
        report = self.candidate()
        current = copy.deepcopy(report)
        current['implementation_sha256'] = self.hashes
        inputs = {p['id']: [151646, 151648, 198] for p in self.suite['problems']}
        c.verify_resume_runtime(report, current, inputs)
        for key in ['packages', 'runtime_contract', 'resolved_generation_config', 'installed_sources']:
            changed = copy.deepcopy(current)
            changed[key] = {'changed': True}
            with self.assertRaises(ValueError):
                c.verify_resume_runtime(report, changed, inputs)
        changed = copy.deepcopy(current)
        changed['implementation_sha256']['run_long_generation.py'] = '0' * 64
        with self.assertRaises(ValueError):
            c.verify_resume_runtime(report, changed, inputs)
        report.pop('recovery_ledger')
        with self.assertRaises(ValueError):
            c.verify_resume_runtime(report, current, inputs)

    def test_recovery_scope_and_current_hashes(self):
        self.assertEqual(c.require_scope(self.status, self.cfg), self.hashes)
        recovery.require_recovery_scope(self.status, self.hashes)
        for key, value in [('authorization', 'PENDING'), ('policy_sha256', '0' * 64)]:
            changed = copy.deepcopy(self.status)
            changed['long_generation_ssh_recovery'][key] = value
            with self.assertRaises(ValueError):
                recovery.require_recovery_scope(changed, self.hashes)
        with self.assertRaises(ValueError):
            recovery.require_recovery_scope(self.status, {})

    def test_running_worker_blocks_before_heavy_import_and_leaves_report(self):
        code = """import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'scripts')
class Block:
 def find_spec(self,fullname,*a,**k):
  if fullname.split('.')[0] in {'torch','numpy','transformers','huggingface_hub'}:raise RuntimeError('HEAVY_IMPORT')
sys.meta_path.insert(0,Block())
import run_long_generation as r
import long_generation_contracts as c
cfg,suite,runtime=c.load_plan()
status=json.loads(Path(sys.argv[2]).read_text())
try:r.execute(SimpleNamespace(output_dir=Path(sys.argv[1]),resume=True,recover_interrupted=True),cfg,suite,runtime,status)
except ValueError as e:print(str(e));sys.exit(2)
"""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'run.json'
            raw = json.dumps(self.source()).encode()
            path.write_bytes(raw)
            statuspath = folder / 'status.json'
            statuspath.write_text(json.dumps(self.status))
            with output_lock(folder, resume=True):
                proc = subprocess.run([sys.executable, '-c', code, str(folder), str(statuspath)],
                                      cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn('실행 중', proc.stdout)
            self.assertNotIn('HEAVY_IMPORT', proc.stderr)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(list(folder.glob('run.before_resume.*')), [])

    def test_invalid_cli_flags_and_unapproved_recovery_leave_no_output(self):
        for flags in [['--recover-interrupted'], ['--execute', '--recover-interrupted'],
                      ['--plan', '--resume', '--recover-interrupted']]:
            with self.assertRaises(SystemExit) as raised:
                runner.main(flags)
            self.assertEqual(raised.exception.code, 2)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'run.json'
            raw = json.dumps(self.source()).encode()
            path.write_bytes(raw)
            status = copy.deepcopy(self.status)
            status.pop('long_generation_ssh_recovery')
            with self.assertRaises(ValueError):
                runner.execute(SimpleNamespace(output_dir=folder, resume=True, recover_interrupted=True),
                               self.cfg, self.suite, self.runtime, status)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(list(folder.glob('run.before_resume.*')), [])


if __name__ == '__main__':
    unittest.main()
