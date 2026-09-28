import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import resolve_asset_revisions as resolver
import reproduction_contracts as contracts

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / 'configs/reproduction_candidate.json').read_text())

    @staticmethod
    def fake_fetch(url):
        if '/api/models/' in url:
            return {'id': url.split('/api/models/')[1], 'sha': 'a' * 40, 'private': False}
        if '/api/datasets/' in url:
            return {'id': url.split('/api/datasets/')[1], 'sha': 'b' * 40, 'private': False}
        if '/resolve/' in url:
            return {'fake_config_fixture': True}
        raise AssertionError(url)

    def test_success_is_not_run_ready(self):
        report = resolver.resolve_assets(self.plan, self.fake_fetch)
        self.assertEqual(report['status'], 'METADATA_RESOLVED_NOT_RUN_READY')
        self.assertEqual(len(report['assets']), 4)

    def test_files_use_resolved_sha(self):
        calls = []
        def get(url):
            calls.append(url)
            return self.fake_fetch(url)
        resolver.resolve_assets(self.plan, get)
        files = [u for u in calls if '/resolve/' in u]
        self.assertEqual(len(files), 3)
        self.assertTrue(all('/' + 'a' * 40 + '/' in u for u in files))
        self.assertFalse(any(u.endswith('.safetensors') or u.endswith('.parquet') for u in calls))

    def test_invalid_sha_fails(self):
        report = resolver.resolve_assets(self.plan, lambda _: {'sha': 'main'})
        self.assertEqual(report['status'], 'METADATA_INCOMPLETE')
        self.assertEqual(len(report['errors']), 4)

    def test_http_failure_recorded(self):
        def fail(url):
            raise HTTPError(url, 503, 'unavailable', None, None)
        report = resolver.resolve_assets(self.plan, fail)
        self.assertEqual(report['errors'][0]['http_status'], 503)

    def test_missing_config_is_incomplete(self):
        def get(url):
            if url.endswith('generation_config.json'):
                raise HTTPError(url, 404, 'not found', None, None)
            return self.fake_fetch(url)
        self.assertEqual(resolver.resolve_assets(self.plan, get)['status'], 'METADATA_INCOMPLETE')

    def test_alias_not_silently_replaced(self):
        report = resolver.resolve_assets(self.plan, lambda _: {'sha': 'a' * 40, 'id': 'other/repo'})
        self.assertEqual(report['status'], 'METADATA_INCOMPLETE')

    def test_bad_repo_rejected(self):
        for repo in ('https://example.com/x', '../other/repo', 'a/b/c'):
            with self.assertRaises(ValueError):
                resolver.metadata_url({'repo_type': 'model', 'repo_id': repo})

    def test_unknown_repo_type_rejected(self):
        with self.assertRaises(ValueError):
            resolver.metadata_url({'repo_type': 'space', 'repo_id': 'a/b'})

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.json'
            resolver.write_new_report(path, {'first': True})
            with self.assertRaises(FileExistsError):
                resolver.write_new_report(path, {'second': True})
            self.assertEqual(json.loads(path.read_text()), {'first': True})

    def test_plan_mode_no_network(self):
        with patch.object(sys, 'argv', ['resolver', '--manifest', str(ROOT / 'configs/reproduction_candidate.json')]), patch.object(resolver, 'resolve_assets', side_effect=AssertionError('network')), patch('builtins.print'):
            self.assertEqual(resolver.main(), 0)

    def test_exact_blocks(self):
        blocks = contracts.exact_token_blocks([[1] * 400 for _ in range(164)])
        self.assertEqual(len(blocks), 128)
        self.assertTrue(all(len(block) == 512 for block in blocks))

    def test_128_documents_not_128_blocks(self):
        with self.assertRaises(ValueError):
            contracts.exact_token_blocks([[1] * 400 for _ in range(128)])

    def test_invalid_tokens(self):
        with self.assertRaises(ValueError):
            contracts.exact_token_blocks([[True]], block_size=1, n_blocks=1)

    def test_eos_exactly_at_cap_not_censored(self):
        out = contracts.termination_record([1, 9], {9}, 2)
        self.assertEqual(out['finish_reason'], 'eos')
        self.assertTrue(out['budget_reached'])
        self.assertFalse(out['right_censored'])

    def test_no_eos_at_cap_censored(self):
        self.assertTrue(contracts.termination_record([1, 2], {9}, 2)['right_censored'])

    def test_unexpected_early_stop(self):
        self.assertEqual(contracts.termination_record([1], {9}, 2)['finish_reason'], 'unexpected_stop')

    def test_after_eos_rejected(self):
        with self.assertRaises(ValueError):
            contracts.termination_record([9, 1], {9}, 2)

    def test_matched_kl_different_lengths(self):
        a, b = contracts.kl_length_counterexample()['arms']
        self.assertAlmostEqual(a['kl_nats'], b['kl_nats'])
        self.assertAlmostEqual(a['entropy_nats'], b['entropy_nats'])
        self.assertAlmostEqual(a['expected_tokens_including_eos'] / b['expected_tokens_including_eos'], 9.5)

if __name__ == '__main__':
    unittest.main()
