from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_tokenizer_contract as audit


class FakeGeneration(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**{'top_k': 50, **kwargs})

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


class FakeTokenizer:
    bos_token = '<｜begin▁of▁sentence｜>'
    eos_token = '<｜end▁of▁sentence｜>'
    bos_token_id = 151646
    eos_token_id = 151643
    pad_token_id = 151643
    is_fast = True
    model_max_length = 16384
    chat_template = 'fixture-template'
    padding_side = 'right'
    duplicate_bos = False
    wrong_mask = False
    wrong_direct = False

    def encode(self, text, add_special_tokens=False):
        specials = {self.bos_token: self.bos_token_id, self.eos_token: self.eos_token_id,
                    '<think>': 151648, '</think>': 151649}
        values = []
        while text:
            match = next((token for token in specials if text.startswith(token)), None)
            if match:
                values.append(specials[match])
                text = text[len(match):]
            else:
                values.append(1000 + ord(text[0]))
                text = text[1:]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        text = self.bos_token + '<｜User｜>' + messages[0]['content'] + '<｜Assistant｜><think>\n'
        if self.duplicate_bos:
            text = self.bos_token + text
        if not tokenize:
            return text
        return self.encode(text) + ([11] if self.wrong_direct else [])

    def __call__(self, text, add_special_tokens=False, truncation=False, padding=False, **kwargs):
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens)
            return {'input_ids': ids, 'attention_mask': [0 if self.wrong_mask else 1] * len(ids)}
        rows = [self.encode(t, add_special_tokens) for t in text]
        longest = max(map(len, rows))
        return {'input_ids': [[self.pad_token_id] * (longest-len(row)) + row for row in rows],
                'attention_mask': [[0] * (longest-len(row)) + [1] * len(row) for row in rows]}


def fixtures():
    manifest = json.loads((ROOT/'configs/reproduction_candidate.json').read_text())
    refs = json.loads((ROOT/'configs/asset_inspection_refs.json').read_text())
    configs = {
        'config.json': {'model_type': 'qwen2', 'bos_token_id': 151643, 'eos_token_id': 151643,
                        'vocab_size': 151936, 'max_position_embeddings': 131072},
        'generation_config.json': {'bos_token_id': 151646, 'eos_token_id': 151643, 'temperature': 0.6, 'top_p': 0.95},
        'tokenizer_config.json': {'chat_template': 'fixture-template', 'model_max_length': 16384},
    }
    rows = [{**a, 'status': 'REVISION_RESOLVED', 'private': False, 'gated': False} for a in refs['assets']]
    model = rows[0]
    model['configuration_files'] = {name: {
        'content': content, 'canonical_sha256': audit.canonical_hash(content),
        'source_url': f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}"
    } for name, content in configs.items()}
    refs['model_configuration_sha256'] = {n: audit.canonical_hash(c) for n, c in configs.items()}
    report = {'status': 'METADATA_RESOLVED_NOT_RUN_READY', 'errors': [], 'assets': rows,
              'manifest_canonical_sha256': audit.canonical_hash(manifest)}
    return report, manifest, refs, configs


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.report, self.manifest, self.refs, self.configs = fixtures()

    def review(self):
        return audit.review_inputs(self.report, self.manifest, self.refs)

    def test_manifest_hash_matches_submitted_report(self):
        self.assertEqual(audit.canonical_hash(self.manifest), '98ff733b0d8391c3a33aa2f561e2d389bd6e482ee53b1b680c1bb6e7e6b6201f')

    def test_valid_mock_report(self):
        self.assertEqual(self.review(), self.configs)

    def test_manifest_drift_rejected(self):
        self.manifest['r1']['top_p'] = 0.8
        with self.assertRaises(ValueError): self.review()

    def test_config_content_mutation_rejected(self):
        self.report['assets'][0]['configuration_files']['config.json']['content']['eos_token_id'] = 7
        with self.assertRaises(ValueError): self.review()

    def test_duplicate_role_rejected(self):
        self.report['assets'][1] = copy.deepcopy(self.report['assets'][0])
        with self.assertRaises(ValueError): self.review()

    def test_wrong_revision_rejected(self):
        self.report['assets'][0]['revision'] = 'main'
        with self.assertRaises(ValueError): self.review()

    def test_main_config_url_rejected(self):
        self.report['assets'][0]['configuration_files']['config.json']['source_url'] = 'https://huggingface.co/main/config.json'
        with self.assertRaises(ValueError): self.review()

    def test_report_error_rejected(self):
        self.report['errors'] = ['network failure']
        with self.assertRaises(ValueError): self.review()

    def test_gated_asset_rejected(self):
        self.report['assets'][0]['gated'] = 'auto'
        with self.assertRaises(ValueError): self.review()

    def test_unknown_role_rejected(self):
        self.report['assets'][0]['role'] = 'another_reference'
        with self.assertRaises(ValueError): self.review()

    def test_config_hash_in_refs_rejected(self):
        self.refs['model_configuration_sha256']['config.json'] = '0' * 64
        with self.assertRaises(ValueError): self.review()

    def test_auto_finds_reviewed_report_not_newest(self):
        with tempfile.TemporaryDirectory() as d:
            chosen = Path(d)/'asset_revisions_old.json'
            reviewed = {**self.report, 'recorded_at_utc': self.refs['source_recorded_at_utc']}
            audit.write_new(chosen, reviewed)
            audit.write_new(Path(d)/'asset_revisions_new.json', {**reviewed, 'recorded_at_utc': 'future'})
            self.assertEqual(audit.find_report(Path(d), self.refs), chosen)

    def test_missing_auto_report_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(audit.ContractError): audit.find_report(Path(d), self.refs)

    def test_ambiguous_auto_reports_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            reviewed = {**self.report, 'recorded_at_utc': self.refs['source_recorded_at_utc']}
            for suffix in ('a', 'b'):
                audit.write_new(Path(d)/f'asset_revisions_{suffix}.json', reviewed)
            with self.assertRaises(audit.ContractError): audit.find_report(Path(d), self.refs)

    def test_write_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result.json'
            audit.write_new(p, {'status': 'first'})
            with self.assertRaises(FileExistsError): audit.write_new(p, {'status': 'second'})
            self.assertEqual(json.loads(p.read_text())['status'], 'first')

    def test_invalid_payload_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result.json'
            with self.assertRaises(ValueError): audit.write_new(p, {'x': float('nan')})
            self.assertFalse(p.exists())


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        _, self.manifest, _, self.configs = fixtures()
        self.tok = FakeTokenizer()

    def probe(self):
        return audit.inspect_tokenizer(self.tok, self.configs, self.manifest, FakeGeneration)

    def test_valid_synthetic_tokenizer(self):
        out = self.probe()
        self.assertEqual(out['left_padding_mask_check'], 'PASS')
        self.assertFalse(out['long_context_validated'])
        self.assertEqual(self.tok.padding_side, 'right')

    def test_default_top_k_is_overridden_not_assumed_disabled(self):
        out = self.probe()
        self.assertEqual(out['inherited_top_k'], 50)
        self.assertEqual(out['explicit_r0_generation_config_candidate_not_executed']['top_k'], 0)

    def test_model_bos_discrepancy_preserved(self):
        out = self.probe()
        self.assertNotEqual(out['source_model_bos_token_id'], out['bos_token_id'])
        self.assertEqual(self.configs['config.json']['bos_token_id'], 151643)

    def test_duplicate_bos_rejected(self):
        self.tok.duplicate_bos = True
        with self.assertRaises(ValueError): self.probe()

    def test_direct_and_two_step_disagreement_rejected(self):
        self.tok.wrong_direct = True
        with self.assertRaises(ValueError): self.probe()

    def test_bad_mask_rejected(self):
        self.tok.wrong_mask = True
        with self.assertRaises(ValueError): self.probe()

    def test_wrong_bos_rejected(self):
        self.tok.bos_token_id = 151643
        with self.assertRaises(ValueError): self.probe()

    def test_think_not_stop_and_not_generated_prefix(self):
        out = self.probe()
        self.assertNotIn(out['eos_token_id'], out['closing_think_ids_not_eos'])
        self.assertIsNone(out['explicit_r0_generation_config_candidate_not_executed']['stop_strings'])

    def test_no_embedding_resize_to_tokenizer_length(self):
        self.configs['config.json']['vocab_size'] = 100
        with self.assertRaises(ValueError): self.probe()


if __name__ == '__main__':
    unittest.main()
