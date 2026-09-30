"""R0가 검토한 자료와 다른 자체 일관 번들을 사용하지 않는지 검사합니다."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_assets as assets
import runtime_contracts as contracts
import run_r0


class PreparedReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.candidate, self.refs, self.policy = assets.load_contracts()
        self.policy = copy.deepcopy(self.policy)
        self.policy["calibration"].update(blocks=1, block_size=2)
        self.blocks = [[1, 2]]
        self.calibration = {"candidate_sha256": assets.canonical_hash(self.candidate),
                            "token_blocks_sha256": assets.canonical_hash(self.blocks),
                            "total_tokens": 2, "used_documents": 1, "counts": {"source_rows": 1}}
        self.development = {"selected_ids_sha256": assets.canonical_hash(["synthetic"]),
                            "counts": {"selected": 1}}
        self.report = {"status": "PREPARED_CANDIDATE_NOT_RUN_APPROVAL", "include_calibration": True,
                       "candidate_sha256": assets.canonical_hash(self.candidate),
                       "inspection_refs_sha256": assets.canonical_hash(self.refs),
                       "policy_sha256": assets.canonical_hash(self.policy),
                       "source_files": [{"role": "synthetic", "revision": "a" * 40}],
                       "packages": {"synthetic": "1"}, "implementation_sha256": {"synthetic.py": "b" * 64}}
        self.write_bundle()
        self.review = {**copy.deepcopy(self.report), "schema_version": 1,
                       "status": "PREPARATION_REPORT_REVIEWED_NOT_RUN_APPROVAL",
                       "calibration": {k: v for k, v in self.calibration.items() if k != "candidate_sha256"},
                       "development": copy.deepcopy(self.development)}

    def write_bundle(self):
        values = {"calibration_manifest.json": self.calibration,
                  "calibration_tokens.json": {"blocks": self.blocks},
                  "development_manifest.json": self.development}
        for name, value in values.items():
            (self.folder / name).write_text(json.dumps(value))
        for name in ("r1_inputs.jsonl", "r1_gold.jsonl"):
            (self.folder / name).write_text('{}\n')
        self.report["artifacts"] = {p.name: assets.file_digests(p) for p in self.folder.iterdir()
                                     if p.name != "preparation_report.json"}
        (self.folder / "preparation_report.json").write_text(json.dumps(self.report))

    def verify(self):
        blocks, report = contracts.load_prepared_calibration(
            self.folder, self.candidate, self.refs, self.policy)
        return contracts.verify_reviewed_preparation(self.folder, report, blocks,
            self.candidate, self.refs, self.policy, self.review)

    def test_matching_bundle_records_review_hash(self):
        self.assertEqual(self.verify(), assets.canonical_hash(self.review))

    def test_self_consistent_replacement_tokens_still_rejected(self):
        self.blocks = [[3, 4]]
        self.calibration["token_blocks_sha256"] = assets.canonical_hash(self.blocks)
        self.write_bundle()
        # 자체 보고서·파일 해시 대조만으로는 바뀐 자료도 통과합니다.
        self.assertEqual(contracts.load_prepared_calibration(
            self.folder, self.candidate, self.refs, self.policy)[0], [[3, 4]])
        with self.assertRaisesRegex(ValueError, "검토한 자료와 다릅니다: artifacts"):
            self.verify()

    def test_source_revision_or_preparation_code_change_rejected(self):
        original = copy.deepcopy(self.report)
        for key, value in (("source_files", [{"revision": "c" * 40}]),
                           ("implementation_sha256", {"synthetic.py": "d" * 64}),
                           ("packages", {"synthetic": "2"})):
            with self.subTest(key=key):
                self.report = {**copy.deepcopy(original), key: value}
                self.write_bundle()
                with self.assertRaisesRegex(ValueError, key):
                    self.verify()

    def test_stale_review_policy_rejected(self):
        self.review["policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "설정 해시"):
            self.verify()

    def test_selected_id_change_rejected_even_with_updated_file_hash(self):
        self.development["selected_ids_sha256"] = assets.canonical_hash(["different"])
        self.write_bundle()
        self.review["artifacts"] = copy.deepcopy(self.report["artifacts"])
        with self.assertRaisesRegex(ValueError, "development.*selected_ids_sha256"):
            self.verify()

    def test_invalid_block_shape_or_boolean_id_rejected(self):
        for blocks in ([[1]], [[True, 2]]):
            with self.subTest(blocks=blocks):
                self.blocks = blocks
                self.calibration["token_blocks_sha256"] = assets.canonical_hash(blocks)
                self.write_bundle()
                self.review["artifacts"] = copy.deepcopy(self.report["artifacts"])
                self.review["calibration"]["token_blocks_sha256"] = assets.canonical_hash(blocks)
                with self.assertRaisesRegex(ValueError, "블록 크기 또는 토큰 형식"):
                    self.verify()

    def test_research_gates_still_precede_review_and_model_loading(self):
        # 사용자 승인을 되돌리지 않고 명시적인 미승인 사례를 구성합니다.
        state = {**contracts.REQUIRED_GATES, "research_approval": "NOT_RECORDED"}
        with patch.object(run_r0, "repository_state") as repo, \
             patch.object(run_r0, "verify_reviewed_preparation") as verify, \
             self.assertRaisesRegex(ValueError, "승인 기록"):
            run_r0.run(self.folder / "run", self.folder, self.folder, "cuda:0",
                       self.candidate, self.refs, self.policy, state)
        repo.assert_not_called()
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
