"""새 고정 prefix 계약·수치 요약만 검사. 모델·기존 AWQ 검사 실행 없음."""
import builtins
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_fixed_prefix_diagnostic as runner
from difficulty_pilot_contracts import canonical_hash, file_hash
from fixed_prefix_diagnostic_contracts import CONFIG, compare_logits, difference_stats, read_inputs, require_authorization


class PrefixContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = json.loads(CONFIG.read_text())
        source = {"use_sliding_window": False, "sliding_window": 4096, "vocab_size": 8}
        self.cfg["vocab_size"] = 8
        self.cfg["model_configuration_sha256"] = canonical_hash(source)
        self.cfg["effective_source_configuration_sha256"] = canonical_hash({**source, "sliding_window": None})
        rows = []
        for ref in self.cfg["inputs"]:
            row = {"problem_id": ref["problem_id"], "arm": "bf16", "seed": 42,
                   "input_ids": [1] * ref["input_token_count"], "generated_ids": [2] * 2048}
            ref["attempt_sha256"] = canonical_hash(row)
            ref["sequence_ids_sha256"] = canonical_hash(row["input_ids"] + row["generated_ids"])
            rows.append(row)
        self.evidence = self.root / "evidence.json"
        self.mask = self.root / "mask.json"
        self.packet = {"cached_model_config": {"configuration": source}, "run_snapshot": {"attempts": rows}}
        self.evidence.write_text(json.dumps(self.packet))
        self.mask.write_text(json.dumps({"status": "DISABLED_FLAG_WINDOW_MASK_OBSERVED"}))
        self.cfg["evidence_packet_sha256"] = file_hash(self.evidence)
        self.cfg["mask_report_sha256"] = file_hash(self.mask)

    def test_positions_and_budget_use_saved_ids_without_mutating_config(self):
        before = copy.deepcopy(self.packet)
        inputs, source, effective = read_inputs(self.evidence, self.mask, self.cfg)
        self.assertEqual([len(x["ids"]) for x in inputs], [2119, 2127])
        self.assertEqual([x["logit_positions"][-1] for x in inputs], [2118, 2126])
        self.assertEqual(sum(len(x["ids"]) for x in inputs) * 6, 25476)
        self.assertEqual(source["sliding_window"], 4096)
        self.assertIsNone(effective["sliding_window"])
        self.assertEqual(json.loads(self.evidence.read_text()), before)

    def test_modified_evidence_and_logit_position_are_rejected(self):
        self.evidence.write_text(self.evidence.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "바이트 해시"):
            read_inputs(self.evidence, self.mask, self.cfg)
        self.cfg["evidence_packet_sha256"] = file_hash(self.evidence)
        self.cfg["inputs"][0]["logit_positions"][0] += 1
        with self.assertRaisesRegex(ValueError, "계약 불일치"):
            read_inputs(self.evidence, self.mask, self.cfg)

    def test_r0_approval_does_not_authorize_new_forward(self):
        status = {"research_approval": "APPROVED", "novelty_gate": "SCOPED_CONTRIBUTION_ACCEPTED",
                  "protocol_gate": "ACCEPTED"}
        with self.assertRaisesRegex(ValueError, "승인 전"):
            require_authorization(status, self.cfg)
        status["fixed_prefix_diagnostic"] = {"authorization": "APPROVED", "config_sha256": canonical_hash(self.cfg)}
        require_authorization(status, self.cfg)
        changed = copy.deepcopy(self.cfg)
        changed["maximum_forward_calls"] = 14
        with self.assertRaises(ValueError):
            require_authorization(status, changed)

    def test_plan_never_imports_model_libraries_or_writes_output(self):
        config = self.root / "protocol.json"
        config.write_text(json.dumps(self.cfg))
        out = self.root / "output"
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "huggingface_hub", "numpy"}:
                raise AssertionError("plan에서 모델/수치 라이브러리 import")
            return original_import(name, *args, **kwargs)
        with patch.object(runner, "CONFIG", config), patch("builtins.__import__", guarded), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--plan", "--evidence", str(self.evidence),
                "--mask-report", str(self.mask), "--output-dir", str(out)]), 0)
        self.assertFalse(out.exists())


class LogitMetrics(unittest.TestCase):
    def test_known_distribution_top1_and_eos_change(self):
        a = np.log([[0.75, 0.25]])
        b = np.log([[0.25, 0.75]])
        result = compare_logits(a, b, [7], 1)
        row = result["per_position"][0]
        self.assertAlmostEqual(row["kl_reference_to_observed"], 0.5 * math.log(3))
        self.assertAlmostEqual(row["total_variation"], 0.5)
        self.assertAlmostEqual(row["eos_probability_delta"], 0.5)
        self.assertAlmostEqual(row["eos_log_probability_delta"], math.log(3))
        self.assertEqual(result["top1_changed_count"], 1)
        self.assertFalse(result["model_equivalence_certified"])

    def test_constant_logit_shift_is_not_distribution_change_or_exact_logits(self):
        a = np.array([[1.0, 2.0, 4.0], [0.0, 0.0, 1.0]])
        exact = compare_logits(a, a.copy(), [1, 5], 2)
        self.assertTrue(exact["exact_equal"])
        shifted = compare_logits(a, a + 2.0, [1, 5], 2)
        self.assertEqual(shifted["status"], "LOGITS_CHANGED_TOP1_UNCHANGED")
        self.assertAlmostEqual(shifted["max_total_variation"], 0.0)
        self.assertAlmostEqual(shifted["mean_kl"], 0.0)
        self.assertAlmostEqual(shifted["max_abs"], 2.0)

    def test_nonfinite_shape_and_zero_reference(self):
        for a, b in [(np.array([[1.0, float("nan")]]), np.ones((1, 2))),
                     (np.ones((1, 2)), np.ones((2, 2)))]:
            with self.assertRaises(ValueError):
                compare_logits(a, b, [1], 1)
        result = difference_stats(np.zeros(2), np.array([0.0, 1.0]))
        self.assertIsNone(result["relative_rms"])
        self.assertEqual(result["max_abs_index"], [1])


if __name__ == "__main__":
    unittest.main()
