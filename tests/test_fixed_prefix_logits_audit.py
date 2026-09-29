"""새 CPU 원시 logit 감사·필터 진단 검사. 모델/준비 검사를 재실행하지 않는다."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_fixed_prefix_logits as audit
from difficulty_pilot_contracts import canonical_hash, file_hash
from fixed_prefix_diagnostic_contracts import PAIRS, STATES, compare_logits


def fixture(folder):
    cfg = {"inputs": [{"problem_id": p, "logit_positions": [1, 3, 5]}
                      for p in ("D2-03", "D3-01")], "vocab_size": 7, "eos_token_id": 6}
    run = {"status": "FIXED_PREFIX_DIAGNOSTICS_COLLECTED_PENDING_REVIEW",
           "config_sha256": canonical_hash(cfg), "repository": {"commit": "frozen"},
           "research_status_sha256": "historical-status", "inputs": cfg["inputs"],
           "implementation_sha256": {}, "forward_records": [], "comparisons": []}
    values = {}
    base = np.asarray([[5, 4, 3, 2, 1, 0, -1], [0, 0, 0, 0, 0, 0, 0],
                       [1, 4, 2, 3, 0, -1, -2]], dtype=np.float32)
    for state_index, state in enumerate(STATES):
        for inp in cfg["inputs"]:
            array = base.copy()
            if state_index >= 2:
                array[:, state_index] += state_index * 0.25
            name = f"logits_{state}_{inp['problem_id']}.npy"
            np.save(folder / name, array, allow_pickle=False)
            values[state, inp["problem_id"]] = array
            run["forward_records"].append({"state": state, "problem_id": inp["problem_id"],
                                           "logit_positions": inp["logit_positions"],
                                           "file": name, "file_sha256": file_hash(folder / name),
                                           "shape": [3, 7], "stored_dtype": "float32"})
    for a, b in PAIRS:
        for inp in cfg["inputs"]:
            problem = inp["problem_id"]
            run["comparisons"].append({"reference": a, "observed": b, "problem_id": problem,
                                       **compare_logits(values[a, problem], values[b, problem],
                                                        inp["logit_positions"], 6)})
    path = folder / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    review = {"source_run_sha256": file_hash(path), "config_sha256": run["config_sha256"],
              "execution_commit": "frozen", "execution_research_status_sha256": "historical-status"}
    return run, cfg, review


class SavedLogitsAuditTests(unittest.TestCase):
    def test_known_filter_and_eos_exclusion(self):
        q, info = audit.filtered_distribution(np.log([0.05, 0.15, 0.3, 0.5]), 0, 1.0, 0.7)
        np.testing.assert_allclose(q, [0, 0, 0.375, 0.625], atol=1e-14)
        self.assertFalse(info["eos_retained"])
        self.assertEqual(info["retained_count"], 2)
        self.assertEqual(info["eos_rank_best"], 4)
        self.assertAlmostEqual(info["eos_full_probability"], 0.05)

    def test_boundary_ties_are_explicit(self):
        q, info = audit.filtered_distribution(np.zeros(4), 0, 1.0, 0.5)
        np.testing.assert_array_equal(q, [0, 0, 0.5, 0.5])
        self.assertTrue(info["split_boundary_tie"])
        self.assertTrue(info["eos_boundary_tie_sensitive"])
        self.assertEqual((info["boundary_tie_count"], info["boundary_tie_kept"]), (4, 2))
        self.assertEqual((info["eos_rank_best"], info["eos_rank_worst"]), (1, 4))

    def test_disjoint_support_has_infinite_kl_but_finite_js(self):
        row = audit.compare_filtered([10, 0, -10], [0, 10, -10], 2)
        self.assertEqual(row["total_variation"], 1.0)
        self.assertTrue(row["kl_is_infinite_due_to_support"])
        self.assertIsNone(row["kl_reference_to_observed"])
        self.assertAlmostEqual(row["jensen_shannon_nats"], math.log(2))
        json.dumps(row, allow_nan=False)

    def test_constant_shift_and_tie_membership(self):
        row = audit.compare_filtered([3, 3, 1, -2], [13, 13, 11, 8], 3)
        self.assertEqual(row["total_variation"], 0)
        self.assertEqual(row["reference_maximizer_count"], 2)
        self.assertTrue(row["observed_argmax_is_reference_maximizer"])
        changed = audit.compare_filtered([3, 3, 1, -2], [2, 3, 1, -2], 3)
        self.assertNotEqual(changed["reference_argmax"], changed["observed_argmax"])
        self.assertTrue(changed["observed_argmax_is_reference_maximizer"])

    def test_recalculation_tolerance_is_not_a_model_tolerance(self):
        audit.check_equal({"rms": 0.3, "same": False}, {"rms": 0.3 + 1e-13, "same": False})
        for actual in ({"rms": 0.31, "same": False}, {"rms": math.nan, "same": False},
                       {"rms": 0.3, "same": 0}, {"rms": 0.3}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                audit.check_equal({"rms": 0.3, "same": False}, actual)

    def test_file_hash_shape_dtype_and_finite_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run, _, _ = fixture(folder)
            rec = run["forward_records"][0]
            original = audit.load_logits(folder, rec)
            self.assertFalse(original.flags.writeable)
            del original
            (folder / rec["file"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "해시"):
                audit.load_logits(folder, rec)
            for array in (np.zeros((2, 7), dtype=np.float32), np.zeros((3, 7), dtype=np.float64),
                          np.full((3, 7), np.nan, dtype=np.float32)):
                np.save(folder / rec["file"], array, allow_pickle=False)
                altered = {**rec, "file_sha256": file_hash(folder / rec["file"])}
                with self.assertRaisesRegex(ValueError, "shape·dtype·유한성"):
                    audit.load_logits(folder, altered)
            with self.assertRaises(ValueError):
                audit.load_logits(folder, {**rec, "file": "../escape.npy"})

    def test_frozen_report_and_implementation_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run, cfg, review = fixture(folder)
            path = folder / "run.json"
            self.assertEqual(audit.validate_run(path, review, cfg), run)
            bad_review = {**review, "execution_research_status_sha256": "current-status"}
            with self.assertRaises(ValueError):
                audit.validate_run(path, bad_review, cfg)
            run["implementation_sha256"] = {"changed.py": "wrong"}
            (folder / "scripts").mkdir()
            (folder / "scripts/changed.py").write_text("pass\n")
            path.write_text(json.dumps(run))
            review["source_run_sha256"] = file_hash(path)
            with patch.object(audit, "ROOT", folder), self.assertRaisesRegex(ValueError, "구현"):
                audit.validate_run(path, review, cfg)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "바이트 해시"):
                audit.validate_run(path, review, cfg)

    def test_complete_saved_array_audit_preserves_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run, cfg, _ = fixture(folder)
            before = {p.name: file_hash(p) for p in folder.iterdir()}
            rows = audit.audit_arrays(run, folder, cfg, progress=lambda _: None)
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["t1_recomputed_matches_report"] for r in rows))
            self.assertEqual(rows[0]["filtered_float64_summary"]["max_total_variation"], 0)
            self.assertEqual(before, {p.name: file_hash(p) for p in folder.iterdir()})
            corrupted = copy.deepcopy(run)
            corrupted["comparisons"][0]["per_position"][0]["reference_top1"] = 6
            with self.assertRaisesRegex(ValueError, "집계 값"):
                audit.audit_arrays(corrupted, folder, cfg, progress=lambda _: None)

    def test_no_overwrite_or_nonfinite_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            audit.write_report(path, {"value": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.write_report(path, {"value": 2})
            self.assertEqual(path.read_bytes(), before)
            target = Path(tmp) / "invalid.json"
            with self.assertRaises(ValueError):
                audit.write_report(target, {"value": math.inf})
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
