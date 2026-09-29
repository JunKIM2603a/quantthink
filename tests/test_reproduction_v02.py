import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import awq_weight_space as awq
import build_development_manifest as split
import evaluate_gsm8k as evaluator


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.dev = [{"question": q, "answer": "private"} for q in ["question A", "question B", "overlap", "question C"]]
        self.confirm = [{"problem": "overlap", "answer": "DO_NOT_EXPOSE"}, {"problem": "held out"}]

    def build(self, dev=None, confirm=None):
        return split.build_manifest(dev if dev is not None else self.dev,
                                    confirm if confirm is not None else self.confirm,
                                    n=2, expected_confirmation_rows=2)

    def test_permutation_and_whitespace_stable(self):
        result = self.build()
        reversed_rows = list(reversed(self.dev))
        reversed_rows[0] = {"question": "  question \n C "}
        self.assertEqual(result, self.build(reversed_rows, list(reversed(self.confirm))))

    def test_no_holdout_overlap_and_no_text_or_answers(self):
        out = self.build()
        self.assertEqual(out["counts"]["excluded_confirmation_overlap"], 1)
        self.assertNotIn(split.question_hash("overlap"), [x["question_sha256"] for x in out["selected"]])
        blob = json.dumps(out)
        for text in ("question A", "DO_NOT_EXPOSE", "private"):
            self.assertNotIn(text, blob)

    def test_confirm_answers_never_accessed(self):
        class QuestionOnly(dict):
            def __getitem__(self, key):
                if key != "problem":
                    raise AssertionError("gold accessed")
                return super().__getitem__(key)
        self.build(confirm=[QuestionOnly(problem=x["problem"]) for x in self.confirm])

    def test_duplicate_development_does_not_change_selection(self):
        a = self.build()
        b = self.build(self.dev + [{"question": " question   A "}])
        self.assertEqual(a["selected"], b["selected"])
        self.assertEqual(b["counts"]["duplicate_development_rows"], 1)

    def test_partial_or_duplicate_confirmation_rejected(self):
        for rows in (self.confirm[:1], [self.confirm[0]] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.build(confirm=rows)

    def test_insufficient_data_not_resampled(self):
        with self.assertRaises(ValueError):
            self.build(dev=[{"question": "overlap"}])

    def test_missing_question_rejected(self):
        with self.assertRaises(ValueError):
            self.build(dev=[{"question": None}])

    def test_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            split.write_new(path, {"first": 1})
            with self.assertRaises(FileExistsError):
                split.write_new(path, {"second": 2})
            self.assertEqual(json.loads(path.read_text()), {"first": 1})

    def test_cli_records_local_bytes_not_source_authentication(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            dev, confirm = folder / "dev.jsonl", folder / "confirm.jsonl"
            dev.write_text("\n".join(json.dumps({"question": f"dev {i}"}) for i in range(110)))
            confirm.write_text("\n".join(json.dumps({"problem": f"held {i}"}) for i in range(500)))
            output = folder / "out.json"
            with patch.object(sys, "argv", ["build", "--development", str(dev),
                                          "--confirmation", str(confirm), "--output", str(output)]):
                self.assertEqual(split.main(), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["counts"]["selected"], 100)
            self.assertEqual(result["status"], "LOCAL_CANDIDATE_NOT_SOURCE_AUTHENTICATED")
            self.assertEqual(len(result["input_file_sha256"]["confirmation"]), 64)


class EvaluatorTests(unittest.TestCase):
    def parse(self, text):
        return evaluator.parse_final_answer(text)

    def test_gold_marker_required(self):
        self.assertEqual(evaluator.gold_value("work\n#### 1,200"), 1200)
        with self.assertRaises(ValueError):
            evaluator.gold_value("work 1200")

    def test_nested_fraction_and_exact_decimal(self):
        self.assertEqual(self.parse(r"work </think> \boxed{\frac{1}{2}}")["value"], "1/2")
        self.assertEqual(evaluator.numeric_value("0.5"), evaluator.numeric_value("1/2"))

    def test_reasoning_correct_number_is_not_final_answer(self):
        result = self.parse(r"42 and \boxed{42}, reconsider </think> \boxed{7}")
        self.assertEqual(result["value"], "7")

    def test_missing_or_multiple_boundary_never_uses_reasoning(self):
        for text in (r"\boxed{42}", "</think>42</think>7", "<think>work</think>42"):
            with self.subTest(text=text):
                self.assertIsNone(self.parse(text)["value"])

    def test_different_boxed_answers_are_ambiguous(self):
        out = self.parse(r"</think>\boxed{42} or \boxed{7}")
        self.assertEqual(out["parse_status"], "AMBIGUOUS_FINAL_ANSWER")

    def test_marked_answer_contradiction_is_ambiguous(self):
        out = self.parse("</think>\\boxed{42}\nFinal answer: 7")
        self.assertEqual(out["parse_status"], "AMBIGUOUS_FINAL_ANSWER")

    def test_same_numeric_boxes_allowed(self):
        self.assertEqual(self.parse(r"</think>\boxed{0.5} and \boxed{1/2}")["value"], "1/2")

    def test_explicit_answer_line_allowed(self):
        self.assertEqual(self.parse("reason</think>The final answer is 42.")["value"], "42")

    def test_arbitrary_last_number_not_extracted(self):
        self.assertIsNone(self.parse("</think>We considered 42 possibilities.")["value"])

    def test_unsupported_and_unbounded_numerics_rejected(self):
        for value in ("NaN", "inf", "1e999", "1/0", "1,2", "12%", "3 meters",
                      "__import__('os')", "2**3", r"\sqrt{4}", "1" * 101):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.numeric_value(value)

    def test_malformed_box_does_not_fall_back(self):
        for text in (r"</think>\boxed{42", r"</think>\boxed{\frac{1}{0}}",
                     r"</think>\boxed{nonsense} 42"):
            with self.subTest(text=text):
                self.assertIsNone(self.parse(text)["value"])

    def test_cap_stops_kept_in_denominator(self):
        gold = [{"id": "a", "answer": "#### 42"}]
        rows = [{"id": "a", "arm": "bf16", "seed": 42,
                 "finish_reason": "length", "generated_text": "thinking 42"}]
        result = evaluator.evaluate_grid(gold, rows, arms=["bf16"], seeds=[42])
        self.assertEqual(result["summary"]["bf16"]["primary_accuracy"], 0)
        self.assertEqual(result["summary"]["bf16"]["budget_stops"], 1)
        self.assertEqual(len(result["items"]), 1)
        self.assertTrue(result["requires_parse_audit"])
        self.assertFalse(result["main_hypothesis_ready"])

    def test_missing_and_error_prevent_primary_comparison(self):
        gold = [{"id": "a", "answer": "#### 42"}, {"id": "b", "answer": "#### 2"}]
        rows = [{"id": "a", "arm": "bf16", "seed": 42, "finish_reason": "error"}]
        result = evaluator.evaluate_grid(gold, rows, arms=["bf16"], seeds=[42])
        self.assertEqual(result["status"], "INCOMPLETE_GRID_NO_PRIMARY_COMPARISON")
        self.assertEqual(result["summary"]["bf16"]["missing"], 1)
        self.assertIsNone(result["summary"]["bf16"]["primary_accuracy"])

    def test_duplicate_and_unknown_generation_rejected(self):
        gold = [{"id": "a", "answer": "#### 42"}]
        row = {"id": "a", "arm": "bf16", "seed": 42, "finish_reason": "eos",
               "generated_text": "</think>42"}
        for rows in ([row, row], [{**row, "id": "unknown"}], [{**row, "seed": True}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluator.evaluate_grid(gold, rows, arms=["bf16"], seeds=[42])

    def test_all_conditions_paired_by_id(self):
        gold = [{"id": "a", "answer": "#### 42"}]
        rows = [{"id": "a", "arm": arm, "seed": seed, "finish_reason": "eos",
                 "generated_text": "</think>42"} for arm in ("bf16", "awq_w3") for seed in (42, 43)]
        result = evaluator.evaluate_grid(gold, list(reversed(rows)))
        self.assertEqual(result["status"], "COMPLETE_GRID_CANDIDATE_SCORING")
        self.assertEqual(len(result["items"]), 4)
        self.assertTrue(all(x["correct"] for x in result["items"]))

    def planned_data(self):
        config = json.loads((ROOT / "configs/reproduction_candidate.json").read_text())
        config["r1"]["n_problems"] = 2
        manifest = split.build_manifest([{"question": "q1"}, {"question": "q2"}],
                                        [{"problem": "held"}], n=2, expected_confirmation_rows=1)
        manifest["manifest_canonical_sha256"] = split.canonical_hash(config)
        gold = [{"id": row["id"], "answer": "#### 42"} for row in manifest["selected"]]
        return config, manifest, gold

    def test_planned_ids_reject_missing_gold_or_changed_plan(self):
        config, manifest, gold = self.planned_data()
        evaluator.validate_planned_ids(gold, manifest, config)
        for rows in (gold[:1], gold + [gold[0]], gold + [{"id": "extra", "answer": "#### 1"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluator.validate_planned_ids(rows, manifest, config)
        changed = copy.deepcopy(config)
        changed["r1"]["sampling_seeds"] = [999]
        with self.assertRaises(ValueError):
            evaluator.validate_planned_ids(gold, manifest, changed)
        corrupt = copy.deepcopy(manifest)
        corrupt["selected"][0]["question_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            evaluator.validate_planned_ids(gold, corrupt, config)

    def test_evaluation_cli_keeps_plan_and_incomplete_grid(self):
        config, manifest, gold = self.planned_data()
        records = [{"id": row["id"], "arm": arm, "seed": seed, "finish_reason": "eos",
                    "generated_text": "</think>42"} for row in gold
                   for arm in config["r1"]["arms"] for seed in config["r1"]["sampling_seeds"]]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "candidate.json").write_text(json.dumps(config))
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (folder / "gold.jsonl").write_text("\n".join(json.dumps(row) for row in gold))
            for count, expected_exit in ((len(records), 0), (len(records) - 1, 2)):
                (folder / "generations.jsonl").write_text("\n".join(json.dumps(row) for row in records[:count]))
                output = folder / f"score_{count}.json"
                argv = ["evaluate", "--candidate", str(folder / "candidate.json"),
                        "--development-manifest", str(folder / "manifest.json"),
                        "--gold", str(folder / "gold.jsonl"), "--generations",
                        str(folder / "generations.jsonl"), "--output", str(output)]
                with patch.object(sys, "argv", argv):
                    self.assertEqual(evaluator.main(), expected_exit)
                report = json.loads(output.read_text())
                self.assertEqual(len(report["items"]), len(records))
                self.assertEqual(report["development_manifest_canonical_sha256"], split.canonical_hash(manifest))
                self.assertFalse(report["main_hypothesis_ready"])
                self.assertTrue(report["requires_source_provenance_review"])
                if expected_exit:
                    self.assertIsNone(report["summary"][records[-1]["arm"]]["primary_accuracy"])


class WeightSpaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.parameters = {"norm.weight": np.array([1.0, 1.5, 0.5]),
                           "gate.weight": rng.normal(size=(4, 3)),
                           "up.weight": rng.normal(size=(4, 3)),
                           "up.bias": rng.normal(size=(4,)),
                           "down.weight": rng.normal(size=(2, 4)),
                           "down.bias": rng.normal(size=(2,))}
        self.ledger = [
            {"kind": "norm_linears", "previous": "norm", "following": ["gate", "up"],
             "scales": [0.25, 2.0, 4.0]},
            {"kind": "linear_linear", "previous": "up", "following": ["down"],
             "scales": [2.0, 0.5, 4.0, 0.25]},
        ]
        self.x = rng.normal(size=(5, 3))

    def forward(self, p):
        x = self.x * p["norm.weight"]
        gate = x @ p["gate.weight"].T
        up = x @ p["up.weight"].T + p["up.bias"]
        hidden = gate / (1 + np.exp(-gate)) * up
        return hidden @ p["down.weight"].T + p["down.bias"]

    def test_function_preserved_with_overlapping_scale_operations(self):
        scaled = awq.apply_scale_ledger(self.parameters, self.ledger)
        np.testing.assert_allclose(self.forward(self.parameters), self.forward(scaled), rtol=1e-12, atol=1e-12)
        self.assertGreater(np.linalg.norm(scaled["up.weight"] - self.parameters["up.weight"]), 0.1)

    def test_reverse_order_recovers_reference_without_mutation(self):
        original = copy.deepcopy(self.parameters)
        scaled = awq.apply_scale_ledger(self.parameters, self.ledger)
        recovered = awq.apply_scale_ledger(scaled, self.ledger, inverse=True)
        for key in original:
            np.testing.assert_array_equal(original[key], self.parameters[key])
            np.testing.assert_allclose(recovered[key], original[key], rtol=1e-14, atol=1e-14)

    def test_perturbation_not_erased_by_canonicalization(self):
        changed = awq.apply_scale_ledger(self.parameters, self.ledger)
        changed["up.weight"][1, 2] += 0.08
        canonical = awq.apply_scale_ledger(changed, self.ledger, inverse=True)
        self.assertAlmostEqual(canonical["up.weight"][1, 2] - self.parameters["up.weight"][1, 2], 0.01)
        np.testing.assert_allclose(self.forward(changed), self.forward(canonical), rtol=1e-12, atol=1e-12)
        result = awq.canonical_error(self.parameters, changed, self.ledger, weight_names=["up.weight"])
        self.assertGreater(result["parameters"][0]["rms_error"], 0)

    def test_unquantized_roundtrip_has_zero_canonical_error(self):
        scaled = awq.apply_scale_ledger(self.parameters, self.ledger)
        result = awq.canonical_error(self.parameters, scaled, self.ledger,
                                     weight_names=["up.weight", "down.weight"])
        self.assertTrue(all(x["rms_error"] < 1e-14 for x in result["parameters"]))

    def test_unsupported_mismatched_or_nonfinite_scales_rejected(self):
        for update in ({"kind": "gelu"}, {"scales": [0, 1, 2]},
                       {"scales": [1, float("nan"), 2]}, {"scales": [1, 2]},
                       {"following": ["up", "up"]}, {"following": ["norm"]}):
            ledger = [{**self.ledger[0], **update}]
            with self.subTest(update=update), self.assertRaises(ValueError):
                awq.apply_scale_ledger(self.parameters, ledger)

    def test_missing_parameter_never_silently_skipped(self):
        partial = {k: v for k, v in self.parameters.items() if k != "gate.weight"}
        with self.assertRaises(KeyError):
            awq.apply_scale_ledger(partial, self.ledger)


if __name__ == "__main__":
    unittest.main()
