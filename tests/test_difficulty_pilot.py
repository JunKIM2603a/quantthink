"""새 난도 문제 정답과 오판 위험 검사. 기존 R0·합성 AWQ·토크나이저 검사를 실행하지 않음."""
from __future__ import annotations

import copy
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from difficulty_pilot_contracts import (canonical_hash, final_answer, load_suite, prompt_for,
                                        rational, score_attempt, summarize)
from reproduction_contracts import termination_record
from replay_r0_awq import LINEARS, _bf16_value, validate_recipe, verify_r0_files
from run_difficulty_pilot import require_scope


class DifficultyPilotTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.suite = load_suite()

    def row(self, problem, arm="bf16", seed=42, *, correct=True, censored=False):
        ids = [11, 151649, 12, 151643]
        if censored:
            ids = [11, 151649] + [12] * (self.cfg["max_new_tokens"] - 2)
        value = problem["answer"] if correct else "-9999"
        return {"problem_id": problem["id"], "level": problem["level"], "arm": arm, "seed": seed,
                "input_ids": [151646, 19, 151648, 198], "input_token_count": 4,
                "input_ids_sha256": canonical_hash([151646, 19, 151648, 198]),
                "prompt_sha256": canonical_hash(prompt_for(self.suite, problem)),
                "generated_ids": ids, "generated_text": "reasoning 999 </think> Final: \\boxed{" + value + "}.",
                **termination_record(ids, {151643}, self.cfg["max_new_tokens"])}

    def report(self):
        return {"config_sha256": canonical_hash(self.cfg), "suite_sha256": canonical_hash(self.suite),
                "attempts": [self.row(p, arm, seed) for p in self.suite["problems"]
                             for arm in self.cfg["arms"] for seed in self.cfg["seeds"]]}

    def test_all_twenty_answers_by_exact_independent_calculation(self):
        dice = [(a, b) for a in range(1, 7) for b in range(1, 7) if a + b >= 9]
        coefficient = [1]
        for _ in range(8):
            result = [0] * (len(coefficient) + 4)
            for power, count in enumerate(coefficient):
                for add in range(5):
                    result[power + add] += count
            coefficient = result
        paths = {(0, 0): 1}
        for x in range(9):
            for y in range(x + 1):
                if (x, y) != (0, 0):
                    paths[x, y] = 0 if (x, y) == (4, 4) else paths.get((x - 1, y), 0) + paths.get((x, y - 1), 0)
        permutations = [p for p in itertools.permutations(range(1, 8)) if p[0] != 1]
        computed = {
            "D1-01": 240 * Fraction(85, 100) * Fraction(110, 100),
            "D1-02": 50 - 6 * 4 - 3 * 2,
            "D1-03": (Fraction(39, 3) + 5) / 2,
            "D1-04": next(w * (w + 5) for w in range(1, 24) if 2 * (2 * w + 5) == 46),
            "D2-01": 120 * Fraction(2, 3) * Fraction(3, 4) + 18,
            "D2-02": Fraction(5, 6) / (Fraction(1, 6) + Fraction(1, 9)),
            "D2-03": Fraction(sum(6 in x for x in dice), len(dice)),
            "D2-04": next(n for n in range(10, 100) if n % 10 + n // 10 == 11 and n - (n % 10 * 10 + n // 10) == 27),
            "D3-01": sum(x + y + z == 14 for x in range(2, 15) for y in range(3, 15) for z in range(1, 15)),
            "D3-02": next(n for n in range(1, 316) if (n % 5, n % 7, n % 9) == (2, 3, 4)),
            "D3-03": Fraction(math.isqrt(21 * 8 * 7 * 6), 21),
            "D3-04": 5 * (5 * (5 * 5 - 3 * 2) - 3 * 5) - 3 * (5 * 5 - 3 * 2),
            "D4-01": sum(sum(x) == 5 and all(not (x[i] and x[(i + 1) % 12]) for i in range(12))
                          for x in itertools.product([0, 1], repeat=12)),
            "D4-02": sum(math.gcd(n, 210) == 1 for n in range(1, 1001)),
            "D4-03": Fraction(sum(sum(x) % 2 == 1 for x in itertools.combinations(range(1, 13), 6)), math.comb(12, 6)),
            "D4-04": Fraction(12, 2) * Fraction(36 - 18, 12 - 2),
            "D5-01": coefficient[20],
            "D5-02": sum((840 * 840) % a == 0 for a in range(1, 841)),
            "D5-03": paths[8, 8],
            "D5-04": Fraction(sum(sum(v == i for i, v in enumerate(p, 1)) == 2 for p in permutations), len(permutations)),
        }
        for problem in self.suite["problems"]:
            with self.subTest(problem=problem["id"]):
                self.assertEqual(rational(problem["answer"]), computed[problem["id"]])

    def test_prompt_contains_only_question_and_instruction(self):
        p = self.suite["problems"][-1]
        prompt = prompt_for(self.suite, p)
        for excluded in [p["id"], p["answer"], p["solution_ko"], p["question_ko"]]:
            self.assertNotIn(excluded, prompt)
        self.assertEqual(self.cfg["maximum_generation_tokens"], len(self.report()["attempts"]) * self.cfg["max_new_tokens"])

    def test_rational_formats_without_code_or_approximation(self):
        for text in [r"\frac{22}{144}", "22/144", r"\dfrac{11}{72}"]:
            self.assertEqual(rational(text), Fraction(11, 72))
        self.assertEqual(rational("224.4"), Fraction(1122, 5))
        self.assertEqual(rational(r"-\frac{1}{2}"), Fraction(-1, 2))
        for invalid in ["__import__('os')", "11/72=0.1528", "sqrt(4)", "1,000"]:
            with self.assertRaises(ValueError):
                rational(invalid)

    def test_intermediate_correct_number_does_not_count(self):
        p = self.suite["problems"][1]
        row = self.row(p, correct=False)
        row["generated_text"] = "20 and \\boxed{20} </think> \\boxed{-9999}"
        self.assertEqual(score_attempt(row, p, self.cfg)["outcome"], "INCORRECT")

    def test_parser_boundary_ambiguity_and_final_revision(self):
        for text in [r"\boxed{20}", r"x </think> \boxed{20} then instead 21", r"x </think> \boxed{20} \boxed{21}",
                     r"x </think> \boxed{20", r"x </think> y </think> \boxed{20}"]:
            self.assertNotEqual(final_answer(text)["parse_status"], "PARSED")
        self.assertEqual(final_answer(r"x </think> \(\boxed{\frac{2}{4}}\). ")["answer"], "1/2")

    def test_cap_correct_box_remains_censored_and_inconclusive(self):
        report = self.report()
        report["attempts"][0] = self.row(self.suite["problems"][0], censored=True)
        summary = summarize(report, self.cfg, self.suite)
        level = summary["levels"][0]
        self.assertEqual(level["verdict"], "INCONCLUSIVE")
        self.assertEqual(level["observed_final_correct_in_censored"], 1)
        self.assertIsNone(summary["pairs"][0]["natural_total_length_delta_awq_minus_bf16"])

    def test_eos_on_final_step_counts_as_eos(self):
        p = self.suite["problems"][0]
        row = self.row(p, censored=True)
        row["generated_ids"][-1] = 151643
        row.update(termination_record(row["generated_ids"], {151643}, self.cfg["max_new_tokens"]))
        self.assertEqual(score_attempt(row, p, self.cfg)["outcome"], "CORRECT")

    def test_both_seeds_required_and_nonmonotonic_levels_preserved(self):
        report = self.report()
        for index, row in enumerate(report["attempts"]):
            if row["problem_id"] in ["D2-01", "D2-02"] and row["arm"] == "bf16" and row["seed"] == 43:
                problem = next(p for p in self.suite["problems"] if p["id"] == row["problem_id"])
                report["attempts"][index] = self.row(problem, seed=43, correct=False)
        summary = summarize(report, self.cfg, self.suite)
        self.assertEqual(summary["levels"][2]["verdict"], "FAIL")
        self.assertEqual(summary["boundaries"]["bf16"]["highest_contiguous_pass_from_D1"], 1)
        self.assertTrue(summary["boundaries"]["bf16"]["nonmonotonic_or_gapped_passes"])
        self.assertEqual(summary["boundaries"]["awq_w3_replay"]["highest_contiguous_pass_from_D1"], 5)

    def test_missing_duplicate_and_input_mismatch(self):
        report = self.report()
        report["attempts"].pop()
        self.assertEqual(summarize(report, self.cfg, self.suite)["missing_attempts"], 1)
        report["attempts"].append(copy.deepcopy(report["attempts"][0]))
        with self.assertRaises(ValueError):
            summarize(report, self.cfg, self.suite)
        report = self.report()
        report["attempts"][1]["input_ids"][1] = 99
        report["attempts"][1]["input_ids_sha256"] = canonical_hash(report["attempts"][1]["input_ids"])
        with self.assertRaises(ValueError):
            summarize(report, self.cfg, self.suite)

    def test_inconsistent_termination_and_think_ids_rejected(self):
        p = self.suite["problems"][0]
        row = self.row(p)
        row["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            score_attempt(row, p, self.cfg)
        row = self.row(p)
        row["generated_ids"][1] = 17
        with self.assertRaises(ValueError):
            score_attempt(row, p, self.cfg)

    def test_no_old_r0_approval_reused_for_new_config(self):
        status = {"novelty_gate": "SCOPED_CONTRIBUTION_ACCEPTED", "protocol_gate": "ACCEPTED", "research_approval": "APPROVED"}
        with self.assertRaises(ValueError):
            require_scope(status, self.cfg)
        status["difficulty_pilot"] = {"authorization": "USER_REQUESTED_BOUNDED_DIFFICULTY_PILOT", "config_sha256": canonical_hash(self.cfg)}
        require_scope(status, self.cfg)
        changed = dict(self.cfg, max_new_tokens=8192)
        with self.assertRaises(ValueError):
            require_scope(status, changed)

    def test_saved_recipe_dtype_fails_closed(self):
        _bf16_value(1.5, positive=True)
        _bf16_value(0, positive=False)
        for value in [1.1, float("nan"), -1, 0]:
            with self.assertRaises(ValueError):
                _bf16_value(value, positive=True)

    def test_recipe_paths_and_dimensions_without_torch(self):
        shapes = {name + ".weight": [128, 128] for name in LINEARS}
        shapes.update({"input_layernorm.weight": [128], "post_attention_layernorm.weight": [128]})
        paths = [("norm_linears", "input_layernorm", list(LINEARS[:3])),
                 ("linear_linear", "self_attn.v_proj", ["self_attn.o_proj"]),
                 ("norm_linears", "post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
                 ("linear_linear", "mlp.up_proj", ["mlp.down_proj"])]
        row = {"layer": 0, "quantized_linears": list(LINEARS),
               "scales": [{"kind": k, "previous": p, "following": f, "scales": [1.0] * 128} for k, p, f in paths],
               "clips": [{"parameter": n, "shape": [128, 1, 1], "max_values": [[[1.0]] for _ in range(128)]}
                         for n in LINEARS if n not in LINEARS[:2]]}
        validate_recipe(row, 0, shapes)
        row["scales"].reverse()
        with self.assertRaises(ValueError):
            validate_recipe(row, 0, shapes)

    def test_original_files_hash_and_read_only_enforcement(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            run = {"repository": {"commit": "abc"}, "status": "R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW", "completed_awq_layers": 28}
            (folder / "run.json").write_text(json.dumps(run))
            for i in range(28):
                (folder / f"awq_layer_{i:03d}.json").write_text("{}")
            from difficulty_pilot_contracts import file_hash
            review = {"source_run_commit": "abc", "source_files": [{"file": p.name, "sha256": file_hash(p)} for p in folder.iterdir()]}
            verify_r0_files(folder, review)
            self.assertEqual(json.loads((folder / "run.json").read_text()), run)
            (folder / "awq_layer_000.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_r0_files(folder, review)

    def test_read_only_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(self.report()))
            before = path.read_bytes()
            out = path.with_name("summary.json")
            cmd = [sys.executable, str(ROOT / "scripts/summarize_difficulty_pilot.py"), "--run", str(path), "--output", str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("D5", result.stdout)
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(json.loads(out.read_text())["model_ready"])
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_throughput_keeps_censored_generations(self):
        report = self.report()
        report["attempts"][0] = self.row(self.suite["problems"][0], censored=True)
        report["attempts"][0]["elapsed_seconds"] = 0.5
        report["attempts"][1]["elapsed_seconds"] = 1.5
        measured = summarize(report, self.cfg, self.suite)["throughput"]["bf16"]
        self.assertEqual(measured["measured_attempts"], 2)
        self.assertEqual(measured["generated_tokens_including_eos"], 4100)
        self.assertEqual(measured["tokens_per_generation_second"], 2050)

    def test_default_plan_cli_never_loads_model_or_requires_result_files(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_difficulty_pilot.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["attempts"], 80)
        self.assertFalse(plan["model_loaded"])
        self.assertFalse(plan["network_used"])
        self.assertFalse(plan["local_r0_files_verified"])


if __name__ == "__main__":
    unittest.main()
