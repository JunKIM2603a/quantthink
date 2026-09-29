"""중단·재개·보존의 CPU 계약/실행기 모의 연동 검사. CUDA·실모델·AWQ 계산 없음."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from difficulty_pilot_contracts import canonical_hash, load_suite, prompt_for
from difficulty_pilot_resume import (ProgressReporter, backup_before_resume, check_parameter_hash,
                                     mark_interrupted, output_lock, planned_order, resume_plan,
                                     verify_compatibility)
from reproduction_contracts import termination_record
import run_difficulty_pilot as runner


class DifficultyResumeTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.suite = load_suite()
        self.inputs = {p["id"]: [151646, 100 + i, 151648, 198] for i, p in enumerate(self.suite["problems"])}

    def report(self, n=12):
        by_id = {p["id"]: p for p in self.suite["problems"]}
        rows = []
        for pid, arm, seed in planned_order(self.cfg, self.suite)[:n]:
            ids = [11, 151649, 12, 151643]
            rows.append({"problem_id": pid, "arm": arm, "seed": seed,
                         "input_ids": self.inputs[pid], "input_ids_sha256": canonical_hash(self.inputs[pid]), "input_token_count": 4,
                         "prompt_sha256": canonical_hash(prompt_for(self.suite, by_id[pid])),
                         "generated_ids": ids, "generated_text": "x</think>\\boxed{" + by_id[pid]["answer"] + "}",
                         **termination_record(ids, {151643}, self.cfg["max_new_tokens"])})
        critical = ["difficulty_pilot_contracts.py", "replay_r0_awq.py", "qwen2_awq_adapter.py", "runtime_assets.py",
                    "runtime_contracts.py", "reproduction_contracts.py"]
        return {"status": "DIFFICULTY_PILOT_STARTED", "attempts": rows,
                "config_sha256": canonical_hash(self.cfg), "suite_sha256": canonical_hash(self.suite),
                "parameter_sha256": {"bf16": "a" * 64, **({"awq_w3_replay": "b" * 64} if n > 40 else {})},
                "repository": {"commit": "previous"}, "research_status_sha256": "old-status",
                "packages": {"torch": "2.7.1+cu118"}, "python": "3.11.16",
                "model_reference": {"repo_id": "fixture", "revision": "pinned"},
                "generation_config": {"max_new_tokens": 4096}, "runtime_flags": {"deterministic_algorithms": False},
                "gpu": {"device": "cuda:0", "name": "fixture", "total_memory_bytes": 1},
                "r0_input_files": ["fixture"], "upstream_source_files": ["fixture"],
                "implementation_sha256": {**{n: "fixed" for n in critical}, "run_difficulty_pilot.py": "old-runner"}}

    def test_twelve_preserved_sixtyeight_remaining(self):
        previous = self.report()
        before = canonical_hash(previous)
        plan = resume_plan(previous, self.cfg, self.suite)
        self.assertEqual((plan["completed"], plan["remaining"]), (12, 68))
        self.assertEqual(plan["pending_keys"][0], ("D2-03", "bf16", 42))
        self.assertEqual(canonical_hash(previous), before)

    def test_censored_and_wrong_completed_outputs_are_not_retried(self):
        previous = self.report()
        row = previous["attempts"][0]
        row["generated_ids"] = [11, 151649] + [12] * 4094
        row["generated_text"] = "x</think>\\boxed{-9}"
        row.update(termination_record(row["generated_ids"], {151643}, 4096))
        self.assertEqual(resume_plan(previous, self.cfg, self.suite)["remaining"], 68)

    def test_duplicates_gaps_and_errors_rejected(self):
        for change in ("duplicate", "gap", "error"):
            r = self.report()
            if change == "duplicate":
                r["attempts"].append(r["attempts"][0])
            elif change == "gap":
                r["attempts"].pop(0)
            else:
                r["attempts"][-1]["finish_reason"] = "error"
            with self.subTest(change=change), self.assertRaises(ValueError):
                resume_plan(r, self.cfg, self.suite)

    def test_runtime_and_input_changes_rejected_runner_change_recordable(self):
        old = self.report()
        current = copy.deepcopy(old)
        current["implementation_sha256"]["run_difficulty_pilot.py"] = "new-runner"
        verify_compatibility(old, current, self.inputs)
        for field in ["packages", "generation_config", "runtime_flags", "model_reference", "gpu"]:
            changed = copy.deepcopy(current)
            changed[field] = {"changed": True}
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_compatibility(old, changed, self.inputs)
        changed = copy.deepcopy(self.inputs)
        changed["D1-01"] = [1]
        with self.assertRaises(ValueError):
            verify_compatibility(old, current, changed)
        current["implementation_sha256"]["replay_r0_awq.py"] = "changed"
        with self.assertRaises(ValueError):
            verify_compatibility(old, current, self.inputs)

    def test_parameter_hash_mismatch_blocks_reuse(self):
        r = self.report()
        check_parameter_hash(r, "bf16", "a" * 64)
        with self.assertRaises(ValueError):
            check_parameter_hash(r, "bf16", "c" * 64)
        self.assertEqual(r["parameter_sha256"]["bf16"], "a" * 64)

    def test_backup_keeps_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            raw = json.dumps(self.report(), ensure_ascii=False, indent=3).encode() + b"\n"
            path.write_bytes(raw)
            proof = backup_before_resume(path)
            self.assertEqual((path.parent / proof["file"]).read_bytes(), raw)
            self.assertEqual(backup_before_resume(path), proof)
            self.assertEqual(path.read_bytes(), raw)

    def test_lock_prevents_second_writer_and_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            with output_lock(folder, resume=False):
                (folder / "run.json").write_text("{}")
                with self.assertRaises(ValueError):
                    with output_lock(folder, resume=True):
                        self.fail("second writer allowed")
            with output_lock(folder, resume=True):
                self.assertEqual((folder / "run.json").read_text(), "{}")
            with self.assertRaises(ValueError):
                with output_lock(folder, resume=False):
                    self.fail("existing output overwritten")

    def test_keyboard_interrupt_metadata_keeps_completed_rows(self):
        r = self.report()
        before = canonical_hash(r["attempts"])
        r["active_attempt"] = {"problem_id": "D2-03", "arm": "bf16", "seed": 42}
        mark_interrupted(r)
        self.assertEqual(r["status"], "DIFFICULTY_PILOT_INTERRUPTED_RESUMABLE")
        self.assertEqual(canonical_hash(r["attempts"]), before)
        self.assertEqual(r["interruptions"][-1]["active_attempt"]["problem_id"], "D2-03")
        self.assertNotIn("active_attempt", r)

    def test_progress_is_time_based_and_does_not_read_token_values(self):
        now, messages = [0], []
        progress = ProgressReporter("fixture", 4096, clock=lambda: now[0], emit=lambda s, **kw: messages.append(s))
        now[0] = 14
        progress.tick(30)
        self.assertEqual(messages, [])
        now[0] = 15
        progress.tick(31)
        self.assertIn("31/4096", messages[0])

    def test_resume_preview_cli_no_model_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(self.report()))
            before = path.read_bytes()
            r = subprocess.run([sys.executable, str(ROOT / "scripts/run_difficulty_pilot.py"), "--resume", "--output-dir", directory], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["remaining"], 68)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def mocked_execution(self, directory, *, interrupt=False, n=12):
        folder = Path(directory)
        old = self.report(n)
        (folder / "run.json").write_text(json.dumps(old))
        for i in range(28):
            (folder / f"awq_layer_{i:03d}.json").write_text("{}")
        current = copy.deepcopy(old)
        current["repository"] = {"commit": "resume-code"}
        current["implementation_sha256"]["run_difficulty_pilot.py"] = "new-runner"
        calls, seeds = [], []

        class Tensor:
            def __init__(self, values):
                self.values, self.device = values, "cuda:0"
                self.shape = (1, len(values[0])) if values and isinstance(values[0], list) else (len(values),)
            def tolist(self): return self.values
            def detach(self): return self
            def cpu(self): return self
            def __getitem__(self, index): return Tensor([11, 151649, 12, 151643])

        class Qwen2ForCausalLM:
            def __init__(self):
                self.model = SimpleNamespace(layers=[SimpleNamespace(named_parameters=lambda: []) for _ in range(28)])
                self.config = SimpleNamespace(max_position_embeddings=131072)
            def cpu(self): return self
            def to(self, device): return self
            def eval(self): return self
            def generate(self, input_ids, **kwargs):
                calls.append((input_ids.values[0][1], seeds[-1]))
                if interrupt:
                    raise KeyboardInterrupt
                # 추가 종료 조건이 항상 False인지 모의 타입으로 확인합니다.
                self_check = kwargs["stopping_criteria"][0](Tensor([[1, 2, 3, 4, 5]]), None)
                if self_check != [False]:
                    raise AssertionError("progress stops generation")
                return Tensor([[1]])

        class GenerationConfig:
            def __init__(self, **kwargs): self.values = kwargs
            def to_dict(self): return self.values

        torch = ModuleType("torch")
        torch.bfloat16, torch.long, torch.bool = "bfloat16", "long", "bool"
        torch.tensor = lambda values, **kw: Tensor(values)
        torch.ones_like = lambda x: x
        torch.zeros = lambda n, **kw: [False] * n
        torch.manual_seed = seeds.append
        torch.get_rng_state = lambda: Tensor([1, 2])
        torch.inference_mode = nullcontext
        torch.cuda = SimpleNamespace(synchronize=lambda *a: None, reset_peak_memory_stats=lambda *a: None,
                                     get_rng_state=lambda *a: Tensor([3, 4]), max_memory_allocated=lambda *a: 1)
        transformers = ModuleType("transformers")
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *a, **k: Qwen2ForCausalLM())
        transformers.GenerationConfig = GenerationConfig
        transformers.StoppingCriteria, transformers.StoppingCriteriaList = object, list
        runtime = ModuleType("runtime_contracts")
        runtime.generated_record = lambda ids, tokenizer, cap: {"generated_ids": ids, "generated_text": "x</think>\\boxed{0}", **termination_record(ids, {151643}, cap)}
        adapter = ModuleType("qwen2_awq_adapter")
        adapter.load_upstream = lambda *a: {}
        args = SimpleNamespace(resume=True, output_dir=folder, r0_dir=folder, upstream_dir=folder)
        modules = {"torch": torch, "transformers": transformers, "runtime_contracts": runtime, "qwen2_awq_adapter": adapter}
        hash_values = iter(["a" * 64, "b" * 64])
        with patch.dict(sys.modules, modules), patch.object(runner, "validate_recipe"), \
             patch.object(runner, "verify_r0_files"), patch.object(runner, "replay", return_value={"status": "mock"}), \
             patch.object(runner, "parameter_hash", side_effect=lambda model: next(hash_values)), redirect_stdout(io.StringIO()):
            if interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute_locked(args, self.cfg, self.suite, current, self.inputs, None, {}, {}, "cuda:0")
            else:
                runner.execute_locked(args, self.cfg, self.suite, current, self.inputs, None, {}, {}, "cuda:0")
        return old, json.loads((folder / "run.json").read_text()), calls

    def test_runner_resumes_12_to_80_without_repeating_completed_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            old, result, calls = self.mocked_execution(directory)
            self.assertEqual(len(calls), 68)
            self.assertEqual(calls[0], (106, 42))
            self.assertEqual(result["attempts"][:12], old["attempts"])
            self.assertEqual(len(result["attempts"]), 80)
            self.assertEqual(result["repository"], old["repository"])
            self.assertEqual(result["execution_segments"][-1]["repository"]["commit"], "resume-code")
            self.assertEqual(result["status"], "DIFFICULTY_PILOT_EXECUTED_PENDING_REVIEW")

    def test_runner_interrupt_does_not_append_or_lose_partial_response(self):
        with tempfile.TemporaryDirectory() as directory:
            old, result, calls = self.mocked_execution(directory, interrupt=True)
            self.assertEqual(result["attempts"], old["attempts"])
            self.assertEqual(result["status"], "DIFFICULTY_PILOT_INTERRUPTED_RESUMABLE")
            self.assertEqual(len(calls), 1)
            self.assertEqual(resume_plan(result, self.cfg, self.suite)["remaining"], 68)

    def test_runner_awq_partial_resume_skips_all_bf16_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            old, result, calls = self.mocked_execution(directory, n=45)
            self.assertEqual(len(calls), 35)
            self.assertEqual(result["attempts"][:45], old["attempts"])
            self.assertTrue(all(r["arm"] == "awq_w3_replay" for r in result["attempts"][45:]))

    def test_recovery_command_restores_only_deleted_trial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            def command(*args):
                return subprocess.run(args, cwd=folder, check=True, capture_output=True, text=True)
            command("git", "init", "-q")
            names = ["configs/difficulty_pilot_v01.json", "docs/DIFFICULTY_PILOT_V01_KO.md",
                     "docs/DIFFICULTY_QUESTIONS_V01_KO.md", "docs/REALISTIC_INPUT_PILOT_PLAN_20260929_KO.md",
                     "fixtures/difficulty_ladder_v01.json", "scripts/difficulty_pilot_contracts.py",
                     "scripts/replay_r0_awq.py", "scripts/run_difficulty_pilot.py", "scripts/summarize_difficulty_pilot.py",
                     "tests/test_difficulty_pilot.py", "other.txt", "removed_elsewhere.txt"]
            for name in names:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            command("git", "add", ".")
            command("git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            for name in names[:10] + ["removed_elsewhere.txt"]:
                (folder / name).unlink()
            (folder / "other.txt").write_text("keep edited")
            (folder / "results/local").mkdir(parents=True)
            (folder / "results/local/run.json").write_text("keep results")
            command("bash", "-c", "git diff --name-only --diff-filter=D -z -- "
                    "configs/difficulty_pilot_v01.json fixtures/difficulty_ladder_v01.json "
                    "'docs/DIFFICULTY_*_V01_KO.md' docs/REALISTIC_INPUT_PILOT_PLAN_20260929_KO.md "
                    "'scripts/*difficulty_pilot*.py' scripts/replay_r0_awq.py tests/test_difficulty_pilot.py "
                    "| xargs -0 -r git restore --source=HEAD --worktree --")
            self.assertTrue(all((folder / name).read_text() == "original" for name in names[:10]))
            self.assertFalse((folder / "removed_elsewhere.txt").exists())
            self.assertEqual((folder / "other.txt").read_text(), "keep edited")
            self.assertEqual((folder / "results/local/run.json").read_text(), "keep results")


if __name__ == "__main__":
    unittest.main()
