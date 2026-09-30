"""새 종료-prefix의 위치·근거·추가 예산 승인 보호. 모델/기존 검사 재실행 없음."""
import builtins
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_termination_prefix_diagnostic as runner
from difficulty_pilot_contracts import canonical_hash, file_hash


def fixture(folder):
    cfg = json.loads(runner.CONFIG.read_text())
    source = {"sliding_window": 4096, "use_sliding_window": False}
    base = {"model_configuration_sha256": canonical_hash(source),
            "effective_source_configuration_sha256": canonical_hash({**source, "sliding_window": None}),
            "eos_token_id": 151643, "vocab_size": 151936}
    packet = {"cached_model_config": {"configuration": source}, "run_snapshot": {"attempts": []}}
    for ref in cfg["inputs"]:
        is_b = ref["arm"] == "bf16"
        row = {"problem_id": "D2-03", "arm": ref["arm"], "seed": 42, "input_ids": [101] * 71,
               "generated_ids": [102] * 2424 + [151643] if is_b else [103] * 4096,
               "finish_reason": "eos" if is_b else "length", "right_censored": not is_b}
        ref["attempt_sha256"] = canonical_hash(row)
        ref["sequence_ids_sha256"] = canonical_hash(row["input_ids"] + row["generated_ids"][:2424])
        packet["run_snapshot"]["attempts"].append(row)
    evidence, old, audit = (folder / n for n in ("evidence.json", "old.json", "audit.json"))
    evidence.write_text(json.dumps(packet))
    old.write_text(json.dumps({"config_sha256": canonical_hash(base)}))
    audit.write_text(json.dumps({"status": "SAVED_LOGITS_AUDITED_FLOAT64_FILTER_DIAGNOSTICS",
                                 "source_run_sha256": file_hash(old), "t1_recalculation": {"all_match": True}}))
    cfg.update(base_fixed_prefix_config_sha256=canonical_hash(base), evidence_packet_sha256=file_hash(evidence),
               source_fixed_prefix_run_sha256=file_hash(old), source_logits_audit_sha256=file_hash(audit))
    return cfg, base, evidence, old, audit


def forbid_heavy_imports(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers", "huggingface_hub"}:
        raise AssertionError("승인/계획 조회에서 heavy import가 발생했습니다.")
    return REAL_IMPORT(name, *args, **kwargs)


REAL_IMPORT = builtins.__import__


class TerminationPrefixTests(unittest.TestCase):
    def test_fixed_anchor_positions_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, base, evidence, old, audit = fixture(Path(tmp))
            before = [file_hash(p) for p in (evidence, old, audit)]
            rows, source, effective = runner.read_inputs(evidence, old, audit, cfg, base)
            self.assertEqual([len(r["ids"]) for r in rows], [2495, 2495])
            self.assertEqual([r["source_next_token_is_eos"] for r in rows], [True, False])
            for row in rows:
                self.assertNotIn(151643, row["ids"])
                self.assertEqual(row["logit_positions"], list(range(2366, 2495)))
            self.assertEqual(sum(len(r["ids"]) for r in rows) * 3, 14970)
            self.assertEqual(source["sliding_window"], 4096)
            self.assertIsNone(effective["sliding_window"])
            self.assertEqual(before, [file_hash(p) for p in (evidence, old, audit)])

    def test_changed_budget_filter_positions_or_source_arm_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, base, evidence, old, audit = fixture(Path(tmp))
            changes = []
            for key, val in (("maximum_forward_calls", 8), ("maximum_sequence_length", 4096),
                             ("new_generation_tokens", 1), ("temperature", 0.7), ("use_cache", True)):
                bad = copy.deepcopy(cfg); bad[key] = val; changes.append(bad)
            bad = copy.deepcopy(cfg); bad["inputs"][0]["logit_positions"][-1] -= 1; changes.append(bad)
            bad = copy.deepcopy(cfg); bad["inputs"].reverse(); changes.append(bad)
            bad = copy.deepcopy(cfg); bad["inputs"][0]["source_next_token_is_eos"] = False; changes.append(bad)
            for bad in changes:
                with self.subTest(config=bad["maximum_forward_calls"]), self.assertRaises(ValueError):
                    runner.read_inputs(evidence, old, audit, bad, base)

    def test_changed_evidence_and_eos_leak_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, base, evidence, old, audit = fixture(Path(tmp))
            evidence.write_text(evidence.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "바이트 해시"):
                runner.read_inputs(evidence, old, audit, cfg, base)
            packet = json.loads(evidence.read_text())
            row = packet["run_snapshot"]["attempts"][0]
            row["generated_ids"][200] = 151643
            evidence.write_text(json.dumps(packet));cfg["evidence_packet_sha256"] = file_hash(evidence)
            with self.assertRaisesRegex(ValueError, "응답"):
                runner.read_inputs(evidence, old, audit, cfg, base)
            cfg["inputs"][0]["attempt_sha256"] = canonical_hash(row)
            cfg["inputs"][0]["sequence_ids_sha256"] = canonical_hash(row["input_ids"] + row["generated_ids"][:2424])
            with self.assertRaisesRegex(ValueError, "EOS"):
                runner.read_inputs(evidence, old, audit, cfg, base)

    def test_new_scope_requires_own_approval_budget_and_code(self):
        cfg = json.loads(runner.CONFIG.read_text())
        state = {"research_approval": "APPROVED", "novelty_gate": "SCOPED_CONTRIBUTION_ACCEPTED",
                 "protocol_gate": "ACCEPTED", "fixed_prefix_diagnostic": {"authorization": "APPROVED"}}
        with self.assertRaisesRegex(ValueError, "별도 승인"):
            runner.require_authorization(state, cfg)
        hashes = {n: file_hash(runner.ROOT / "scripts" / n) for n in runner.IMPLEMENTATION}
        state["termination_prefix_diagnostic"] = {"authorization": "APPROVED", "config_sha256": canonical_hash(cfg),
                                                  "remaining_forward_calls": 6, "implementation_sha256": hashes}
        self.assertEqual(runner.require_authorization(state, cfg), hashes)
        for key, val in (("remaining_forward_calls", 0), ("config_sha256", "wrong"),
                         ("authorization", "PENDING"), ("implementation_sha256", {})):
            bad = copy.deepcopy(state);bad["termination_prefix_diagnostic"][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.require_authorization(bad, cfg)

    def test_plan_has_no_model_import_or_output_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);cfg, base, evidence, old, audit = fixture(folder)
            config, basefile = folder / "config.json", folder / "base.json"
            config.write_text(json.dumps(cfg));basefile.write_text(json.dumps(base))
            (folder / "docs").mkdir();(folder / "docs/research_status.json").write_text("{}")
            output = folder / "new-output";out = io.StringIO()
            with patch.object(runner, "ROOT", folder), patch.object(runner, "CONFIG", config), \
                    patch.object(runner, "BASE_CONFIG", basefile), patch("builtins.__import__", forbid_heavy_imports), \
                    contextlib.redirect_stdout(out):
                code = runner.main(["--evidence", str(evidence), "--source-run", str(old),
                                    "--audit", str(audit), "--output-dir", str(output), "--plan"])
            self.assertEqual(code, 0)
            result = json.loads(out.getvalue())
            self.assertEqual(result["maximum_forward_calls"], 6)
            self.assertEqual(result["authorization"], "PENDING")
            self.assertNotIn('"ids"', out.getvalue())
            self.assertFalse(output.exists())

    def test_unapproved_execute_stops_before_missing_inputs_and_heavy_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);(folder / "docs").mkdir()
            (folder / "docs/research_status.json").write_text(json.dumps({"research_approval": "APPROVED"}))
            output = folder / "output";error = io.StringIO()
            with patch.object(runner, "ROOT", folder), patch("builtins.__import__", forbid_heavy_imports), \
                    contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                runner.main(["--execute", "--output-dir", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("별도 승인", error.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
