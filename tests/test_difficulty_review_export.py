"""기존 결과 내보내기의 보존·불일치 검출만 확인. 모델/준비 검사는 실행하지 않음."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from difficulty_pilot_contracts import canonical_hash, load_suite, prompt_for, summarize
from export_difficulty_review import build_packet, cached_model_config
from reproduction_contracts import termination_record


class DifficultyReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.cfg, self.suite = load_suite()
        rows = []
        for arm in self.cfg["arms"]:
            for problem in self.suite["problems"]:
                for seed in self.cfg["seeds"]:
                    ids = [11, 151649, 12, 151643]
                    text = "생각 </think> \\boxed{" + problem["answer"] + "}."
                    if not rows:
                        text += " 단위 설명"  # 형식 미판정도 원문 보존
                    rows.append({"problem_id": problem["id"], "arm": arm, "seed": seed,
                                 "input_ids": [151646, 19], "input_token_count": 2,
                                 "input_ids_sha256": canonical_hash([151646, 19]),
                                 "prompt_sha256": canonical_hash(prompt_for(self.suite, problem)),
                                 "generated_ids": ids, "generated_text": text,
                                 **termination_record(ids, {151643}, 4096)})
        rows[-1]["generated_ids"] = [11] * 4096
        rows[-1]["generated_text"] = "검열된 생각"
        rows[-1].update(termination_record(rows[-1]["generated_ids"], {151643}, 4096))
        self.report = {"config_sha256": canonical_hash(self.cfg),
                       "suite_sha256": canonical_hash(self.suite), "attempts": rows}
        self.run = self.folder / "run.json"
        self.summary = self.folder / "summary.json"
        self.save()

    def save(self):
        self.run.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")
        self.summary.write_text(json.dumps(summarize(self.report, self.cfg, self.suite),
                                           ensure_ascii=False), encoding="utf-8")

    def test_complete_ids_text_and_original_bytes_preserved(self):
        before = [p.read_bytes() for p in (self.run, self.summary)]
        packet = build_packet(self.run, self.summary)
        self.assertEqual(packet["run_snapshot"], self.report)
        self.assertEqual(packet["source_files"]["run.json"]["sha256"], hashlib.sha256(before[0]).hexdigest())
        self.assertEqual(len(packet["run_snapshot"]["attempts"]), 80)
        self.assertEqual(before, [p.read_bytes() for p in (self.run, self.summary)])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_stale_summary_and_invalid_termination_rejected(self):
        summary = json.loads(self.summary.read_text())
        summary["missing_attempts"] = 1
        self.summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "summary.json"):
            build_packet(self.run, self.summary)
        self.save()
        self.report["attempts"][0]["generated_ids"][-1] = 100
        self.run.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "종료 필드"):
            build_packet(self.run, self.summary)

    def test_resume_backup_prefix_and_missing_backup(self):
        old = copy.deepcopy(self.report)
        old["attempts"] = old["attempts"][:12]
        data = json.dumps(old).encode()
        digest = hashlib.sha256(data).hexdigest()
        backup = self.folder / f"run.before_resume.{digest}.json"
        backup.write_bytes(data)
        self.report["execution_segments"] = [{"completed_at_start": 12,
            "completed_attempts_sha256": canonical_hash(old["attempts"]),
            "resume_backup": {"file": backup.name, "sha256": digest}}]
        self.save()
        packet = build_packet(self.run, self.summary)
        self.assertEqual(packet["resume_backup_evidence"][0]["preserved_attempts"], 12)
        self.assertEqual(packet["resume_backup_evidence"][0]["backup_snapshot"], old)
        self.report["attempts"][0]["generated_text"] += " changed"
        self.save()
        with self.assertRaisesRegex(ValueError, "완료 응답"):
            build_packet(self.run, self.summary)
        backup.unlink()
        self.assertEqual(build_packet(self.run, self.summary)["resume_backup_evidence"][0]["status"],
                         "BACKUP_NOT_FOUND_NOT_VERIFIED")

    def test_cli_writes_once_and_refuses_overwrite(self):
        output = self.folder / "review.json"
        cmd = [sys.executable, str(ROOT / "scripts/export_difficulty_review.py"),
               "--run", str(self.run), "--summary", str(self.summary), "--output", str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = output.read_bytes()
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), data)

    def test_cached_config_is_offline_and_hash_checked(self):
        reference = json.loads((ROOT / "configs/asset_inspection_refs.json").read_text())
        asset = next(a for a in reference["assets"] if a["role"] == "bf16_reference")
        self.report["model_reference"] = asset
        path = self.folder / "config.json"
        path.write_text('{"sliding_window":4096,"use_sliding_window":false}')
        reference["model_configuration_sha256"]["config.json"] = canonical_hash(json.loads(path.read_text()))
        mock_root = self.folder / "repo"
        (mock_root / "configs").mkdir(parents=True)
        (mock_root / "configs/asset_inspection_refs.json").write_text(json.dumps(reference))
        download = Mock(return_value=str(path))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}), \
                patch("export_difficulty_review.ROOT", mock_root):
            result = cached_model_config(self.report)
            self.assertEqual(result["status"], "PINNED_CACHED_CONFIG_MATCH")
            download.assert_called_once_with(asset["repo_id"], "config.json", revision=asset["revision"],
                                             token=False, local_files_only=True)
            path.write_text('{"sliding_window":8192}')
            with self.assertRaisesRegex(ValueError, "설정 해시"):
                cached_model_config(self.report)
            download.side_effect = OSError("캐시 없음")
            self.assertEqual(cached_model_config(self.report)["status"], "CACHED_CONFIG_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
