"""고정 Pile의 큰 행과 calibration 길이 제외에 대한 회귀 검사."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_assets as assets
import prepare_runtime_assets as prepare


class JsonlLimitTests(unittest.TestCase):
    def test_large_text_is_read_then_excluded_without_shifting_source_rows(self):
        _, _, policy = assets.load_contracts()
        policy = copy.deepcopy(policy)
        policy["calibration"].update(blocks=1, block_size=3)
        huge = "x" * (assets.DEFAULT_JSONL_MAX_LINE_CHARACTERS + 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            path.write_text(json.dumps({"text": huge}) + '\n{"text":"abc"}\n')
            stats, encoded = {}, []
            def encode(text):
                encoded.append(text)
                return [1, 2, 3]
            rows = assets.iter_jsonl(path, statistics=stats,
                max_line_characters=assets.CALIBRATION_JSONL_MAX_LINE_CHARACTERS)
            blocks, report = assets.select_calibration(rows, encode, set(), policy)
            self.assertEqual(blocks, [[1, 2, 3]])
            self.assertEqual(encoded, ["abc"])
            self.assertEqual(report["documents"][0]["source_row"], 1)
            self.assertEqual(report["counts"]["excluded_text_length"], 1)
            self.assertEqual(stats["source_rows"], 2)
            self.assertEqual(stats["rows_above_default_line_limit"], 1)

    def test_large_metadata_does_not_discard_eligible_short_text(self):
        _, _, policy = assets.load_contracts()
        policy = copy.deepcopy(policy)
        policy["calibration"].update(blocks=1, block_size=3)
        obj = {"text": "abc", "metadata": "m" * assets.DEFAULT_JSONL_MAX_LINE_CHARACTERS}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            path.write_text(json.dumps(obj) + "\n")
            rows = assets.iter_jsonl(path,
                max_line_characters=assets.CALIBRATION_JSONL_MAX_LINE_CHARACTERS)
            _, report = assets.select_calibration(rows, lambda _: [1, 2, 3], set(), policy)
            self.assertEqual(report["counts"]["eligible_unique_documents"], 1)
            self.assertEqual(report["documents"][0]["source_row"], 0)

    def test_line_limit_reports_file_row_and_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.jsonl"
            path.write_text('{}\n' + json.dumps({"text": "x" * 50}) + '\n{}\n')
            rows = assets.iter_jsonl(path, max_line_characters=32)
            self.assertEqual(next(rows), (0, {}))
            with self.assertRaisesRegex(ValueError, r"행 길이 상한.*sample.jsonl.*source_row=1.*33자 > 32자"):
                next(rows)

    def test_row_limit_exact_eof_passes_and_excess_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.jsonl"
            path.write_text('{}\n{}')
            self.assertEqual(len(list(assets.iter_jsonl(path, max_rows=2))), 2)
            with self.assertRaisesRegex(ValueError, r"행 수 상한.*source_row=1.*max_rows=1"):
                list(assets.iter_jsonl(path, max_rows=1))

    def test_exact_character_limit_and_unicode_are_not_byte_limits(self):
        line = json.dumps({"text": "한글😃"}, ensure_ascii=False) + "\n"
        self.assertGreater(len(line.encode()), len(line))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unicode.jsonl"
            path.write_text(line, encoding="utf-8")
            self.assertEqual(list(assets.iter_jsonl(path, max_line_characters=len(line)))[0][1],
                             {"text": "한글😃"})
            with self.assertRaisesRegex(ValueError, "행 길이 상한"):
                list(assets.iter_jsonl(path, max_line_characters=len(line) - 1))

    def test_invalid_or_blank_json_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.jsonl"
            for tail in ('{"text":', '\n', '[]\n'):
                path.write_text('{}\n' + tail)
                with self.assertRaisesRegex(ValueError, "source_row=1"):
                    list(assets.iter_jsonl(path))

    def test_invalid_limits_rejected(self):
        for value in (0, -1, True, 3.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(assets.iter_jsonl("unused", max_line_characters=value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(assets.iter_jsonl("unused", max_rows=value))

    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "zstandard 없음")
    def test_large_row_spanning_zstd_frames_preserves_following_row(self):
        import zstandard
        content = (json.dumps({"text": "x" * (assets.DEFAULT_JSONL_MAX_LINE_CHARACTERS + 1)})
                   + '\n{"text":"after"}\n').encode()
        split = len(content) // 2
        compressor = zstandard.ZstdCompressor()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frames.jsonl.zst"
            path.write_bytes(compressor.compress(content[:split]) + compressor.compress(content[split:]))
            rows = assets.iter_jsonl(path, compressed=True,
                max_line_characters=assets.CALIBRATION_JSONL_MAX_LINE_CHARACTERS)
            first, second = list(rows)
            self.assertEqual(len(first[1]["text"]), assets.DEFAULT_JSONL_MAX_LINE_CHARACTERS + 1)
            self.assertEqual(second, (1, {"text": "after"}))


class CalibrationLengthTests(unittest.TestCase):
    def test_over_512_tokens_are_excluded_without_truncation_or_model_warning(self):
        _, _, policy = assets.load_contracts()
        policy = copy.deepcopy(policy)
        policy["calibration"].update(blocks=1, block_size=2)
        calls = []
        def encode(text, **kwargs):
            calls.append(kwargs)
            return [1] * 513 if text == "long tokens" else [2, 3]
        tokenizer = SimpleNamespace(encode=encode)
        blocks, report = assets.select_calibration(
            [(0, {"text": "long tokens"}), (1, {"text": "short"})],
            lambda text: prepare.encode_calibration_text(tokenizer, text), set(), policy)
        self.assertEqual(blocks, [[2, 3]])
        self.assertEqual(report["documents"][0]["source_row"], 1)
        self.assertEqual(report["counts"]["excluded_token_length"], 1)
        self.assertEqual(calls, [{"add_special_tokens": False, "truncation": False, "verbose": False}] * 2)

    def test_exclusion_counts_reconcile_without_changing_selection(self):
        _, _, policy = assets.load_contracts()
        policy = copy.deepcopy(policy)
        policy["calibration"].update(blocks=1, block_size=3, max_text_characters=10,
                                     max_document_tokens=3)
        rows = list(enumerate({"text": x} for x in ("  ", "x" * 11, "held", " abc ", "abc", "defg")))
        _, report = assets.select_calibration(rows, lambda x: list(x.encode()),
                                              {assets.question_hash("held")}, policy)
        counts = report["counts"]
        self.assertEqual(counts["source_rows"], sum(v for k, v in counts.items() if k != "source_rows"))
        self.assertEqual(report["documents"][0]["source_row"], 3)

    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "zstandard 없음")
    def test_all_pipeline_accepts_large_excluded_row_and_records_limits(self):
        import zstandard
        candidate, refs, policy = assets.load_contracts()
        candidate, policy = copy.deepcopy(candidate), copy.deepcopy(policy)
        candidate["r1"]["n_problems"] = 2
        policy.update(expected_development_rows=2, expected_confirmation_rows=1)
        policy["calibration"].update(blocks=1, block_size=3)
        dev = [{"question": f"dev {i}", "answer": f"#### {i}"} for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {role: root / role for role in ("development", "confirmation", "awq_calibration")}
            paths["development"].write_text(json.dumps(dev))
            paths["confirmation"].write_text('{"problem":"held"}\n')
            content = (json.dumps({"text": "x" * (assets.DEFAULT_JSONL_MAX_LINE_CHARACTERS + 1)})
                       + '\n{"text":"abc"}\n').encode()
            paths["awq_calibration"].write_bytes(zstandard.ZstdCompressor().compress(content))
            by_repo = {a["repo_id"]: a for a in refs["assets"] if a["role"] in paths}
            def info(repo_id, **kwargs):
                asset = by_repo[repo_id]
                path = paths[asset["role"]]
                return SimpleNamespace(sha=asset["revision"], siblings=[SimpleNamespace(
                    rfilename=policy["source_paths"][asset["role"]], size=path.stat().st_size,
                    lfs=None, blob_id=assets.file_digests(path)["git_blob_sha1"])])
            def download(repo_id, **kwargs):
                return paths[by_repo[repo_id]["role"]]
            hub = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(repo_info=info), hf_hub_download=download)
            tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
            deps = {"huggingface-hub": "0.36.2", "pyarrow": "20.0.0",
                    "zstandard": "0.23.0", "transformers": "4.51.3"}
            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 patch.object(prepare, "package_versions", return_value=deps), \
                 patch.object(prepare, "read_development", return_value=dev), \
                 patch.object(prepare, "load_tokenizer", return_value=(tokenizer, {"test_double": True})), \
                 patch("sys.stdout", new_callable=io.StringIO):
                result = prepare.prepare_data(root / "output", candidate, refs, policy, include_calibration=True)
            cal = json.loads((root / "output/calibration_manifest.json").read_text())
            self.assertEqual(result["status"], "PREPARED_CANDIDATE_NOT_RUN_APPROVAL")
            self.assertEqual(cal["jsonl_reader"]["rows_above_default_line_limit"], 1)
            self.assertEqual(cal["jsonl_reader"]["max_line_characters"], assets.CALIBRATION_JSONL_MAX_LINE_CHARACTERS)
            self.assertEqual(cal["counts"]["excluded_text_length"], 1)
            self.assertEqual(cal["documents"][0]["source_row"], 1)
            self.assertEqual(len(result["source_files"]), 3)
            for name, hashes in result["artifacts"].items():
                self.assertEqual(hashes, assets.file_digests(root / "output" / name))


if __name__ == "__main__":
    unittest.main()
