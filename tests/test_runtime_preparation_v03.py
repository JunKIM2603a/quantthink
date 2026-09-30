import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_assets as assets
import runtime_contracts as contracts
import prepare_runtime_assets as prepare
import qwen2_awq_adapter as adapter
import run_r0


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.asset = {"role": "development", "repo_id": "test/public", "repo_type": "dataset",
                      "revision": "a" * 40}
        self.raw = b'{"question":"synthetic"}\n'
        self.sibling = SimpleNamespace(rfilename="train.jsonl", size=len(self.raw), lfs=None,
                                       blob_id=hashlib.sha1(b"blob " + str(len(self.raw)).encode() + b"\0" + self.raw).hexdigest())
        self.info = SimpleNamespace(sha=self.asset["revision"], siblings=[self.sibling])

    def test_git_blob_header_is_verified(self):
        plan = assets.source_plan(self.info, self.asset, "train.jsonl")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.jsonl"
            path.write_bytes(self.raw)
            proof = assets.verify_source(path, plan)
            self.assertEqual(proof["verification"], "BYTES_MATCH_PINNED_HUB_METADATA")
            self.assertEqual(proof["observed_sha256"], hashlib.sha256(self.raw).hexdigest())
            path.write_bytes(self.raw.replace(b"synthetic", b"tampered!"))
            with self.assertRaises(ValueError):
                assets.verify_source(path, plan)

    def test_lfs_content_sha256_not_pointer_blob(self):
        self.sibling.lfs = SimpleNamespace(size=len(self.raw), sha256=hashlib.sha256(self.raw).hexdigest())
        self.sibling.blob_id = "f" * 40
        plan = assets.source_plan(self.info, self.asset, "train.jsonl")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lfs"
            path.write_bytes(self.raw)
            self.assertEqual(assets.verify_source(path, plan)["digest_algorithm"], "sha256")

    def test_mismatched_revision_or_missing_metadata_fails(self):
        variants = [
            SimpleNamespace(sha="b" * 40, siblings=[self.sibling]),
            SimpleNamespace(sha=self.info.sha, siblings=[]),
            SimpleNamespace(sha=self.info.sha, siblings=[self.sibling, self.sibling]),
        ]
        for info in variants:
            with self.subTest(info=info), self.assertRaises(ValueError):
                assets.source_plan(info, self.asset, "train.jsonl")
        for changed in ({"size": None}, {"blob_id": None},
                        {"lfs": SimpleNamespace(size=1, sha256="a" * 64)}):
            sibling = SimpleNamespace(**{**vars(self.sibling), **changed})
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                assets.source_plan(SimpleNamespace(sha=self.info.sha, siblings=[sibling]),
                                   self.asset, "train.jsonl")

    def test_download_uses_fixed_revision_and_no_token(self):
        plan = assets.source_plan(self.info, self.asset, "train.jsonl")
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source"
            path.write_bytes(self.raw)
            def downloader(**kwargs):
                calls.append(kwargs)
                return path
            paths, proofs = assets.download_sources([plan], downloader)
            self.assertEqual(paths["development"], path)
        self.assertEqual(calls[0]["revision"], "a" * 40)
        self.assertIs(calls[0]["token"], False)
        self.assertEqual(calls[0]["endpoint"], "https://huggingface.co")
        self.assertEqual(len(proofs), 1)

    def test_size_limit_fails_before_download(self):
        api = SimpleNamespace(repo_info=lambda **kwargs: self.info)
        policy = {"source_paths": {"development": "train.jsonl"}, "max_dataset_download_bytes": 1}
        with self.assertRaises(ValueError):
            assets.plan_sources(api, {"assets": [self.asset]}, policy, ["development"])

    def test_no_http_during_default_inspection(self):
        with patch.object(prepare, "prepare_data", side_effect=AssertionError), \
             patch.object(prepare, "fetch_upstream", side_effect=AssertionError), \
             patch.object(sys, "argv", ["prepare"]), patch("sys.stdout", new_callable=io.StringIO) as stream:
            self.assertEqual(prepare.main(), 0)
        result = json.loads(stream.getvalue())
        self.assertFalse(result["network_used"])
        self.assertFalse(result["model_weights_loaded"])

    def test_explicit_online_required(self):
        with patch.object(prepare, "fetch_upstream") as fetch, \
             patch.object(sys, "argv", ["prepare", "--mode", "upstream", "--output-dir", "unused"]), \
             patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as caught:
            prepare.main()
        self.assertEqual(caught.exception.code, 2)
        fetch.assert_not_called()

    def test_upstream_download_and_subsequent_tamper(self):
        data = b"def synthetic(): pass\n"
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        policy = {"upstream": {"repository": "test/repo", "revision": "a" * 40,
                              "files": [{"path": "methods/awq/module.py", "git_blob_sha1": digest}]}}
        calls = []
        def opener(url, timeout):
            calls.append((url, timeout))
            return io.BytesIO(data)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "up"
            self.assertEqual(len(assets.fetch_upstream(folder, policy, opener=opener)), 1)
            assets.fetch_upstream(folder, policy, opener=lambda *a, **kw: self.fail("network repeated"))
            (folder / "methods/awq/module.py").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                assets.verify_upstream(folder, policy)
        self.assertIn("/" + "a" * 40 + "/", calls[0][0])

    def test_upstream_wrong_download_is_not_saved(self):
        policy = {"upstream": {"repository": "test/repo", "revision": "a" * 40,
                              "files": [{"path": "module.py", "git_blob_sha1": "b" * 40}]}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "up"
            with self.assertRaises(ValueError):
                assets.fetch_upstream(folder, policy, opener=lambda *a, **k: io.BytesIO(b"wrong"))
            self.assertFalse((folder / "module.py").exists())

    def test_upstream_partial_download_resumes_without_overwriting_verified_file(self):
        data = {"a.py": b"a=1\n", "b.py": b"b=2\n"}
        entries = [{"path": name, "git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()}
            for name, value in data.items()]
        policy = {"upstream": {"repository": "test/repo", "revision": "a" * 40, "files": entries}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.py").write_bytes(data["a.py"])
            calls = []
            def opener(url, timeout):
                calls.append(url)
                return io.BytesIO(data[url.rsplit("/", 1)[-1]])
            assets.fetch_upstream(folder, policy, opener=opener)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].endswith("/b.py"))
            self.assertEqual((folder / "a.py").read_bytes(), data["a.py"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.candidate, _, self.policy = assets.load_contracts()
        self.candidate = copy.deepcopy(self.candidate)
        self.policy = copy.deepcopy(self.policy)
        self.candidate["r1"]["n_problems"] = 2
        self.policy.update(expected_development_rows=3, expected_confirmation_rows=1)
        self.dev = [{"question": f"dev {i}", "answer": f"work #### {i}"} for i in range(3)]
        self.confirm = [{"problem": "held"}]
        self.policy["calibration"].update(blocks=2, block_size=3, max_document_tokens=4,
                                           pool_documents=3, max_text_characters=30)
        self.encode = lambda text: [ord(c) for c in text]

    def test_development_outputs_pair_ids_and_keep_gold_local(self):
        manifest, inputs, gold = assets.development_bundle(self.dev, self.confirm, self.candidate, self.policy)
        self.assertEqual([x["id"] for x in inputs], [x["id"] for x in gold])
        self.assertEqual(len(inputs), 2)
        self.assertNotIn("answer", json.dumps(manifest))
        self.assertTrue(all("answer" not in row for row in inputs))
        self.assertEqual(manifest["manifest_canonical_sha256"], assets.canonical_hash(self.candidate))

    def test_partial_source_or_conflicting_duplicate_gold_rejected(self):
        with self.assertRaises(ValueError):
            assets.development_bundle(self.dev[:2], self.confirm, self.candidate, self.policy)
        wrong = [self.dev[0], {**self.dev[0], "answer": "#### 99"}, self.dev[1]]
        with self.assertRaises(ValueError):
            assets.development_bundle(wrong, self.confirm, self.candidate, self.policy)

    def test_development_pipeline_verifies_bytes_and_can_retry_missing_dependency(self):
        _, refs, _ = assets.load_contracts()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            dev_path, confirm_path = folder / "dev", folder / "confirm"
            dev_path.write_text(json.dumps(self.dev))
            confirm_path.write_text(json.dumps(self.confirm[0]) + "\n")
            by_role = {"development": dev_path, "confirmation": confirm_path}
            by_repo = {x["repo_id"]: x for x in refs["assets"] if x["role"] in by_role}
            def info(repo_id, **kwargs):
                asset = by_repo[repo_id]
                path = by_role[asset["role"]]
                sibling = SimpleNamespace(rfilename=self.policy["source_paths"][asset["role"]],
                    size=path.stat().st_size, lfs=None, blob_id=assets.file_digests(path)["git_blob_sha1"])
                return SimpleNamespace(sha=asset["revision"], siblings=[sibling])
            def download(repo_id, **kwargs):
                return by_role[by_repo[repo_id]["role"]]
            hub = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(repo_info=info),
                                  hf_hub_download=download)
            output = folder / "prepared"
            with patch.object(prepare, "package_versions", return_value={"huggingface-hub": None}):
                with self.assertRaises(ValueError):
                    prepare.prepare_data(output, self.candidate, refs, self.policy, include_calibration=False)
            self.assertFalse(output.exists())
            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 patch.object(prepare, "package_versions", return_value={"huggingface-hub": "0.36.2", "pyarrow": "20.0.0"}), \
                 patch.object(prepare, "read_development", side_effect=lambda p: json.loads(p.read_text())), \
                 patch("sys.stdout", new_callable=io.StringIO):
                result = prepare.prepare_data(output, self.candidate, refs, self.policy, include_calibration=False)
            self.assertEqual(len(result["source_files"]), 2)
            self.assertFalse(result["generation_run"])
            manifest = json.loads((output / "development_manifest.json").read_text())
            self.assertEqual(manifest["status"], "SOURCE_FILES_VERIFIED_CANDIDATE_NOT_ACCEPTED")
            self.assertEqual(manifest["counts"]["selected"], 2)
            for name, recorded in result["artifacts"].items():
                self.assertEqual(assets.file_digests(output / name), recorded)

    def test_exact_calibration_budget_and_recorded_partial_document(self):
        rows = [(0, {"text": " abc "}), (1, {"text": "defg"}), (2, {"text": "hi"})]
        blocks, report = assets.select_calibration(rows, self.encode, set(), self.policy)
        self.assertEqual([len(row) for row in blocks], [3, 3])
        self.assertEqual(sum(row["used_tokens"] for row in report["documents"]), 6)
        self.assertEqual(report["token_blocks_sha256"], assets.canonical_hash(blocks))
        self.assertTrue(any(row["used_tokens"] < row["original_tokens"] for row in report["documents"]))

    def test_calibration_ranking_is_independent_of_iteration_order(self):
        rows = [(i, {"text": text}) for i, text in enumerate(["abc", "def", "ghi", "jkl", "mno"])]
        a = assets.select_calibration(rows, self.encode, set(), self.policy)
        b = assets.select_calibration(list(reversed(rows)), self.encode, set(), self.policy)
        self.assertEqual(a, b)

    def test_calibration_excludes_questions_duplicates_and_long_documents(self):
        rows = list(enumerate([{"text": "held"}, {"text": " abc "}, {"text": "abc"},
                               {"text": "def"}, {"text": "a" * 50}]))
        blocks, report = assets.select_calibration(rows, self.encode, {assets.question_hash("held")}, self.policy)
        self.assertEqual(report["counts"]["excluded_exact_question_overlap"], 1)
        self.assertEqual(report["counts"]["duplicate_documents"], 1)
        self.assertEqual(report["counts"]["eligible_unique_documents"], 2)
        self.assertNotIn(0, [x["source_row"] for x in report["documents"]])

    def test_insufficient_calibration_not_resampled(self):
        with self.assertRaises(ValueError):
            assets.select_calibration([(0, {"text": "abc"})], self.encode, set(), self.policy)

    def test_invalid_text_and_token_ids_rejected(self):
        with self.assertRaises(ValueError):
            assets.select_calibration([(0, {"missing": "value"})], self.encode, set(), self.policy)
        with self.assertRaises(ValueError):
            assets.select_calibration([(0, {"text": "abc"})], lambda _: [True], set(), self.policy)

    def test_jsonl_blank_invalid_and_row_limit_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input"
            path.write_text('{"text":"abc"}\n{"text":"def"}\n')
            self.assertEqual([i for i, _ in assets.iter_jsonl(path)], [0, 1])
            with self.assertRaises(ValueError):
                list(assets.iter_jsonl(path, max_rows=1))
            for text in ('{"text":"abc"}\n\n', '[]\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    list(assets.iter_jsonl(path))

    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "zstandard 없음")
    def test_zstd_multiple_frames_are_all_read(self):
        import zstandard
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl.zst"
            compressor = zstandard.ZstdCompressor()
            path.write_bytes(compressor.compress(b'{"text":"abc"}\n')
                             + compressor.compress(b'{"text":"def"}\n'))
            self.assertEqual(len(list(assets.iter_jsonl(path, compressed=True))), 2)


class RuntimeTests(unittest.TestCase):
    def test_unapproved_run_stops_before_environment_or_model(self):
        # 실제 저장소의 승인 상태가 바뀌어도 미승인 실행 차단을 검사합니다.
        status = {**contracts.REQUIRED_GATES, "research_approval": "NOT_RECORDED"}
        with patch.object(run_r0, "repository_state") as repo, \
             patch.object(run_r0, "load_upstream") as upstream, self.assertRaises(ValueError):
            run_r0.run(None, None, None, "cuda:0", {}, {}, {}, status)
        repo.assert_not_called()
        upstream.assert_not_called()

    def test_each_gate_required(self):
        good = dict(contracts.REQUIRED_GATES)
        contracts.require_gates(good)
        for key in good:
            bad = dict(good)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.require_gates(bad)

    def test_r0_inspect_does_not_execute(self):
        with patch.object(run_r0, "run", side_effect=AssertionError), \
             patch.object(sys, "argv", ["r0"]), patch("sys.stdout", new_callable=io.StringIO) as stream:
            self.assertEqual(run_r0.main(), 0)
        result = json.loads(stream.getvalue())
        status = json.loads((ROOT / "docs/research_status.json").read_text())
        self.assertEqual(result["unmet_gates"], contracts.unmet_gates(status))
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["network_used"])

    def test_calibration_shape_and_id_validation(self):
        adapter.validate_shape([[0, 2], [1, 0]], expected_shape=(2, 2), vocab_size=3)
        for blocks in ([[0]], [[0, 3], [1, 0]], [[0, True], [1, 0]], [[0, -1], [1, 0]]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                adapter.validate_shape(blocks, expected_shape=(2, 2), vocab_size=3)

    def test_prompt_has_one_bos_and_prefilled_think(self):
        seen = []
        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                seen.append((messages, kwargs))
                return [151646, 151644, 42, 151645, 151648, 198]
        self.assertEqual(contracts.prompt_ids(Tok(), "synthetic")[-2:], [151648, 198])
        self.assertTrue(seen[0][1]["tokenize"])
        self.assertTrue(seen[0][1]["add_generation_prompt"])

    def test_bos_duplication_and_eos_in_prompt_fail(self):
        for ids in ([151646, 151646, 151648, 198], [151646, 151643, 151648, 198], [1, 151648, 198]):
            tok = SimpleNamespace(apply_chat_template=lambda *a, **k: ids)
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                contracts.prompt_ids(tok, "synthetic")

    def test_termination_preserves_think_and_counts_eos(self):
        seen = []
        def decode(ids, **kwargs):
            seen.append((ids, kwargs))
            return "reason</think>42"
        tok = SimpleNamespace(decode=decode)
        result = contracts.generated_record([9, 151649, 42, 151643], tok, 4)
        self.assertEqual(result["finish_reason"], "eos")
        self.assertEqual(result["generated_token_count_including_eos"], 4)
        self.assertFalse(result["right_censored"])
        self.assertEqual(seen[0][0], [9, 151649, 42])
        self.assertFalse(seen[0][1]["skip_special_tokens"])

    def test_after_eos_padding_not_silently_trimmed(self):
        tok = SimpleNamespace(decode=lambda *a, **k: "unused")
        with self.assertRaises(ValueError):
            contracts.generated_record([9, 151643, 151643], tok, 4)

    def test_prepared_tokens_modified_after_manifest_rejected(self):
        candidate, refs, policy = assets.load_contracts()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            artifacts = {
                "r1_inputs.jsonl": "synthetic\n", "r1_gold.jsonl": "synthetic\n",
                "development_manifest.json": "{}",
                "calibration_manifest.json": json.dumps({"token_blocks_sha256": assets.canonical_hash([[1]]),
                                                          "candidate_sha256": assets.canonical_hash(candidate)}),
                "calibration_tokens.json": json.dumps({"blocks": [[1]]}),
            }
            for name, content in artifacts.items():
                (folder / name).write_text(content)
            report = {"status": "PREPARED_CANDIDATE_NOT_RUN_APPROVAL", "include_calibration": True,
                      "candidate_sha256": assets.canonical_hash(candidate), "policy_sha256": assets.canonical_hash(policy),
                      "inspection_refs_sha256": assets.canonical_hash(refs),
                      "artifacts": {name: assets.file_digests(folder / name) for name in artifacts}}
            assets.write_new(folder / "preparation_report.json", report)
            self.assertEqual(contracts.load_prepared_calibration(folder, candidate, refs, policy)[0], [[1]])
            (folder / "calibration_tokens.json").write_text('{"blocks":[[2]]}')
            with self.assertRaises(ValueError):
                contracts.load_prepared_calibration(folder, candidate, refs, policy)


HAS_TORCH_STACK = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


@unittest.skipUnless(HAS_TORCH_STACK, "PyTorch·Transformers가 없어 실제 Qwen2 CPU 연결 검사는 미실행")
class Qwen2CpuConnectionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        self.torch = torch
        config = Qwen2Config(vocab_size=32, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            max_position_embeddings=32, use_cache=False)
        config._attn_implementation = "sdpa"
        self.model = Qwen2ForCausalLM(config).eval()

    def test_real_first_layer_kwargs_preserve_rope_and_hook_removed(self):
        ids = self.torch.tensor([[1, 2, 3, 4]])
        hidden, kwargs = adapter.capture_first_input(self.model, ids, "cpu")
        self.assertEqual(tuple(hidden.shape), (1, 4, 128))
        self.assertEqual(len(kwargs["position_embeddings"]), 2)
        with self.torch.no_grad():
            output = self.model.model.layers[0](hidden, **kwargs)[0]
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(len(self.model.model.layers[0]._forward_pre_hooks), 0)

    def test_real_error_not_swallowed_and_hook_removed(self):
        with patch.object(self.model, "forward", side_effect=ValueError("real failure")):
            with self.assertRaisesRegex(ValueError, "real failure"):
                adapter.capture_first_input(self.model, self.torch.tensor([[1]]), "cpu")
        self.assertEqual(len(self.model.model.layers[0]._forward_pre_hooks), 0)

    def test_qwen_scale_paths_include_gqa_skip(self):
        t = self.torch
        layer = self.model.model.layers[0]
        values = [
            ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), t.ones(128)),
            ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), t.ones(128)),
            ("mlp.up_proj", ("mlp.down_proj",), t.ones(256)),
        ]
        ledger = adapter.scale_ledger(layer, values)
        self.assertEqual(len(ledger), 3)
        self.assertNotIn("self_attn.v_proj", [r["previous"] for r in ledger])
        with self.assertRaises(ValueError):
            adapter.scale_ledger(layer, values[::-1])


if __name__ == "__main__":
    unittest.main()
