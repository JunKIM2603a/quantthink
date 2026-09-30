# Asset/configuration review v0.1 — 2026-09-29

**METADATA INTERNALLY VERIFIED; TOKENIZER PASS USER-REPORTED; SESSION 01 OPEN.** This review does not accept the candidate scientific protocol, freeze H1/H2, or authorize model runs. Read `research_status.json` and `REPRODUCTION_PLAN_V01.md` with this document.

The user supplied a successful online tokenizer report at `2026-09-29T00:13:10.027340+00:00`. Repository hashes and submitted fields were checked in [the tokenizer follow-up](TOKENIZER_CONTRACT_REVIEW_20260929.md). The original preparation details below are retained as history; the local tokenizer task is complete.

## Evidence and reference scope

The user pasted a successful `METADATA_RESOLVED_NOT_RUN_READY` report recorded at `2026-09-28T23:25:35.768653+00:00`. The report is execution evidence supplied by the user, not assistant access to local saved bytes. All four asset entries have immutable revision SHAs and no reported errors. The submitted manifest hash matches the current `reproduction_candidate.json`: `98ff733b0d8391c3a33aa2f561e2d389bd6e482ee53b1b680c1bb6e7e6b6201f`.

The assistant recomputed the canonical hashes of the three supplied configuration objects using a local transcription of the pasted fields; all three match the submitted hashes. This establishes internal content/hash consistency, not independent authentication or a live download. Raw reports, full tokenizer assets and that local transcription are not committed.

`configs/asset_inspection_refs.json` preserves these model/dataset revision references for inspection. It is **not** a dependency lock, a verified model-weight snapshot, a dataset-example manifest, or final acceptance of the run configuration. Do not resolve a fresh `main` to replace these references without an explicit review.

## Findings from supplied configuration content

| Item | Observed source values | Handling in the preparation/probe |
|---|---|---|
| BOS | Model config: 151643; generation config: 151646; tokenizer config supplies the named begin-of-sentence token | Preserve both source values. Test actual tokenizer encoding against the generation BOS, and build input IDs through the pinned template. Do not edit the upstream config to conceal the discrepancy. |
| EOS/PAD | EOS is 151643 in both numeric configs; tokenizer names the same string for EOS and PAD | Verify actual token IDs; use tokenizer-produced attention masks, not an assumption that every EOS-valued position is padding. |
| Reasoning prefix | Single-user template ends with `<｜Assistant｜><think>\n` | Do not append a second opening marker. The prefilled opening marker is input, not generated reasoning tokens. Treat closing `</think>` as a reasoning boundary, not whole-response EOS. |
| Context metadata | Tokenizer limit 16384; model position capacity 131072; candidate total budget 40960 | These are separate declarations. Do not silently truncate, raise the tokenizer limit, or certify long-context runtime on this basis. No long-context run occurs in this probe. |
| Model class | Model architecture is Qwen2ForCausalLM; tokenizer class is LlamaTokenizerFast | Inspect the actual tokenizer as declared. Do not substitute a Qwen tokenizer or resize model embeddings based only on class names. |
| Sampling | Generation file states T=0.6/top-p=0.95 but omits top-k; existing candidate explicitly sets top-k=0 | Verify inheritance and explicit overrides. This is prevention of a possible implementation error, not an observed model-run failure. |

The 4.51.3 Transformers source initializes `GenerationConfig.top_k` to 50 when absent: [official source](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/generation/configuration_utils.py). The [official chat-template guide](https://huggingface.co/docs/transformers/v4.51.3/en/chat_templating) explains why two-step template rendering/tokenization must avoid adding special tokens twice. These are external source checks; the model-specific values above come from the user's report.

## Prepared CPU tokenizer probe

`scripts/audit_tokenizer_contract.py` first validates the existing report against the unchanged candidate hash, recorded revisions and configuration hashes. By default it selects the unique local report with the recorded timestamp/hash; `--metadata` supports an explicit existing path. It never silently chooses the newest file or overwrites an output.

Without `--online`, the script performs only local JSON review using the standard library. With `--online`, it uses the declared Transformers 4.51.3/Jinja2 3.1.6 inspection profile. It downloads only pinned configuration/tokenizer assets with `token=False` and `trust_remote_code=False`. It disables framework imports in its own process; it does not load weights, read dataset rows, generate answers, call a GPU or alter the user's CUDA installation.

The online probe verifies the three configuration hashes against downloaded content, actual BOS/EOS/PAD IDs, a single BOS, exact user/assistant/think formatting for two synthetic fixtures, agreement between direct and two-step tokenization, left-padded attention masks, and explicit top-k=0. It records the actual tokenizer/dependency versions and synthetic prompt IDs. A library/configuration mismatch is a failed probe, not a reason to silently change the scientific setup.

## Earlier preparation validation

- 25 new CPU unit tests passed using test doubles, not the real pretrained tokenizer or live HTTP.
- Syntax checks passed for the new script and tests.
- The offline CLI passed on a local transcription of the user's supplied configuration fields, with `METADATA_REVIEW_PASS_TOKENIZER_NOT_RUN`.
- The actual pinned-tokenizer online probe was **not run** in the assistant environment. Container network retrieval was unavailable. No successful package installation, model or GPU run is claimed.
- The earlier test suites were not rerun in this update; their historical results remain separate.

## Completed local action (retained for reproducibility)

The following was the requested tokenizer-only action from the existing `quantthink` environment and working branch. No rerun is requested after the reported pass:

```bash
git pull --ff-only &&
python -m unittest discover -s tests -p 'test_tokenizer_contract.py' -v &&
python -m pip install -r requirements/tokenizer-audit.txt &&
python -m pip check &&
python scripts/audit_tokenizer_contract.py --online
```

The new report is written under `results/local/tokenizer_contract_<timestamp>.json`. Success is `TOKENIZER_CONTRACT_PASS_NOT_MODEL_READY`; failure is `TOKENIZER_AUDIT_INCOMPLETE`, with no fallback to another revision. A missing/ambiguous local report requires `--metadata` pointing to the existing report, not another revision-resolution run.

Session 01 remains open. The user-reported tokenizer pass closes the concrete input-format subtask. The two primary-text access blockers were resolved in the [novelty follow-up](NOVELTY_FOLLOWUP_20260929.md); final contribution/novelty judgment, scientific protocol acceptance, research approval, AWQ adapter validation and eventual model reproduction remain open. The next model step remains R0 only after the required gates are satisfied; R1 and H1 are not started by this command.
