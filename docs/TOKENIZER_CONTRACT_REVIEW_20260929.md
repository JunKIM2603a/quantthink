# Tokenizer contract review — 2026-09-29

**USER-REPORTED PASS; REPOSITORY HASHES AND SUBMITTED FIELDS CONSISTENT.**

This records the report supplied by the user with timestamp `2026-09-29T00:13:10.027340+00:00`. Its reported status is `TOKENIZER_CONTRACT_PASS_NOT_MODEL_READY`. The reviewed repository head is `8335ef867b7a48132e32badd3f60308e8fa15c4f`; the report does not contain its generating Git commit, so that provenance is not inferred.

## Verification actually performed

The assistant recomputed canonical JSON hashes from the repository files and compared them with the submitted fields:

| Reference | Recomputed hash / result |
|---|---|
| `configs/reproduction_candidate.json` | `98ff733b0d8391c3a33aa2f561e2d389bd6e482ee53b1b680c1bb6e7e6b6201f` — matches |
| `configs/asset_inspection_refs.json` | `97f5b7bf6cb6888c716a3838f21c224e7920b1df05c5996efe1321d39cd15276` — matches |
| Reference model revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` — matches |
| Declared inspection package versions | Transformers 4.51.3 and Jinja2 3.1.6 — match |
| Submitted input arrays | Counts 22 and 33; one BOS each; neither contains EOS |
| Candidate generation fields | R0 cap 128; T=0.6, top-p=0.95, top-k=0, repetition penalty 1.0 — match |

The two repository configuration files remain unchanged so the report's references stay valid. Their historical pending-check labels describe the inspection snapshot, not the latest milestone; current evidence lives in `docs/research_status.json`.

Verification used a local transcription of selected submitted fields. It is not an independent rerun, inspection of the user's original file bytes, a tokenizer-vocabulary hash audit, or evidence that the complete preceding shell command chain passed. No raw diagnostic report is committed.

## Rules for the future generation implementation

| Observed report evidence | Implementation consequence |
|---|---|
| Tokenizer BOS=151646; model config BOS=151643 | Use the pinned tokenizer's actual input IDs. Preserve the source discrepancy in provenance. |
| Actual BOS count=1; adding special tokens again gives 2 | Use direct chat-template tokenization, or tokenize rendered text with `add_special_tokens=False`. |
| EOS=PAD=151643; left-padding check passes | Preserve tokenizer-produced attention masks. Token identity alone cannot distinguish active content from padding. |
| `</think>` maps to 151649 | Use it as the reasoning boundary. Whole-response EOS remains 151643; retain the final-answer region. |
| Opening `<think>` is already in the input prefix | Do not append another opening marker or count the prefilled marker as generated reasoning. |
| Inherited top-k=50; explicit candidate top-k=0 | Pass the candidate overrides explicitly in future generation; do not rely on the model generation file alone. |
| Tokenizer metadata=16384; model capacity=131072; candidate budget=40960 | These metadata do not certify a 40960-token runtime. Long-context feasibility is still untested. |

The report records `LlamaTokenizerFast`, tokenizers 0.21.4, huggingface-hub 0.36.2 and NumPy 1.26.4. These are observed tokenizer-inspection versions, not an accepted AWQ/inference environment lock.

## Readiness after this milestone

| Preparation item | Current evidence / remaining action |
|---|---|
| Public asset references | Reviewed metadata and pinned inspection references |
| Input format and CPU tokenizer | User-reported pass, checked against the repository |
| Scientific contribution scope | Final integration/decision remains open |
| BF16/AWQ runtime and transformation | Select an executable adapter/version plan, then validate scale folding and canonical weight-error space under the approved run plan |
| Calibration and data separation | Record exact token blocks and disjoint example manifests |
| Evaluation | Complete answer-parser fixtures and blinded overthinking rubric |
| R0/R1 plan | Candidate exists; acceptance and required research-run approval are not recorded |
| Long context, model generation, H1/H2 | No new execution or hypothesis evidence |

The immediate local tokenizer task is complete; no rerun or fresh metadata resolution is requested. The next preparation deliverable is the executable reproduction specification covering the remaining runtime, AWQ, data and evaluator items. Existing `REPRODUCTION_PLAN_V01.md` remains a candidate.

Session 01 remains open under `AGENTS.md` and `SESSION_PLAN.md`. This report does not authorize a model experiment, freeze the candidate, or change the original hypotheses. R0 is the next model-level engineering check only after the relevant gates are satisfied.
