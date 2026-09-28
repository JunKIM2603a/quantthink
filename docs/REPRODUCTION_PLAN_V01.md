# QuantThink R0/R1 candidate v0.1 — 2026-09-29

Status note: this candidate and its preparation history are retained. Later asset/tokenizer status is in `research_status.json`; the two primary-text access limitations in the source audit below are superseded by [the Session 01 continuation](NOVELTY_FOLLOWUP_20260929.md). The candidate remains unaccepted and no model run is recorded.

**PROPOSED, NOT ACCEPTED, NOT RUN READY.** This advances Session 01; it neither changes its exit criteria nor freezes H1/H2. The owner's instruction to continue supports preparation. Required academic/research-run approval is still not recorded. All settings introduced here are design proposals, not recovered settings from Lotfi's exact runtime.

## 1. Scope decision

Keep the scientific question: can reference-prefix-KL-matched Gaussian **weight** perturbations reproduce AWQ-induced length inflation and verified correct-answer abandonment? Do not claim that matched-noise controls, generic compression-induced verbosity, or non-termination analysis are new. A scalar KL match is an operational comparison, not sufficient identification of a mechanism.

R0 will eventually check loaders, prompts and termination logging using two synthetic non-benchmark prompts with 128 output tokens. R1 is a development premise check using BF16 and AWQ-W3 only. Neither tests H1. Main H1 margins, sample-size/power decisions and directional H2 remain unfrozen in `PROTOCOL_DRAFT.md`.

## 2. Expanded source audit

These are primary-source inspections, not replications or a complete systematic review. Previously inspected facts not revisited are not silently re-certified.

| Source, version and inspected scope | Supported statement | Consequence |
|---|---|---|
| [Lotfi 2606.00206v1](https://arxiv.org/html/2606.00206v1), sections 3, 4.1–4.3, 7, 8.1–8.2 | Reports longer reasoning and correct-answer non-commitment; calibration description uses 128 sequences of 512 Pile-uncopyrighted tokens; discusses generic perturbations as future work. | Premise, not new contribution. Its exact runtime/output cap is still unverified. |
| [Sathyanarayanan 2609.23125](https://arxiv.org/abs/2609.23125), indexed primary abstract only | Activation-error Gaussian and sign-randomization controls already exist for induction/retrieval damage. | Broad control-design novelty overlaps. Full text still unavailable; no claim about all experiments absent from the manuscript. |
| [Atkinson 2602.17691v1](https://arxiv.org/html/2602.17691v1), section 3.2, quantization-penalty passage | Describes temperature-equivalent quantization noise as a heuristic. | Not a theorem that Gaussian weight noise or quantization always raises entropy. |
| [2606.02011v1](https://arxiv.org/html/2606.02011v1), appendix D and F | Temperature can change looping, budget hits and length; severe collapse can persist. | Increasing temperature is not automatically a lengthening intervention. |
| [2509.12464v1](https://arxiv.org/html/2509.12464v1), introduction and sections 5.2, 6 | Pruning can also lengthen reasoning; effects depend on budgets and calibration. | Non-quantization verbosity is already known. Do not sell H1 as the first example of that fact. |
| [Lian 2606.25519v2](https://arxiv.org/html/2606.25519v2), section 7.2 | Calibration can jointly change accuracy and token inflation. | Calibration must be documented; not proof of equal-error directional causation. |
| [e-CUSUM 2607.11317v1](https://arxiv.org/html/2607.11317v1), abstract and sections 1, 2, 7–8 | In its small GSM8K FP16/INT4 study, non-termination differs from verbatim looping; accuracy improvement remains inconclusive. | Explicit finish reasons, label validity and budget accounting are essential. Merely measuring non-termination is not new. |

A further lead, `2609.06473` (Steering Under Compression), surfaced through secondary discovery; direct primary retrieval failed. It is **UNVERIFIED_LEAD**, not an audited result or a resolved collision. Its matched-effect/length-steering scope needs primary verification before claims overlap is judged.

Queries included exact paper IDs/titles; `reasoning KL-matched`; `quantization overthinking Gaussian`; `quantization termination reasoning`; and `quantization Gaussian noise chain-of-thought`. Many results were irrelevant. The searched/read sources did not establish a direct completed study of the complete proposed comparison, but this is **not** evidence of global absence. Primary arXiv HTML/PDF and author-site attempts did not provide 2609.23125 full text. Container network retrieval also failed. No proxy summary was treated as the missing manuscript.

**Novelty gate stays OPEN_WITH_ADJACENT_CONTROL_OVERLAP.** The next concrete source artifact needed is the primary PDF for 2609.23125, not another unbounded repetition of the same queries. We can prepare metadata and measurement contracts without claiming this gate passed.

## 3. New upstream calibration finding

At [Liu upstream commit bf947e29f52e3f666e3263efac149dae0ac18d00](https://github.com/ruikangliu/Quantized-Reasoning-Models/tree/bf947e29f52e3f666e3263efac149dae0ac18d00), inspected:

- `methods/awq/run_awq.py`, blob `3587aabc90426711e691a1629a026a4539e13c91`: calls AWQ search, reloads the original model, then applies scales/clips and pseudo-quantization. The `--w_asym` help wording is misleading; the actual code sets `zero_point=args.w_asym`.
- `methods/awq/calib_data.py`, blob `a17467d065474fc212343730f6c910a43f433813`: shuffles pile-val-backup with seed 42, collects `n_samples` nonempty documents of at most 512 encoded tokens, concatenates, then returns `floor(total_tokens/block_size)` blocks. **128 documents are not necessarily 128 blocks.** For the constructed 400-token-per-document example, 128 documents produce only 100 blocks; this is arithmetic, not a measured source-dataset count.
- `methods/awq/pre_quant.py`, blob `58a3bd5eab434641d8c31ca3c29b0990bdd29cf3`: its pileval path consumes those returned blocks, then searches/applies scaling and clipping. No runtime compatibility or canonical-weight conversion was tested here.

Propose explicit 128 x 512 = 65,536 calibration tokens from a pinned Pile-uncopyrighted snapshot, with ordered document IDs, token hashes, tokenizer revision and preprocessing. Do not silently replace that corpus by pile-val-backup. Record the difference from upstream; call this an **adapted controlled replication**, not exact source reproduction. `exact_token_blocks()` implements only the final packing contract; it is not an AWQ quantizer or a complete calibration loader.

## 4. Candidate minimal design

The machine-readable choices are in `configs/reproduction_candidate.json`.

| Item | Candidate choice / limitation |
|---|---|
| Model | DeepSeek-R1-Distill-Qwen-1.5B, one canonical BF16 checkpoint/tokenizer revision. Immutable revisions await metadata collection and review. |
| Arms | BF16; AWQ-W3 g128 asymmetric, BF16 activations/KV. No noise/temperature arms in R1. |
| Runtime | Prefer the same Transformers/SDPA stack for both arms, one GPU per worker, TP=1. This is a candidate, not a verified compatible alternative to upstream custom inference. Engine/package pins and AWQ canonicalization are unresolved. |
| R0 | Two synthetic prompts, 128 output tokens, loader/format check only. Fixtures to be committed before execution. |
| R1 development data | Propose 100 GSM8K `main/train` questions, two sampling seeds (42,43), after normalized-question deduplication and exact-text exclusion against MATH-500. This is deliberately **not** MATH-500 reproduction. Selection hash algorithm/seed is in the config; actual IDs still need committing before a run. |
| Confirmation | Keep MATH-500 untouched for the original primary comparison. Do not tune/generate on it in R1. Hash-based exclusion does not prove absence of semantic duplicates or pretraining contamination. |
| Sampling | T=0.6, top-p=0.95, top-k disabled, no penalties; same chat template/user-only math instruction in both arms. Exact rendered-template hash and EOS IDs require review of pinned configs, not guesses. |
| Budget | R1: 32,768 generated tokens, input at most 8,192, total at most 40,960. Reject oversized inputs; no silent truncation. These are our proposed caps, not Lotfi's recovered caps. Configuration capacity does not establish available GPU memory. |
| Logging | Raw token IDs, counts including EOS, rendered prompt hash, EOS/budget/unexpected stop, final-answer region, evaluator version, labels, seeds, elapsed time and peak memory. `</think>` is a reasoning boundary, not automatically whole-response EOS. |
| Token upper bound | 100 questions x 2 arms x 2 seeds x 32,768 = 13,107,200 generated tokens, excluding quantization/calibration/R0 and any future rerun. This is a ceiling, not predicted usage or measured GPU-hours. |

Why GSM8K development: leave all 500 primary questions available for independent confirmation. Domain transfer is a limitation. A weak GSM8K pilot cannot disprove MATH-500 effects. Changing the development dataset later requires a recorded amendment, not selective substitution after seeing outcomes.

## 5. R1 outcomes and acceptance proposal

Engineering validity first: every scheduled attempt gets a record; OOM/exception/unknown stops are explicit and make the affected pair incomplete, not silently dropped examples. Configuration-only capacity is not long-context validation. Both arms use the same inputs and budgets. Validate the AWQ reparameterization without quantization before computing canonical weight-error norms.

For complete R1 data, report capped mean length over **all** generations, arithmetic arm means and their ratio, accuracy difference, EOS and budget-limit rates. Aggregate sampling replicates within each question, then use paired problem-level bootstrap (10,000 resamples) for development confidence intervals; show each seed separately. Two fixed sampling seeds do not support claims over all random seeds. Do not calculate primary length only on correct or terminated responses.

Proposed development outcomes: `PREMISE_SIGNAL` requires a positive lower 95% CI for log length ratio and a negative upper 95% CI for accuracy difference (AWQ minus BF16). Otherwise report the observed pattern or `INCONCLUSIVE`; this is not H1 rejection. A large cap-hit rate limits the claim to **budgeted cost/non-termination**, not unlimited expected reasoning length. No automatic budget extension or change of quantizer.

For overthinking, separate `VALID_SOLUTION_THEN_WRONG_FINAL`, `VALID_SOLUTION_THEN_NO_FINAL`, `INCIDENTAL_ANSWER_STRING`, `OTHER_ERROR`, `UNCERTAIN`. A regex hit is only a candidate. Freeze a blinded rubric, final-answer parser fixtures and review procedure before judging R1; review negatives as well as positives. Denominator is all attempted/completed problems as explicitly reported, separate from share among failures. Annotation uncertainty must remain visible; a length/accuracy signal alone is not validation of the overthinking endpoint. H1 with that endpoint requires annotation validation and matched-control validation later.

No numerical H1 equivalence threshold or sample-size guarantee is accepted here. Original ±20% and ±5% remain unvalidated proposals. H2 is preserved; making it exploratory is still a pending decision.

## 6. A mathematical sanity check, not an LLM result

For a memoryless toy generator with tokens `[EOS,A,B]`, set p=(1/3,1/3,1/3), q_A=(0.05,0.475,0.475), q_B=(0.475,0.05,0.475). At every shared prefix both arms have exactly the same KL(p||q) and entropy (they permute q's coordinates under a uniform p), but expected length including EOS is 1/q(EOS): 20 versus 2.1053.

The CPU script evaluates KL=0.3962587858 nats and entropy=0.8570050649 nats in both arms. This elementary counterexample rules out the universal implication 'equal mean KL/entropy guarantees equal length'. **It does not refute the empirical Gaussian-vs-AWQ H1, identify a real-model mechanism, or establish novelty.** It motivates recording termination probability on common prefixes and avoiding the interpretation that matching itself forces equivalent behavior.

## 7. Actual preparation and next action

Implemented public metadata collector, exact token-block packing and termination contracts, and the toy sanity example. New CPU-only tests pass with mocked HTTP responses; no live collector success, model, AWQ calibration, long-context or GPU experiment is claimed. Existing preflight tests were not rerun in this preparation sandbox.

Run `python scripts/resolve_asset_revisions.py --online --output results/local/asset_revisions_v01.json` once on the user's connected machine. It reads official public metadata for four repositories plus three small model config JSONs at the resolved model SHA. It neither reads dataset rows nor downloads weights/tokenizer vocabulary, installs packages, calls inference or modifies tracked config/status. No auth token is needed or sent. Outputs are candidates for review, not an accepted runtime lock. A missing config/metadata response remains incomplete, with no silent fallback.

Session 01 remains open: primary full-text overlap review, immutable asset/config review, accepted executable calibration/runtime/evaluation manifests, and required research-run approval remain outstanding. No changes to original stage exits; no unapproved scientific run is provided by this PR.
