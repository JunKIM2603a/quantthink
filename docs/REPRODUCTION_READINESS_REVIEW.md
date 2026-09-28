# Reproduction readiness review — 2026-09-29

Status: **DESIGN REVIEW ONLY; NOT AN APPROVED RUN PLAN.** Read together with `PROTOCOL_DRAFT.md`, `EVIDENCE_AUDIT.md`, and `research_status.json`. Original H1/H2 and session exit criteria are unchanged.

## 1. User-reported environment milestone

The user submitted a preflight report at commit `95e7be8b125eb3307b7054a931c7a7033d7c5dfb` with a clean worktree: Python 3.11.16, PyTorch 2.7.1+cu118, NumPy 1.26.4, CUDA available and two RTX 4090 devices with BF16 support. The subsequent supplied output reports `GPU_TENSOR_SMOKE_PASS` for individual 256x256 BF16 all-ones matrix products on both devices.

This is user-submitted execution evidence, not assistant access to the hardware or verification of the saved file bytes. It verifies neither NCCL/peer communication, model loading, attention kernels, long-context throughput, quantization, nor H1/H2. The two devices were tested individually, not in a distributed or simultaneous workload. Raw diagnostics remain local. No further basic CUDA installation or repeat of the same tensor check is needed unless the environment changes.

## 2. Newly inspected upstream implementation

Repository: `ruikangliu/Quantized-Reasoning-Models`, the code for **Liu et al.**, not a verified exact release of Lotfi et al.'s experiments. Main resolved to `bf947e29f52e3f666e3263efac149dae0ac18d00` during this audit. Files were read live; this is a code-inspection reference, not an approved QuantThink runtime pin or a claim that this upstream stack was installed.

| Inspected file | Verified source detail | Consequence / proposed safeguard |
|---|---|---|
| [README.md](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/README.md) | Describes fake AWQ W3/W4, real-quantization scripts/model zoo for INT4, and installations of customized inference dependencies. | Do not assume its released real W4 checkpoint is AWQ-W3, or install its full dependency stack into the working CUDA environment without review. |
| [scripts/quantization/awq.sh](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/scripts/quantization/awq.sh) | Loops over bits 3 and 4; passes `--w_groupsize 128 --w_asym`. | Extract one explicit W3 condition for the initial comparison rather than launch this wrapper unchanged. |
| [scripts/inference/inference.sh](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/scripts/inference/inference.sh) | Loops over six dataset entries; a call accepts one seed, default 42. | Do not interpret one invocation as three-seed evaluation or as a minimal MATH-only run. |
| [inference.py](https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/inference.py) | Defaults both `max_new_tokens` and `max_model_length` to 32768; sets tensor parallel size to the visible CUDA device count; registers custom fake-quantized models in vLLM. | Two visible GPUs would request TP=2 in this script. Independent one-GPU workers require explicit device isolation. Validate custom checkpoint/scaling semantics before loading through a different stack. Input-inclusive context limits and output budgets must be recorded separately. |
| `inference.py`, JSON saving block | Saves full prompt, generated text, gold and metrics; this block does not explicitly save token IDs or finish reasons. | Our runner needs prompt/generated token counts, termination reasons, caps and reproducibility metadata; do not assume all required observables are in the default output. |

Source blob SHAs at inspection: README `6b3bc42ee5ed7376b94b7fed4f0489761da5e6ab`; AWQ wrapper `d23ca31963868b6a404e0d62cc2dfd2e62b7a83c`; inference wrapper `356f09fba1e97522ea47027dfccc83da3e833e12`; inference module `7440d80c54d6e82ee0aebf895551544afb491a60`.

### Do not conflate the upstream default with the source experiment

[Lotfi et al. v1](https://arxiv.org/html/2606.00206v1), sections 3 and 8.1, reports T=0.6/top-p=0.95 and AWQ calibration with 128 Pile-uncopyrighted sequences of length 512. The inspected Liu README points to a different named Pile source. Calibration IDs and preprocessing must therefore be checked, not inferred from the word Pile.

Lotfi section 4.1 reports an average AWQ-W3 LiveCodeBench trace length of 38.8k. That number cannot establish a 32768-token cap for those same traces. The inspected Liu code's defaults are not evidence that Lotfi used the same cap. Source-exact maximum generation length is still unresolved; if we choose our own budget, label the study as an adapted controlled replication and report truncation explicitly.

A previous conversation report used 61.2%/12.9k as the unmodified BF16 MATH baseline. In Lotfi Table 1 those are the AWQ-W3 **+penalty** values; the BF16 base row is 85.6%/5.2k. Preserve that distinction. These are author-reported values, not our target-based model tuning criteria or local observations.

## 3. Bounded novelty follow-up

The arXiv indexed primary abstract for [2609.23125](https://arxiv.org/abs/2609.23125) was found again. It describes activation-quantization controls using per-channel-magnitude-matched Gaussian noise and sign-randomized quantization errors on induction/retrieval. This is real adjacent overlap in control design; neither matched noise nor sign randomization is our novel invention.

Direct abstract, HTML v1/v2, PDF with/without version and export-PDF retrieval were attempted again and did not yield an accessible full manuscript. An indexed author-homepage entry corroborates the title, but is not a substitute for a full-text review. No full-text or code-level conclusion about that paper's use or non-use of KL matching, reasoning length or abandonment is established.

Additional queries included `"quantization" "reasoning" "KL-matched"`, `"overthinking" "Gaussian" "quantization"`, and `"quantization" "termination" "noise" reasoning`. Results were sparse/noisy and did not resolve the full-text gap. Absence from these searches is not a novelty PASS. The gate remains open, not rejected and not passed.

Candidate contribution, pending review: test whether matched held-out reference-prefix divergence predicts free-generation length inflation and verified correct-answer abandonment. That is a stronger and narrower question than merely asking whether noise can damage a model. Mean KL remains a scalar match rather than a causal identification guarantee.

## 4. Proposed minimal sequence after gates and approval

| Stage | Proposed scope | What it can establish |
|---|---|---|
| R0 — engineering smoke | 1.5B, BF16 and validated AWQ-W3, a tiny non-evaluation prompt fixture, short output limit, same inference stack | Loader, format, checkpoint/scaling, and termination-log functionality only. No overthinking or accuracy claim. |
| R1 — exploratory premise check | The same two arms on a predeclared development set excluded from independent confirmation; fixed sampling policy and budgets; independently assigned one-GPU jobs | Whether the setup shows length inflation, accuracy/termination changes, and evaluable failure traces. R1 does not test H1. |
| R2 — matched controls | Fixed random weight draws with separate matching and validation data; then held-out behavioral evaluation | Eventual H1 testing after matching validity, endpoint labels, margins, sample size and power are specified. |

Do not start with all benchmarks, 7B, tensor parallelism, or every quantizer. Do not lower a generation cap silently when a run is slow, and do not remove cap-hit samples from the primary length analysis. No automatic switch to GPTQ after a weak AWQ effect: diagnose and record any protocol change before another run.

Before R1 is runnable, record the exact model/tokenizer revisions, quantizer implementation and tensor coverage, calibration manifest, prompt template and stop tokens, data separation, sample size/seeds, context/output budgets, evaluator fixtures, annotation rubric, and predeclared premise-check outcomes. An inconclusive small pilot stays inconclusive. The original H1 20% equivalence and 5% matching margins are not validated by this document.

## 5. Next action and session state

Basic local environment preparation is complete to the limited extent above. The next work is source/code provenance and acceptance of a concrete minimal reproduction manifest, not another CUDA setup cycle. The full-text novelty gap and research-run approval remain unrecorded. `next_session_ready` remains false. No model was downloaded or run as part of this review.
