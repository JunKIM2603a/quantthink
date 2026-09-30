# QuantThink protocol — draft, not preregistered

The project question and original hypotheses are preserved below. Corrections are proposed design safeguards, not silent changes to a frozen experiment. No main-test data have been examined locally.

## Original proposal retained

H1: On DeepSeek-R1-Distill-Qwen-1.5B, increases in CoT length and overthinking error under AWQ-W3 can be reproduced within 20% by Gaussian weight perturbations matched to its mean reference-token KL within 5%.

H2 (original directional version): At branching/reconsideration positions, quantization decreases entropy while Gaussian noise and higher temperature increase it.

The numerical margins, estimands, power, and original H2 direction are **NOT VALIDATED / NOT FROZEN**. A proposed safer role for H2 is exploratory until matched-prefix evidence supports a directional confirmatory test. This change requires an explicit recorded decision.

## Interpretation safeguards

Equal mean KL is one scalar constraint, not equality of local error allocation or sequence distributions. A matched noise arm reproducing an effect supports that arm in the tested regime, not every compression method. Non-equivalence or failed matching does not identify a quantization-specific mechanism. Distinguish a confidence interval inside an equivalence region, a meaningful difference, and an inconclusive interval. An inconclusive result is not automatic publication value.

For fixed logits on an untruncated full-vocabulary softmax, raising positive temperature increases entropy. Static Gaussian weight noise need not do so. Mean entropy can also shift downward under a particular quantizer, so an exclusively upward temperature match may not exist. Temperature is an output-distribution control, not a weight-error intervention.

## Reference, quantizer and generation

Primary model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; replication model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`; checkpoint/tokenizer revisions remain to be pinned. Use one canonical BF16 model per scale and an identical inference/evaluation stack within a comparison.

Initial quantizer: AWQ-W3, group size 128, weight-only. The inspected Lotfi v1 section 8.1 uses 128 Pile-uncopyrighted sequences of length 512 for AWQ, while GPTQ uses a different reasoning calibration recipe. Treat method/calibration differences as confounders. Upstream code commit, calibration IDs and preprocessing, exact scaling/clipping, and layer coverage must be recorded before reproduction.

Fake/dequantized BF16 inference is permitted for studying a specified weight transformation, but is not assumed numerically identical to native low-bit kernels and cannot support native latency/memory claims. AWQ reparameterization and scale folding must be verified before comparing canonical BF16 weight errors or injecting noise. Do not mistake plain round-to-nearest weights for AWQ.

Proposed reproduction decoding: T=0.6 and top-p=0.95. Prompt template, special tokens, stopping rules, generated-token budget, total-context limit, and evaluator are pending. The original proposal's 32K budget is not yet verified as the source budget. A short loader smoke is an engineering check only, not an overthinking replication.

## Controls and disjoint data

Use separate quantizer calibration, noise/temperature matching, matching validation, exploratory evaluation, and independent confirmation sets. Record IDs and exclusion rules. Do not retune noise or thresholds on the final MATH-500 outcomes. If a pilot uses final-test examples, exclude those examples from independent confirmation and report the reduced denominator, or adopt a distinct held-out confirmation dataset.

Planned arms: BF16; AWQ-W3; weight-error-matched Gaussian; reference-prefix KL-matched Gaussian; temperature/entropy control. A sign-randomized quantization-error control is a proposed extension motivated by 2609.23125, not claimed as a new control-design invention.

For the weight-error arm, predefine tensor/channel scope and scaling. For KL matching, keep each Gaussian draw fixed as its scale changes, perturb the same declared tensors, and save noise seeds. Inspect the scale-to-KL curve before assuming monotonicity or using bisection. A failed bracket is MATCH_FAILED, not evidence against H1. Global and layerwise matches are distinct arms, not interchangeable descriptions.

Define diagnostic KL as `KL(p_BF16 || p_arm)` on identical frozen BF16-generated reasoning prefixes, using full-vocabulary distributions before top-p filtering, natural logarithms, and a declared temperature. Proposed diagnostic temperature is 1.0, with generation-temperature KL as sensitivity analysis; final choice is pending. Predeclare token masking, sampled positions, prompt-vs-token weighting, and BF16-prefix generation policy. Report held-out match residuals and high-entropy/late-position strata. Matching only the training-calibration mean is insufficient.

Reference-prefix matching does not constrain prefixes visited only by a perturbed model. Use a separate on-policy/union-prefix sensitivity analysis rather than redefining the primary match after seeing outcomes. Layerwise error, accuracy, EOS/termination probability, and length caps are diagnostic covariates, not variables to tune away post hoc.

## Outcomes and statistical units

Track final-answer accuracy, raw generated tokens, reasoning-only tokens where parseable, termination reason, max-budget hit rate, repetitive-loop indicators, and verified overthinking errors. Do not drop truncated generations or calculate only successful-generation length as the primary metric.

Overthinking is not merely occurrence of the correct numeral in a trace. Distinguish a valid intermediate solution followed by a wrong committed answer, a valid solution followed by truncation/no answer, and an incidental/rejected numeral match. Regex can nominate cases but cannot validate this primary endpoint. Freeze the annotation rubric, label blinded examples, inspect both predicted-positive and predicted-negative cases, and report label uncertainty and denominators. Keep overthinking per all problems separate from its share among failures.

The H1 estimands must specify whether 20% refers to absolute length, inflation ratio, or the increase over BF16. A candidate is the between-arm difference in increases over BF16, assessed separately for length and error risk. Relative margins are unstable when the quantization effect is near zero. Use a justified smallest meaningful effect and design-stage power simulation before choosing final margins; do not widen them to obtain equivalence. Declare whether both co-primary outcomes must pass.

Problems and perturbation/sampling seeds are clustered/crossed units, not independent tokens. Use problem-level paired or appropriate hierarchical analyses, report noise-seed variation, and predefine multiplicity treatment. A token-level sign test that treats all tokens as independent is not the H2 default. Select branch positions from a common reference rule, not different generated-token subsets for each arm. Local entropy change, branch probability mass, and termination effects are separate measurements; causal mechanism claims require an additional intervention.

## Minimal next sequence (not an approved model run)

1. Close the source/novelty scope and approve the reproduction protocol. Environment inspection and CPU tests may proceed now.
2. Pin a compatible isolated inference environment, model/tokenizer/quantizer revisions, and evaluation fixtures. Run a tiny loader/format smoke, then a predeclared BF16/AWQ reproduction pilot with matched budgets and termination logging.
3. Measure total throughput, peak memory and output lengths on the actual two GPUs. Forecast GPU-hours from total generated tokens divided by measured aggregate tokens/second, plus matching/quantization overhead. Do not assert the original 15–25 GPU-hour estimate as measured.
4. Only after reproduction and label validity, calibrate matched controls on disjoint data; use independent held-out runs for the eventual H1 test. Avoid launching 7B or a four-benchmark grid at this stage.
