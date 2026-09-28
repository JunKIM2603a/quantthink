# Initial evidence / novelty audit — 2026-09-29

This is a scoped initial audit, not a completed systematic review or a local replication. `VERIFIED_TEXT` means the stated passage was read in a primary source; it does not certify the paper's experiments. `INDEXED_PRIMARY_ABSTRACT_ONLY` is deliberately weaker. No completed Deep Research report was available in the inspected repository.

| Primary source and scope inspected | What the source supports | Implication for QuantThink |
|---|---|---|
| [Lotfi et al., 2606.00206v1](https://arxiv.org/html/2606.00206v1), sections 3, 4.1–4.3, 6–8.1; VERIFIED_TEXT | For 1.5B/MATH-500, AWQ-W3: accuracy 85.6% to 47.0%, CoT 5.2k to 23.4k, overthinking errors 19 to 139. Section 7 suggests other perturbations may affect uncertain positions; section 6 does not claim a complete causal explanation. | Premise is source-supported, H1 is not established. The 19/139 counts concern judged failures, not token-string matches. Reproduce the reported calibration recipe rather than any arbitrary AWQ checkpoint. |
| [Liu et al., 2504.04823](https://arxiv.org/html/2504.04823), section 4.1; VERIFIED_TEXT, exact current revision not pinned | Aggressive W3G128/W4A4KV4 can lengthen outputs; milder settings often do not. | Do not frame Liu and Lotfi as a simple contradiction. Different regimes matter. |
| [ReSET, 2606.13233v1](https://arxiv.org/html/2606.13233v1), sections 3 and B.2/B.4; VERIFIED_TEXT | NVFP4 entropy changes depend on token/step uncertainty. Very high-entropy tokens in uncertain steps can sharpen. | This does not establish AWQ-W3 entropy signs or a general Gaussian-noise contrast. Its native B200 setup is not the planned 4090 weight-only experiment. |
| [Lian et al., 2606.25519v2](https://arxiv.org/html/2606.25519v2), section 7.2; VERIFIED_TEXT | For Qwen3-4B INT3 GPTQ, calibration changes jointly affect accuracy and length. | The calibration observation is not a matched-error demonstration that direction rather than magnitude caused the change. |
| [Sathyanarayanan, 2609.23125](https://arxiv.org/abs/2609.23125), indexed primary abstract; INDEXED_PRIMARY_ABSTRACT_ONLY | Activation quantization is compared with per-channel-magnitude-matched Gaussian noise and sign-randomized error for induction/retrieval damage. | The broad matched-noise/sign-control idea is already present. This is adjacent control-design overlap, not yet proof that KL-matched weight-noise overthinking has been studied. Full text is a blocking review item. |

The newly located source's indexed submission date is 2026-09-19. Direct arXiv abstract/HTML/PDF retrieval failed in this session. Do not describe its full manuscript, code, or claim boundaries as fully inspected. Inspect a primary full text before closing the novelty gate.

## Search scope and remaining gaps

Initial web queries included `"reasoning" "quantization" "KL-matched"`, `"overthinking" "Gaussian" "quantization"`, `"quantization" "matched noise" reasoning`, and the exact title of 2609.23125. Some query results were irrelevant; search absence is not evidence of global novelty. Primary-source passages above were used for claims, not secondary summaries.

Pending: full-text overlap review of 2609.23125; broader weight-noise / compression / termination literature; code and checkpoint provenance; stable paper versions; original generation budget and judge protocol; a defensible contribution beyond a new control arm. The original report's 2602.17691, 2606.02011, and 2509.12464 were not re-audited here.

## Current decision

**OPEN_WITH_ADJACENT_CONTROL_OVERLAP.** Continue design and environment preparation. Do not declare novelty PASS, cancel the topic, or start the full experiment matrix on this evidence alone. A candidate differentiator to test is whether a held-out reference-prefix divergence match predicts sequence-level termination and correct-answer abandonment; this is a proposed scope, not a verified novel finding.
