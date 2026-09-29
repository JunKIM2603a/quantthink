# QuantThink research operating rules

Read `docs/research_status.json`, `docs/SESSION_PLAN.md`, and the relevant protocol before doing work. Inspect the live branch, commit, and results rather than relying on chat memory. Never treat a planned or running external research task as a completed report.

1. Distinguish source-reported findings, local observations, proposed hypotheses, and interpretations. Record primary sources and the actual verification depth.
2. Keep the same canonical BF16 reference. Verify AWQ reparameterization/scale folding before calculating weight errors against that reference.
3. Treat model revision, quantizer/calibration, dtype/kernel, decoding, context budget, evaluator, and seeds as potential confounders.
4. Separate quantizer calibration, control matching, exploratory evaluation, and held-out confirmation. Do not tune against final-test outcomes.
5. Do not assume Gaussian noise raises entropy, that equal mean KL makes perturbations identical, or that fake quantization is numerically identical to every native kernel.
6. Preserve the original H1/H2 in the draft; proposed revisions require an explicit decision before being called frozen. Pilot analysis is not confirmatory testing.
7. Report equivalence, meaningful differences, and inconclusive results separately. A negative result does not automatically establish an alternative mechanism or publication value.
8. Do not launch scientific model runs before the source/novelty and smoke-protocol gates, and the required research approval, are recorded. Local environment inspection is allowed while review is open.
9. Keep large artifacts, tokens, credentials, raw local diagnostics, and personal or workplace information out of this public repository. Do not publish full third-party manuscripts or restricted benchmark data.
10. Prefer reviewable branch/PR changes. Do not force-push, merge, or change repository visibility without authorization.
11. 사용자에게 전달하는 새 문서와 갱신 문서는 한국어로 작성한다. 코드 식별자, 파일명, 논문 제목, 원문 인용은 필요한 경우 원어를 유지한다. 기존 영문 문서는 과거 기록이며, 새 안내에서는 최신 한글 문서를 우선 연결한다.

## Session transitions

At each substantive results review, compare evidence with the current session's exit criteria. If met, explicitly tell the user that the session is complete, provide the next title, and fill the handoff template. Otherwise list the remaining blockers and stay in the current session. Do not claim continuous monitoring, remote GPU access, or an automatic ChatGPT title change. Maintain session/status documents alongside actual evidence, not ahead of it.
