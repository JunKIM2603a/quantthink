# QuantThink

Investigating why quantized reasoning models overthink through matched perturbation controls.

**Status: research-design review, not an established finding. No model experiment has been run or verified in this repository.**

The planned question is whether increases in reasoning length and overthinking under aggressive quantization can be reproduced by random weight perturbations matched on reference-prefix token-level divergence. A secondary question concerns local entropy and branching behavior. The initial model is DeepSeek-R1-Distill-Qwen-1.5B, with a planned 7B replication and MATH-500 as the primary benchmark.

## Start here

- [Current research status](docs/research_status.json)
- [Session plan and exit criteria](docs/SESSION_PLAN.md)
- [Initial evidence and novelty audit](docs/EVIDENCE_AUDIT.md)
- [Protocol draft: unresolved choices are explicit](docs/PROTOCOL_DRAFT.md)
- [Session handoff template](docs/HANDOFF_TEMPLATE.md)
- [Research operating rules](AGENTS.md)

The broad matched-noise control idea already appears in activation-quantization research (arXiv:2609.23125). Its full-text overlap with the proposed reasoning study remains to be reviewed. Do not claim that a Gaussian-noise comparison is itself new.

## Local preparation only

Python 3.10+ is required for the diagnostic script. It installs nothing, downloads no models, makes no remote uploads, and does not alter the software environment. It optionally imports an already installed PyTorch in a subprocess to inspect visible CUDA devices.

```bash
python -m unittest discover -s tests -v
python scripts/preflight.py --require-cuda --min-gpus 2
```

The report is written to `results/local/preflight.json`, which is gitignored. Review it before sharing. Exit code 2 means the requested CUDA visibility check did not pass; the report is still written. `CUDA_VISIBLE` does not establish BF16 inference, memory capacity, AWQ compatibility, or research readiness. Existing reports are not overwritten unless `--overwrite` is supplied. Use `--skip-torch` for a metadata-only inspection; it cannot pass a required CUDA check.

Do not install an arbitrary collection of latest inference packages before reviewing this report. A compatible, isolated inference environment and pinned versions will be selected during Session 02.

## Scientific guardrails

Long output, looping, truncation, and abandoning a correct intermediate answer are different outcomes. Mean-KL matching is a controlled comparison, not proof of a causal mechanism. Failed equivalence is not evidence of a quantization-specific cause. Fake/dequantized BF16 evaluation is not a measurement of native low-bit throughput. See the draft protocol before implementing model experiments.
