"""CPU-only design contracts, not AWQ implementation or LLM evidence."""
from __future__ import annotations
import json
import math
from collections.abc import Iterable


def exact_token_blocks(documents: Iterable[list[int]], *, block_size: int = 512, n_blocks: int = 128) -> list[list[int]]:
    """Pack ordered pretokenized input to an exact budget; fail if insufficient.

    Tokenization, document selection and separator policy belong to the manifest.
    This utility neither downloads data nor recreates an upstream calibration set.
    """
    if type(block_size) is not int or type(n_blocks) is not int or min(block_size, n_blocks) < 1:
        raise ValueError("Positive integer sizes required")
    needed = block_size * n_blocks
    tokens: list[int] = []
    for doc in documents:
        if not isinstance(doc, list) or any(type(t) is not int or t < 0 for t in doc):
            raise ValueError("Expected lists of nonnegative integer token ids")
        tokens.extend(doc[:needed - len(tokens)])
        if len(tokens) == needed:
            break
    if len(tokens) != needed:
        raise ValueError(f"Insufficient calibration tokens: {len(tokens)} < {needed}")
    return [tokens[i:i + block_size] for i in range(0, needed, block_size)]


def termination_record(generated_ids: list[int], eos_ids: set[int], budget: int) -> dict:
    """One unpadded response; EOS on the last permitted step is observed, not censored."""
    if type(budget) is not int or budget < 1 or not generated_ids or not eos_ids:
        raise ValueError("Require positive budget, nonempty response and EOS set")
    if any(type(t) is not int or t < 0 for t in generated_ids) or any(type(t) is not int or t < 0 for t in eos_ids):
        raise ValueError("Token ids must be nonnegative integers")
    if len(generated_ids) > budget or any(t in eos_ids for t in generated_ids[:-1]):
        raise ValueError("Padded/over-budget/after-EOS responses require explicit normalization")
    eos = generated_ids[-1] in eos_ids
    cap = len(generated_ids) == budget
    return {"generated_token_count_including_eos": len(generated_ids),
            "finish_reason": "eos" if eos else "length" if cap else "unexpected_stop",
            "budget_reached": cap, "right_censored": cap and not eos}


def kl_length_counterexample() -> dict:
    """Elementary memoryless categorical example; not a test of QuantThink H1."""
    p = (1 / 3,) * 3  # indices: EOS, token A, token B
    rows = []
    for name, q in (("A", (0.05, 0.475, 0.475)), ("B", (0.475, 0.05, 0.475))):
        rows.append({"arm": name, "kl_nats": sum(a * math.log(a / b) for a, b in zip(p, q)),
                     "entropy_nats": -sum(b * math.log(b) for b in q), "eos_probability": q[0],
                     "expected_tokens_including_eos": 1 / q[0]})
    return {"scope": "Mathematical sanity example; not empirical LLM results or a novelty claim", "arms": rows}


if __name__ == "__main__":
    print(json.dumps(kl_length_counterexample(), indent=2))
