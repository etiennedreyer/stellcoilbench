"""Guardrails, safe mode, and batch composition (GA proposer)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from stellcoilbench.validate_config import validate_ci_case

from .ga import (
    _config_hash_short,
    _rng,
    explore_case,
    mutate_case,
)

__all__ = ["check_guardrails", "is_safe_mode", "propose_batch"]

# BO proposer is imported lazily inside propose_batch to keep bayes_opt optional.


def check_guardrails(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check guardrails.  Returns (should_stop, reason)."""
    gr = policy.get("guardrails", {})
    stats = ctx.get("failure_stats", {})

    fail_rate = stats.get("fail_rate", 0.0)
    max_fail_rate = gr.get("max_fail_rate", 0.6)
    if fail_rate > max_fail_rate:
        return True, f"fail_rate {fail_rate:.2f} > {max_fail_rate}"

    mcrc = stats.get("most_common_reason_count", 0)
    max_mcrc = gr.get("max_common_failure_count", 12)
    if mcrc > max_mcrc:
        reason = stats.get("most_common_reason", "?")
        return True, f"failure reason '{reason}' repeated {mcrc} > {max_mcrc} times"

    critical_classes = set(gr.get("critical_failure_classes", []))
    max_crit = gr.get("max_critical_class_count", 10)
    for cls, cnt in stats.get("failure_classes", {}).items():
        if cls in critical_classes and cnt >= max_crit:
            return True, f"critical failure class '{cls}' repeated {cnt} >= {max_crit}"

    return False, ""


def is_safe_mode(ctx: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    """Return True if the proposer should operate in safe mode."""
    sm = policy.get("safe_mode", {})
    threshold = sm.get("threshold", 0.35)
    fail_rate = ctx.get("failure_stats", {}).get("fail_rate", 0.0)
    return fail_rate > threshold


def propose_batch(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    batch_size: int = 8,
    seed: int | None = None,
    mode: str = "ga",
) -> List[Dict[str, Any]]:
    """Propose a batch of cases.

    Parameters
    ----------
    mode:
        ``"ga"`` (default) — mutation + exploration genetic algorithm.
        ``"bo"`` — Bayesian Optimisation warm-started from submission history.
    """
    if mode == "bo":
        from .bo import propose_batch_bo
        return propose_batch_bo(ctx, policy, batch_size=batch_size, seed=seed)

    rng = _rng(seed)
    safe = is_safe_mode(ctx, policy)

    exploit_frac = policy.get("exploit_fraction", 0.5)
    exploit_count = int(math.floor(exploit_frac * batch_size))
    explore_count = batch_size - exploit_count

    parents = ctx.get("top_parents", [])
    recent_hashes = set(ctx.get("recent_config_hashes", []))

    cases: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    attempts = 0
    while len(cases) < exploit_count and attempts < exploit_count * 5:
        attempts += 1
        if not parents:
            break
        parent = rng.choice(parents)
        child = mutate_case(parent, policy, rng, safe=safe)

        h = _config_hash_short(child.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue

        errors = validate_ci_case(child, policy=policy)
        if errors:
            continue

        seen_hashes.add(h)
        cases.append(child)

    attempts = 0
    while len(cases) < exploit_count + explore_count and attempts < explore_count * 5:
        attempts += 1
        child = explore_case(policy, rng, safe=safe)

        h = _config_hash_short(child.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue

        errors = validate_ci_case(child, policy=policy)
        if errors:
            continue

        seen_hashes.add(h)
        cases.append(child)

    extra_attempts = 0
    while len(cases) < batch_size and extra_attempts < batch_size * 5:
        extra_attempts += 1
        child = explore_case(policy, rng, safe=safe)

        h = _config_hash_short(child.get("case_config", {}))
        if h in recent_hashes or h in seen_hashes:
            continue

        errors = validate_ci_case(child, policy=policy)
        if not errors:
            seen_hashes.add(h)
            cases.append(child)

    return cases[:batch_size]
