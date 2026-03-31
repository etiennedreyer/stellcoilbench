"""CLI entry point for the batch proposer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stellcoilbench.path_utils import load_yaml

# Ensure tools/ and repo root are on path for build_context
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from build_context import build_context  # noqa: E402

from .proposer import check_guardrails, propose_batch
from .llm import propose_batch_llm_direct


def main() -> int:  # pragma: no cover
    """Main entry point for the batch proposer."""
    parser = argparse.ArgumentParser(
        description="Batch proposer for the nonstop CI autopilot. "
        "Supports GA (default) and LLM-powered modes."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Number of cases to propose."
    )
    parser.add_argument(
        "--done-dir",
        type=Path,
        default=None,
        help="Directory containing completed case summaries (legacy, used when "
        "--submissions-dir and --failures-file are not both provided).",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("submissions"),
        help="Root submissions directory for successful runs (Option C). "
        "When provided with --failures-file, used instead of --done-dir.",
    )
    parser.add_argument(
        "--failures-file",
        type=Path,
        default=Path("policy/autopilot_failures.json"),
        help="Path to autopilot failures JSON (Option C). "
        "When provided with --submissions-dir, used instead of --done-dir.",
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=Path("cases/pending"),
        help="Directory for new pending cases.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policy/proposer_policy.yaml"),
        help="Path to proposer_policy.yaml.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed cases to stdout without writing files.",
    )
    parser.add_argument(
        "--mode",
        choices=["ga", "bo", "llm"],
        default="ga",
        help="Proposer mode: 'ga' (default), 'bo' (Bayesian Optimisation), or 'llm'.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM proposer (direct mode; requires KB_LLM_* env vars). Equivalent to --mode llm.",
    )
    parser.add_argument(
        "--verify-llm",
        action="store_true",
        help="When combined with --llm, fail loudly (exit 1) if the LLM is "
        "unavailable instead of silently falling back to the GA proposer. "
        "Useful for CI to confirm the LLM secret is configured.",
    )
    args = parser.parse_args()

    pause_file = _REPO_ROOT / "PAUSE_AUTORUN"
    if pause_file.exists():
        print("PAUSE_AUTORUN file exists. Exiting without proposing.", file=sys.stderr)
        return 0

    pending = args.pending_dir
    if not args.dry_run and pending.is_dir() and any(pending.glob("*.json")):
        print(
            "Pending directory is not empty. Waiting for current batch to finish.",
            file=sys.stderr,
        )
        return 0

    if not args.policy.exists():
        print(f"ERROR: policy file not found: {args.policy}", file=sys.stderr)
        return 1
    policy = load_yaml(path=args.policy)

    done_dir = args.done_dir or Path("cases/done")
    ctx = build_context(
        done_dir,
        args.policy,
        submissions_root=args.submissions_dir,
        failures_path=args.failures_file,
    )

    should_stop, reason = check_guardrails(ctx, policy)
    if should_stop:
        print(f"GUARDRAIL TRIGGERED: {reason}", file=sys.stderr)
        cooldown = policy.get("cooldown", {})
        if cooldown.get("write_pause_file", False):
            pause_file.write_text(f"Guardrail: {reason}\n")
            print(f"Created {pause_file}", file=sys.stderr)
        return 0

    if args.llm and args.verify_llm:
        try:
            from knowledge.llm_client import is_available

            if not is_available():
                print(
                    "ERROR: --verify-llm: LLM is not configured. "
                    "Set ANTHROPIC_API_KEY (or KB_LLM_* env vars).",
                    file=sys.stderr,
                )
                return 1
            print("LLM verified: available.", file=sys.stderr)
        except ImportError as e:
            print(
                f"ERROR: --verify-llm: cannot import LLM client: {e}", file=sys.stderr
            )
            return 1

    reasoning_history_path = args.policy.parent / "llm_reasoning_history.json"
    mode = "llm" if args.llm else args.mode
    if mode == "llm":
        cases = propose_batch_llm_direct(
            ctx,
            policy,
            batch_size=args.batch_size,
            seed=args.seed,
            reasoning_history_path=reasoning_history_path,
        )
    else:
        cases = propose_batch(ctx, policy, batch_size=args.batch_size, seed=args.seed, mode=mode)

    if args.dry_run:
        print(f"Proposer: {mode}", file=sys.stderr)
        print(json.dumps(cases, indent=2))
        return 0

    pending.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cid = case["case_id"]
        out_path = pending / f"{cid}.json"
        out_path.write_text(json.dumps(case, indent=2))
        print(f"Wrote {out_path}")

    print(f"Proposed {len(cases)} cases.", file=sys.stderr)
    return 0
