"""Bayesian Optimisation proposer for stellcoilbench.

Warm-starts a Gaussian-process surrogate from the existing submissions
database, then uses the UCB/EI/POI acquisition function to suggest the
next batch of case configurations.

Design choices (see plan discussion for rationale):
- Objective: ``-log10(avg_BdotN_over_B)`` — threshold-independent, higher
  is better, directly measures coil quality.
- Infeasible runs (unlinked coils, interlinked coils, missing metric) are
  excluded from the GP dataset rather than imputed.
- Discrete parameters (ncoils, order) are encoded as continuous and rounded
  at decode time — adequate for the small integer ranges used here.
- All threshold parameters are log-transformed before being passed to the
  GP so the kernel sees a roughly uniform-density space.
- Batch proposals use sequential hallucination: after each ``suggest()``
  call the predicted mean is temporarily registered so the next suggestion
  isn't a repeat.
- Falls back to the GA proposer when history is thinner than
  ``bo_params.min_history``.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from stellcoilbench.validate_config import validate_ci_case
except ImportError:
    _repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo / "src"))
    from stellcoilbench.validate_config import validate_ci_case

from .ga import _config_hash_short, _new_case_id, _rng, explore_case

log = logging.getLogger(__name__)

__all__ = [
    "load_bo_history",
    "compute_bo_target",
    "build_bo_optimizer",
    "propose_batch_bo",
]

# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

# Continuous BO dimensions and their log-space bounds (at ARIES-CS reactor
# scale, matching the policy's exploration ranges).
_THRESHOLD_PARAMS: List[Tuple[str, float, float]] = [
    ("length_threshold",   100.0,  200.0),
    ("cc_threshold",         0.5,    1.5),
    ("cs_threshold",         0.7,    2.0),
    ("curvature_threshold",  0.5,   10.0),
    ("msc_threshold",        0.5,   10.0),
    ("force_threshold",     50.0,  500.0),
]

# ncoils encoded as a float; rounded to nearest int at decode time.
_NCOILS_RANGE = (3.0, 7.0)
# order encoded as index into this list.
_ORDER_CHOICES = [4, 6, 8]


def _pbounds(policy: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Build the bayes_opt pbounds dict from policy exploration ranges."""
    expl = policy.get("exploration", {})
    bounds: Dict[str, Tuple[float, float]] = {}

    for param, default_lo, default_hi in _THRESHOLD_PARAMS:
        range_key = f"{param}_range"
        lo, hi = expl.get(range_key, [default_lo, default_hi])
        # BO operates in log space; map back at decode time.
        bounds[f"log_{param}"] = (math.log(lo), math.log(hi))

    # ncoils_cont in [3,7]
    ncoils_choices = expl.get("ncoils_choices", [3, 4, 5, 6, 7])
    bounds["ncoils_cont"] = (float(min(ncoils_choices)), float(max(ncoils_choices)))

    # order encoded as index in [0, len-1]
    order_choices = expl.get("order_choices", _ORDER_CHOICES)
    bounds["order_idx"] = (0.0, float(len(order_choices) - 1))

    return bounds


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def compute_bo_target(metrics: Dict[str, Any]) -> Optional[float]:
    """Compute BO target value from a results.json metrics dict.

    Returns ``-log10(avg_BdotN_over_B)`` (higher is better), or ``None``
    if the run is infeasible or the metric is unavailable.
    """
    if not metrics.get("coils_linked_to_surface", False):
        return None
    if abs(float(metrics.get("final_linking_number", 1.0))) >= 0.5:
        return None
    flux = metrics.get("avg_BdotN_over_B")
    if flux is None or not math.isfinite(float(flux)) or float(flux) <= 0:
        return None
    return -math.log10(float(flux))


def _surface_dir_name(surface_input: str) -> str:
    """Strip 'input.' prefix to match the submissions directory name."""
    s = surface_input
    if s.startswith("input."):
        s = s[len("input."):]
    # strip .focus / .nc suffixes too
    for suffix in (".focus", ".nc"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def load_bo_history(
    surface: str,
    submissions_dir: Path,
) -> List[Dict[str, Any]]:
    """Scan submissions for a surface and return GP-ready (params, target) records.

    Each record is ``{"params": {bo_dim: float, ...}, "target": float}``.
    Records for failed/infeasible runs are excluded.

    ``surface`` should be the raw surface name as it appears in ``case.yaml``
    (e.g. ``"input.LandremanPaul2021_QA"``).
    """
    surface_dirname = _surface_dir_name(surface)
    surface_path = submissions_dir / surface_dirname
    if not surface_path.is_dir():
        log.warning("BO history: submissions dir not found: %s", surface_path)
        return []

    records: List[Dict[str, Any]] = []

    for case_yaml_path in surface_path.rglob("case.yaml"):
        run_dir = case_yaml_path.parent
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue

        try:
            with open(case_yaml_path) as f:
                case_cfg = yaml.safe_load(f)
            with open(results_path) as f:
                results = json.load(f)
        except Exception as exc:
            log.debug("BO history: skipping %s (%s)", run_dir, exc)
            continue

        metrics = results.get("metrics", {})
        target = compute_bo_target(metrics)
        if target is None:
            continue

        # Extract input parameters from case.yaml
        obj = case_cfg.get("coil_objective_terms", {})
        coils = case_cfg.get("coils_params", {})

        try:
            params: Dict[str, float] = {}
            valid = True
            for param, default_lo, default_hi in _THRESHOLD_PARAMS:
                val = obj.get(param)
                if val is None or float(val) <= 0:
                    valid = False
                    break
                params[f"log_{param}"] = math.log(float(val))

            if not valid:
                continue

            ncoils = coils.get("ncoils")
            order = coils.get("order")
            if ncoils is None or order is None:
                continue

            params["ncoils_cont"] = float(int(ncoils))

            order_choices = _ORDER_CHOICES
            if int(order) in order_choices:
                params["order_idx"] = float(order_choices.index(int(order)))
            else:
                # Use nearest match
                dists = [abs(int(order) - o) for o in order_choices]
                params["order_idx"] = float(dists.index(min(dists)))

        except (TypeError, ValueError) as exc:
            log.debug("BO history: param extraction failed for %s (%s)", run_dir, exc)
            continue

        records.append({"params": params, "target": target})

    log.info("BO history: loaded %d feasible records for surface '%s'", len(records), surface)
    return records


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def build_bo_optimizer(
    history: List[Dict[str, Any]],
    pbounds: Dict[str, Tuple[float, float]],
    acquisition_fn: str = "ucb",
    kappa: float = 2.576,
    xi: float = 0.0,
    seed: Optional[int] = None,
) -> Any:
    """Build and warm-start a BayesianOptimization instance.

    Parameters
    ----------
    history:
        Records returned by :func:`load_bo_history`.
    pbounds:
        Parameter bounds dict for BayesianOptimization.
    acquisition_fn:
        ``"ucb"``, ``"ei"``, or ``"poi"``.
    kappa:
        Exploration weight for UCB.
    xi:
        Exploration offset for EI/POI.
    seed:
        Random seed.

    Returns
    -------
    BayesianOptimization
        Fitted optimizer with history registered.
    """
    from bayes_opt import BayesianOptimization, acquisition as acq_mod

    acq_fn_lower = acquisition_fn.lower()
    if acq_fn_lower == "ucb":
        acq = acq_mod.UpperConfidenceBound(kappa=kappa)
    elif acq_fn_lower == "ei":
        acq = acq_mod.ExpectedImprovement(xi=xi)
    elif acq_fn_lower in ("poi", "pi"):
        acq = acq_mod.ProbabilityOfImprovement(xi=xi)
    else:
        raise ValueError(f"Unknown acquisition_fn '{acquisition_fn}'. Use 'ucb', 'ei', or 'poi'.")

    bo = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        acquisition_function=acq,
        random_state=seed,
        allow_duplicate_points=True,
        verbose=0,
    )

    # Clamp history params to pbounds before registering (some older runs may
    # have been generated with slightly different ranges).
    skipped = 0
    for record in history:
        clamped = {}
        in_bounds = True
        for dim, val in record["params"].items():
            lo, hi = pbounds.get(dim, (-1e9, 1e9))
            if val < lo - 1e-6 or val > hi + 1e-6:
                in_bounds = False
                break
            clamped[dim] = max(lo, min(hi, val))
        if not in_bounds:
            skipped += 1
            continue
        bo.register(params=clamped, target=record["target"])

    if skipped:
        log.debug("BO warm-start: skipped %d out-of-bounds history records", skipped)

    return bo


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

def _decode_params(
    raw: Dict[str, float],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert BO parameter dict back into ``coil_objective_terms`` + ``coils_params``."""
    expl = policy.get("exploration", {})
    order_choices = expl.get("order_choices", _ORDER_CHOICES)
    ncoils_choices = expl.get("ncoils_choices", [3, 4, 5, 6, 7])

    obj: Dict[str, Any] = {}
    for param, _, _ in _THRESHOLD_PARAMS:
        log_val = raw[f"log_{param}"]
        obj[param] = round(math.exp(log_val), 4)

    ncoils_cont = raw["ncoils_cont"]
    ncoils = int(round(ncoils_cont))
    ncoils = max(min(ncoils_choices), min(max(ncoils_choices), ncoils))

    order_idx = int(round(raw["order_idx"]))
    order_idx = max(0, min(len(order_choices) - 1, order_idx))
    order = order_choices[order_idx]

    return {"coil_objective_terms": obj, "ncoils": ncoils, "order": order}


def _build_case_config(
    decoded: Dict[str, Any],
    surface: str,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a full case_config dict from decoded BO parameters."""
    expl = policy.get("exploration", {})
    obj = decoded["coil_objective_terms"]

    # Populate objective term types (mirrors explore_case logic)
    coil_objective_terms: Dict[str, Any] = {
        "total_length": "l2_threshold",
        "coil_curvature": "lp_threshold",
        "coil_curvature_p": 2,
        "coil_mean_squared_curvature": "l2_threshold",
        "coil_arclength_variation": "l2_threshold",
        "linking_number": "",
    }
    if expl.get("include_force", False):
        coil_objective_terms["coil_coil_force"] = "lp_threshold"
    if expl.get("include_torque", False):
        coil_objective_terms["coil_coil_torque"] = "lp_threshold"
    if expl.get("include_torsion", False):
        coil_objective_terms["coil_torsion"] = "lp_threshold"
        coil_objective_terms["coil_torsion_p"] = 2

    # Merge in threshold values from BO
    coil_objective_terms.update(obj)

    ncoils = decoded["ncoils"]
    order = decoded["order"]
    max_iterations = expl.get("max_iterations", 500)

    case_config: Dict[str, Any] = {
        "description": f"BO proposal: {surface} ncoils={ncoils} order={order}",
        "surface_params": {
            "surface": surface,
            "range": "half period",
        },
        "coils_params": {
            "ncoils": ncoils,
            "order": order,
        },
        "optimizer_params": {
            "algorithm": "augmented_lagrangian",
            "max_iterations": max_iterations,
            "verbose": True,
        },
        "coil_objective_terms": coil_objective_terms,
    }

    fc = policy.get("fourier_continuation", {})
    if fc and fc.get("enabled") and fc.get("orders"):
        case_config["fourier_continuation"] = {
            "enabled": True,
            "orders": list(fc["orders"]),
        }

    return case_config


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def propose_batch_bo(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    batch_size: int = 8,
    seed: Optional[int] = None,
    submissions_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Propose a batch of cases using Bayesian Optimisation.

    Parameters
    ----------
    ctx:
        Context dict from ``build_context()`` (same as GA proposer).
    policy:
        Policy dict loaded from ``proposer_policy.yaml``.
    batch_size:
        Number of cases to propose.
    seed:
        Random seed for reproducibility.
    submissions_dir:
        Path to the submissions directory. Defaults to
        ``<repo_root>/submissions``.

    Returns
    -------
    list of case dicts
        Each dict has the same schema as GA proposer output.
        Falls back to GA-style exploration for any cases that cannot be
        filled by BO (thin history, validation failures, etc.).
    """
    bo_cfg = policy.get("bo_params", {})
    min_history = int(bo_cfg.get("min_history", 5))
    acquisition_fn = bo_cfg.get("acquisition", "ucb")
    kappa = float(bo_cfg.get("kappa", 2.576))
    xi = float(bo_cfg.get("xi", 0.0))

    # Determine target surface (most-explored surface in submissions, or override)
    target_surface = bo_cfg.get("target_surface") or None
    if target_surface is None:
        surface_counts = ctx.get("surface_exploration_counts", {})
        expl_surfaces = policy.get("exploration", {}).get("surfaces", [])
        if surface_counts and expl_surfaces:
            # Pick the most-explored surface that is in the current surface list
            target_surface = max(
                (s for s in expl_surfaces if s in surface_counts or True),
                key=lambda s: surface_counts.get(s, 0),
                default=None,
            )
        if target_surface is None and expl_surfaces:
            target_surface = expl_surfaces[0]

    if target_surface is None:
        log.warning("BO proposer: no target surface found, falling back to GA")
        return _ga_fallback(ctx, policy, batch_size, seed)

    # Locate submissions directory
    if submissions_dir is None:
        submissions_dir = Path(__file__).resolve().parents[2] / "submissions"

    # Load history
    history = load_bo_history(target_surface, submissions_dir)

    if len(history) < min_history:
        log.info(
            "BO proposer: only %d history records for '%s' (need %d), falling back to GA",
            len(history), target_surface, min_history,
        )
        return _ga_fallback(ctx, policy, batch_size, seed)

    bounds = _pbounds(policy)
    bo = build_bo_optimizer(
        history=history,
        pbounds=bounds,
        acquisition_fn=acquisition_fn,
        kappa=kappa,
        xi=xi,
        seed=seed,
    )

    rng = _rng(seed)
    recent_hashes = set(ctx.get("recent_config_hashes", []))
    caps = policy.get("resource_caps", {})
    expl = policy.get("exploration", {})
    max_iterations = expl.get("max_iterations", 500)
    resource = {
        "max_total_iterations": min(max_iterations, caps.get("max_total_iterations", 10000)),
        "timeout_minutes": caps.get("timeout_minutes_max", 60),
    }

    cases: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    bo_attempts = 0
    max_bo_attempts = batch_size * 6

    while len(cases) < batch_size and bo_attempts < max_bo_attempts:
        bo_attempts += 1

        try:
            raw_params = bo.suggest()
        except Exception as exc:
            log.debug("BO suggest failed: %s", exc)
            break

        decoded = _decode_params(raw_params, policy)
        case_config = _build_case_config(decoded, target_surface, policy)

        h = _config_hash_short(case_config)
        if h in recent_hashes or h in seen_hashes:
            # Jitter to escape duplicate
            _jitter_params(raw_params, bounds, rng, scale=0.05)
            continue

        new_case_id = _new_case_id()
        errors = validate_ci_case(
            {"case_id": new_case_id, "case_config": case_config, "resource": resource},
            policy=policy,
        )
        if errors:
            log.debug("BO proposal failed validation: %s", errors)
            continue

        new_seed = rng.randint(0, 2**31 - 1)
        case = {
            "case_id": new_case_id,
            "parent_ids": [],
            "tags": ["bo"],
            "proposer_mode": "bo",
            "resource": resource,
            "case_config": case_config,
            "random_seed": new_seed,
        }
        seen_hashes.add(h)
        cases.append(case)

        # Hallucination: register GP posterior mean so next suggest() differs.
        try:
            bo.register(params=raw_params, target=float(bo._gp.predict([list(raw_params.values())])[0]))
        except Exception:
            # If GP isn't fitted yet (shouldn't happen with warm-start, but be safe)
            pass

    # Fill remaining slots with GA-style exploration
    if len(cases) < batch_size:
        log.info(
            "BO proposer: filling %d remaining slots with GA exploration",
            batch_size - len(cases),
        )
        fallback = _ga_fallback(ctx, policy, batch_size - len(cases), seed)
        cases.extend(fallback)

    return cases[:batch_size]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jitter_params(
    params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    rng: Any,
    scale: float = 0.05,
) -> None:
    """In-place add small gaussian noise to params (clamped to bounds)."""
    for k in params:
        lo, hi = bounds[k]
        span = hi - lo
        params[k] = max(lo, min(hi, params[k] + rng.gauss(0, scale * span)))


def _ga_fallback(
    ctx: Dict[str, Any],
    policy: Dict[str, Any],
    n: int,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    """Generate ``n`` GA exploration cases as fallback."""
    from .proposer import propose_batch
    rng = _rng(seed)
    fallback_seed = rng.randint(0, 2**31 - 1)
    # Use the standard GA proposer but limit to exploration only
    cases = propose_batch(ctx, policy, batch_size=n, seed=fallback_seed)
    return cases
