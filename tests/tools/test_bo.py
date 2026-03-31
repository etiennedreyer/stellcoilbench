"""Unit tests for the Bayesian Optimisation proposer (tools/propose_batch/bo.py)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import yaml

# Stub simsopt and stellcoilbench.path_utils so tests run without a full
# simsopt installation.  The BO proposer itself doesn't need simsopt; the
# dependency is only transitive through ga.py → path_utils → simsopt.geo.
def _stub(name: str) -> MagicMock:
    mod = MagicMock(spec=ModuleType)
    mod.__name__ = name
    mod.__path__ = []
    mod.__spec__ = None
    sys.modules[name] = mod
    return mod

for _name in [
    "simsopt",
    "simsopt.geo",
    "simsopt.field",
    "simsopt.mhd",
    "simsopt._core",
]:
    if _name not in sys.modules:
        _stub(_name)

# Ensure src and repo root are on path before the stellcoilbench import.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

# Stub path_utils before stellcoilbench is imported so its __init__ never
# tries to pull in simsopt.geo.SurfaceRZFourier.
_path_utils_stub = _stub("stellcoilbench.path_utils")
_path_utils_stub.get_surface_filename = MagicMock(return_value="input.LandremanPaul2021_QA")

from tools.propose_batch.bo import (
    _decode_params,
    _pbounds,
    build_bo_optimizer,
    compute_bo_target,
    load_bo_history,
    propose_batch_bo,
)

# ---------------------------------------------------------------------------
# Minimal policy fixture matching proposer_policy.yaml structure
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_policy():
    return {
        "batch_size": 4,
        "exploit_fraction": 0.5,
        "top_k_parents": 3,
        "resource_caps": {
            "max_total_iterations": 500,
            "timeout_minutes_max": 30,
        },
        "mutation": {
            "threshold_sigma": 0.1,
            "structural_mutation_prob": 0.2,
            "ncoils_choices": [3, 4, 5, 6, 7],
            "order_choices": [4],
            "max_iterations": 500,
            "dof_perturbation": 0.0,
        },
        "fourier_continuation": {"enabled": True, "orders": [4, 8, 16]},
        "exploration": {
            "use_default_thresholds": False,
            "length_threshold_range": [100.0, 200.0],
            "cc_threshold_range": [0.5, 1.5],
            "cs_threshold_range": [0.7, 2.0],
            "curvature_threshold_range": [0.5, 10.0],
            "msc_threshold_range": [0.5, 10.0],
            "force_threshold_range": [50.0, 500.0],
            "ncoils_choices": [3, 4, 5, 6, 7],
            "order_choices": [4],
            "max_iterations": 500,
            "include_force": True,
            "include_torque": False,
            "include_torsion": False,
            "dof_perturbation": 0.0,
            "surfaces": ["input.LandremanPaul2021_QA"],
            "algorithms": ["augmented_lagrangian"],
        },
        "guardrails": {
            "sliding_window": 30,
            "max_fail_rate": 0.6,
            "max_common_failure_count": 12,
            "max_critical_class_count": 10,
            "critical_failure_classes": [],
        },
        "safe_mode": {"threshold": 0.35, "preferred_surfaces": ["input.LandremanPaul2021_QA"]},
        "bo_params": {
            "acquisition": "ucb",
            "kappa": 2.576,
            "xi": 0.0,
            "min_history": 2,
        },
    }


@pytest.fixture()
def minimal_ctx():
    return {
        "failure_stats": {"fail_rate": 0.0},
        "top_parents": [],
        "recent_config_hashes": [],
        "surface_exploration_counts": {"input.LandremanPaul2021_QA": 10},
    }


# ---------------------------------------------------------------------------
# Helpers for building fake submission directories
# ---------------------------------------------------------------------------

def _make_submission(
    root: Path,
    surface_dir: str,
    run_name: str,
    ncoils: int = 4,
    order: int = 4,
    cc: float = 0.8,
    cs: float = 1.3,
    length: float = 150.0,
    curvature: float = 2.0,
    msc: float = 1.5,
    force: float = 200.0,
    avg_bdotn: float = 0.005,
    linked: bool = True,
    linking_number: float = 0.0,
) -> None:
    """Write a minimal case.yaml + results.json pair under root/surface_dir/run_name/."""
    run_dir = root / surface_dir / "auto" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    case = {
        "description": f"test {run_name}",
        "surface_params": {"surface": f"input.{surface_dir}", "range": "half period"},
        "coils_params": {"ncoils": ncoils, "order": order},
        "optimizer_params": {"algorithm": "augmented_lagrangian", "max_iterations": 500},
        "coil_objective_terms": {
            "total_length": "l2_threshold",
            "coil_curvature": "lp_threshold",
            "coil_curvature_p": 2,
            "coil_mean_squared_curvature": "l2_threshold",
            "coil_arclength_variation": "l2_threshold",
            "linking_number": "",
            "coil_coil_force": "lp_threshold",
            "length_threshold": length,
            "cc_threshold": cc,
            "cs_threshold": cs,
            "curvature_threshold": curvature,
            "msc_threshold": msc,
            "force_threshold": force,
        },
        "fourier_continuation": {"enabled": True, "orders": [4, 8, 16]},
    }
    (run_dir / "case.yaml").write_text(yaml.dump(case))

    results = {
        "metadata": {"contact": "auto", "iterations_used": 500, "walltime_sec": 60.0},
        "metrics": {
            "avg_BdotN_over_B": avg_bdotn,
            "coils_linked_to_surface": linked,
            "final_linking_number": linking_number,
            "final_squared_flux": avg_bdotn ** 2,
        },
        "reactor_scale_metrics": {},
    }
    (run_dir / "results.json").write_text(json.dumps(results))


# ---------------------------------------------------------------------------
# compute_bo_target
# ---------------------------------------------------------------------------

class TestComputeBoTarget:
    def test_returns_negative_log10_flux(self):
        metrics = {
            "avg_BdotN_over_B": 0.01,
            "coils_linked_to_surface": True,
            "final_linking_number": 0.0,
        }
        target = compute_bo_target(metrics)
        assert target == pytest.approx(-math.log10(0.01))  # +2.0

    def test_unlinked_coils_returns_none(self):
        metrics = {
            "avg_BdotN_over_B": 0.01,
            "coils_linked_to_surface": False,
            "final_linking_number": 0.0,
        }
        assert compute_bo_target(metrics) is None

    def test_interlinked_coils_returns_none(self):
        metrics = {
            "avg_BdotN_over_B": 0.01,
            "coils_linked_to_surface": True,
            "final_linking_number": 0.8,
        }
        assert compute_bo_target(metrics) is None

    def test_missing_flux_returns_none(self):
        metrics = {
            "coils_linked_to_surface": True,
            "final_linking_number": 0.0,
        }
        assert compute_bo_target(metrics) is None

    def test_zero_flux_returns_none(self):
        metrics = {
            "avg_BdotN_over_B": 0.0,
            "coils_linked_to_surface": True,
            "final_linking_number": 0.0,
        }
        assert compute_bo_target(metrics) is None

    def test_higher_flux_gives_lower_target(self):
        good = compute_bo_target({"avg_BdotN_over_B": 0.001, "coils_linked_to_surface": True, "final_linking_number": 0.0})
        bad = compute_bo_target({"avg_BdotN_over_B": 0.1, "coils_linked_to_surface": True, "final_linking_number": 0.0})
        assert good > bad  # better coils → higher BO target


# ---------------------------------------------------------------------------
# load_bo_history
# ---------------------------------------------------------------------------

class TestLoadBoHistory:
    def test_loads_feasible_records(self, tmp_path):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "run_001")
        _make_submission(tmp_path, "LandremanPaul2021_QA", "run_002", avg_bdotn=0.002)
        records = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        assert len(records) == 2

    def test_filters_infeasible_unlinked(self, tmp_path):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "good", linked=True)
        _make_submission(tmp_path, "LandremanPaul2021_QA", "bad", linked=False)
        records = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        assert len(records) == 1

    def test_filters_interlinked(self, tmp_path):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "good", linking_number=0.0)
        _make_submission(tmp_path, "LandremanPaul2021_QA", "bad", linking_number=0.9)
        records = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        assert len(records) == 1

    def test_returns_empty_for_missing_surface_dir(self, tmp_path):
        records = load_bo_history("input.NonExistent", tmp_path)
        assert records == []

    def test_record_has_all_expected_params(self, tmp_path):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "run_001", ncoils=5, order=4)
        records = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        assert len(records) == 1
        params = records[0]["params"]
        assert "log_length_threshold" in params
        assert "log_cc_threshold" in params
        assert "log_cs_threshold" in params
        assert "log_curvature_threshold" in params
        assert "log_msc_threshold" in params
        assert "log_force_threshold" in params
        assert "ncoils_cont" in params
        assert "order_idx" in params
        assert params["ncoils_cont"] == 5.0

    def test_target_matches_compute_bo_target(self, tmp_path):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "run_001", avg_bdotn=0.003)
        records = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        expected = -math.log10(0.003)
        assert records[0]["target"] == pytest.approx(expected, rel=1e-6)

    def test_strips_input_prefix_from_surface(self, tmp_path):
        # Submissions directory uses the bare name, not 'input.' prefix
        _make_submission(tmp_path, "LandremanPaul2021_QA", "run_001")
        records = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# build_bo_optimizer
# ---------------------------------------------------------------------------

class TestBuildBoOptimizer:
    def test_warmstart_registers_all_feasible_records(self, tmp_path, minimal_policy):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "r1")
        _make_submission(tmp_path, "LandremanPaul2021_QA", "r2", avg_bdotn=0.002)
        history = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        bounds = _pbounds(minimal_policy)
        bo = build_bo_optimizer(history, bounds, seed=42)
        assert len(bo.res) == 2

    def test_suggests_point_in_bounds(self, tmp_path, minimal_policy):
        for i in range(5):
            _make_submission(tmp_path, "LandremanPaul2021_QA", f"r{i}", avg_bdotn=0.001 * (i + 1))
        history = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        bounds = _pbounds(minimal_policy)
        bo = build_bo_optimizer(history, bounds, seed=42)
        suggestion = bo.suggest()
        for dim, (lo, hi) in bounds.items():
            assert lo <= suggestion[dim] <= hi, f"{dim} out of bounds"

    def test_accepts_ei_acquisition(self, tmp_path, minimal_policy):
        _make_submission(tmp_path, "LandremanPaul2021_QA", "r1")
        history = load_bo_history("input.LandremanPaul2021_QA", tmp_path)
        bounds = _pbounds(minimal_policy)
        bo = build_bo_optimizer(history, bounds, acquisition_fn="ei", seed=42)
        assert bo is not None

    def test_rejects_unknown_acquisition(self, tmp_path, minimal_policy):
        history = []
        bounds = _pbounds(minimal_policy)
        with pytest.raises(ValueError, match="Unknown acquisition_fn"):
            build_bo_optimizer(history, bounds, acquisition_fn="banana")

    def test_out_of_bounds_history_skipped(self, minimal_policy):
        bounds = _pbounds(minimal_policy)
        # Create a record with a param way outside bounds
        bad_record = {
            "params": {k: -999.0 for k in bounds},
            "target": 3.0,
        }
        bo = build_bo_optimizer([bad_record], bounds, seed=42)
        assert len(bo.res) == 0


# ---------------------------------------------------------------------------
# _decode_params
# ---------------------------------------------------------------------------

class TestDecodeParams:
    def test_roundtrip_continuous_thresholds(self, minimal_policy):
        bounds = _pbounds(minimal_policy)
        # Mid-point of each bound
        raw = {k: (lo + hi) / 2 for k, (lo, hi) in bounds.items()}
        decoded = _decode_params(raw, minimal_policy)
        obj = decoded["coil_objective_terms"]
        # All threshold values should be positive
        for param in ["length_threshold", "cc_threshold", "cs_threshold",
                      "curvature_threshold", "msc_threshold", "force_threshold"]:
            assert obj[param] > 0

    def test_ncoils_rounded_to_integer(self, minimal_policy):
        bounds = _pbounds(minimal_policy)
        raw = {k: (lo + hi) / 2 for k, (lo, hi) in bounds.items()}
        raw["ncoils_cont"] = 4.6
        decoded = _decode_params(raw, minimal_policy)
        assert decoded["ncoils"] == 5

    def test_order_mapped_from_index(self, minimal_policy):
        bounds = _pbounds(minimal_policy)
        raw = {k: (lo + hi) / 2 for k, (lo, hi) in bounds.items()}
        raw["order_idx"] = 0.0
        decoded = _decode_params(raw, minimal_policy)
        assert decoded["order"] == 4  # first element of order_choices


# ---------------------------------------------------------------------------
# propose_batch_bo (integration)
# ---------------------------------------------------------------------------

class TestProposeBatchBo:
    def test_returns_correct_batch_size(self, tmp_path, minimal_policy, minimal_ctx):
        for i in range(10):
            _make_submission(tmp_path, "LandremanPaul2021_QA", f"run_{i:03d}",
                             avg_bdotn=0.001 * (i + 1))
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=3, seed=42,
            submissions_dir=tmp_path,
        )
        assert len(cases) == 3

    def test_cases_have_required_keys(self, tmp_path, minimal_policy, minimal_ctx):
        for i in range(5):
            _make_submission(tmp_path, "LandremanPaul2021_QA", f"run_{i:03d}")
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=2, seed=0,
            submissions_dir=tmp_path,
        )
        for case in cases:
            assert "case_id" in case
            assert "case_config" in case
            assert "resource" in case
            assert "random_seed" in case

    def test_bo_tagged_cases(self, tmp_path, minimal_policy, minimal_ctx):
        # Use diverse submissions (varying ncoils, thresholds, flux) so the GP
        # can suggest novel points that pass the hash-novelty check.
        configs = [
            dict(ncoils=3, cc=0.6, cs=0.9, length=110.0, curvature=3.0, msc=2.0, force=100.0, avg_bdotn=0.008),
            dict(ncoils=4, cc=0.8, cs=1.3, length=150.0, curvature=2.0, msc=1.5, force=200.0, avg_bdotn=0.005),
            dict(ncoils=5, cc=1.0, cs=1.6, length=170.0, curvature=1.5, msc=1.0, force=300.0, avg_bdotn=0.003),
            dict(ncoils=6, cc=1.2, cs=1.8, length=190.0, curvature=1.0, msc=0.8, force=400.0, avg_bdotn=0.002),
            dict(ncoils=7, cc=1.4, cs=2.0, length=200.0, curvature=0.8, msc=0.6, force=450.0, avg_bdotn=0.001),
        ]
        for i, cfg in enumerate(configs):
            _make_submission(tmp_path, "LandremanPaul2021_QA", f"run_{i:03d}", **cfg)
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=2, seed=1,
            submissions_dir=tmp_path,
        )
        # BO should have produced at least one case (possibly filled remainder with GA)
        assert len(cases) == 2
        bo_cases = [c for c in cases if "bo" in c.get("tags", [])]
        assert len(bo_cases) > 0

    def test_falls_back_to_ga_when_no_history(self, tmp_path, minimal_policy, minimal_ctx):
        # No submissions at all — must fall back gracefully
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=3, seed=7,
            submissions_dir=tmp_path,
        )
        assert len(cases) == 3

    def test_falls_back_when_history_below_min(self, tmp_path, minimal_policy, minimal_ctx):
        # Only 1 submission, min_history=2 → fallback
        _make_submission(tmp_path, "LandremanPaul2021_QA", "run_001")
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=2, seed=3,
            submissions_dir=tmp_path,
        )
        assert len(cases) == 2

    def test_no_duplicate_case_ids_in_batch(self, tmp_path, minimal_policy, minimal_ctx):
        for i in range(10):
            _make_submission(tmp_path, "LandremanPaul2021_QA", f"run_{i:03d}",
                             avg_bdotn=0.001 * (i + 1))
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=4, seed=99,
            submissions_dir=tmp_path,
        )
        ids = [c["case_id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_surface_in_case_config(self, tmp_path, minimal_policy, minimal_ctx):
        for i in range(5):
            _make_submission(tmp_path, "LandremanPaul2021_QA", f"run_{i:03d}")
        cases = propose_batch_bo(
            minimal_ctx, minimal_policy, batch_size=2, seed=5,
            submissions_dir=tmp_path,
        )
        for case in cases:
            surface = case["case_config"].get("surface_params", {}).get("surface", "")
            assert surface != ""
