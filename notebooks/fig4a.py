"""Reproduce Fig 4a from the stellcoilbench paper (Landreman-Paul QA surface).

Each dot is one coil optimization run from the submissions database:
  x-axis : avg_BdotN_over_B  — how accurately the coils reproduce the target
                                magnetic field (lower = better coils)
  y-axis : final_max_curvature — how sharply the coils bend (lower = more buildable)
  color  : final_total_length — total length of all unique coils (shorter = cheaper)

The star marks the 3-coil QA solution highlighted in the paper.

Optional --gp flag additionally produces a second figure: an 8×8 corner/pair plot
of the GP posterior across all parameter pairs.  Lower triangle = 2-D contourf
slice (all other dims held at median); diagonal = 1-D slice; upper triangle = same
runs as scatter coloured by actual outcome so you can check GP accuracy.

Usage
-----
  # Fig 4a only:
  python fig4a.py

  # Fig 4a + GP corner plot (saves both to disk):
  python fig4a.py --gp --out fig4a.png --out-gp gp_corner.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import yaml

# ── repo path setup ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

SUBMISSIONS_DIR = _REPO / "submissions" / "LandremanPaul2021_QA"
POLICY_PATH     = _REPO / "policy" / "proposer_policy.yaml"
SURFACE_NAME    = "input.LandremanPaul2021_QA"

# Published 3-coil QA result from Landreman & Paul (2022) — the star in Fig 4a.
# Values read from the paper figure.
# Paper star: flux and curvature read from Fig 4a; curvature already in unit-R0 units
# in the paper (R0≈1 m so κ*R0 ≈ κ numerically).
PAPER_STAR = dict(flux=2.0e-3, curvature=2.9, label="3-coil QA\n(Landreman & Paul 2022)")


# ── data loading ───────────────────────────────────────────────────────────────

def load_qa_runs() -> list[dict]:
    """Return list of dicts with flux, curvature, length, ncoils for every
    feasible QA run (coils linked, no coil–coil interlinking)."""
    runs = []
    for rf in SUBMISSIONS_DIR.rglob("results.json"):
        try:
            d   = json.loads(rf.read_text())
            m   = d["metrics"]
            if not m.get("coils_linked_to_surface"):
                continue
            if abs(float(m.get("final_linking_number", 1))) >= 0.5:
                continue
            flux   = m.get("avg_BdotN_over_B")
            curv   = m.get("final_max_curvature")
            length = m.get("final_total_length")
            R0     = m.get("_cached_thresholds", {}).get("major_radius", 1.0)
            if flux is None or curv is None or length is None:
                continue
            if not (math.isfinite(flux) and flux > 0):
                continue

            # Read ncoils from the companion case.yaml
            cf = rf.parent / "case.yaml"
            ncoils = None
            if cf.exists():
                ncoils = yaml.safe_load(cf.read_text()).get(
                    "coils_params", {}
                ).get("ncoils")

            # Normalise to unit major radius (R0 ≈ 1.01 m for QA, so effect is small
            # but matches the paper's "scaled to 1 m major radius" convention).
            # length_per_coil normalises for varying ncoils, matching the paper's
            # "total coil length per half-field period" for a fixed-ncoils policy.
            length_norm = (length / R0) / ncoils if ncoils else length / R0
            curv_norm   = curv * R0

            runs.append(dict(
                flux=flux,
                curvature=curv_norm,
                length=length_norm,
                ncoils=ncoils,
            ))
        except Exception:
            pass
    return runs


# ── GP landscape helpers ────────────────────────────────────────────────────────

def build_gp_on_runs(runs: list[dict], policy: dict):
    """Fit a GP to the QA run outcomes and return (bo, bounds, dim_order)."""
    from tools.propose_batch.bo import (
        load_bo_history,
        build_bo_optimizer,
        _pbounds,
    )

    # Point policy at QA surface
    policy = dict(policy)
    policy["exploration"] = dict(policy.get("exploration", {}))
    policy["exploration"]["surfaces"] = [SURFACE_NAME]

    history = load_bo_history(SURFACE_NAME, _REPO / "submissions")
    print(f"  GP warm-start: {len(history)} feasible QA runs loaded.")

    bounds    = _pbounds(policy)
    dim_order = list(bounds.keys())
    bo        = build_bo_optimizer(history, bounds, seed=42)
    bo._gp.fit(bo._space.params, bo._space.target)
    print("  GP fitted.")
    return bo, bounds, dim_order


def gp_slice_curvature_vs_flux(bo, bounds, dim_order, n: int = 80):
    """Compute a 2-D GP posterior slice over (log_curvature_threshold, log_length_threshold).

    All other dimensions are held at their median value.  We use these two
    because they correspond directly to the y-axis (curvature) and colorbar
    (length) of Fig 4a — the GP is predicting whether a given *constraint
    setting* will yield a good result, and these two settings most directly
    govern the coil geometry visible in the plot.
    """
    lc = np.linspace(*bounds["log_curvature_threshold"], n)   # → y-axis proxy
    ll = np.linspace(*bounds["log_length_threshold"],    n)   # → color proxy
    median = {k: (lo + hi) / 2 for k, (lo, hi) in bounds.items()}

    Xg, Yg = np.meshgrid(ll, lc)   # (length, curvature)
    pts = []
    for length_val, curv_val in zip(Xg.ravel(), Yg.ravel()):
        p = dict(median)
        p["log_length_threshold"]    = length_val
        p["log_curvature_threshold"] = curv_val
        pts.append([p[k] for k in dim_order])

    Z = bo._gp.predict(pts).reshape(n, n)
    return np.exp(Xg), np.exp(Yg), Z   # back to physical units


def gp_next_suggestions(bo, bounds, dim_order, n: int = 6):
    """Return n BO suggestions (with hallucination so they differ)."""
    suggestions = []
    for _ in range(n):
        raw  = bo.suggest()
        pred = float(bo._gp.predict([[raw[k] for k in dim_order]])[0])
        bo.register(params=raw, target=pred)
        suggestions.append(raw)
    return suggestions


# ── plotting ───────────────────────────────────────────────────────────────────

def plot(runs: list[dict], gp: bool, out: str | None):
    flux   = np.array([r["flux"]      for r in runs])
    curv   = np.array([r["curvature"] for r in runs])
    length = np.array([r["length"]    for r in runs])

    vmin, vmax = np.percentile(length, [5, 95])
    norm_len = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if gp:
        # Two-panel layout: left = Fig 4a outcome scatter,
        # right = GP landscape in threshold space (where the GP actually lives)
        fig, (ax, ax_gp) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    # ── LEFT PANEL: Fig 4a scatter (outcome space) ────────────────────────────
    sc = ax.scatter(
        flux, curv,
        c=length, cmap="viridis", norm=norm_len,
        s=10, alpha=0.7, linewidths=0,
        zorder=3, label=f"Feasible runs  (N = {len(runs):,})",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Coil length per coil / R₀  [dimensionless]", fontsize=10)

    ax.scatter(
        PAPER_STAR["flux"], PAPER_STAR["curvature"],
        marker="*", s=350, c="#7B2D8B", edgecolors="k", linewidths=0.5,
        zorder=7, label=PAPER_STAR["label"],
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\langle |\mathbf{B}\cdot\hat{n}| \rangle / \langle B \rangle$",
                  fontsize=13)
    ax.set_ylabel(r"Max Curvature ($\kappa R_0$)  [dimensionless]", fontsize=12)
    ax.set_title("Landreman–Paul QA surface\ncoil optimization landscape", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)

    # ── RIGHT PANEL: GP posterior in threshold space ───────────────────────────
    if gp:
        print("Building GP posterior — this takes ~20 s …")
        policy = yaml.safe_load(POLICY_PATH.read_text())
        bo, bounds, dim_order = build_gp_on_runs(runs, policy)

        # 2-D slice: length_threshold (x) vs curvature_threshold (y).
        # These are the *inputs* to the optimizer — the constraints you impose.
        # The GP predicts what quality will result from each combination.
        n = 80
        ll = np.linspace(*bounds["log_length_threshold"],    n)
        lc = np.linspace(*bounds["log_curvature_threshold"], n)
        median = {k: (lo + hi) / 2 for k, (lo, hi) in bounds.items()}
        Xg, Yg = np.meshgrid(ll, lc)
        pts = []
        for lv, cv in zip(Xg.ravel(), Yg.ravel()):
            p = dict(median)
            p["log_length_threshold"]    = lv
            p["log_curvature_threshold"] = cv
            pts.append([p[k] for k in dim_order])
        Z = bo._gp.predict(pts).reshape(n, n)

        cf = ax_gp.contourf(np.exp(Xg), np.exp(Yg), Z,
                            levels=25, cmap="plasma", alpha=0.85)
        cbar_gp = fig.colorbar(cf, ax=ax_gp)
        cbar_gp.set_label(
            "GP predicted  –log₁₀(BdotN/B)\n(higher = predicted better field accuracy)",
            fontsize=9,
        )

        # Overlay actual runs in threshold space — read their input thresholds
        # from the history records loaded by load_bo_history
        from tools.propose_batch.bo import load_bo_history
        history = load_bo_history(SURFACE_NAME, _REPO / "submissions")
        h_len  = [math.exp(h["params"]["log_length_threshold"])    for h in history]
        h_curv = [math.exp(h["params"]["log_curvature_threshold"]) for h in history]
        h_tgt  = [h["target"] for h in history]
        ax_gp.scatter(h_len, h_curv, c=h_tgt, cmap="plasma",
                      vmin=Z.min(), vmax=Z.max(),
                      s=8, alpha=0.5, edgecolors="none", zorder=3,
                      label=f"Existing runs (actual outcome)\nN={len(history)}")

        # BO next suggestions
        sugs = gp_next_suggestions(bo, bounds, dim_order, n=5)
        ax_gp.scatter(
            [math.exp(s["log_length_threshold"])    for s in sugs],
            [math.exp(s["log_curvature_threshold"]) for s in sugs],
            marker="*", s=320, c="cyan", edgecolors="k", linewidths=0.6,
            zorder=6, label="BO next suggestions",
        )

        ax_gp.set_xlabel("Max allowed total coil length  (m, reactor scale)", fontsize=11)
        ax_gp.set_ylabel("Max allowed coil curvature  (m⁻¹, reactor scale)", fontsize=11)
        ax_gp.set_title(
            "What the GP learned from the data\n"
            "Background = predicted field quality for each constraint setting",
            fontsize=10,
        )
        ax_gp.legend(fontsize=8, loc="upper right")
        ax_gp.grid(True, ls="--", lw=0.4, alpha=0.5)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


# ── GP corner / pair plot ──────────────────────────────────────────────────────

# Human-readable axis labels for each GP dimension.
_DIM_LABELS = {
    "log_length_threshold":    "Length\nthreshold (m)",
    "log_cc_threshold":        "Coil–coil\ngap (m)",
    "log_cs_threshold":        "Coil–surface\ngap (m)",
    "log_curvature_threshold": "Curvature\nlimit (m⁻¹)",
    "log_msc_threshold":       "Mean sq.\ncurvature (m⁻²)",
    "log_force_threshold":     "Force\nlimit (N/m)",
    "ncoils_cont":             "N coils",
    "order_idx":               "Fourier\norder idx",
}

# For log-space dims we show exp(x); for linear dims (ncoils, order) we show x.
_IS_LOG = {k: k.startswith("log_") for k in _DIM_LABELS}

# Dimensions that are discrete in practice (GP treats them as continuous, but
# only integer values are physically meaningful).  Maps dim name → list of
# integer tick values to mark with dashed lines in each subplot.
_DISCRETE_TICKS = {
    "ncoils_cont": [3, 4, 5, 6, 7],
    "order_idx":   [0, 1, 2],
}


def _to_display(k: str, v: float) -> float:
    return math.exp(v) if _IS_LOG[k] else v


def _add_discrete_lines(ax, ki: str, kj: str, xlim, ylim) -> None:
    """Add semi-transparent dashed lines at integer positions for discrete dims."""
    kw = dict(color="white", alpha=0.45, lw=0.8, ls="--", zorder=4)
    if kj in _DISCRETE_TICKS:   # discrete dim on x-axis → vertical lines
        for v in _DISCRETE_TICKS[kj]:
            if xlim[0] <= v <= xlim[1]:
                ax.axvline(v, **kw)
    if ki in _DISCRETE_TICKS:   # discrete dim on y-axis → horizontal lines
        for v in _DISCRETE_TICKS[ki]:
            if ylim[0] <= v <= ylim[1]:
                ax.axhline(v, **kw)


def plot_gp_corner(bo, bounds: dict, dim_order: list, history: list,
                   out: str | None) -> None:
    """Corner plot of the GP posterior across all non-degenerate parameter pairs.

    Lower triangle : 2-D contourf slice (all other dims held at median).
    Diagonal       : 1-D posterior slice through the median point.
    Upper triangle : scatter of real runs coloured by actual outcome —
                     lets you verify the GP has learned correctly.
    """
    # Drop dimensions where all history values are identical (degenerate dims,
    # e.g. order_idx when only one Fourier order is in use).
    active_dims = []
    for k in dim_order:
        vals = [h["params"][k] for h in history]
        if max(vals) - min(vals) > 1e-9:
            active_dims.append(k)
        else:
            print(f"  Dropping degenerate dimension: {k} (all values = {vals[0]:.3g})")

    n_dim = len(active_dims)
    n_1d  = 60
    n_2d  = 35

    median = {k: (lo + hi) / 2 for k, (lo, hi) in bounds.items()}

    grids = {k: np.linspace(*bounds[k], n_1d) for k in active_dims}

    h_params = {k: np.array([_to_display(k, h["params"][k]) for h in history])
                for k in active_dims}
    h_target = np.array([h["target"] for h in history])
    t_min, t_max = h_target.min(), h_target.max()

    fig, axes = plt.subplots(n_dim, n_dim,
                             figsize=(2.0 * n_dim, 1.8 * n_dim))
    fig.suptitle(
        f"GP posterior — all pairwise 2-D marginals  (QA surface, {len(history):,} warm-start runs)\n"
        "Lower triangle: GP predicted field quality (bright = better)  |  "
        "Diagonal: 1-D slice  |  Upper triangle: actual run outcomes",
        fontsize=9, y=1.002,
    )

    print(f"  Computing {n_dim*(n_dim-1)//2} GP slices ({n_2d}×{n_2d} each) …")

    for row in range(n_dim):
        for col in range(n_dim):
            ax = axes[row, col]
            ki = active_dims[row]
            kj = active_dims[col]

            if row == col:
                # ── Diagonal: 1-D slice ──────────────────────────────────
                xs = grids[ki]
                pts = [[p[k] for k in dim_order]
                       for p in ({**median, ki: x} for x in xs)]
                ys = bo._gp.predict(pts)
                xd = [_to_display(ki, x) for x in xs]
                ax.plot(xd, ys, color="#e05c00", lw=1.5)
                ax.set_ylabel("–log₁₀\n(BdotN/B)", fontsize=5)
                ax.grid(True, ls="--", lw=0.3, alpha=0.5)
                # Dashed vertical lines at discrete integer values
                if ki in _DISCRETE_TICKS:
                    for v in _DISCRETE_TICKS[ki]:
                        if xd[0] <= v <= xd[-1]:
                            ax.axvline(v, color="gray", alpha=0.5, lw=0.8, ls="--")

            elif row > col:
                # ── Lower triangle: 2-D heatmap (imshow fills every pixel) ──
                xg = np.linspace(*bounds[kj], n_2d)
                yg = np.linspace(*bounds[ki], n_2d)
                Xg, Yg = np.meshgrid(xg, yg)
                pts = [[{**median, kj: xv, ki: yv}[k] for k in dim_order]
                       for xv, yv in zip(Xg.ravel(), Yg.ravel())]
                Z = bo._gp.predict(pts).reshape(n_2d, n_2d)
                xd_lo = _to_display(kj, xg[0]);  xd_hi = _to_display(kj, xg[-1])
                yd_lo = _to_display(ki, yg[0]);  yd_hi = _to_display(ki, yg[-1])
                # imshow: origin="lower" so y increases upward; aspect="auto"
                # to fill the axes rectangle exactly.
                ax.imshow(Z, origin="lower", aspect="auto", cmap="plasma",
                          extent=[xd_lo, xd_hi, yd_lo, yd_hi],
                          vmin=Z.min(), vmax=Z.max(), interpolation="bilinear")
                ax.scatter(h_params[kj], h_params[ki], c=h_target,
                           cmap="plasma", vmin=Z.min(), vmax=Z.max(),
                           s=1.5, alpha=0.35, linewidths=0, zorder=3)
                # Clip axes to the imshow extent — scatter points outside the
                # current policy bounds would otherwise float on white background.
                ax.set_xlim(xd_lo, xd_hi)
                ax.set_ylim(yd_lo, yd_hi)
                # _add_discrete_lines(ax, ki, kj,
                #                     xlim=(xd_lo, xd_hi),
                #                     ylim=(yd_lo, yd_hi))

            else:
                # ── Upper triangle: actual run scatter ───────────────────
                # Clip to policy bounds so axes match the lower-triangle panels.
                xd_lo = _to_display(kj, bounds[kj][0])
                xd_hi = _to_display(kj, bounds[kj][1])
                yd_lo = _to_display(ki, bounds[ki][0])
                yd_hi = _to_display(ki, bounds[ki][1])
                ax.scatter(h_params[kj], h_params[ki], c=h_target,
                           cmap="viridis", vmin=t_min, vmax=t_max,
                           s=1.5, alpha=0.4, linewidths=0)
                ax.set_xlim(xd_lo, xd_hi)
                ax.set_ylim(yd_lo, yd_hi)
                # _add_discrete_lines(ax, ki, kj,
                #                     xlim=(xd_lo, xd_hi), ylim=(yd_lo, yd_hi))

            ax.tick_params(labelsize=5)
            if row == n_dim - 1:
                ax.set_xlabel(_DIM_LABELS.get(kj, kj), fontsize=6)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(_DIM_LABELS.get(ki, ki), fontsize=6)
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  Saved {out}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gp", action="store_true",
                        help="Also produce the GP corner plot.")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Save Fig 4a to this path (default: display).")
    parser.add_argument("--out-gp", default=None, metavar="PATH",
                        help="Save GP corner plot to this path (default: display).")
    args = parser.parse_args()

    print("Loading QA submissions …")
    runs = load_qa_runs()
    print(f"  {len(runs):,} feasible runs loaded.")

    # Always produce Fig 4a (left panel only when --gp not set)
    plot(runs, gp=False, out=args.out)

    if args.gp:
        print("Building GP for corner plot …")
        policy = yaml.safe_load(POLICY_PATH.read_text())
        bo, bounds, dim_order = build_gp_on_runs(runs, policy)

        from tools.propose_batch.bo import load_bo_history
        history = load_bo_history(SURFACE_NAME, _REPO / "submissions")

        plot_gp_corner(bo, bounds, dim_order, history, out=args.out_gp)


if __name__ == "__main__":
    main()
