"""Generate the NKG lateral distribution figure for Chapter 2 of the thesis.

The figure replaces the placeholder `lateralDistribution.png` that the advisor
inserted as a sketch. It is generated directly from the NKG function as written
in the thesis (Eq. 2.4), so the curve in the right panel is the same expression
the text defines, not a redrawn approximation of a published figure.

Everything is expressed in units of the Moliere radius r_M. That is deliberate:
r_M in metres depends on the air density at the observation altitude, and no
verified value for 5,300 m is on hand, so the figure avoids committing to one.
In these units the curve is exact and needs no external source.

Output: thesis/figures/lateral_distribution.png
"""

from __future__ import annotations

from math import gamma
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import condor_style as style

OUT = Path(__file__).resolve().parents[1] / "thesis" / "figures" / "lateral_distribution.png"

# Shower age values. s = 1 is shower maximum; s < 1 is a young shower still
# developing, s > 1 an old one past maximum and already attenuating.
AGES = [0.8, 1.0, 1.2]
AGE_COLORS = [style.PROTON, style.GAMMA, style.ACCENT]


def nkg(r_over_rm: np.ndarray, s: float) -> np.ndarray:
    """NKG particle density in units of N_e / r_M^2, as a function of r / r_M.

    Thesis Eq. (2.4):
        rho(r) = N_e / (2 pi r_M^2) * C(s) * (r/r_M)^(s-2) * (1 + r/r_M)^(s-4.5)
        C(s)   = Gamma(4.5 - s) / [Gamma(s) Gamma(4.5 - 2s)]
    """
    c_s = gamma(4.5 - s) / (gamma(s) * gamma(4.5 - 2.0 * s))
    return (c_s / (2.0 * np.pi)) * r_over_rm ** (s - 2.0) * (1.0 + r_over_rm) ** (s - 4.5)


def main() -> None:
    style.apply_style("paper")

    # Wide and shallow on purpose: at \linewidth in the thesis this keeps the
    # figure box short enough to share a page with body text instead of
    # pushing the surrounding paragraphs onto a page of their own.
    fig, (ax_map, ax_prof) = plt.subplots(
        1, 2, figsize=(9.8, 3.15), gridspec_kw={"width_ratios": [1.0, 1.3]}
    )

    # ---- left panel: 2-D footprint of the density at shower maximum --------
    extent = 2.2  # in units of r_M
    n = 700
    grid = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(grid, grid)
    rr = np.hypot(xx, yy)
    rr = np.clip(rr, 1e-3, None)  # the NKG diverges at r = 0
    dens = np.log10(nkg(rr, 1.0))

    ax_map.set_facecolor("white")
    ax_map.grid(False)
    im = ax_map.pcolormesh(
        xx,
        yy,
        dens,
        cmap=style.CMAP_SEQ,
        shading="auto",
        rasterized=True,
        vmin=-3.0,
        vmax=1.0,
    )

    # Rings of constant radius, to make the radial fall-off legible as
    # structure rather than as a smooth blur. Each label sits at a different
    # angle on its own ring, so they cannot collide with one another.
    for frac, ang in ((0.5, 55.0), (1.0, 0.0), (2.0, -45.0)):
        ax_map.add_patch(
            plt.Circle(
                (0, 0), frac, fill=False, ec="#4A4A4A", lw=0.7, ls=(0, (4, 3)), zorder=3
            )
        )
        rad = np.deg2rad(ang)
        ax_map.text(
            frac * np.cos(rad),
            frac * np.sin(rad),
            rf"${frac:g}\,r_M$",
            fontsize=7,
            color="#2E2E2E",
            ha="center",
            va="center",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )

    ax_map.plot(0, 0, marker="+", ms=9, mew=1.6, color="#1A1A1A", zorder=5)
    ax_map.text(
        -0.13,
        0.0,
        "core",
        fontsize=7.5,
        color="#1A1A1A",
        ha="right",
        va="center",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
    )

    ax_map.set_aspect("equal")
    ax_map.set_xlim(-extent, extent)
    ax_map.set_ylim(-extent, extent)
    ax_map.set_xlabel(r"$x / r_M$")
    ax_map.set_ylabel(r"$y / r_M$")
    ax_map.set_title("Particle density on the ground", pad=7)

    cb = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}\,\rho\;[N_e\,r_M^{-2}]$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ---- right panel: radial profile for three shower ages ----------------
    r = np.logspace(-2, np.log10(10.0), 400)
    for s, color in zip(AGES, AGE_COLORS):
        label = rf"$s = {s:g}$"
        if s == 1.0:
            label += "  (maximum)"
        ax_prof.loglog(r, nkg(r, s), color=color, lw=1.8, label=label)

    ax_prof.set_xlabel(r"lateral distance from the core, $r / r_M$")
    ax_prof.set_ylabel(r"$\rho(r)\;[N_e\,r_M^{-2}]$")
    ax_prof.set_title("Radial profile for three shower ages", pad=7)
    ax_prof.legend(frameon=False, loc="upper right")
    ax_prof.set_xlim(1e-2, 10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
