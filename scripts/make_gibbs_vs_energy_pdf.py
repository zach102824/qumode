#!/usr/bin/env python3
"""Two-page PDF: why Gibbs finds the bitstring when ⟨H⟩ does not."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from qumode_vqe.eta import SampledTailEta, weighted_quantile
from qumode_vqe.vqe import gibbs_objective

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "gibbs_vs_energy.pdf"

E = np.array([-12.0, -8.0, 80.0])
NAMES = ["ground\n$E=-12$", "trap\n$E=-8$", "wall\n$E=+80$"]
SHORT = ["GS", "trap", "wall"]
# Start: mode on the trap, 37% still on the penalty wall.
P0 = np.array([0.08, 0.55, 0.37])
# Same L1 move of 0.37 probability:
PA = np.array([0.08, 0.92, 0.00])  # wall -> trap; decoder still trap
PB = np.array([0.45, 0.18, 0.37])  # trap -> GS; decoder becomes GS
# Head-to-head: lower energy but wrong bits vs higher energy but correct bits.
PT = np.array([0.10, 0.90, 0.00])  # trap-clean
PG = np.array([0.52, 0.08, 0.40])  # GS-leaky (wall leftover)

GS, TRAP, WALL = "#1b7f6e", "#c9960a", "#c0392b"
COLORS = [GS, TRAP, WALL]
MUTED = "#5c656b"
INK = "#1a1a1a"
BODY = "#222222"


def energy(p: np.ndarray) -> float:
    return float(np.dot(p, E))


def mode_name(p: np.ndarray) -> str:
    return SHORT[int(np.argmax(p))]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig_lines(fig, x: float, y: float, lines: list[str], *, size: float = 9.5, dy: float = 0.020, weight: str = "normal") -> None:
    for i, line in enumerate(lines):
        fig.text(
            x,
            y - i * dy,
            line,
            fontsize=size,
            fontweight=weight,
            color=INK if weight == "bold" else BODY,
            va="top",
            ha="left",
        )


def bar_hist(ax, p: np.ndarray, title: str) -> None:
    x = np.arange(3)
    ax.bar(x, p, color=COLORS, width=0.62, edgecolor="white", linewidth=0.7, zorder=2)
    ax.set_xticks(x, NAMES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Born probability")
    ax.set_title(title, pad=8)
    for i, v in enumerate(p):
        ax.text(i, v + 0.03, f"{100 * v:.0f}%", ha="center", va="bottom", fontsize=9, color="#222")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ececec", lw=0.7)


def bar_energy(ax) -> None:
    x = np.arange(3)
    ax.bar(x, E, color=COLORS, width=0.62, edgecolor="white", linewidth=0.7, zorder=2)
    ax.set_xticks(x, NAMES)
    ax.set_ylabel("bitstring energy")
    ax.set_title(r"Landscape: wall $+80$ vs GS–trap gap $4$", pad=8)
    ax.axhline(0.0, color="0.35", lw=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ececec", lw=0.7)


def _fmt_delta(v: float) -> str:
    if abs(v) < 0.005:
        return f"{v:.4f}"
    return f"{v:.2f}"


def delta_bars(ax, values, ylabel, title, ylim: tuple[float, float]) -> None:
    ax.bar([0, 1], values, color=[MUTED, GS], width=0.5, edgecolor="white", linewidth=0.7, zorder=2)
    ax.set_xticks([0, 1], ["A   wall → trap\nwrong bitstring", "B   trap → GS\ncorrect bitstring"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.axhline(0.0, color="0.35", lw=0.7)
    ax.set_ylim(*ylim)
    span = ylim[1] - ylim[0]
    for i, v in enumerate(values):
        color = "#333" if i == 0 else GS
        if abs(v) > 0.22 * span:
            ax.text(i, 0.5 * v, _fmt_delta(v), ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        else:
            ax.text(i, 0.06 * span, _fmt_delta(v), ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ececec", lw=0.7)


def add_table(ax, cells, cols, highlight_col=None) -> None:
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=cols,
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d9dee2")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#eef2f3")
            cell.set_text_props(fontweight="bold", color="#243036")
        elif highlight_col is not None and c == highlight_col:
            cell.set_facecolor("#e7f4f0")
        else:
            cell.set_facecolor("white")


def page1(pdf: PdfPages, eta: float, q05: float, q25: float, w: np.ndarray, preview: Path | None) -> None:
    fig = plt.figure(figsize=(8.5, 11.0), facecolor="white")
    fig_lines(fig, 0.08, 0.965, ["Why Gibbs finds the bitstring when energy does not"], size=14, weight="bold")
    fig_lines(
        fig,
        0.08,
        0.928,
        [
            r"Optimizer cost:  $f=-\ln\langle e^{-\eta E}\rangle$   with sampled_tail  $\eta=\ln 20/(Q_{25}-Q_{05})$.",
            r"Toy knapsack slice, three bitstrings: ground state $E=-12$, a wrong feasible packing (trap) $E=-8$,",
            r"and an infeasible penalty wall $E=+80$.  Start with the histogram mode on the trap and 37% of shots on the wall.",
            rf"sampled_tail on that histogram:  $Q_{{05}}={q05:.2f}$,  $Q_{{25}}={q25:.2f}$,  $\eta={eta:.2f}$.",
            rf"Boltzmann weight vs GS:  trap ${w[1]:.1e}$,  wall ${w[2]:.0e}$.  Hits are read from the mode, not from $\langle H\rangle$.",
        ],
        dy=0.019,
    )

    ax_h = fig.add_axes([0.10, 0.52, 0.36, 0.28])
    bar_hist(ax_h, P0, "Start histogram  (most likely = trap)")
    ax_e = fig.add_axes([0.56, 0.52, 0.36, 0.28])
    bar_energy(ax_e)

    fig_lines(
        fig,
        0.08,
        0.485,
        [
            r"At this $\eta$ the wall (and almost the trap) drops out of $f$.  Energy $\langle H\rangle=\sum p_i E_i$ still feels the $+80$."
        ],
        dy=0.018,
    )
    ax_b = fig.add_axes([0.08, 0.28, 0.84, 0.18])
    add_table(
        ax_b,
        [
            [r"Boltzmann $e^{-\eta(E-E_{\mathrm{GS}})}$", "1", f"{w[1]:.2e}", f"{w[2]:.0e}"],
            [
                r"contribution  $p_i\times$ weight",
                f"{P0[0] * w[0]:.3f}",
                f"{P0[1] * w[1]:.2e}",
                f"{P0[2] * w[2]:.0e}",
            ],
        ],
        ["", "GS", "trap", "wall"],
    )

    fig_lines(
        fig,
        0.08,
        0.235,
        [
            r"Because $\langle H\rangle$ is linear, a shot on the wall costs $+80$ while moving trap $\to$ GS is only a gap of $4$.",
            r"SPSA on energy is dominated by clearing infeasible shots, even if the most likely bitstring stays wrong.",
            r"Gibbs with this $\eta$ is (up to a constant) $-\ln p_{\mathrm{GS}}$ once the wall is exponentially off.",
            r"That is why a histogram can have higher $\langle H\rangle$ and still be preferred: leftover wall shots barely enter $f$.",
        ],
        dy=0.020,
    )
    pdf.savefig(fig)
    if preview is not None:
        fig.savefig(preview / "page1.png", dpi=140)
    plt.close(fig)


def page2(
    pdf: PdfPages,
    e0: float,
    eA: float,
    eB: float,
    g0: float,
    gA: float,
    gB: float,
    eT: float,
    eG: float,
    gT: float,
    gG: float,
    preview: Path | None,
) -> None:
    fig = plt.figure(figsize=(8.5, 11.0), facecolor="white")
    fig_lines(fig, 0.08, 0.965, ["Same-size move, then a higher-energy but correct bitstring"], size=13.5, weight="bold")
    fig_lines(
        fig,
        0.08,
        0.928,
        [
            r"Hold $\eta$ fixed (SPSA does this for both probes of a step).  Move the same probability mass $0.37$ two ways.",
            r"A: dump the wall onto the trap (decoder stays trap).   B: dump trap mass onto the ground state (decoder becomes GS).",
        ],
        dy=0.019,
    )

    ax_de = fig.add_axes([0.10, 0.58, 0.36, 0.26])
    delta_bars(
        ax_de,
        [eA - e0, eB - e0],
        r"change in $\langle H\rangle$",
        "Energy SPSA sees the wall",
        (-38.0, 10.0),
    )
    ax_dg = fig.add_axes([0.56, 0.58, 0.36, 0.26])
    delta_bars(
        ax_dg,
        [gA - g0, gB - g0],
        r"change in $-\ln\langle e^{-\eta E}\rangle$",
        "Gibbs SPSA sees the bitstring",
        (-2.15, 0.45),
    )

    fig_lines(
        fig,
        0.08,
        0.545,
        [
            f"A lowers energy by {e0 - eA:.1f} and the decoded bitstring stays the trap.  "
            f"B lowers energy by only {e0 - eB:.1f}, but Gibbs by {g0 - gB:.2f}.",
            f"B has much higher energy than A ({eB:.1f} vs {eA:.1f}) and is still the Gibbs winner: the mode is now GS.",
        ],
        dy=0.019,
    )

    ax_ht = fig.add_axes([0.10, 0.24, 0.36, 0.24])
    bar_hist(ax_ht, PT, "Trap-clean   (energy winner)")
    ax_hg = fig.add_axes([0.56, 0.24, 0.36, 0.24])
    bar_hist(ax_hg, PG, "GS-leaky   (Gibbs winner)")

    fig_lines(
        fig,
        0.08,
        0.205,
        [
            f"Head-to-head at the same $\\eta$: trap-clean has lower energy ({eT:.2f} vs {eG:.2f}) but the wrong bitstring.",
            r"GS-leaky still has 40% of shots on the wall, so $\langle H\rangle$ looks terrible, yet Gibbs ranks it better because the mode is GS.",
        ],
        dy=0.018,
    )
    ax_cmp = fig.add_axes([0.08, 0.035, 0.84, 0.15])
    add_table(
        ax_cmp,
        [
            ["$p$  (GS, trap, wall)", "0.10,  0.90,  0", "0.52,  0.08,  0.40"],
            [r"$\langle H\rangle$  (lower looks better)", f"{eT:.2f}", f"{eG:.2f}"],
            [r"Gibbs $f$  (lower is better)", f"{gT:.2f}", f"{gG:.2f}"],
            ["most likely bitstring", "trap  (wrong)", "GS  (correct)"],
            ["energy would pick", "yes", "no"],
            ["Gibbs would pick", "no", "yes"],
        ],
        ["", "Trap-clean", "GS-leaky"],
        highlight_col=2,
    )
    pdf.savefig(fig)
    if preview is not None:
        fig.savefig(preview / "page2.png", dpi=140)
    plt.close(fig)


def main() -> None:
    style()
    eta = float(SampledTailEta().initialize(E, P0).eta)
    q05 = weighted_quantile(E, P0, 0.05)
    q25 = weighted_quantile(E, P0, 0.25)
    w = np.exp(-eta * (E - E[0]))
    e0, eA, eB = energy(P0), energy(PA), energy(PB)
    g0, gA, gB = gibbs_objective(P0, E, eta), gibbs_objective(PA, E, eta), gibbs_objective(PB, E, eta)
    eT, eG = energy(PT), energy(PG)
    gT, gG = gibbs_objective(PT, E, eta), gibbs_objective(PG, E, eta)

    preview_dir = os.environ.get("GIBBS_PDF_PREVIEW")
    preview = Path(preview_dir) if preview_dir else None
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        page1(pdf, eta, q05, q25, w, preview)
        page2(pdf, e0, eA, eB, g0, gA, gB, eT, eG, gT, gG, preview)

    print(f"Wrote {OUT}")
    print(f"  eta={eta:.4f}")
    print(f"  A: dE={eA - e0:.3f}  dG={gA - g0:.5f}  <H>={eA:.2f}  mode={mode_name(PA)}")
    print(f"  B: dE={eB - e0:.3f}  dG={gB - g0:.4f}  <H>={eB:.2f}  mode={mode_name(PB)}")
    print(f"  trap-clean <H>={eT:.2f}  Gibbs={gT:.3f}  mode={mode_name(PT)}")
    print(f"  GS-leaky   <H>={eG:.2f}  Gibbs={gG:.3f}  mode={mode_name(PG)}")


if __name__ == "__main__":
    main()
