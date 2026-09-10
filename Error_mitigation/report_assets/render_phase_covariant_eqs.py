#!/usr/bin/env python3
"""Render phase-covariant teaching equations for the GDR deck.

pdflatex → qlmanage PNG → crop, same recipe as render_slide_eqs.py.
Falls back to matplotlib mathtext if pdflatex is unavailable.

Writes under Error_mitigation/report_assets/figures/:
  phase_covariant_eqs.png        definition (commutes with cavity phase)
  phase_covariant_pos_eqs.png    |+⟩ vs |−⟩, same q = Mp
  phase_covariant_neg_eqs.png    D(α), q+ ≠ q−, no single M
  phase_covariant_pos_hist.png   two identical 3-bin histograms after loss
  phase_covariant_neg_hist.png   two different 3-bin histograms after D(α)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SLIDE_EQ_WIDTH_IN = 11.6
QL_MAX_PX = 3600

FIGURES: dict[str, str] = {
    "phase_covariant_eqs": r"""
\mathcal{E}\bigl(e^{i\phi\hat{n}}\rho\,e^{-i\phi\hat{n}}\bigr)
      = e^{i\phi\hat{n}}\,\mathcal{E}(\rho)\,e^{-i\phi\hat{n}}
""",
    "phase_covariant_pos_eqs": r"""
|+\rangle &= \frac{|0\rangle+|2\rangle}{\sqrt{2}},
      \qquad
      |-\rangle = \frac{|0\rangle-|2\rangle}{\sqrt{2}},
      \qquad
      p_0=p_2=\tfrac12 \\[0.50em]
\langle 0|\rho_+|2\rangle &= -\langle 0|\rho_-|2\rangle,
      \qquad
      q_+=q_-=Mp
""",
    "phase_covariant_neg_eqs": r"""
|+\rangle &= \frac{|0\rangle+|2\rangle}{\sqrt{2}},
      \qquad
      |-\rangle = \frac{|0\rangle-|2\rangle}{\sqrt{2}} \\[0.40em]
D(\alpha) &\approx I + \alpha(a^\dagger - a)
      \qquad(\alpha\in\mathbb{R},\ \text{small}) \\[0.40em]
\langle 1|D(\alpha)|0\rangle &\approx +\alpha
      \quad\text{(create a photon)},
      \qquad
      \langle 1|D(\alpha)|2\rangle \approx -\alpha\sqrt{2}
      \quad\text{(destroy one)} \\[0.45em]
\langle 1|D(\alpha)|+\rangle
      &= \frac{\langle 1|D|0\rangle+\langle 1|D|2\rangle}{\sqrt{2}}
      \approx \frac{\alpha}{\sqrt{2}}(1-\sqrt{2})
      \qquad\text{paths cancel} \\[0.35em]
\langle 1|D(\alpha)|-\rangle
      &= \frac{\langle 1|D|0\rangle-\langle 1|D|2\rangle}{\sqrt{2}}
      \approx \frac{\alpha}{\sqrt{2}}(1+\sqrt{2})
      \qquad\text{paths add}
""",
}

MATHTEXT: dict[str, list[str]] = {
    "phase_covariant_eqs": [
        r"$\mathcal{E}(e^{i\phi\hat{n}}\rho e^{-i\phi\hat{n}})"
        r"=e^{i\phi\hat{n}}\mathcal{E}(\rho)e^{-i\phi\hat{n}}$",
    ],
    "phase_covariant_pos_eqs": [
        r"$|+\rangle=(|0\rangle+|2\rangle)/\sqrt{2},\quad"
        r"|-\rangle=(|0\rangle-|2\rangle)/\sqrt{2},\quad"
        r"p_0=p_2=1/2$",
        r"$\langle 0|\rho_+|2\rangle=-\langle 0|\rho_-|2\rangle,"
        r"\qquad q_+=q_-=Mp$",
    ],
    "phase_covariant_neg_eqs": [
        r"$|+\rangle=(|0\rangle+|2\rangle)/\sqrt{2},\quad"
        r"|-\rangle=(|0\rangle-|2\rangle)/\sqrt{2}$",
        r"$D(\alpha)\approx I+\alpha(a^\dagger-a)\quad(\alpha$ real, small$)$",
        r"$\langle 1|D(\alpha)|0\rangle\approx+\alpha$ (create),\quad"
        r"$\langle 1|D(\alpha)|2\rangle\approx-\alpha\sqrt{2}$ (destroy)",
        r"$\langle 1|D(\alpha)|+\rangle="
        r"(\langle 1|D|0\rangle+\langle 1|D|2\rangle)/\sqrt{2}"
        r"\approx\frac{\alpha}{\sqrt{2}}(1-\sqrt{2})$  cancel",
        r"$\langle 1|D(\alpha)|-\rangle="
        r"(\langle 1|D|0\rangle-\langle 1|D|2\rangle)/\sqrt{2}"
        r"\approx\frac{\alpha}{\sqrt{2}}(1+\sqrt{2})$  add",
    ],
}


def _tex_document(body: str) -> str:
    return rf"""\documentclass[12pt]{{article}}
\usepackage{{fix-cm}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage[
  paperwidth={SLIDE_EQ_WIDTH_IN}in,
  paperheight=5.2in,
  margin=0.22in,
  headheight=0pt,
  headsep=0pt,
  footskip=0pt
]{{geometry}}
\pagestyle{{empty}}
\setlength{{\abovedisplayskip}}{{3pt}}
\setlength{{\belowdisplayskip}}{{3pt}}
\setlength{{\abovedisplayshortskip}}{{2pt}}
\setlength{{\belowdisplayshortskip}}{{2pt}}
\begin{{document}}
\fontsize{{22}}{{28}}\selectfont
\begin{{align*}}
{body.strip()}
\end{{align*}}
\end{{document}}
"""


def _find_pdflatex() -> str | None:
    for candidate in (
        shutil.which("pdflatex"),
        "/Library/TeX/texbin/pdflatex",
        "/usr/texbin/pdflatex",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _crop_white(path: Path, pad: int = 18) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img.convert("L"))
    ink = arr < 248
    if not ink.any():
        raise RuntimeError(f"rasterized page is blank: {path}")
    ys, xs = np.where(ink)
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(img.width, int(xs.max()) + pad + 1),
        min(img.height, int(ys.max()) + pad + 1),
    )
    return img.crop(box)


def _pad_to_slide_width(img: Image.Image, min_width: int) -> Image.Image:
    left, right, tb = 12, 16, 6
    width = max(min_width, img.width + left + right)
    height = img.height + 2 * tb
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(img, (left, tb))
    return canvas


def render_pdflatex(name: str, body: str) -> Image.Image:
    pdflatex = _find_pdflatex()
    if pdflatex is None:
        raise FileNotFoundError("pdflatex not found")
    qlmanage = "/usr/bin/qlmanage"
    if not Path(qlmanage).is_file():
        raise FileNotFoundError("qlmanage not found")

    tex_source = _tex_document(body)
    with tempfile.TemporaryDirectory(prefix=f"gdr_eq_{name}_") as tmp:
        td = Path(tmp)
        tex_path = td / f"{name}.tex"
        tex_path.write_text(tex_source, encoding="utf-8")
        proc = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={td}",
                str(tex_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        pdf = td / f"{name}.pdf"
        if proc.returncode != 0 or not pdf.is_file():
            raise RuntimeError(proc.stdout[-2000:] + proc.stderr[-800:])
        subprocess.run(
            [qlmanage, "-t", "-s", str(QL_MAX_PX), "-o", str(td), str(pdf)],
            check=True,
            capture_output=True,
            text=True,
        )
        thumb = td / f"{name}.pdf.png"
        if not thumb.is_file():
            raise FileNotFoundError(thumb)
        return _pad_to_slide_width(_crop_white(thumb), QL_MAX_PX)


def _crop_white_array(buf: np.ndarray, pad: int) -> Image.Image:
    img = Image.fromarray(buf).convert("RGB")
    arr = np.asarray(img.convert("L"))
    ink = arr < 248
    if not ink.any():
        raise RuntimeError("empty mathtext render")
    ys, xs = np.where(ink)
    return img.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(img.width, int(xs.max()) + pad + 1),
            min(img.height, int(ys.max()) + pad + 1),
        )
    )


def render_mathtext(name: str) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "cm",
            "text.usetex": False,
            "figure.facecolor": "white",
        }
    )
    strips: list[Image.Image] = []
    for line in MATHTEXT[name]:
        fig = plt.figure(figsize=(14.0, 1.05), dpi=240, facecolor="white")
        fig.text(0.0, 0.5, line, fontsize=26, color="black", ha="left", va="center")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        strips.append(_crop_white_array(buf, pad=14))
    gap, left, right = 16, 10, 18
    width = left + max(s.width for s in strips) + right
    height = gap + sum(s.height + gap for s in strips)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = gap
    for strip in strips:
        canvas.paste(strip, (left, y))
        y += strip.height + gap
    return _pad_to_slide_width(canvas, 240 * 12)


def render_one(name: str, body: str) -> tuple[Path, str]:
    (OUT / f"{name}.tex").write_text(_tex_document(body), encoding="utf-8")
    try:
        img = render_pdflatex(name, body)
        engine = "pdflatex"
    except Exception as exc:
        print(f"{name}: pdflatex failed ({exc!r}); falling back to mathtext")
        img = render_mathtext(name)
        engine = "mathtext"
    dest = OUT / f"{name}.png"
    dest2 = OUT / f"{name}_2x.png"
    img.save(dest, "PNG")
    w, h = img.size
    img.resize((w * 2, h * 2), Image.Resampling.LANCZOS).save(dest2, "PNG")
    print(f"{name}: {engine} {img.size} -> {dest}")
    return dest, engine


def _expm(G: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import expm

        return expm(G)
    except ImportError:
        X = np.eye(G.shape[0], dtype=complex)
        term = np.eye(G.shape[0], dtype=complex)
        for k in range(1, 48):
            term = term @ G / k
            X = X + term
        return X


def _displacement_fock(alpha: complex, cutoff: int = 16) -> np.ndarray:
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(1, cutoff):
        a[n - 1, n] = np.sqrt(n)
    return _expm(alpha * a.T.conj() - np.conj(alpha) * a)


def _loss_column(n: int, eta: float, nmax: int = 3) -> np.ndarray:
    from math import comb

    q = np.zeros(nmax + 1)
    for k in range(min(n, nmax) + 1):
        q[k] = comb(n, k) * (eta**k) * ((1.0 - eta) ** (n - k))
    return q


def render_hist_pair(
    name: str,
    q_plus: np.ndarray,
    q_minus: np.ndarray,
    *,
    left_title: str,
    right_title: str,
    ymax: float,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.edgecolor": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )
    bins = np.arange(len(q_plus))
    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 2.35), dpi=220, facecolor="white", sharey=True
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.18)
    for ax, q, lab in ((axes[0], q_plus, left_title), (axes[1], q_minus, right_title)):
        ax.bar(bins, q, color="black", width=0.55)
        ax.set_title(lab, fontsize=14, pad=6, color="black")
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{int(i)}" for i in bins])
        ax.set_xlabel("n", fontsize=12)
        ax.set_ylim(0.0, ymax)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3)
    axes[0].set_ylabel("q", fontsize=12)
    dest = OUT / f"{name}.png"
    dest2 = OUT / f"{name}_2x.png"
    fig.savefig(dest, dpi=220)
    fig.savefig(dest2, dpi=440)
    plt.close(fig)
    print(f"{name}: hist {Image.open(dest).size} -> {dest}")
    return dest


def render_histograms() -> None:
    # Same η = 0.8 as the |2⟩ numerical slide. Superposition column of M.
    eta = 0.8
    q_loss = 0.5 * _loss_column(0, eta, nmax=2) + 0.5 * _loss_column(2, eta, nmax=2)
    render_hist_pair(
        "phase_covariant_pos_hist",
        q_loss,
        q_loss.copy(),
        left_title=r"$|+\rangle$ after loss",
        right_title=r"$|-\rangle$ after loss",
        ymax=0.70,
    )
    # Small real displacement: interference into n = 1 has opposite sign.
    U = _displacement_fock(0.45)
    plus = (U[:, 0] + U[:, 2]) / np.sqrt(2)
    minus = (U[:, 0] - U[:, 2]) / np.sqrt(2)
    q_plus = np.abs(plus[:3]) ** 2
    q_minus = np.abs(minus[:3]) ** 2
    render_hist_pair(
        "phase_covariant_neg_hist",
        q_plus,
        q_minus,
        left_title=r"$|+\rangle$ after $D(\alpha)$",
        right_title=r"$|-\rangle$ after $D(\alpha)$",
        ymax=0.70,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engines = set()
    for name, body in FIGURES.items():
        _, engine = render_one(name, body)
        engines.add(engine)
    render_histograms()
    print("engines:", ", ".join(sorted(engines)))


if __name__ == "__main__":
    main()
