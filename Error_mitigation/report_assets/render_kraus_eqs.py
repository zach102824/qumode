#!/usr/bin/env python3
"""Render the paper amplitude-damping Kraus operators for the GDR deck.

Writes:
  Error_mitigation/report_assets/figures/kraus_operators.png
  Error_mitigation/report_assets/figures/kraus_operators_2x.png
  Error_mitigation/report_assets/figures/kraus_operators.tex
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

# Source of truth: same physics as paper_amplitude_damping_kraus / Eqs. (37)–(39).
TEX_SOURCE = r"""\documentclass[12pt]{article}
\usepackage{fix-cm}
\usepackage{amsmath}
\usepackage[
  paperwidth=11.6in,
  paperheight=4.6in,
  margin=0.28in,
  headheight=0pt,
  headsep=0pt,
  footskip=0pt
]{geometry}
\pagestyle{empty}
\setlength{\abovedisplayskip}{4pt}
\setlength{\belowdisplayskip}{4pt}
\setlength{\abovedisplayshortskip}{2pt}
\setlength{\belowdisplayshortskip}{2pt}
\begin{document}
\fontsize{22}{28}\selectfont
\begin{align*}
\gamma &= 1 - \exp(-\kappa\tau) \\[0.45em]
K_j &= \sqrt{\gamma^j / j!}\; \exp(-\kappa\tau\,\hat{n}/2)\; a^j
      \quad (j=1,\ldots,L-1) \\[0.45em]
K_0 &= \sqrt{I - \sum_{j\ge 1} K_j^\dagger K_j} \\[0.45em]
\rho &\leftarrow \sum_j K_j \rho K_j^\dagger
      \quad \text{on that cavity.}
\end{align*}
\end{document}
"""

MATHTEXT_LINES = [
    r"$\gamma = 1 - \exp(-\kappa\tau)$",
    r"$K_j = \sqrt{\gamma^j / j!}\; \exp(-\kappa\tau\,\hat{n}/2)\; a^j"
    r"\quad (j=1,\ldots,L-1)$",
    r"$K_0 = \sqrt{I - \sum_{j\geq 1} K_j^\dagger K_j}$",
    r"$\rho \leftarrow \sum_j K_j \rho K_j^\dagger$"
    r"$\quad$ on that cavity.",
]


def _find_pdflatex() -> str | None:
    for candidate in (
        shutil.which("pdflatex"),
        "/Library/TeX/texbin/pdflatex",
        "/usr/texbin/pdflatex",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _crop_white(path: Path, pad: int = 22) -> Image.Image:
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


def render_pdflatex() -> Image.Image:
    pdflatex = _find_pdflatex()
    if pdflatex is None:
        raise FileNotFoundError("pdflatex not found")
    qlmanage = "/usr/bin/qlmanage"
    if not Path(qlmanage).is_file():
        raise FileNotFoundError("qlmanage not found")

    with tempfile.TemporaryDirectory(prefix="kraus_eqs_") as tmp:
        td = Path(tmp)
        tex_path = td / "kraus_operators.tex"
        tex_path.write_text(TEX_SOURCE, encoding="utf-8")
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
        pdf = td / "kraus_operators.pdf"
        if proc.returncode != 0 or not pdf.is_file():
            raise RuntimeError(proc.stdout[-1500:] + proc.stderr[-500:])
        subprocess.run(
            [qlmanage, "-t", "-s", "3600", "-o", str(td), str(pdf)],
            check=True,
            capture_output=True,
            text=True,
        )
        thumb = td / "kraus_operators.pdf.png"
        if not thumb.is_file():
            raise FileNotFoundError(thumb)
        return _crop_white(thumb)


def render_mathtext() -> Image.Image:
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
    for line in MATHTEXT_LINES:
        fig = plt.figure(figsize=(14.0, 1.05), dpi=240, facecolor="white")
        fig.text(0.0, 0.5, line, fontsize=28, color="black", ha="left", va="center")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        strips.append(_crop_white_array(buf, pad=16))
    gap, left, right = 22, 10, 20
    width = left + max(s.width for s in strips) + right
    height = gap + sum(s.height + gap for s in strips)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = gap
    for strip in strips:
        canvas.paste(strip, (left, y))
        y += strip.height + gap
    return canvas


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kraus_operators.tex").write_text(TEX_SOURCE, encoding="utf-8")
    try:
        img = render_pdflatex()
        engine = "pdflatex"
    except Exception as exc:
        print(f"pdflatex path failed ({exc!r}); falling back to mathtext")
        img = render_mathtext()
        engine = "mathtext"
    dest = OUT / "kraus_operators.png"
    dest2 = OUT / "kraus_operators_2x.png"
    img.save(dest, "PNG")
    w, h = img.size
    img.resize((w * 2, h * 2), Image.Resampling.LANCZOS).save(dest2, "PNG")
    print(engine)
    print(dest)
    print(dest2)
    print(f"size={img.size}")


if __name__ == "__main__":
    main()
