#!/usr/bin/env python3
"""Render the classical-reshuffling numerical example for the GDR deck.

Writes:
  Error_mitigation/report_assets/figures/reshuffle_eqs.png
  Error_mitigation/report_assets/figures/reshuffle_eqs_2x.png
  Error_mitigation/report_assets/figures/reshuffle_eqs.tex
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

TEX_BODY = r"""
&(1)\quad
  |2\rangle:\ 
  p=(0,\,0,\,1,\,0,\ldots)^{\mathsf T}
  \\[0.50em]
&(2)\quad
  q_k=\binom{2}{k}\eta^{k}(1-\eta)^{2-k},
  \quad \eta=0.8,\ k=0,1,2,
  \quad q_{k>2}=0
  \\[0.50em]
&(3)\quad
  q=Mp=M_{\cdot,2}
  =(0.04,\,0.32,\,0.64,\,0,\ldots)^{\mathsf T}
  \\[0.50em]
&(4)\quad
  P(2\to 2)=\eta^{2}=0.64,\quad
  P(2\to 1)=2\eta(1-\eta)=0.32,\quad
  P(2\to 0)=(1-\eta)^{2}=0.04
"""

TEX_SOURCE = rf"""\documentclass[12pt]{{article}}
\usepackage{{fix-cm}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage[
  paperwidth=11.6in,
  paperheight=4.8in,
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
{TEX_BODY.strip()}
\end{{align*}}
\end{{document}}
"""

MATHTEXT_LINES = [
    r"$(1)\quad |2\rangle:\ p=(0,\,0,\,1,\,0,\ldots)^{\mathrm{T}}$",
    r"$(2)\quad q_k=\binom{2}{k}\eta^{k}(1-\eta)^{2-k},"
    r"\quad \eta=0.8,\ k=0,1,2,\quad q_{k>2}=0$",
    r"$(3)\quad q=Mp=M_{\cdot,2}"
    r"=(0.04,\,0.32,\,0.64,\,0,\ldots)^{\mathrm{T}}$",
    r"$(4)\quad P(2\to 2)=\eta^{2}=0.64,\quad "
    r"P(2\to 1)=2\eta(1-\eta)=0.32,\quad "
    r"P(2\to 0)=(1-\eta)^{2}=0.04$",
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


def _crop_white(path: Path, pad: int = 18) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img.convert("L"))
    ink = arr < 248
    if not ink.any():
        raise RuntimeError(f"rasterized page is blank: {path}")
    ys, xs = np.where(ink)
    return img.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(img.width, int(xs.max()) + pad + 1),
            min(img.height, int(ys.max()) + pad + 1),
        )
    )


def _pad_banner(img: Image.Image, *, banner_aspect: float = 12.0 / 2.45) -> Image.Image:
    left, right, tb = 10, 22, 8
    width = img.width + left + right
    height = img.height + 2 * tb
    min_width = int(round(height * banner_aspect))
    canvas = Image.new("RGB", (max(width, min_width), height), (255, 255, 255))
    canvas.paste(img, (left, tb))
    return canvas


def render_pdflatex() -> Image.Image:
    pdflatex = _find_pdflatex()
    if pdflatex is None:
        raise FileNotFoundError("pdflatex not found")
    qlmanage = "/usr/bin/qlmanage"
    if not Path(qlmanage).is_file():
        raise FileNotFoundError("qlmanage not found")

    with tempfile.TemporaryDirectory(prefix="reshuffle_eqs_") as tmp:
        td = Path(tmp)
        tex_path = td / "reshuffle_eqs.tex"
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
        pdf = td / "reshuffle_eqs.pdf"
        if proc.returncode != 0 or not pdf.is_file():
            raise RuntimeError(proc.stdout[-1500:] + proc.stderr[-500:])
        subprocess.run(
            [qlmanage, "-t", "-s", "3600", "-o", str(td), str(pdf)],
            check=True,
            capture_output=True,
            text=True,
        )
        thumb = td / "reshuffle_eqs.pdf.png"
        if not thumb.is_file():
            raise FileNotFoundError(thumb)
        return _pad_banner(_crop_white(thumb))


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


def render_mathtext() -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Calibri",
            "font.size": 26,
            "mathtext.fontset": "cm",
            "text.color": "black",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "text.usetex": False,
        }
    )
    strips: list[Image.Image] = []
    for line in MATHTEXT_LINES:
        fig = plt.figure(figsize=(16.0, 1.05), dpi=240, facecolor="white")
        fig.text(
            0.0,
            0.5,
            line,
            fontsize=29,
            color="black",
            ha="left",
            va="center",
            math_fontfamily="cm",
        )
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        strips.append(_crop_white_array(buf, pad=10))
    gap, left, right = 18, 8, 24
    width = left + max(s.width for s in strips) + right
    height = gap + sum(s.height + gap for s in strips)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = gap
    for strip in strips:
        canvas.paste(strip, (left, y))
        y += strip.height + gap
    return _pad_banner(canvas)


def main() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reshuffle_eqs.tex").write_text(TEX_SOURCE, encoding="utf-8")
    try:
        img = render_pdflatex()
        engine = "pdflatex"
    except Exception as exc:
        print(f"pdflatex path failed ({exc!r}); falling back to mathtext")
        img = render_mathtext()
        engine = "mathtext"
    dest = OUT / "reshuffle_eqs.png"
    dest2 = OUT / "reshuffle_eqs_2x.png"
    img.save(dest, "PNG")
    w, h = img.size
    img.resize((w * 2, h * 2), Image.Resampling.LANCZOS).save(dest2, "PNG")
    print(engine)
    print(dest)
    print(dest2)
    print(f"size={img.size}")
    return dest


if __name__ == "__main__":
    main()
