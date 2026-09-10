#!/usr/bin/env python3
"""Render leftover GDR-deck equations as LaTeX figures.

Same recipe as render_kraus_eqs.py: pdflatex → qlmanage PNG → crop.
Falls back to matplotlib mathtext if pdflatex is unavailable.

Writes PNG (+ optional .tex) under Error_mitigation/report_assets/figures/.
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

# Display width target (inches) used to pad cropped ink so fig_h() stays sane.
SLIDE_EQ_WIDTH_IN = 11.6
QL_MAX_PX = 3600
# Half-slide columns: pad less so 5.8in placement keeps design-size type.
COLUMN_MIN_PX = 1900
COLUMN_FIGS = {
    "ecd_gaussian_eqs",
    "snap_gaussian_eqs",
    "birth_death_eqs",
    "tvd_eqs",
}

# name -> align* body (no surrounding document)
FIGURES: dict[str, str] = {
    "lindblad_intro_eqs": r"""
\mathcal{E} &= \exp(\mathcal{L}\tau)
      \qquad\text{$H=0$ Liouvillian on the local space} \\[0.55em]
\rho &\leftarrow \sum_j K_j\rho K_j^\dagger
""",
    "lindblad_cavity_eqs": r"""
c_\downarrow &= \sqrt{\kappa(n_{th}+1)}\,a
      \qquad\text{photon loss, }n\to n-1 \\[0.45em]
c_\uparrow &= \sqrt{\kappa n_{th}}\,a^\dagger
      \qquad\text{thermal heating, }n\to n+1
      \quad(n_{th}=0\text{ off}) \\[0.45em]
c_\phi &= \sqrt{\kappa_\phi}\,\hat{n}
      \qquad\text{number dephasing, no hops in }n \\[0.45em]
\kappa\cdot\tau &= \kappa\tau
      \qquad\text{for one application}
""",
    "lindblad_transmon_eqs": r"""
c_{T_1} &= \sqrt{1/T_1}\,\sigma_-
      \qquad\text{amplitude damping, }|1\rangle\to|0\rangle \\[0.45em]
c_{\phi q} &= \sqrt{1/(2T_\phi)}\,\sigma_z \\[0.45em]
\frac{1}{T_2} &= \frac{1}{2T_1}+\frac{1}{T_\phi}
""",
    "circuit_noise_apply_eqs": r"""
\rho &\leftarrow |\psi_0\rangle\langle\psi_0|
      \quad\text{then, once per application:}
      \quad \rho\leftarrow U\rho U^\dagger
      \quad\text{then}\quad \rho\leftarrow\mathcal{E}(\rho) \\[0.55em]
p(q,n,m) &= \langle qnm|\rho|qnm\rangle
""",
    "circuit_noise_kraus_eqs": r"""
\rho &\leftarrow \sum_j K_j\rho K_j^\dagger
      \qquad\text{local $\{K_j\}$; identity on the other registers}
""",
    "kerr_eqs": r"""
U &= \exp\bigl(-i K\tau\,\hat{n}(\hat{n}-1)/2\bigr)
      \qquad K=2\pi\times 500\,\mathrm{Hz}
""",
    "control_error_eqs": r"""
\beta &\to 1.01\,|\beta|\,e^{i\arg\beta}
      \qquad
      \theta\to 1.01\,\theta
      \qquad
      \rho\leftarrow U_{\mathrm{wrong}}\rho\,U_{\mathrm{wrong}}^\dagger
""",
    "readout_pobs_eqs": r"""
p_{\mathrm{obs}}(q',n',m')
      &= \sum_{q,n,m}
      C_q[q',q]\,C_1[n',n]\,C_2[m',m]\,p(q,n,m)
""",
    "zne_scale_eqs": r"""
\kappa\tau &\in\{0.003,\,0.03,\,0.1\}
      \qquad
      \text{ZNE scales }\kappa\tau,\;\kappa_\phi,\;1/T_1^q,\;1/T_2^q
""",
    "transfer_matrix_eqs": r"""
q_{\mathrm{obs}} &\approx Mp
      \qquad
      M=C_q\otimes C_1\otimes C_2
""",
    "gdr_run_eqs": r"""
M &= C_q\otimes C_1\otimes C_2
      \qquad
      q_{\mathrm{obs}}\approx Mp
      \qquad
      p\ge 0,\;\sum p=1
""",
    "birth_death_eqs": r"""
\Gamma_\downarrow(n) &= (-\ln\eta)\,(n_{th}+1)\,n \\[0.45em]
\Gamma_\uparrow(n) &= (-\ln\eta)\,n_{th}\,(n+1)
""",
    "tvd_eqs": r"""
\mathrm{TVD}(p,q) &= \tfrac12\sum_i |p_i-q_i|
""",
    "ecd_gaussian_eqs": r"""
\mathrm{ECD} &\to D(\pm\beta/2)\ \text{plus a bit flip} \\[0.45em]
|\psi\rangle &= |q\rangle\,|\alpha_1\rangle\,|\alpha_2\rangle
""",
    "snap_gaussian_eqs": r"""
\theta_n &\approx bn \qquad(\theta_0=0) \\[0.45em]
\exp(ib\hat{n}) &\quad\text{is a phase rotation (Gaussian)}
""",
    "kraus_caption_eqs": r"""
K_0 &\neq \exp(-\kappa\tau\,\hat{n}/2)
      \qquad\text{(infinite-dimension no-jump)} \\[0.45em]
\sum_j K_j^\dagger K_j &= I
      \qquad
      \eta=e^{-\kappa\tau}
""",
    "reshuffle_coherence_eqs": r"""
|\psi\rangle &= \frac{|0\rangle+|2\rangle}{\sqrt{2}}
      \qquad
      p_0=p_2=\tfrac12
      \qquad
      \langle 0|\rho|2\rangle\text{ off-diagonal, never measured}
""",
}

# matplotlib fallback, one mathtext line per display row
MATHTEXT: dict[str, list[str]] = {
    "lindblad_intro_eqs": [
        r"$\mathcal{E} = \exp(\mathcal{L}\tau)"
        r"\quad H=0$ Liouvillian on the local space",
        r"$\rho \leftarrow \sum_j K_j\rho K_j^\dagger$",
    ],
    "lindblad_cavity_eqs": [
        r"$c_\downarrow = \sqrt{\kappa(n_{th}+1)}\,a"
        r"\quad$ photon loss, $n\to n-1$",
        r"$c_\uparrow = \sqrt{\kappa n_{th}}\,a^\dagger"
        r"\quad$ thermal heating, $n\to n+1$ ($n_{th}=0$ off)",
        r"$c_\phi = \sqrt{\kappa_\phi}\,\hat{n}"
        r"\quad$ number dephasing, no hops in $n$",
        r"$\kappa\cdot\tau = \kappa\tau$\quad for one application",
    ],
    "lindblad_transmon_eqs": [
        r"$c_{T_1} = \sqrt{1/T_1}\,\sigma_-$"
        r"$\quad$ amplitude damping, $|1\rangle\to|0\rangle$",
        r"$c_{\phi q} = \sqrt{1/(2T_\phi)}\,\sigma_z$",
        r"$\frac{1}{T_2} = \frac{1}{2T_1}+\frac{1}{T_\phi}$",
    ],
    "circuit_noise_apply_eqs": [
        r"$\rho \leftarrow |\psi_0\rangle\langle\psi_0|$"
        r" then, once per application: "
        r"$\rho\leftarrow U\rho U^\dagger$"
        r" then $\rho\leftarrow\mathcal{E}(\rho)$",
        r"$p(q,n,m) = \langle qnm|\rho|qnm\rangle$",
    ],
    "circuit_noise_kraus_eqs": [
        r"$\rho \leftarrow \sum_j K_j\rho K_j^\dagger$"
        r"$\quad$ local $\{K_j\}$; identity on the other registers",
    ],
    "kerr_eqs": [
        r"$U = \exp(-i K\tau\,\hat{n}(\hat{n}-1)/2)"
        r"\qquad K=2\pi\times 500\,\mathrm{Hz}$",
    ],
    "control_error_eqs": [
        r"$\beta \to 1.01\,|\beta|\,e^{i\arg\beta}"
        r"\qquad \theta\to 1.01\,\theta"
        r"\qquad \rho\leftarrow U_{\mathrm{wrong}}\rho\,U_{\mathrm{wrong}}^\dagger$",
    ],
    "readout_pobs_eqs": [
        r"$p_{\mathrm{obs}}(q',n',m')"
        r"= \sum_{q,n,m} C_q[q',q]\,C_1[n',n]\,C_2[m',m]\,p(q,n,m)$",
    ],
    "zne_scale_eqs": [
        r"$\kappa\tau \in\{0.003,\,0.03,\,0.1\}$"
        r"$\quad$ ZNE scales $\kappa\tau,\;\kappa_\phi,\;1/T_1^q,\;1/T_2^q$",
    ],
    "transfer_matrix_eqs": [
        r"$q_{\mathrm{obs}} \approx Mp"
        r"\qquad M=C_q\otimes C_1\otimes C_2$",
    ],
    "gdr_run_eqs": [
        r"$M = C_q\otimes C_1\otimes C_2"
        r"\qquad q_{\mathrm{obs}}\approx Mp"
        r"\qquad p\ge 0,\;\sum p=1$",
    ],
    "birth_death_eqs": [
        r"$\Gamma_\downarrow(n) = (-\ln\eta)\,(n_{th}+1)\,n$",
        r"$\Gamma_\uparrow(n) = (-\ln\eta)\,n_{th}\,(n+1)$",
    ],
    "tvd_eqs": [
        r"$\mathrm{TVD}(p,q) = \frac{1}{2}\sum_i |p_i-q_i|$",
    ],
    "ecd_gaussian_eqs": [
        r"$\mathrm{ECD} \to D(\pm\beta/2)$ plus a bit flip",
        r"$|\psi\rangle = |q\rangle\,|\alpha_1\rangle\,|\alpha_2\rangle$",
    ],
    "snap_gaussian_eqs": [
        r"$\theta_n \approx bn \quad (\theta_0=0)$",
        r"$\exp(ib\hat{n})$ is a phase rotation (Gaussian)",
    ],
    "kraus_caption_eqs": [
        r"$K_0 \neq \exp(-\kappa\tau\,\hat{n}/2)$"
        r"$\quad$ (infinite-dimension no-jump)",
        r"$\sum_j K_j^\dagger K_j = I"
        r"\qquad \eta=e^{-\kappa\tau}$",
    ],
    "reshuffle_coherence_eqs": [
        r"$|\psi\rangle = (|0\rangle+|2\rangle)/\sqrt{2}"
        r"\qquad p_0=p_2=1/2"
        r"\qquad \langle 0|\rho|2\rangle$ off-diagonal, never measured",
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
    """Left-align ink on a white banner so a 12in placement keeps natural height."""
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
        cropped = _crop_white(thumb)
        min_w = COLUMN_MIN_PX if name in COLUMN_FIGS else QL_MAX_PX
        return _pad_to_slide_width(cropped, min_w)


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
    min_w = COLUMN_MIN_PX if name in COLUMN_FIGS else 240 * 12
    return _pad_to_slide_width(canvas, min_w)


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engines = set()
    for name, body in FIGURES.items():
        _, engine = render_one(name, body)
        engines.add(engine)
    print("engines:", ", ".join(sorted(engines)))


if __name__ == "__main__":
    main()
