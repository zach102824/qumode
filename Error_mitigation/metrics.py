"""Histogram comparison metrics against an ideal |q, n, m> distribution."""

from __future__ import annotations

import numpy as np

from qumode_vqe.measurement import energy_from_histogram
from qumode_vqe.vqe import gibbs_objective


def _as_prob(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 0.0, None)
    total = float(q.sum())
    if total <= 0.0:
        return np.full(q.shape, 1.0 / q.size, dtype=float)
    return q / total


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TVD = (1/2) Σ |p − q|. Zero iff p = q."""
    return 0.5 * float(np.abs(_as_prob(p) - _as_prob(q)).sum())


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance √(1 − Σ √(p q)), in [0, 1]."""
    overlap = float(np.sqrt(_as_prob(p) * _as_prob(q)).sum())
    return float(np.sqrt(max(1.0 - overlap, 0.0)))


def compare_histograms(
    p_mit: np.ndarray | None,
    p_ideal: np.ndarray,
    energy_tensor: np.ndarray,
    ground_qnm: tuple[int, int, int],
    *,
    energy_mit: float | None = None,
    gibbs_eta: float = 1.0,
) -> dict:
    """Metrics of a mitigated histogram (or scalar energy) vs the ideal one."""
    p_id = _as_prob(p_ideal)
    e_ideal = energy_from_histogram(p_id, energy_tensor)
    gq, gn, gm = (int(v) for v in ground_qnm)
    pgs_ideal = float(p_id[gq, gn, gm])
    gibbs_ideal = float(gibbs_objective(p_id, energy_tensor, gibbs_eta))
    out: dict = {
        "tvd": None,
        "hellinger": None,
        "energy_ideal": e_ideal,
        "energy_mit": None,
        "dE": None,
        "p_gs_ideal": pgs_ideal,
        "p_gs_mit": None,
        "dPgs": None,
        "gibbs_ideal": gibbs_ideal,
        "gibbs_mit": None,
        "dGibbs": None,
        "has_histogram": p_mit is not None,
    }
    if p_mit is not None:
        p = _as_prob(p_mit)
        e_mit = energy_from_histogram(p, energy_tensor) if energy_mit is None else float(energy_mit)
        pgs = float(p[gq, gn, gm])
        g_mit = float(gibbs_objective(p, energy_tensor, gibbs_eta))
        out.update(
            {
                "tvd": total_variation(p, p_id),
                "hellinger": hellinger(p, p_id),
                "energy_mit": e_mit,
                "dE": abs(e_mit - e_ideal),
                "p_gs_mit": pgs,
                "dPgs": abs(pgs - pgs_ideal),
                "gibbs_mit": g_mit,
                "dGibbs": abs(g_mit - gibbs_ideal),
            }
        )
        return out
    if energy_mit is None:
        return out
    e_mit = float(energy_mit)
    out["energy_mit"] = e_mit
    out["dE"] = abs(e_mit - e_ideal)
    return out
