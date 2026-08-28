"""Adaptive Gibbs inverse temperature from the sampled energy tail.

η is chosen so the probability-weighted 5% and 25% energy quantiles of the
current Born histogram differ by a Gibbs factor of 20, then EMA-smoothed.
The exact ground energy is not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ETA_MIN = 1e-4
ETA_MAX = 50.0
LN20 = math.log(20.0)
DEFAULT_REFRESH_EVERY = 5
EMA_ALPHA = 0.35


def clamp_eta(eta: float) -> tuple[float, bool]:
    x = float(eta)
    if not math.isfinite(x):
        return ETA_MIN, True
    clamped = min(max(x, ETA_MIN), ETA_MAX)
    return clamped, clamped != x


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    total = float(w.sum())
    if v.size == 0 or total <= 0.0:
        return float("nan")
    w = w / total
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    cdf = np.cumsum(w[order])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return float(np.interp(float(q), cdf, v))


def robust_scale(values: np.ndarray, weights: np.ndarray) -> float:
    q05 = weighted_quantile(values, weights, 0.05)
    q25 = weighted_quantile(values, weights, 0.25)
    q75 = weighted_quantile(values, weights, 0.75)
    iqr = max(q75 - q25, 0.0)
    tail = max(q25 - q05, 0.0)
    return max(tail, 0.25 * iqr, 1e-8)


def eta_from_tail(q05: float, q25: float, floor: float) -> tuple[float, str | None]:
    span = float(q25 - q05)
    fallback = None
    if not math.isfinite(span) or span <= 1e-12:
        span = max(float(floor), 1e-8)
        fallback = "degenerate_tail"
    eta, clamped = clamp_eta(LN20 / span)
    if clamped:
        fallback = fallback or "clamped"
    return eta, fallback


@dataclass
class EtaState:
    eta: float
    scale: float | None = None
    fallback: str | None = None
    clamped: bool = False
    step: int = 0


@dataclass
class SampledTailEta:
    """Adaptive η from probability-weighted 5%/25% energy quantiles."""

    refresh_every: int = DEFAULT_REFRESH_EVERY
    ema: float = EMA_ALPHA
    name: str = "sampled_tail"
    eta: float = 1.0
    history: list[dict] = field(default_factory=list)
    n_clamps: int = 0
    n_fallbacks: int = 0
    last: EtaState | None = None

    def _commit(self, state: EtaState) -> EtaState:
        self.eta = float(state.eta)
        self.last = state
        if state.clamped:
            self.n_clamps += 1
        if state.fallback:
            self.n_fallbacks += 1
        self.history.append(
            {
                "step": int(state.step),
                "eta": float(state.eta),
                "scale": None if state.scale is None else float(state.scale),
                "fallback": state.fallback,
                "clamped": bool(state.clamped),
            }
        )
        return state

    def _from_hist(self, energy_tensor: np.ndarray, probs: np.ndarray, step: int) -> EtaState:
        e = np.asarray(energy_tensor, dtype=float)
        p = np.asarray(probs, dtype=float)
        q05 = weighted_quantile(e, p, 0.05)
        q25 = weighted_quantile(e, p, 0.25)
        floor = robust_scale(e, p)
        target, fallback = eta_from_tail(q05, q25, floor)
        if self.last is not None:
            target = (1.0 - self.ema) * self.eta + self.ema * target
        eta, clamped = clamp_eta(target)
        if clamped:
            fallback = fallback or "clamped"
        span = q25 - q05
        return EtaState(
            eta=eta,
            scale=max(span, floor) if math.isfinite(span) else floor,
            fallback=fallback,
            clamped=clamped,
            step=step,
        )

    def initialize(self, energy_tensor: np.ndarray, probs: np.ndarray | None = None) -> EtaState:
        if probs is None:
            probs = np.full(np.asarray(energy_tensor).shape, 1.0)
        return self._commit(self._from_hist(energy_tensor, probs, 0))

    def maybe_update(
        self,
        step: int,
        total_steps: int,
        energy_tensor: np.ndarray,
        probs: np.ndarray,
    ) -> EtaState:
        del total_steps
        k = int(step)
        if k > 1 and (k - 1) % int(self.refresh_every) != 0:
            assert self.last is not None
            return self.last
        return self._commit(self._from_hist(energy_tensor, probs, k))

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "eta": float(self.eta),
            "n_clamps": int(self.n_clamps),
            "n_fallbacks": int(self.n_fallbacks),
            "history": list(self.history),
        }
