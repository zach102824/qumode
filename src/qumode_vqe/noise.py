"""Error-model configuration and local generators for the hybrid device.

Photon loss is implemented in exactly one of two mutually exclusive ways:

* ``LossModel.PAPER_KRAUS`` — truncated amplitude-damping Kraus operators
  of Eqs. (37)–(40), matching the published noisy notebook.
* ``LossModel.LINDBLAD`` — unified local Lindblad generators including
  thermal excitation and dephasing.

Measurement errors are *not* applied here; they act on the final histogram
in :mod:`qumode_vqe.measurement`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import qutip as qt

from .channels import (
    apply_full_unitary,
    apply_kraus_local,
    apply_local_unitary,
    cross_kerr_phases,
    destroy_matrix,
    dispersive_phases,
    idle_kerr_unitary,
    lindblad_kraus,
    num_matrix,
    paper_amplitude_damping_kraus,
)


class TimingMode(str, Enum):
    PER_UER_LAYER = "per_uer_layer"
    PER_ECD_PAIR = "per_ecd_pair"


class LossModel(str, Enum):
    NONE = "none"
    PAPER_KRAUS = "paper_kraus"
    LINDBLAD = "lindblad"


# Eickbusch et al., Nat. Phys. 18, 1464 (2022): cavity T1 = 436 µs.
# Conservative ECD-block duration used in the paper's Appendix: 0.65 µs.
DEFAULT_T1_CAV = 436e-6
DEFAULT_TAU_ECD = 0.65e-6
# Typical transmon coherence on the same platform (tens of microseconds).
DEFAULT_T1_Q = 50e-6
DEFAULT_T2_Q = 30e-6


@dataclass
class NoiseConfig:
    timing: TimingMode = TimingMode.PER_UER_LAYER
    loss_model: LossModel = LossModel.NONE
    # If set, this dimensionless κτ is used per noise application (paper convention).
    kappa_tau: float | None = None
    nth_cav: float = 0.0
    kappa_phi: float = 0.0  # cavity number-dephasing rate (1/s); ignored by PAPER_KRAUS
    t1_cav: float = DEFAULT_T1_CAV
    tau_ecd: float = DEFAULT_TAU_ECD
    t1_q: float = DEFAULT_T1_Q
    t2_q: float = DEFAULT_T2_Q
    nth_q: float = 0.0
    enable_transmon: bool = False
    rotation_rel_error: float = 0.0
    ecd_amp_rel_error: float = 0.0
    ecd_phase_error: float = 0.0
    kerr: float = 0.0  # self-Kerr, rad/s
    cross_kerr: float = 0.0
    chi_dispersive: float = 0.0
    transmon_leakage: bool = False
    dims: tuple[int, int, int] = (2, 8, 8)

    def __post_init__(self) -> None:
        if self.transmon_leakage:
            raise NotImplementedError(
                "Transmon leakage requires a qutrit (192-dimensional) simulation "
                "and is left as an optional extension."
            )
        if self.loss_model is LossModel.PAPER_KRAUS:
            if self.nth_cav != 0.0 or self.kappa_phi != 0.0:
                raise ValueError(
                    "PAPER_KRAUS is pure loss. Use LossModel.LINDBLAD for thermal "
                    "occupation or cavity dephasing (do not combine both loss paths)."
                )
        if self.t2_q > 2.0 * self.t1_q + 1e-15:
            raise ValueError("Physical T2 cannot exceed 2 T1.")

    @property
    def kappa_cav(self) -> float:
        return 1.0 / self.t1_cav

    @property
    def tau_application(self) -> float:
        """Physical idle time associated with one noise-channel application."""
        if self.timing is TimingMode.PER_ECD_PAIR:
            return self.tau_ecd
        return 2.0 * self.tau_ecd

    @property
    def n_applications(self) -> int:
        """Number of noise applications for a depth-5 circuit is set by the simulator."""
        return 1

    def kappa_tau_used(self) -> float:
        if self.kappa_tau is not None:
            return float(self.kappa_tau)
        return self.kappa_cav * self.tau_application

    def t_phi_q(self) -> float:
        """Pure-dephasing time from 1/T2 = 1/(2 T1) + 1/Tφ."""
        inv = 1.0 / self.t2_q - 1.0 / (2.0 * self.t1_q)
        if inv <= 0.0:
            return np.inf
        return 1.0 / inv

    def cumulative_kappa_t(self, ndepth: int) -> float:
        n_app = ndepth if self.timing is TimingMode.PER_UER_LAYER else 2 * ndepth
        return n_app * self.kappa_tau_used()

    def is_identity(self) -> bool:
        if self.loss_model is LossModel.NONE and not self.enable_transmon:
            coherent = (
                abs(self.kerr)
                + abs(self.cross_kerr)
                + abs(self.chi_dispersive)
                + abs(self.rotation_rel_error)
                + abs(self.ecd_amp_rel_error)
                + abs(self.ecd_phase_error)
            )
            return coherent == 0.0
        if self.loss_model is LossModel.PAPER_KRAUS and self.kappa_tau_used() == 0.0 and not self.enable_transmon:
            return abs(self.kerr) + abs(self.cross_kerr) + abs(self.chi_dispersive) == 0.0
        return False


def paper_loss_config(
    kappa_tau: float,
    timing: TimingMode = TimingMode.PER_UER_LAYER,
    dims: tuple[int, int, int] = (2, 8, 8),
) -> NoiseConfig:
    return NoiseConfig(
        timing=timing,
        loss_model=LossModel.PAPER_KRAUS,
        kappa_tau=float(kappa_tau),
        enable_transmon=False,
        dims=dims,
    )


def realistic_lindblad_config(
    timing: TimingMode = TimingMode.PER_ECD_PAIR,
    **kwargs,
) -> NoiseConfig:
    """Default rates from the hardware cited in the paper, Lindblad path."""
    cfg = NoiseConfig(
        timing=timing,
        loss_model=LossModel.LINDBLAD,
        enable_transmon=True,
        nth_cav=kwargs.pop("nth_cav", 0.01),
        kappa_phi=kwargs.pop("kappa_phi", 0.0),
    )
    return replace(cfg, **kwargs) if kwargs else cfg


def comprehensive_config(
    timing: TimingMode = TimingMode.PER_ECD_PAIR,
    *,
    kappa_tau: float | None = None,
    **kwargs,
) -> NoiseConfig:
    """Combined cQED error model (the project's 'comprehensive' / typical-device case).

    Includes, after each ECD–rotation pair:

    * Lindblad cavity photon loss with thermal occupation ``nth_cav=0.01``
    * transmon amplitude and phase damping (T1 = 50 µs, T2 = 30 µs)
    * cavity self-Kerr (500 Hz)
    * 1% static ECD-amplitude and qubit-rotation errors

    Default cavity κτ is ``τ_ECD / T1_cav ≈ 0.0015`` per ECD pair (10 applications
    at depth 5). Pass ``kappa_tau`` to override that per-application value.
    """
    extras = {
        "rotation_rel_error": 0.01,
        "ecd_amp_rel_error": 0.01,
        "kerr": 2.0 * np.pi * 500.0,
    }
    if kappa_tau is not None:
        extras["kappa_tau"] = float(kappa_tau)
    extras.update(kwargs)
    return realistic_lindblad_config(timing, **extras)


def noise_as_dict(cfg: NoiseConfig) -> dict:
    """JSON-serializable snapshot of the channels that are actually on."""
    return {
        "timing": cfg.timing.value,
        "loss_model": cfg.loss_model.value,
        "kappa_tau_used": cfg.kappa_tau_used(),
        "tau_application_s": cfg.tau_application,
        "cumulative_kappa_t_nd5": cfg.cumulative_kappa_t(5),
        "nth_cav": cfg.nth_cav,
        "kappa_phi": cfg.kappa_phi,
        "t1_cav": cfg.t1_cav,
        "tau_ecd": cfg.tau_ecd,
        "enable_transmon": cfg.enable_transmon,
        "t1_q": cfg.t1_q,
        "t2_q": cfg.t2_q,
        "nth_q": cfg.nth_q,
        "rotation_rel_error": cfg.rotation_rel_error,
        "ecd_amp_rel_error": cfg.ecd_amp_rel_error,
        "ecd_phase_error": cfg.ecd_phase_error,
        "kerr": cfg.kerr,
        "cross_kerr": cfg.cross_kerr,
        "chi_dispersive": cfg.chi_dispersive,
    }


class ChannelCache:
    """Precomputed local Kraus sets for a fixed NoiseConfig."""

    def __init__(self, config: NoiseConfig):
        self.config = config
        self.dims = tuple(int(d) for d in config.dims)
        self.kraus_cav1: list[np.ndarray] = []
        self.kraus_cav2: list[np.ndarray] = []
        self.kraus_qubit: list[np.ndarray] = []
        self._build()

    def _build(self) -> None:
        l1, l2 = self.dims[1], self.dims[2]
        tau = self.config.tau_application
        kt = self.config.kappa_tau_used()

        if self.config.loss_model is LossModel.PAPER_KRAUS:
            self.kraus_cav1 = paper_amplitude_damping_kraus(kt, l1)
            self.kraus_cav2 = paper_amplitude_damping_kraus(kt, l2)
        elif self.config.loss_model is LossModel.LINDBLAD:
            kappa = kt / tau if tau > 0 else self.config.kappa_cav
            self.kraus_cav1 = _cavity_lindblad_kraus(
                kappa, self.config.nth_cav, self.config.kappa_phi, tau, l1
            )
            self.kraus_cav2 = _cavity_lindblad_kraus(
                kappa, self.config.nth_cav, self.config.kappa_phi, tau, l2
            )

        if self.config.enable_transmon:
            self.kraus_qubit = _transmon_lindblad_kraus(
                self.config.t1_q, self.config.t2_q, self.config.nth_q, tau
            )

        self.kerr_u1 = idle_kerr_unitary(self.config.kerr, tau, l1)
        self.kerr_u2 = idle_kerr_unitary(self.config.kerr, tau, l2)
        self.cross_u = (
            cross_kerr_phases(self.config.cross_kerr, tau, self.dims)
            if self.config.cross_kerr
            else None
        )
        self.disp_u = (
            dispersive_phases(self.config.chi_dispersive, tau, self.dims)
            if self.config.chi_dispersive
            else None
        )

    def apply(self, rho: np.ndarray) -> np.ndarray:
        out = rho
        if self.config.kerr:
            out = apply_local_unitary(out, self.kerr_u1, 1, self.dims)
            out = apply_local_unitary(out, self.kerr_u2, 2, self.dims)
        if self.cross_u is not None:
            out = apply_full_unitary(out, self.cross_u)
        if self.disp_u is not None:
            out = apply_full_unitary(out, self.disp_u)
        if self.kraus_qubit:
            out = apply_kraus_local(out, self.kraus_qubit, 0, self.dims)
        if self.kraus_cav1:
            out = apply_kraus_local(out, self.kraus_cav1, 1, self.dims)
        if self.kraus_cav2:
            out = apply_kraus_local(out, self.kraus_cav2, 2, self.dims)
        return out


def _cavity_lindblad_kraus(
    kappa: float,
    nth: float,
    kappa_phi: float,
    tau: float,
    n_fock: int,
) -> list[np.ndarray]:
    c_ops: list[np.ndarray] = []
    a = destroy_matrix(n_fock)
    adag = a.conj().T
    n_op = num_matrix(n_fock)
    if kappa > 0.0:
        c_ops.append(np.sqrt(kappa * (nth + 1.0)) * a)
        if nth > 0.0:
            c_ops.append(np.sqrt(kappa * nth) * adag)
    if kappa_phi > 0.0:
        c_ops.append(np.sqrt(kappa_phi) * n_op)
    if not c_ops:
        return [np.eye(n_fock, dtype=complex)]
    return lindblad_kraus(c_ops, tau, n_fock)


def _transmon_lindblad_kraus(t1: float, t2: float, nth: float, tau: float) -> list[np.ndarray]:
    # QuTiP / standard: |0> = ground, σ- = |0><1|.
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sp = sm.conj().T
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    gamma1 = 1.0 / t1
    c_ops = [
        np.sqrt(gamma1 * (nth + 1.0)) * sm,
    ]
    if nth > 0.0:
        c_ops.append(np.sqrt(gamma1 * nth) * sp)
    inv_phi = 1.0 / t2 - 1.0 / (2.0 * t1)
    if inv_phi > 0.0:
        t_phi = 1.0 / inv_phi
        # c = sqrt(1/(2 Tφ)) σz  ⇒ coherence decay extra 1/Tφ
        c_ops.append(np.sqrt(1.0 / (2.0 * t_phi)) * sz)
    return lindblad_kraus(c_ops, tau, 2)


def transmon_c_ops_qutip(t1: float, t2: float, nth: float = 0.0) -> list[qt.Qobj]:
    sm = qt.sigmam()
    gamma1 = 1.0 / t1
    ops = [np.sqrt(gamma1 * (nth + 1.0)) * sm]
    if nth > 0.0:
        ops.append(np.sqrt(gamma1 * nth) * qt.sigmap())
    inv_phi = 1.0 / t2 - 1.0 / (2.0 * t1)
    if inv_phi > 0.0:
        t_phi = 1.0 / inv_phi
        ops.append(np.sqrt(1.0 / (2.0 * t_phi)) * qt.sigmaz())
    return ops
