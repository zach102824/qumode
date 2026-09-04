"""Circuit-noise families, readout levels, and ZNE scaling for GDR tests.

A noisy run is the product of three independent pieces:

1. Circuit noise (``NoiseConfig``) applied between ECD/SNAP layers.
2. Readout confusion (``MeasurementConfig``) applied only to the final
   ``|q, n, m>`` histogram, never to the density matrix.
3. Finite-shot multinomial sampling of that histogram.

ZNE via ``scale_noise`` stretches circuit rates. It does **not** scale
readout: idle-time folding does not change the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from qumode_vqe.measurement import (
    MeasurementConfig,
    is_column_stochastic,
    nearest_neighbor_fock_confusion,
    qubit_bitflip_confusion,
)
from qumode_vqe.noise import (
    LossModel,
    NoiseConfig,
    TimingMode,
    comprehensive_config,
    paper_loss_config,
)

NFOCK = 8
DEFAULT_KAPPA_TAU = (0.003, 0.03, 0.1)
SMOKE_KAPPA_TAU = (0.003,)

# Asymmetric qubit readout: |1> → |0> relaxation during the integration
# window dominates |0> → |1>. Fock n → n±1 from bitwise photon-number
# misassignment (Curtis et al., PRA 103, 023705). p_nn=0.03 is chosen so
# readout-only raw TVD is the same order as loss-only raw TVD at κτ=0.003.
READOUT_SPECS: dict[str, dict[str, float]] = {
    "ideal": {"p01": 0.0, "p10": 0.0, "p_nn": 0.0},
    "readout_realistic": {"p01": 0.01, "p10": 0.03, "p_nn": 0.03},
    "readout_strong": {"p01": 0.03, "p10": 0.08, "p_nn": 0.10},
}

CIRCUIT_FAMILIES = ("loss", "loss_thermal_dephasing", "comprehensive")


@dataclass(frozen=True)
class ReadoutSpec:
    level: str
    p01: float
    p10: float
    p_nn: float
    n_shots: int
    seed: int | None = None

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "p01": float(self.p01),
            "p10": float(self.p10),
            "p_nn": float(self.p_nn),
            "n_shots": int(self.n_shots),
            "seed": None if self.seed is None else int(self.seed),
        }


def circuit_noise(family: str, kappa_tau: float, dims: tuple[int, int, int] = (2, 8, 8)) -> NoiseConfig:
    """Build one of the three circuit-noise families at a given κτ."""
    name = str(family).lower()
    kt = float(kappa_tau)
    if name == "loss":
        # Pure photon loss, paper Kraus, one application per UER/SNAP layer.
        # Phase-covariant: binomial unfolding should be exact up to interleaving.
        return paper_loss_config(kt, timing=TimingMode.PER_UER_LAYER, dims=dims)
    if name == "loss_thermal_dephasing":
        # Still phase-covariant (loss + heating + number dephasing). Tests
        # that factorial moments are *not* a pure η^k rescaling once nth>0.
        cfg = paper_loss_config(kt, timing=TimingMode.PER_UER_LAYER, dims=dims)
        tau = cfg.tau_application
        kappa_phi = 0.5 * kt / tau if tau > 0.0 else 0.0
        return NoiseConfig(
            timing=TimingMode.PER_UER_LAYER,
            loss_model=LossModel.LINDBLAD,
            kappa_tau=kt,
            nth_cav=0.05,
            kappa_phi=float(kappa_phi),
            enable_transmon=False,
            dims=dims,
        )
    if name == "comprehensive":
        # Loss + nth=0.01 + transmon T1/T2 + Kerr + 1% coherent control errors.
        # Ancilla errors break phase covariance. Default timing is per ECD pair.
        return comprehensive_config(timing=TimingMode.PER_ECD_PAIR, kappa_tau=kt, dims=dims)
    raise ValueError(f"unknown circuit-noise family {family!r}")


def readout_spec(level: str, n_shots: int, seed: int | None = None) -> ReadoutSpec:
    key = str(level).lower()
    if key not in READOUT_SPECS:
        raise ValueError(f"unknown readout level {level!r}; expected one of {tuple(READOUT_SPECS)}")
    spec = READOUT_SPECS[key]
    return ReadoutSpec(
        level=key,
        p01=float(spec["p01"]),
        p10=float(spec["p10"]),
        p_nn=float(spec["p_nn"]),
        n_shots=int(n_shots),
        seed=seed,
    )


def readout_config(spec: ReadoutSpec, n_fock: int = NFOCK) -> MeasurementConfig:
    """Column-stochastic confusion on qubit and each Fock register."""
    if spec.level == "ideal" or (spec.p01 == 0.0 and spec.p10 == 0.0 and spec.p_nn == 0.0):
        return MeasurementConfig(n_shots=int(spec.n_shots), seed=spec.seed)
    qubit_c = qubit_bitflip_confusion(spec.p01, spec.p10)
    fock_c = nearest_neighbor_fock_confusion(int(n_fock), spec.p_nn)
    for name, mat in (("qubit", qubit_c), ("fock", fock_c)):
        if not is_column_stochastic(mat):
            raise ValueError(f"{name} confusion is not column-stochastic")
    return MeasurementConfig(
        qubit_c=qubit_c,
        fock1_c=fock_c,
        fock2_c=fock_c.copy(),
        n_shots=int(spec.n_shots),
        seed=spec.seed,
    )


def readout_as_dict(spec: ReadoutSpec) -> dict:
    return spec.as_dict()


def is_trivial_readout(spec: ReadoutSpec) -> bool:
    return spec.level == "ideal" or (spec.p01 == 0.0 and spec.p10 == 0.0 and spec.p_nn == 0.0)


def scale_noise(cfg: NoiseConfig, scale: float) -> NoiseConfig:
    """Multiply circuit error rates by ``scale``. Readout is untouched.

    Scales ``kappa_tau`` (or ``1/T1_cav`` if κτ is implicit), cavity
    dephasing, and qubit ``1/T1``, ``1/T2``. Thermal occupation is left
    as a bath property.
    """
    s = float(scale)
    if s <= 0.0:
        raise ValueError(f"noise scale must be positive, got {scale}")
    updates: dict = {
        "kappa_phi": cfg.kappa_phi * s,
        "t1_q": cfg.t1_q / s,
        "t2_q": cfg.t2_q / s,
    }
    if cfg.kappa_tau is not None:
        updates["kappa_tau"] = cfg.kappa_tau * s
    else:
        updates["t1_cav"] = cfg.t1_cav / s
    return replace(cfg, **updates)


def family_description(family: str) -> str:
    return {
        "loss": (
            "Pure photon loss (paper amplitude-damping Kraus) after each "
            "UER/SNAP layer. Phase-covariant. Oracle binomial kernel."
        ),
        "loss_thermal_dephasing": (
            "Lindblad cavity loss with nth=0.05 and number dephasing "
            "κ_φ τ = 0.5 κτ. Phase-covariant; heating spoils η^k moment scaling."
        ),
        "comprehensive": (
            "Device-like: Lindblad loss (nth=0.01), transmon T1/T2, cavity "
            "self-Kerr, 1% ECD-amplitude and rotation errors. Not phase-covariant."
        ),
    }[str(family).lower()]
