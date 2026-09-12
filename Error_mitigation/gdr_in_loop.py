"""GDR-in-loop helpers for the default noiseless+noisy protocol.

Policy B: fit ``gdr_param`` once per (Hamiltonian, κτ, ansatz) from Gaussian
twins of a random same-depth circuit, then reuse the same M for every SPSA
iterate. The SPSA cost is the Gibbs objective of the unfolded histogram.

In-loop histograms apply readout_realistic confusion but not extra multinomial
shot noise (shot noise would drown the SPSA gradient). Twin fitting still uses
finite shots, matching the official GDR driver.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qumode_vqe.hamiltonian import ground_qnm_from_tensor
from qumode_vqe.measurement import (
    apply_confusion,
    decode_bitstring,
    energy_from_histogram,
    joint_probabilities,
    most_likely_qnm,
)
from qumode_vqe.params import random_parameters, random_snap_parameters
from qumode_vqe.vqe import HybridSimulator, ground_bitstring_from_tensor, ground_prob

from Error_mitigation.mitigation import (
    confusion_from_measurement,
    fit_gdr_param,
    observe_histogram,
    params_to_kernels,
    unfold,
)
from Error_mitigation.noise_models import circuit_noise, readout_config, readout_spec
from Error_mitigation.twins import build_twins, designed_twin_plan

PROTOCOL_FAMILY = "comprehensive"
PROTOCOL_READOUT = "readout_realistic"
PROTOCOL_KAPPA_TAU = (0.003, 0.03, 0.1)
PROTOCOL_N_TRAIN = 40
PROTOCOL_TWIN_SHOTS = 8192
PROTOCOL_SEED_BASE = 4000


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def m_cache_path(
    cache_dir: Path,
    *,
    ansatz: str,
    ndepth: int,
    hid: int,
    kappa_tau: float,
    n_train: int,
) -> Path:
    kt = str(kappa_tau).replace(".", "p")
    name = f"gdr_param_{ansatz}_L{int(ndepth)}_h{int(hid):03d}_kt{kt}_n{int(n_train)}.json"
    return Path(cache_dir) / name


def fit_gdr_param_policy_b(
    *,
    ansatz: str,
    ndepth: int,
    hid: int,
    kappa_tau: float,
    energy_tensor: np.ndarray,
    nfocks: tuple[int, int] = (8, 8),
    n_train: int = PROTOCOL_N_TRAIN,
    n_shots: int = PROTOCOL_TWIN_SHOTS,
    seed: int = PROTOCOL_SEED_BASE,
    cache_dir: Path | None = None,
    fit_maxiter: int = 200,
) -> dict:
    """Fit M once from span twins of a random vacuum circuit. Cache to disk."""
    ansatz = str(ansatz).lower()
    ndepth = int(ndepth)
    hid = int(hid)
    kt = float(kappa_tau)
    dims = (2, int(nfocks[0]), int(nfocks[1]))
    cache_path = None if cache_dir is None else m_cache_path(
        cache_dir, ansatz=ansatz, ndepth=ndepth, hid=hid, kappa_tau=kt, n_train=n_train
    )
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        theta = np.asarray(payload["theta"], dtype=float)
        cq, c1, c2 = params_to_kernels(theta, dims)
        payload["kernels"] = (cq, c1, c2)
        payload["from_cache"] = True
        return payload

    rng = np.random.default_rng(int(seed) + 17 * (0 if ansatz == "ecd" else 1) + 1009 * hid)
    if ansatz == "snap":
        x_target = random_snap_parameters(ndepth, nfocks, rng)
    else:
        x_target = random_parameters(ndepth, rng)

    tensor = np.asarray(energy_tensor, dtype=float)
    ground_qnm = ground_qnm_from_tensor(tensor)
    sim_ideal = HybridSimulator(
        ndepth=ndepth,
        nfocks=nfocks,
        target_qnm=ground_qnm,
        energy_tensor=tensor,
        ansatz=ansatz,
        cost_kind="gibbs",
    )
    t_list, scales = designed_twin_plan(int(n_train), ndepth)
    twins = build_twins(sim_ideal, x_target, rng, t_free_list=t_list, mag_scales=scales)

    noise = circuit_noise(PROTOCOL_FAMILY, kt, dims=dims)
    sim_noisy = HybridSimulator(
        ndepth=ndepth,
        nfocks=nfocks,
        noise=noise,
        target_qnm=ground_qnm,
        energy_tensor=tensor,
        ansatz=ansatz,
        cost_kind="gibbs",
    )
    spec = readout_spec(PROTOCOL_READOUT, n_shots=int(n_shots), seed=int(seed) + hid)
    p_ideals = [np.asarray(t.p_ideal, dtype=float) for t in twins]
    q_obs = []
    for i, twin in enumerate(twins):
        rho = sim_noisy.density_matrix(twin.x)
        p_phys = joint_probabilities(rho, dims)
        q_obs.append(observe_histogram(p_phys, spec, dims, seed=int(seed) + 10_000 * hid + i))

    theta, info = fit_gdr_param(
        p_ideals, q_obs, noise, spec, ndepth, dims, maxiter=int(fit_maxiter)
    )
    cq, c1, c2 = params_to_kernels(theta, dims)
    payload = {
        "ansatz": ansatz,
        "ndepth": ndepth,
        "hamiltonian_id": hid,
        "kappa_tau": kt,
        "n_train": int(n_train),
        "n_shots": int(n_shots),
        "seed": int(seed),
        "m_policy": "B",
        "method": "gdr_param",
        "family": PROTOCOL_FAMILY,
        "readout": PROTOCOL_READOUT,
        "theta": np.asarray(theta, dtype=float).tolist(),
        "fit": info,
        "noise": {
            "kappa_tau_used": noise.kappa_tau_used(),
            "kappa_phi": noise.kappa_phi,
            "kappa_phi_tau": noise.kappa_phi * noise.tau_application,
        },
        "from_cache": False,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    payload["kernels"] = (cq, c1, c2)
    return payload


def make_gdr_forward(
    *,
    theta: np.ndarray,
    dims: tuple[int, int, int],
    energy_tensor: np.ndarray,
    readout_level: str = PROTOCOL_READOUT,
    n_shots: int | None = None,
    shot_seed: int | None = None,
):
    """Return ``forward(sim, x)`` that unfolds a noisy+readout histogram with fixed M.

    ``n_shots is None`` (default): apply confusion only, no multinomial sampling.
    """
    theta = np.asarray(theta, dtype=float)
    cq, c1, c2 = params_to_kernels(theta, dims)
    spec = readout_spec(readout_level, n_shots=int(n_shots or 0), seed=shot_seed)
    meas = readout_config(spec, n_fock=int(dims[1]))
    tensor = np.asarray(energy_tensor, dtype=float)
    gs = ground_qnm_from_tensor(tensor)
    gs_bits = ground_bitstring_from_tensor(tensor)

    def forward(sim, xvec):
        xvec = np.asarray(xvec, dtype=float)
        rho = sim.density_matrix(xvec)
        p_phys = joint_probabilities(rho, dims)
        energy_physical = float(np.real(np.trace(sim._h_np @ rho)))
        if n_shots is not None and int(n_shots) > 0:
            q_raw = observe_histogram(p_phys, spec, dims, seed=int(shot_seed or 0))
        else:
            q_raw = apply_confusion(p_phys, meas.qubit_c, meas.fock1_c, meas.fock2_c)
        p_mit = unfold(q_raw, cq, c1, c2)
        ml_raw = most_likely_qnm(q_raw)
        bits_raw = decode_bitstring(*ml_raw, partition=sim.partition)
        extra = {
            "p_gs_phys": float(p_phys[gs]),
            "p_gs_raw": ground_prob(q_raw, tensor),
            "p_gs_mit": ground_prob(p_mit, tensor),
            "most_likely_bitstring_raw": bits_raw,
            "success_raw": bits_raw == gs_bits,
            "energy_hist_raw": energy_from_histogram(q_raw, tensor),
            "energy_hist_mit": energy_from_histogram(p_mit, tensor),
        }
        return {
            "probs": p_mit,
            "energy_physical": energy_physical,
            "extra": extra,
        }

    return forward


def kernels_from_theta(theta, dims: tuple[int, int, int] = (2, 8, 8)):
    return params_to_kernels(np.asarray(theta, dtype=float), dims)
