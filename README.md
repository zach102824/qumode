# Qumode ECD-VQE Simulator

Classical simulation of the binary-knapsack ECD-VQE experiment from
Dutta et al., *Solving Constrained Optimization Problems Using Hybrid
Qubit-Qumode Quantum Devices* (arXiv:2501.11735). The noiseless and
photon-loss models follow the paper and the authors' published notebooks;
additional transmon, cavity, coherent-control, and measurement errors are
included as a more complete cQED error model.

The simulated device is **one transmon + two microwave cavities** with
Fock cutoffs \(L_1=L_2=8\) (Hilbert dimension \(2\times8\times8=128\))
and an ECD-rotation ansatz of depth \(N_d=5\) (**10 ECD gates**, 40 real
parameters).

## Setup

```bash
conda env create -f environment.yml
conda activate qumode
pip install -e ".[dev]"
```

The environment name is `qumode` as requested. Exact resolved versions are
pinned in `environment.yml` (QuTiP 5.3.1, NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.11.1, pytest 9.1.1).

## Run tests

Fast deterministic checks (Hamiltonian, circuit, channels, measurement,
stored reference parameters). Slow optimization tests are skipped by default:

```bash
conda activate qumode
pytest
pytest -m slow
```

## Run the experiment

```bash
python scripts/run_experiment.py --outdir results
```

Useful flags:

- `--mode {all,noiseless,paper-loss,comprehensive}` — which sweeps to run
- `--maxiter 80` — BFGS iterations (paper figures use ~80–200)
- `--full` — 200 noiseless iterations and noisy reoptimization
- `--seed 0` — RNG seed for random initializations
- `--skip-optimize` — only evaluate the stored reference parameters

Outputs are written under `results/` as JSON summaries, NPZ arrays, and
PNG plots.

## Gibbs VQE (adaptive initial state)

`scripts/Gibbs_and_adaptive_optim.py` runs noiseless Gibbs VQE. Each trial
starts from one uniform random preparation (qubit \(R_y\) plus two truncated
coherents) and a random ECD ansatz. The default budget is **70 joint SPSA
steps** on all 45 coordinates. Preparation is **not** frozen. Gibbs η is always
`sampled_tail` (histogram quantiles; the exact ground energy is not used).

```bash
python scripts/Gibbs_and_adaptive_optim.py --n-knapsack 20 --n-ising 20 --workers 7 --outdir results
```

`--spsa-iter N` with `N>0` is an optional ablation that freezes prep after
the joint stage. `--compare-prep-schedules` replays that ablation against
the current joint-70 default. `results/gibbs_mixed_40.json` is the older
freeze 20+50 suite; `results/gibbs_schedule_abc.json` includes the joint-70
run that is now the default.

## What is being simulated

The Binary Knapsack instance of Eq. (23) is mapped to the seven-qubit
diagonal Hamiltonian of Eq. (25), then to the hybrid basis
\(|q,n,m\rangle\) with \(|0,6,0\rangle\) the exact ground state (energy
\(-12\)). Expectation values are photon-number / Pauli-\(Z\) histograms.

Noise can be applied **after each UER layer** (5 applications; paper
Appendix A) or **after each ECD–rotation pair** (10 applications; closer
to gate timing). Photon loss is implemented either as the paper's truncated
Kraus channel (Eqs. 37–40) **or** as a unified local Lindblad model, never
both at once.

## Reference

Code and numerical targets are checked against
https://github.com/CQDMQD/codes_qumode_qubo
(`bkp_ecd_vqe.ipynb`, `bkp_ecd_vqe_noise.ipynb`), retrieved 2026-08-20.
The published noiseless BFGS run reaches approximately \(-11.99606\) after
200 iterations rather than the exact ground energy \(-12\).
