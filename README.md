# Qumode ECD-VQE Simulator

Classical simulation of hybrid qubit–qumode ECD-VQE for constrained
optimization, following Dutta et al., *Solving Constrained Optimization
Problems Using Hybrid Qubit-Qumode Quantum Devices*
([arXiv:2501.11735](https://arxiv.org/abs/2501.11735)).

The noiseless and photon-loss models match the paper and the authors’
notebooks. This repo also adds:

- a more complete cQED error model (transmon, cavity, coherent control, readout)
- **Gibbs VQE** with an adaptive product-state preparation and a histogram-only η
- random **knapsack** and **diagonal Ising** instances in the same 7-bit embedding
- the paper’s **multiple-constraint** instance (Eqs. 29–32)

Package: `qumode-vqe`. Source: [github.com/zach102824/qumode](https://github.com/zach102824/qumode).

## Device and ansatz

Default hardware is **one transmon + two microwave cavities** with Fock
cutoffs \(L_1=L_2=8\) (Hilbert dimension \(2\times8\times8=128\)).

The ECD–rotation ansatz (Eq. 22) has depth \(N_d=5\): **10 ECD gates**,
**40 real parameters** in the paper polar layout
\((\lvert\beta\rvert,\arg\beta,\theta,\varphi)\). A Cartesian
\((\mathrm{Re}\,\beta,\mathrm{Im}\,\beta)\) layout is available for bounded
optimizers. Each UER layer is

```text
U_ER = ECD_2(β2) R(θ2, φ2) ECD_1(β1) R(θ1, φ1)
```

Energy VQE starts from vacuum. Gibbs VQE also optimizes a five-parameter
product state \(R_y(\theta)\lvert0\rangle\otimes\lvert\alpha_1\rangle\otimes\lvert\alpha_2\rangle\)
(truncated coherents; \(\lvert\alpha\rvert^2\) uniform on the available Fock
range). That is **45 real coordinates** in the default joint SPSA run.

| Instance | Qubits | Hybrid basis | Depth | Exact ground |
|---|---|---|---|---|
| Binary knapsack, Eqs. (23)–(26) | 7 (4 items + 3 slack) | \(\lvert q,n,m\rangle\), \(L=(8,8)\), partition \((1,3,3)\) | \(N_d=5\) | \(\lvert0,6,0\rangle\), \(E=-12\), bits `0110000` |
| Multiple constraints, Eqs. (29)–(32) | 6 | \(L=(4,8)\), partition \((1,2,3)\) | \(N_d=10\) | \(\lvert1,0,4\rangle\), \(E=1\), bits `100100` |

QUBO energies are photon-number / Pauli-\(Z\) histograms on \(\lvert q,n,m\rangle\).
Random knapsacks stay in the paper encoding (4 items + 3 slack); λ is raised
when needed so the QUBO ground state is a feasible packing. Random Ising
instances are 7 local \(Z\) fields plus 12 \(ZZ\) couplings, RMS-normalized.

## Layout

```text
src/qumode_vqe/     simulator (Hamiltonian, circuit, noise, VQE, Gibbs η)
scripts/            experiment drivers
tests/              pytest suite
docs/               paper PDF and Gibbs-vs-energy note
results/            suite JSON, NPZ, and plots
```

## Setup

```bash
conda env create -f environment.yml
conda activate qumode
pip install -e ".[dev]"
```

Python 3.11. The environment name is `qumode`. Pins in `environment.yml`:
QuTiP 5.3.1, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1, pytest 9.1.1.

After the editable install you can use either
`python scripts/run_experiment.py` or the console script `qumode-experiment`.

## Tests

Fast checks (Hamiltonian encodings, ECD circuit, channels, measurement,
truncation, stored notebook parameters, Gibbs/SPSA guards) run by default.
Long BFGS / noisy-VQE regressions are marked `slow` and skipped:

```bash
conda activate qumode
pytest
pytest -m slow
```

## Paper energy VQE (BFGS)

Reproduce the notebook-style **energy** objective \(\langle H\rangle\) from
vacuum, with optional photon-loss and cQED ablations:

```bash
python scripts/run_experiment.py --outdir results
# same: qumode-experiment --outdir results
```

Flags:

- `--mode {all,noiseless,paper-loss,comprehensive}`
- `--maxiter 80` — BFGS iterations (paper figures use ~80–200)
- `--full` — 200 noiseless iterations and noisy reoptimization
- `--seed 0`
- `--skip-optimize` — evaluate the stored reference vector only

Paper figure scripts (Figs. 4, 5, 8, 9, 14 and a noisy Fig. 4 overlay):

```bash
python scripts/make_paper_figures.py
python scripts/run_noisy_fig4.py
```

`run_noisy_fig4.py` compares noiseless BFGS to paper Kraus loss
(\(\kappa\tau\simeq 0.003\) per UER layer) and a typical-device Lindblad model
(cavity T1/\(n_{\mathrm{th}}\), transmon T1/T2, Kerr, 1% coherent-control errors).

The published noiseless BFGS vector in `src/qumode_vqe/data/reference.json`
reaches about \(-11.99606\) after 200 iterations, not the exact \(-12\).

## Gibbs VQE (adaptive preparation)

`scripts/Gibbs_and_adaptive_optim.py` is **noiseless Gibbs VQE**. It does not
use a known ground-state label during optimization.

Default protocol (`optimize_gibbs_adaptive`):

1. Draw one uniform random preparation \((\theta,\alpha_1,\alpha_2)\) and a
   random ECD ansatz.
2. Joint SPSA on all **45** coordinates for **70** steps. Prep is **not** frozen
   (`DEFAULT_JOINT_STEPS=70`, `DEFAULT_ANSATZ_STEPS=0`).
3. Cost is the Gibbs objective \(f=-\ln\langle e^{-\eta E}\rangle\) on the
   Born histogram. η is always `sampled_tail`: probability-weighted 5% and 25%
   energy quantiles of the current histogram, set so those quantiles differ by
   a Gibbs factor of 20, EMA-smoothed, refreshed every five unperturbed steps,
   and held fixed for both SPSA probes of a step.

`--spsa-iter N` with `N>0` is an optional ablation: freeze prep after the joint
stage and continue the **same** ansatz vector for N more steps.

```bash
# Paper BKP, 50 random starts, joint-70 (writes results/gibbs_adaptive_noiseless.json)
python scripts/Gibbs_and_adaptive_optim.py --outdir results

# 20 random knapsacks + 20 random 7-spin Ising, one start each
python scripts/Gibbs_and_adaptive_optim.py --n-knapsack 20 --n-ising 20 --workers 7 --outdir results
```

The mixed suite writes `results/gibbs_mixed_40.json` and
`results/hamiltonians.json`. Follow-up jobs reuse those Hamiltonians and starts:

```bash
# Vacuum + ⟨H⟩ + SPSA on the saved Hamiltonians (budget = outer+spsa, 70 by default)
python scripts/Gibbs_and_adaptive_optim.py --energy-baseline --outdir results

# Same 40 H and starts: joint-70 vs freeze 20+50 vs scaled-prep 20+50
python scripts/Gibbs_and_adaptive_optim.py --compare-prep-schedules --outdir results

# Fully joint SPSA at several step budgets (default 40 50 80 100 150 200)
python scripts/Gibbs_and_adaptive_optim.py --joint-step-sweep --outdir results
```

Useful flags: `--outer-iter`, `--spsa-iter`, `--workers`, `--seed-base`,
`--ham-seed`, `--n-hamiltonians` (paper BKP plus nine knapsack-like variants),
`--n-trials`, `--prep-step-scale` (SPSA gain on the five prep coordinates),
`--joint-steps`, `--hamiltonians`, `--starts`.

`docs/gibbs_vs_energy.pdf` (from `scripts/make_gibbs_vs_energy_pdf.py`) is a
short note on why the Gibbs cost can decode the ground bitstring when mean
energy does not.

### Local mixed-suite snapshot

Numbers below are from the development 20 knapsack + 20 Ising suite (one start
each, same seeds), stored under `results/`. Success means the decoded
\(\lvert q,n,m\rangle\) is an exact ground state.

| Run | Hits |
|---|---|
| Energy SPSA, vacuum, 70 steps | 0 / 40 |
| Gibbs freeze 20+50 (`gibbs_mixed_40.json`) | 24 / 40 |
| Gibbs joint-70 (current default) | 26 / 40 |
| Gibbs joint-100 / 150 / 200 | 29 / 31 / 32 of 40 |

On that suite, extra joint steps only **added** hits (no joint-70 success was
lost at a larger budget). `--compare-prep-schedules` also showed joint-70
slightly ahead of freeze 20+50 and of scaled-prep 20+50.

## Noise and measurement

Noise can be applied **after each UER layer** (5 applications; paper Appendix A)
or **after each ECD–rotation pair** (10 applications; closer to gate timing).
Photon loss is either the paper’s truncated Kraus channel (Eqs. 37–40) **or**
a unified local Lindblad model, never both at once.

Readout errors act only on the final histogram (qubit bit-flip and
nearest-neighbor Fock confusion), not on the density matrix.

## Reference

Numerical targets are checked against
https://github.com/CQDMQD/codes_qumode_qubo
(`bkp_ecd_vqe.ipynb`, `bkp_ecd_vqe_noise.ipynb`), retrieved 2026-08-20.

MIT License. Copyright (c) 2026 Zekun He.
