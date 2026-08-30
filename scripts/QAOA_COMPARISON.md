# Qubit QAOA / HEA vs ECD-VQE on the 40-Hamiltonian suite

This note compares noiseless **qubit** QAOA (and a hardware-efficient ansatz)
to the stored ECD-VQE Gibbs suite. It is a baseline, not a claim that one
ansatz is more expressive.

The Gibbs cost \(f=-\ln\langle e^{-\eta E}\rangle\) is Li et al.,
*Phys. Rev. Research* **2**, 023074 (2020). The \(\eta\) schedule
(`sampled_tail`) is **this repo's heuristic**, not Dutta or Li.

Dutta et al. (arXiv:2501.11735) reported that ECD-VQE (\(N_d=5\), 40 ECD
params) beat qubit QAOA (\(p=20\), 40 params) on the *paper* knapsack under
the energy objective \(\langle H\rangle\) and BFGS. This suite asks whether
that gap still appears on **these** 40 Hamiltonians under **this** Gibbs +
SPSA protocol.

## What is matched

| Item | Setting |
|---|---|
| Hamiltonians | Stored `results/hamiltonians.json` / `.npz`. 20 random knapsack + 20 random 7-spin diagonal Ising. Not regenerated. |
| Embedding | Same 7-bit map, partition \((1,3,3)\). Qubit 0 is MSB (`bits_from_qnm` / `qubit_index_from_bits`). Dim 128. |
| Success | Most-likely computational-basis bitstring is an exact ground (`atol=1e-8`). |
| Cost (Gibbs runs) | \(f=-\ln\langle e^{-\eta E}\rangle\) via `gibbs_objective`. |
| \(\eta\) | `sampled_tail` in `src/qumode_vqe/eta.py`: probability-weighted 5% / 25% energy quantiles, \(\eta_{\mathrm{raw}}=\ln 20/(Q_{25}-Q_{05})\), EMA \(\alpha=0.35\), clamp \([10^{-4},50]\), refresh every 5 **unperturbed** SPSA steps, held fixed for both \(\pm c\) probes. |
| Optimizer | `run_spsa`: \(a=0.2\), \(c=0.15\), \(A=10\), \(\alpha=0.602\), \(\gamma=0.101\), 70 steps. Not reimplemented. |
| Starts | One random start per Hamiltonian (`n_trials_per_hamiltonian=1`). |
| Seeds | `seed_base=3000` (mixed-suite convention). Knapsack: `seed_base + 100·hid + protocol_offset`. Ising: `seed_base + 10_000 + 100·hid + protocol_offset`. Offsets: QAOA Gibbs p=20 `+30000`, QAOA energy p=20 `+31000`, HEA Gibbs `+32000`, QAOA Gibbs p=22 `+33000`. |
| Primary param count | QAOA \(p=20\) → 40 real \((\gamma,\beta)\), matching Dutta and matching ECD's 40 ECD parameters. |
| HEA param count | \(n(L+1)=7\cdot 6=42\) (Ry on each qubit, \(L=5\) nearest-neighbour CZ layers, plus a final Ry layer). |
| HEA CZ packing | Even then odd commuting groups: CZ(0,1), CZ(2,3), CZ(4,5) (q6 idle), then CZ(1,2), CZ(3,4), CZ(5,6) (q0 idle). Same unitary as sequential CZ(\(i,i+1\)); depth-2 packing only. |

QAOA native initial state is \(\lvert+\rangle^{\otimes 7}\). HEA starts from
\(\lvert 0\rangle^{\otimes 7}\).

## What cannot be matched

- ECD lives in the hybrid product space \(\lvert q,n,m\rangle\) with a
  five-parameter product-state prep \(R_y(\theta)\lvert 0\rangle\otimes\lvert\alpha_1\rangle\otimes\lvert\alpha_2\rangle\)
  plus ECD–rotation generators. Gibbs ECD therefore has **45** live
  coordinates in the joint run (5 prep + 40 ECD). The primary QAOA match is
  **40** parameters, as in Dutta. QAOA \(p=22\) (44 params) is an extra
  sensitivity check against those 45 coordinates, not the fair match.
- QAOA / HEA use a different generator set. QAOA is \(\{H_C,H_M\}\) with
  \(H_M=\sum_i X_i\) and \(H_C=\mathrm{diag}(E)\). HEA is single-qubit \(R_y\)
  plus nearest-neighbour CZ. Neither is the ECD–rotation ansatz.
- The Hilbert spaces are identified only through the binary embedding.
  That is the same 128-dim diagonal cost, not the same variational manifold.
- Dutta's QAOA comparison used BFGS and \(\langle H\rangle\) on the paper
  knapsack. This baseline uses SPSA and (for the Gibbs runs) \(f\), on 40
  random instances. A QAOA loss under BFGS/energy is not a theorem about
  ECD expressivity at 40 parameters.

## How to run

```bash
python scripts/qaoa_baseline.py --outdir results --workers 7
# smoke: python scripts/qaoa_baseline.py --limit 2 --outdir results
```

Writes `results/qaoa_gibbs_p20.json`, `results/qaoa_energy_p20.json`,
`results/hea_gibbs.json`, extra `results/qaoa_gibbs_p22.json`, and the
even/odd HEA rerun `results/hea_gibbs_evenodd.json` (does not overwrite
`hea_gibbs.json`).

```bash
# HEA-only even/odd rerun (same seeds as hea_gibbs)
python scripts/qaoa_baseline.py --protocols hea_gibbs_evenodd --outdir results --workers 7
```

## Results

Success = decoded most-likely bitstring is an exact ground of the energy
tensor. ECD numbers are from stored JSON, not rerun. Qubit rows are from
`scripts/qaoa_baseline.py` on this suite (`seed_base=3000`, 70 SPSA steps,
one start each).

Joint-70 ECD is taken from `results/gibbs_schedule_abc.json` (`joint70`
block, 26/40). `results/gibbs_joint_step_sweep.json` exists but its budgets
are 40/50/80/100/150/200, not 70. Freeze 20+50 is `results/gibbs_mixed_40.json`
(24/40). Energy SPSA vacuum is `results/energy_spsa_baseline.json` (0/40).

| Run | All | Knapsack | Ising |
| --- | --- | --- | --- |
| ECD Gibbs freeze 20+50 | 24/40 | 13/20 | 11/20 |
| ECD Gibbs joint-70 | 26/40 | 12/20 | 14/20 |
| ECD energy SPSA vacuum 70 | 0/40 | 0/20 | 0/20 |
| QAOA p=20 Gibbs 70 | 0/40 | 0/20 | 0/20 |
| QAOA p=20 energy 70 | 0/40 | 0/20 | 0/20 |
| HEA L=5 Gibbs 70 (42 params) | 38/40 | 19/20 | 19/20 |
| HEA L=5 Gibbs 70 even/odd CZ (42 params) | 38/40 | 19/20 | 19/20 |
| QAOA p=22 Gibbs 70 (extra) | 0/40 | 0/20 | 0/20 |

The even/odd HEA rerun (`results/hea_gibbs_evenodd.json`) matches the
stored sequential `hea_gibbs.json` exactly: same 38/40, same mean
\(P(\mathrm{gs})=0.611\), same misses (knapsack H1 and Ising H19). That
is required: all CZs commute, so the packing is only a depth-2 rewrite.

Mean \(P(\mathrm{ground})\) on the qubit runs: QAOA \(p=20\) Gibbs 0.012,
QAOA \(p=20\) energy 0.010, HEA Gibbs 0.611, QAOA \(p=22\) Gibbs 0.009
(uniform would be \(1/128\approx 0.008\)). HEA knapsack packings were
feasible on 20/20 instances. HEA misses: knapsack H1 (most-likely
`1000100`, \(E=0.4\) vs \(E_{\min}=-10\)) and Ising H19 (most-likely
`0101010`, gap \(0.17\)).

### What the numbers say

Under **this** protocol, the Dutta-style fair match (QAOA \(p=20\), 40
params, one random start, 70 SPSA steps) does **not** recover grounds:
0/40 on both Gibbs and \(\langle H\rangle\). That is the same hit rate as
ECD energy SPSA from vacuum, and far below ECD Gibbs (24/40 freeze, 26/40
joint). Adding two QAOA layers (\(p=22\), 44 params) stayed at 0/40.

That is not a general statement that a 40-parameter qubit ansatz is weaker
than ECD. The 42-parameter HEA, with the same Gibbs cost, \(\eta\), SPSA
gains, and budget, hits **38/40** and concentrates (\(P(\mathrm{gs})\approx 0.61\)).
QAOA from \(\gamma\sim U(0,2\pi)\), \(\beta\sim U(0,\pi)\) at \(p=20\) stays
near-uniform: a deep random QAOA circuit plus SPSA did not concentrate on
these landscapes. An adiabatic-style small-angle QAOA start was not used;
the suite convention is one random start.

So: ECD Gibbs still beats **this** QAOA baseline by a wide margin, which is
compatible with Dutta's QAOA comparison being optimizer/init limited rather
than a theorem about ECD expressivity. A different qubit generator set
(HEA) wins under the same Gibbs protocol. These are hit counts on one
start each, not a landscape or expressivity proof.
