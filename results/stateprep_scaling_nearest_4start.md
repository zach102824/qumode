# ECD vs HEA state-preparation scaling

Noiseless statevector fidelities for **state preparation**, not Gibbs VQE.
Cost is `1-F`. Optimizer is L-BFGS-B with independent random starts;
the table keeps the best start. Numbers are from
`results/stateprep_scaling.json` (this run), not invented.

This is **not** a claim that ECD is a new gate, that cats are new, or that
Gibbs / `sampled_tail` helps state prep. Dutta et al. arXiv:2501.11735 is a
**binary knapsack / constrained-optimization** experiment (the wrong problem
class for this note) and is cited only as contrast.

## Setup

| Item | Setting |
|---|---|
| Target (primary) | Even two-legged cat |C_α⟩ ∝ |α⟩+|−α⟩, normalized on the truncated Fock space |
| α | 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 |
| Cutoff L | smallest L ≥ |α|²+8|α|+16 with truncation infidelity < 0.0001, cap 64 |
| Same L | ECD and HEA use the same cutoff at each α |
| ECD | Single oscillator + one transmon. Layer R(θ, φ) then `ECD(β)`. N_d in [1, 2, 3]. Terminal rotation: True (4 N_d+2 real params). |
| ECD objective | F = |⟨g, target|ψ⟩|² |
| HEA matched | Binary Fock index, MSB-first, unused levels padded with 0. Layers chosen so n(L+1) is closest to the ECD budget 4 N_d = 8 (primary N_d=2). |
| HEA unconstrained | extra: n_layers=5 → n·6 parameters |
| Starts / maxiter / seed | 4 random starts, L-BFGS-B maxiter=200, seed=0 |
| Constructive ECD | Ry(π/2), `ECD(β=2α)`, X-basis post-select |+⟩ |
| Negative control | Fock |n⟩, n=round(α²) clipped to [1, L-2] |

## Cutoffs

| α | L | n_qubits | F_truncation | 1−F_trunc | n = round(α²) |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 25 | 5 | 1.000000e+00 | 1.110e-16 | 1 |
| 1.5 | 31 | 5 | 1.000000e+00 | 0.000e+00 | 2 |
| 2.0 | 36 | 6 | 1.000000e+00 | 0.000e+00 | 4 |
| 2.5 | 43 | 6 | 1.000000e+00 | 4.441e-16 | 6 |
| 3.0 | 49 | 6 | 1.000000e+00 | 4.441e-16 | 9 |
| 3.5 | 57 | 6 | 1.000000e+00 | 1.110e-16 | 12 |
| 4.0 | 64 | 6 | 1.000000e+00 | 0.000e+00 | 16 |

## Even cat (best start)

| α | L | constructive F | ECD N_d=1 | ECD N_d=2 | ECD N_d=3 | HEA matched | HEA extra (L=5) | ECD N_d=2 params | HEA matched params |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 25 | 1.000000 | 0.648054 | 0.948471 | 0.987284 | 0.999996 | 0.999996 | 10 | 10 |
| 1.5 | 31 | 1.000000 | 0.506166 | 0.944564 | 0.991288 | 0.998678 | 0.998678 | 10 | 10 |
| 2.0 | 36 | 1.000000 | 0.500169 | 0.964555 | 0.975525 | 0.882678 | 0.984863 | 10 | 6 |
| 2.5 | 43 | 1.000000 | 0.500002 | 0.500002 | 0.813524 | 0.628288 | 0.970482 | 10 | 6 |
| 3.0 | 49 | 1.000000 | 0.500000 | 0.983474 | 0.675663 | 0.836319 | 0.970893 | 10 | 6 |
| 3.5 | 57 | 1.000000 | 0.500000 | 0.895123 | 0.562021 | 0.781455 | 0.970906 | 10 | 6 |
| 4.0 | 64 | 1.000000 | 0.500000 | 0.500000 | 0.527821 | 0.641755 | 0.966868 | 10 | 6 |

Constructive F is the **post-selected** cavity fidelity. Variational ECD F is the optimized `joint` fidelity (unitary, no post-select). HEA F is |⟨C_α|ψ_HEA⟩|² on the binary Fock register.

## Negative control: Fock |n⟩ (best start)

| α | n | L | ECD N_d=2 | HEA matched | ECD N_d=8 extra |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 1 | 25 | 0.735759 | 1.000000 | 0.999995 |
| 1.5 | 2 | 31 | 0.541341 | 1.000000 | 0.991173 |
| 2.0 | 4 | 36 | 0.390734 | 1.000000 | 0.958747 |
| 2.5 | 6 | 43 | 0.222435 | 1.000000 | 0.829790 |
| 3.0 | 9 | 49 | 0.263511 | 1.000000 | — |
| 3.5 | 12 | 57 | 0.228736 | 1.000000 | — |
| 4.0 | 16 | 64 | 0.198435 | 1.000000 | — |

ECD N_d=8 (extra) is run only for n≤8 (Eickbusch-scale; ≲10 ECD for |7⟩). Larger n is skipped as not cheap at these cutoffs.

## Extra: 4-legged compass cat

| α | L | ECD N_d=2 | ECD N_d=4 | HEA matched |
|---:|---:|---:|---:|---:|
| 2.0 | 36 | 0.490916 | 0.932724 | 0.901187 |
| 3.0 | 49 | 0.355433 | 0.672310 | 0.879988 |

## Thesis

Constructive ECD (Ry(π/2) + ECD(2α) + X-basis post-select) has min F=1.000000 on every completed α, so the O(1) existence proof holds: a two-legged cat is K=2-sparse in the coherent-state basis and one ECD prepares it; leftover infidelity is Fock truncation (here ≲1e-4 by construction, and numerically ~0 at these L). Variational ECD N_d=2 is a different question — unitary, no post-select, optimizing joint fidelity to |g⟩⊗|C_α⟩ with four random L-BFGS-B starts. Best-start F: α=1.0: 0.9485, α=1.5: 0.9446, α=2.0: 0.9646, α=2.5: 0.5000, α=3.0: 0.9835, α=3.5: 0.8951, α=4.0: 0.5000. Matched HEA (n(L+1) closest to 8 ECD params): α=1.0: 1.0000, α=1.5: 0.9987, α=2.0: 0.8827, α=2.5: 0.6283, α=3.0: 0.8363, α=3.5: 0.7815, α=4.0: 0.6418. ECD N_d=2 wins 3 α-points, HEA wins 4, ties 0. **HEA wins the matched-budget even-cat comparison** on this suite (including the small-α points, where 5-qubit HEA with 10 parameters can fit the truncated cat). HEA does degrade as |α| and L grow (F=1.000 at α=1.0 → F=0.642 at α=4.0 on the matched budget), while unconstrained HEA (n_layers=5, extra) stays at min F=0.967. Variational ECD N_d=1 saturates at F≈1/2 (one ECD leaves |g,−α⟩+|e,α⟩, which cannot be |g⟩⊗|C_α⟩). Several N_d=2/3 starts also land on that F=1/2 even/odd trap; when a start escapes, F can stay high (best N_d=2 F=0.983 at α=3.0). That is an optimizer / disentangling issue, not missing cat expressivity — the constructive circuit already has F=1. Negative control (Fock |n⟩, n≈|α|²): HEA wins 7/7 against ECD N_d=2 — HEA can prepare a computational-basis bitstring with a product of Ry(π) gates, while ECD depth must grow with n (Eickbusch needed ≲10 ECD for |7⟩). ECD N_d=8 (extra, n≤8 only) improves Fock fidelity but still loses to HEA. **Thesis split: accepted for the constructive O(1) circuit, rejected for variational matched-budget ECD vs HEA.** Numbers were not retuned to force an ECD win. Dutta arXiv:2501.11735 is the wrong problem class (binary knapsack VQE).

## Citations (not claimed as this work)

- Eickbusch et al., *Nat. Phys.* (2022); arXiv:2111.06414 — ECD gate and Fock-state compilation (≲10 ECD for |7⟩, F>0.99).
- Krastanov et al., *Phys. Rev. A* **92**, 040303 (2015) — SNAP / oscillator control.
- Analytic ECD circuits: arXiv:2504.19992.
- N-fold cat lower bound: arXiv:2608.07696.
- Dutta et al., arXiv:2501.11735 — **contrast only**: binary knapsack ECD-VQE, not state prep.
