# Why ECD Gibbs “works” on mixed p-spin but not 4-SAT

Investigation only. Official GDR defaults were not changed.
Evidence: `results/gibbs_{mixed_p_spin,four_sat}_{ecd,snap}*.json`,
`near_e0_filter.json`, Hamiltonian NPZs, and the diagnostics in this folder
(`ecd_vs_mixed_diag.{json,txt}`, `ecd_l5_eta_ablation_h000.{json,txt}`).
Replay of stored `(prep, x)` recovers JSON `energy_physical` to 1e-8, so the
numbers below are not a measurement-wire bug.

Gate used here and in PR #12: `energy_physical − E0 ≤ 0.5`. For 4-SAT, `E0 = 0`.

## Answer for Zach

ECD on 4-SAT **does** find the ground-state bitstring (`success_gs=true` on
203/216 L4 fleet trials). It **does not** put enough probability on that
state. 4-SAT is a unique, locally rigid SAT: every Hamming-1 neighbor of the
GS costs energy **1**, and H000 has **62/128** basis states at `E=1`. Leftover
mass therefore sits on a huge first-excited plateau, so

```text
⟨H⟩ ≈ 1 − P(GS)
```

H000 ECD best scouts have `P(GS) ≈ 0.23–0.32` → `⟨H⟩ ≈ 0.76–0.83`. SNAP L3
on the same H reaches `P(GS) = 0.67` → `⟨H⟩ = 0.376` and clears the gate.
No `--four-sat` ECD load/dim/measurement bug. Changing Gibbs η at ECD L5
does **not** get H000 under 0.5 (best of 8 arms × 5 trials: **0.780**).

The mixed p-spin “ECD worked” memory is mostly **exact-GS bitstring hits**
(L4: 18/20), which 4-SAT ECD also has. Mixed p-spin Gibbs ECD is **0/20**
near-E0 at every depth 1–5 (mean deficit ≈ 4.3). PR #9’s two strict passers
were **energy L-BFGS-B** polish, which the 4-SAT pipeline did not run.

## What is actually the same

Both families use the hybrid encoding `1 qubit + 2 qumodes`, `nfocks=(8,8)`,
`partition=(1,3,3)`, Hilbert dim **128**. `--four-sat` and `--mixed-p-spin`
share `run_saved_hamiltonian_suite` in `scripts/Gibbs_and_adaptive_optim_ECD.py`
and the same `energy_tensor_from_z_terms` loader. SNAP is the same script
with `ansatz="snap"`. Vacuum start, 200 joint SPSA, `sampled_tail` η.

| | ECD L4 | ECD L5 | ECD L8 | SNAP L2 | SNAP L3 |
|--|--:|--:|--:|--:|--:|
| ansatz params | 32 | 40 | 64 | 36 | 54 |
| primitive gates | 16 | 20 | 32 | 8 | 12 |

4-SAT untilted GS energy is exactly 0 (identity shift 1.125 on H000 is
already in the Pauli expansion). The 1e-10 diagonal tilt only moves `Emin`
to ~1e-8.

## Landscapes

| | 4-SAT (20 H) | mixed p-spin (20 H) |
|--|--|--|
| `Emin` | 0 | mean −6.43 |
| spread | mean 3.45 | mean 13.54 |
| gap | **1.0** (all) | mean 0.41 (min 0.045) |
| unique energies | ~4–6 | **128** (no degeneracy) |
| `n_ground` | 1 | 1 |
| Hamming-1 ΔE | **all 7 neighbors = 1** | mean 4.95, min 0.24 |
| Pauli | ~37 terms, weight ≤4, plus identity | 91 Gaussian 2/3/4-body, no identity |
| H000 occupancy | 1 @0, **62 @1**, 11 @2, 3 @3 | 128 distinct values |

Local rigidity is the 4-SAT design (`Hamiltonians/four_sat.py`): each
single-bit flip falsifies a planted clause. Mixed p-spin neighbors are
**not** cheap on average (mean ΔE 4.95). The distinctive 4-SAT cost of
leakage is the **degenerate E=1 plateau**, not a smaller Hamming-1 field.

## Gibbs numbers (noiseless)

`success` / `success_gs` = most-likely bitstring is a ground state, **not**
the 0.5 energy gate.

### Mixed p-spin ECD (1 trial / H, seed 3000, 200 SPSA)

| depth | GS hits | near-E0 | mean deficit | best deficit | mean η |
|--:|--:|--:|--:|--:|--:|
| L4 | 18/20 | **0/20** | 4.28 | 2.06 | 5.71 |
| L5 | 16/20 | **0/20** | 4.37 | 2.93 | 3.03 |

SNAP L2: 14/20 GS hits, still 0/20 near-E0, mean deficit 4.04.

PR #9 near-E0 ECD used extra **energy L-BFGS-B** on top of Gibbs:
H004 deficit **0.326**, H009 **0.449**; H000 only 0.756. 4-SAT GDR loaded
Gibbs `(prep, x)` directly (`CONCLUSION.md`: “no energy L-BFGS-B”).

### 4-SAT ECD H000 scout (8 trials, seed 3100)

L1–L8: **0/8** near-E0 at every depth. Best `⟨H⟩ = 0.757` at L6.
400-step L6 retry: best 0.851 (worse). Replay: L6 `P(GS)=0.324`,
L5 `P(GS)=0.308`, L8 `P(GS)=0.234`.

### 4-SAT fleet (8 trials, seed 4000; ECD extras seed 5000)

| | depth | GS hits | H with a trial ≤0.5 | best `⟨H⟩` | mean `⟨H⟩` | mean η* |
|--|--:|--|--|--:|--:|--:|
| SNAP | L3 | 153/160 | **20/20** | 0.147 (H010) | 0.572 | 12.5 |
| ECD | L4 | 203/216 | **2/20** (H016 0.326, H010 0.483) | 0.326 | 0.807 | 7.6 |

H018 ECD 0.501 is the borderline kept in GDR, not a strict pass.
H000 ECD fleet best 0.831, `P(GS)=0.205`.
SNAP H000 L3: `⟨H⟩=0.376`, `P(GS)=0.666`. The one ECD passer we replayed
(H016) has `P(GS)=0.699` — same concentration SNAP reaches routinely.

Final adaptive η is the same order on both families (mixed ECD ~3–6,
4-SAT ECD ~4–8). 4-SAT **η0** is often clamped at 50 because vacuum is a
delta on one energy; that is expected `sampled_tail` behavior, not a
broken E0=0 formula. Production never uses `1/(0.05|E0|)`, which would
diverge at `E0=0`.

## Why SNAP concentrates and ECD does not

ECD is conditional **displacements**. `D(β)` couples neighboring Fock
states, and the binary Fock encoding maps those neighbors to neighboring
bitstrings — exactly the Hamming-1 shell that 4-SAT prices at +1.
SNAP is **number-diagonal**: a per-Fock phase plus a displacement can
peak a single `|q,n,m⟩`. That matches a unique computational-basis SAT
solution. Parameter count is not the story: ECD L8 (64 params, 32 gates)
is worse than SNAP L3 (54 params, 12 gates) on H000.

On the original knapsack (paper_result), ECD Gibbs was the *better*
ansatz (84% GS hits vs SNAP L1 0% / L2 80%). The 4-SAT reversal is
landscape + gate set, not a 4-SAT-only wiring error.

## η at ECD L5 on 4-SAT H000 — no

Question: does a different Gibbs η get ECD `ndepth=5` under the 0.5 gate
on H000?

Run: `Error_mitigation/out_four_sat/run_ecd_l5_eta_ablation.py`
(5 paired trials, seed 6100, 200 joint SPSA, same random ECD starts
across arms). Arms reuse `paper_result/run_eta_ablation.py` plus typical
logged finals (mixed ECD L5 mean η=3.0, 4-SAT ECD 7.2, 4-SAT SNAP 12.5).

| policy | near-E0 | best `⟨H⟩` | mean `⟨H⟩` | mean η |
|--|--:|--:|--:|--:|
| sampled_tail adaptive (production) | 0/5 | 0.837 | 0.920 | 6.58 |
| sampled_tail frozen | 0/5 | 0.781 | 0.895 | 50 |
| fixed 0.1 (soft) | 0/5 | 1.109 | 1.123 | 0.10 |
| fixed 1.667 (paper BKP) | 0/5 | 0.996 | 1.029 | 1.67 |
| fixed 3.0 (mixed ECD L5) | 0/5 | 0.869 | 0.987 | 3.0 |
| fixed 7.2 (4-SAT ECD fleet) | 0/5 | 0.784 | 0.948 | 7.2 |
| fixed 12.5 (4-SAT SNAP fleet) | 0/5 | **0.780** | 0.892 | 12.5 |
| fixed 26 (paper sharp) | 0/5 | 0.781 | 0.895 | 26 |

**No arm clears 0.5.** Soft η is strictly worse (cost ≈ `⟨H⟩`, no
pressure to grow `P(GS)`). Sharp / frozen-at-50 / η=12.5 are tied around
0.78 — the same ballpark as the L5 scout (0.774) under production
`sampled_tail`. Frozen and η=26 match trial-by-trial because a vacuum
delta already clamps `sampled_tail` to `ETA_MAX=50`, and both objectives
have already saturated to `−ln P(GS)`.

No spot-check of H016 / H002: nothing on the hard scout was promising
(best 0.780 vs gate 0.5). A thin `fixed_eta` / `eta_adaptive=False` hook
exists on `optimize_gibbs_adaptive` for this script only; defaults remain
`sampled_tail` adaptive.

## Ruled out

1. **Different Hilbert / truncation / qubit-qumode encoding.** Same dim,
   nfocks, partition, loader.
2. **`--four-sat` ECD wire-up bug (wrong H, wrong dim, wrong measurement).**
   Untilted GS energy 0; replay `⟨H⟩` matches JSON; `tests/test_four_sat.py`
   and the new fixed-η tests pass.
3. **Gibbs cost algebra breaking at `E0=0`.** `f = −ln⟨e^{−η(E−Emin)}⟩ + η Emin`
   with `Emin≈0` is just `−ln(P0 + P1 e^{−η} + ⋯)`. Adaptive η on 4-SAT
   ends at ~3–8, not a degenerate-zero pathology. Oracle `1/(0.05|E0|)` is
   not used.
4. **Insufficient depth / SPSA budget as the root cause.** H000 fails
   L1–L8 and a 400-step L6 retry. L5 η ablation at 200 steps does not
   help.
5. **“Mixed p-spin ECD Gibbs already meets the 0.5 energy gate.”** It
   does not (0/20 at L1–L5). Historical success is GS-bitstring hits
   (true on 4-SAT too) plus later energy L-BFGS-B on a few H.
6. **Cheap Hamming-1 leakage unique to mixed p-spin.** Mixed Hamming-1
   ΔE is larger on average, not smaller. 4-SAT’s penalty is the
   **62-fold E=1 plateau**.

## Not claimed / not fixed

ECD can still pass 4-SAT on a few H when it happens to concentrate
(`P(GS)≳0.7`, H016). That is luck of starts, not a recipe. No ECD ansatz
or GDR default was changed. Optional next probes (not done): energy
L-BFGS-B polish on 4-SAT ECD, or a Fock-peaking ECD variant — both would
be new work, not a bugfix.
