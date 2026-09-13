# ECD system-size scaling (isolated)

Noiseless Gibbs **ECD only** bitstring success vs system size `n=7…15`
(and `n=16+` only if wall-clock allows). All code and results for this
study live in this folder. Upstream `src/`, `scripts/`, `Hamiltonians/`,
and `Error_mitigation/` are not edited except when pulling already-merged
default-protocol fixes from `main`.

`n=7` L* is **prior data** from [PR #14](https://github.com/zach102824/qumode/pull/14)
(ECD L4 = 186/200). `n=8…11` is the finished [PR #15](https://github.com/zach102824/qumode/pull/15)
ladder (all L*=4, ≥90%). **Do not rerun those cells.** This folder now
continues at `n=12` by growing the hardware (add a transmon+cavity pair
when the register is full; `FOCK_CUTOFF=8`).

## Question

At fixed clause density ≈ `18/7 ≈ 2.57`, how does ECD depth `L` needed for
**≥ 90%** most-likely-bitstring success (`k/200`) grow past n=11?

Success = `most_likely_bitstring == ground_bitstring`.
Each `(n, L)` cell is **20 Hamiltonians × 10 trials = 200**.
Cost is Gibbs `-ln⟨e^{-ηE}⟩` with production `sampled_tail` η.
No SNAP. No GDR. No noise (κ_φ is irrelevant for this noiseless ladder).

## Canonical protocol (this restart)

| item | value |
|------|-------|
| trials | 10 / Hamiltonian, 20 H |
| L sweep | start at 4, +1 until ≥90%; n≥12 **soft cap L=12** (n=8…11 used L=40, all hit at 4) |
| SPSA | joint **200**, `a = 0.2 × √(37 / n_params)` (production `a=0.2` is the n=7 L=4 reference), `c=0.15, A=10, α=0.602, γ=0.101` |
| η | `sampled_tail` (5%/25% quantiles, EMA, no known `E_min`) |
| prep init | vacuum |
| ham seed | `27700 + 100 n` |
| trial seed | `41000 + 1000 n + 10 hid + trial` |

The previous 70-SPSA L=3…20 cells are **superseded**. They live in
`results_70spsa_superseded/` and are ignored by resume / `SCALE_CONCLUSION.md`.

## Hardware embedding

Each transmon = 1 logical bit. Each cavity at `FOCK_CUTOFF=8` = 3 Fock
bits (binary, MSB first). When n exceeds the current capacity, append
**one transmon and one cavity**. Idle modes (0 assigned bits) stay
vacuum and are **omitted from the simulated tensor**.

Locked n=7…11 map (bit 0 = MSB):

| bits | mode | dim | role |
|------|------|-----|------|
| 0 | T0 | 2 | transmon |
| 1 | T1 | 2 | transmon |
| 2–4 | C0 | 8 | cavity, 3 Fock bits (MSB first) |
| 5–7 | C1 | 8 | cavity |
| 8–10 | C2 | 8 | cavity |

| n | hardware plan | simulated | dims | dim | pairs | notes |
|---|---------------|-----------|------|-----|-------|-------|
| 7 | 2T×3C parent | 1T×2C | `(2,8,8)` | 128 | 2 | production subspace; T1,C2 idle |
| 8 | 2T×3C | 2T×2C | `(2,2,8,8)` | 256 | 4 | C2 idle |
| 9–11 | 2T×3C | 2T×3C | `(2,2,8,8,8)` | 2048 | 6 | C2 bits 1…3; n=11 exact fill |
| 12 | 3T×4C | 3T×3C | `(2,2,2,8,8,8)` | 4096 | 9 | T2 added; leftover 9 bits fill C0–C2; C3 idle |
| 13–15 | 3T×4C | 3T×4C | `(2,2,2,8,8,8,8)` | 32768 | 12 | C3 bits 1…3 |
| 16 | 4T×5C | 4T×4C | 4T×4C | 65536 | 16 | T3+C4 added; C4 idle |
| 17–19 | 4T×5C | 4T×5C | 4T×5C | 524288 | 20 | usually hopeless as a full 200-trial cell |

`n_params = n_T + 2 n_C + 4 L n_T n_C` on the **simulated** register.
ECD stays local-gate statevector (no `dim²` dense unitaries).

ECD layer: one `ECD(β) R(θ, φ)` pair on every live transmon–cavity edge,
cavity-major then transmon. For `n=7` that is exactly the production UER
order (T0–C0, then T0–C1). `L` such layers. Prep is a product
`⊗ Ry(θ_i)|0⟩ ⊗ |α_j⟩`, jointly SPSA-optimized from vacuum (200 steps,
production gains).

## Hamiltonians

Fresh unique-planted 4-SAT → diagonal Ising, same construction as
`Hamiltonians/four_sat.py`, written under `system_size_scaling/Hamiltonians/n{N}/`.
Clause density target `round(n · 18/7)` → `{18,21,23,26,28,31,33,36,39}` for `n=7…15`.
The uniqueness loop may land in a band around that target (not the old
fixed 12–20 window).

## Commands

From the repo root:

```bash
# 20 instances for one n (or n=12…15). Does not touch n=7…11.
python -m system_size_scaling.generate_hamiltonians --n 12
python -m system_size_scaling.generate_hamiltonians --all

# one n: start at L=4, increment until ≥90% or L=12 (n≥12)
python -m system_size_scaling.run_one_n --n 12

# higher-n ladder n=12 → 15 (generates missing Hamiltonians)
python -m system_size_scaling.run_ladder

# optional scout when dim 32768 × 12 pairs is too slow: 5 H × 4 trials
python -m system_size_scaling.run_one_n --n 13 --scout

# smoke: 1 H × 1 trial × L=4 × 2 SPSA steps
python -m system_size_scaling.run_one_n --n 8 --L 4 --max-hamiltonians 1 --n-trials 1 --outer-iter 2 --no-sweep

# unit tests
python -m pytest system_size_scaling/tests -q
```

`--workers` defaults to 1 (in-process). A multi-process pool is slower here
unless BLAS is pinned; use `--workers 1`. Results land in
`results/n{N}_L{LL}.json` and `results/n{N}_summary.json`.
`SCALE_CONCLUSION.md` is rewritten after each depth.
Resume skips a cell only if it is complete **and** `outer_iter=200`.
