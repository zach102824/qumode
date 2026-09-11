# ECD system-size scaling (isolated)

Noiseless Gibbs **ECD only** bitstring success vs system size `n=7…11`.
All code and results for this study live in this folder. Upstream
`src/`, `scripts/`, `Hamiltonians/`, and `Error_mitigation/` are not edited.

## Question

At fixed clause density ≈ `18/7 ≈ 2.57`, how does ECD depth `L` needed for
**≥ 90%** most-likely-bitstring success (`k/200`) grow from `n=8` to `n=11`?
`n=7` L* is **prior data** from [PR #14](https://github.com/zach102824/qumode/pull/14)
(ECD L4 = 186/200), not re-run here.

Success = `most_likely_bitstring == ground_bitstring`.
Each `(n, L)` cell is **20 Hamiltonians × 10 trials = 200**.
Cost is Gibbs `-ln⟨e^{-ηE}⟩` with production `sampled_tail` η.
No SNAP. No GDR. No noise.

## Hardware embedding

Target Hilbert space: **2 transmons × 3 cavities × 8 levels = dim 2048**.

Canonical 11-bit map (bit 0 = MSB):

| bits | mode | dim | role |
|------|------|-----|------|
| 0 | T0 | 2 | transmon |
| 1 | T1 | 2 | transmon |
| 2–4 | C0 | 8 | cavity, 3 Fock bits (MSB first) |
| 5–7 | C1 | 8 | cavity |
| 8–10 | C2 | 8 | cavity |

| n | active modes | dims | dim | notes |
|---|--------------|------|-----|-------|
| 7 | T0, C0, C1 | `(2,8,8)` | 128 | production-like **subspace** of 2048 (T1 and C2 idle) |
| 8 | T0, T1, C0, C1 | `(2,2,8,8)` | 256 | C2 idle; subspace of 2048 |
| 9 | T0, T1, C0, C1, C2 | `(2,2,8,8,8)` | 2048 | C2 contributes 1 logical bit |
| 10 | all five | 2048 | C2 contributes 2 logical bits |
| 11 | all five | 2048 | **exact fill** |

Fock decoding uses the lowest `n_bits` of the occupation (MSB-first among
those bits). Idle hardware modes stay in vacuum and are omitted from the
simulated tensor.

ECD layer: one `ECD(β) R(θ, φ)` pair on every live transmon–cavity edge,
cavity-major then transmon. For `n=7` that is exactly the production UER
order (T0–C0, then T0–C1). `L` such layers. Prep is a product
`⊗ Ry(θ_i)|0⟩ ⊗ |α_j⟩`, jointly SPSA-optimized from vacuum (70 steps,
production gains).

## Hamiltonians

Fresh unique-planted 4-SAT → diagonal Ising, same construction as
`Hamiltonians/four_sat.py`, written under `system_size_scaling/Hamiltonians/n{N}/`.
Clause density target `round(n · 18/7)` → `{18,21,23,26,28}` for `n=7…11`.
The uniqueness loop may land in a band around that target (not the old
fixed 12–20 window).

## Commands

From the repo root:

```bash
# 20 instances for one n (or n=8…11)
python -m system_size_scaling.generate_hamiltonians --n 8
python -m system_size_scaling.generate_hamiltonians --all

# one n: start at L=4, increment until ≥90% or L=20
python -m system_size_scaling.run_one_n --n 8

# live ladder n=8 → 11 (generates missing Hamiltonians)
python -m system_size_scaling.run_ladder

# smoke: 1 H × 1 trial × L=4 × 2 SPSA steps
python -m system_size_scaling.run_one_n --n 8 --L 4 --max-hamiltonians 1 --n-trials 1 --outer-iter 2 --no-sweep

# unit tests
python -m pytest system_size_scaling/tests -q
```

`--workers` defaults to 1 (in-process). A multi-process pool is slower here
unless BLAS is pinned; use `--workers 1`. Results land in
`results/n{N}_L{LL}.json` and `results/n{N}_summary.json`.
`SCALE_CONCLUSION.md` is rewritten after each n.

## Protocol knobs

| item | value |
|------|-------|
| trials | 10 / Hamiltonian, 20 H |
| L sweep | n≥8: 4 … 20, stop at ≥ 90%; n=7 not re-run |
| SPSA | joint 70, `a=0.2, c=0.15, A=10, α=0.602, γ=0.101` |
| η | `sampled_tail` (5%/25% quantiles, EMA, no known `E_min`) |
| prep init | vacuum |
| ham seed | `27700 + 100 n` |
| trial seed | `41000 + 1000 n + 10 hid + trial` |
