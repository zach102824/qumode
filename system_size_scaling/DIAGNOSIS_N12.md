# n=12 ECD 72.5% — diagnosis

Question: why did noiseless ECD bitstring success drop from n=11 L=4 **188/200 (94%)** to n=12 L=4 **145/200 (72.5%)** at the same 200-SPSA protocol with `a = 0.2 √(37/n_params)`? Zach does **not** want a deeper-L sweep. Diagnose n=12 only, the same way PR #15 diagnosed the n=8 collapse.

Hypotheses (verify, do not assume): under-training again (200 steps for 153 params) **or** a T2 wiring bug (first extra transmon).

n=13–15 L=5 work was cancelled — ignored here. L stays 4.

## Symptom (live ladder, unchanged)

| n | L | k/N | success | n_params | a | dim | pairs | mean p(GS) |
|---|---|-----|---------|----------|---|-----|-------|------------|
| 11 | 4 | 188/200 | 0.940 | 104 | 0.1193 | 2048 | 6 | 0.0610 |
| 12 | 4 | 145/200 | 0.725 | 153 | 0.0984 | 4096 | 9 | 0.0116 |
| 12 | 5 | 100/200 | 0.500 | 189 | 0.0885 | 4096 | 9 | 0.0044 |

n=12 L=5 is worse, same shape as n=8 when the SPSA budget was too small.

## 1. Embedding round-trip (n=12)

- Exhaustive logical encode→decode **PASS** (4096 strings, failures=0).
- Exhaustive hybrid labels (dim 4096): decode→encode→decode failures=0.
- Planted GS energy at encode is 0. Hybrid E=0 count mean 1.0 (unique=True).
- Modes ['T0', 'T1', 'T2', 'C0', 'C1', 'C2'], dim [2, 2, 2, 8, 8, 8], pairs=9, idle=['C3'].

n=12 is an exact fill of T0+T1+T2+C0+C1+C2 (dim 4096 = 2^12): **no Fock aliasing**. A decoding bug cannot explain the 72.5%.

## 2. Diff vs working n=11 (T2 wiring)

Wiring / pair-order / decode **PASS**.

- `ecd_pairs` is 3T×3C = 9, cavity-major then transmon, remapped axes [[0, 3], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [0, 5], [1, 5], [2, 5]]. **PASS**.
- Idle C3 omitted from the tensor: idle=['C3'] (**PASS**). Simulated dims=[2, 2, 2, 8, 8, 8] (not 3T×4C / 32768).
- Param-count formula `n_T + 2 n_C + 4 L n_T n_C` → 153 at L=4 (**PASS**).
- Occupation decode vs `bits_from_qnm` analog (transmons then Fock MSB-first): n=12 mismatches=0 (**PASS**); n=11 mismatches=0 (**PASS**).
- Prep `Ry(π)` on T2 (other prep params 0) occupies `[0, 0, 1, 0, 0, 0]` (want `[0,0,1,0,0,0]`) **PASS**.
- One ECD(β=1) on (T2, C0) from vacuum puts p(T2=1)=0.9999999999999999 (**PASS**).
- Bit partition n=11: `['T0', 'T1', 'C0[fock bit 2]', 'C0[fock bit 1]', 'C0[fock bit 0]', 'C1[fock bit 2]', 'C1[fock bit 1]', 'C1[fock bit 0]', 'C2[fock bit 2]', 'C2[fock bit 1]', 'C2[fock bit 0]']`. n=12 inserts T2 as logical bit 2 (True), so C0 shifts from bits 2–4 to 3–5. That is the intended map — n=12 Hamiltonians are 12-bit instances, not n=11 plus a flag.

- QuTiP is not installed in this environment. n=7 production fidelity is the PR #15 record: **1.000000000000** (DIAGNOSIS.md). n=11 has no production 2T×3C path to compare.

Live n=12 L=4 cell, sliced for a T2 bug (would show up as a bit-2 failure mode):

- Success vs planted T2 bit: T2=0 → 23/30 (76.7%), T2=1 → 122/170 (71.8%).
- Bit-wise ML==GS accuracy (uniform, T2 is not special): b0=0.865, b1=0.865, b2=0.875, b3=0.900, b4=0.880, b5=0.865, b6=0.890, b7=0.855, b8=0.855, b9=0.895, b10=0.870, b11=0.860.
- Misses where T2 bit is wrong: 25/55. Miss Hamming mean 5.545454545454546 (random ≈ 6).
- p(GS) on hits mean 0.015378383902723846 (max 0.08252953249841069); on misses mean 0.0016367247758699722. n=11 mean p(GS)=0.060992788639245836 (max 0.2787603209078439).

Argmax is already right 72.5% of the time, but the wavefunction is still almost flat (hit p(GS) ≈ 0.015, n=11 ≈ 0.061). That is the n=8 under-training shape, not a T2 decode/pair bug.

## 3. Random-init success (no SPSA)

Uniform 1/2^12 = 0.000244. Observed 1/4000 = 0.00025 (ratio to uniform 1.024). mean p(GS)=0.0002337770681431419.

Order-1/4096, so argmax is not stuck on a single garbage label and the planted GS is not accidentally dark.

## Root cause

Embedding is not the bug: exhaustive 4096-string encode→decode passes, planted hybrid ground energy is unique (dim=4096=2^12), C3 is idle/omitted. T2 wiring is not the bug: 9 remapped 3T×3C pairs, prep Ry on T2, ECD(T2,C0) flips T2, decode matches the bits_from_qnm analog on all 4096 labels. Live 145/200 is not T2-asymmetric (T2=0 success 0.7666666666666667, T2=1 success 0.7176470588235294); bit-wise accuracy is flat. Hit p(GS) stays tiny (0.015378383902723846). Random-init success 0.00025 is order-1/4096 (0.00024).

## Fix / n=12 L=4

Optimizer fleet not finished; do not change the live n=12 protocol yet.

