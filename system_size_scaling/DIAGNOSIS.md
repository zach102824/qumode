# n=8 ECD depth collapse — diagnosis

Question: why did noiseless ECD bitstring success on n=8 fall as L grew
(70 SPSA: L4 39/200 → L5 12/200 → ~0), and is that a bug?

## 1. Embedding round-trip (n=8, n=9)

- n=8: exhaustive logical encode→decode **PASS** (256 strings). Planted GS energy at encode is 0. Hybrid E=0 count mean 1.0 (unique=True).
- n=9: exhaustive logical encode→decode **PASS** (512 strings). Planted GS energy at encode is 0. Hybrid E=0 count mean 4.0 (unique=False).

n=8 is an exact fill of T0+T1+C0+C1 (dim 256 = 2^8): **no Fock aliasing**. A decoding bug cannot explain the L-collapse on n=8.

n=9: C2 has 1 logical bit on an 8-level cavity, so several Fock states decode to the same bitstring and share energy 0. That is a n=9+ caveat, not the n=8 collapse.

## 2. Diff vs working n=7 production path

- Occupation decode vs `bits_from_qnm` / `qnm_from_bits`: **PASS** (mismatches=0).
- L=4 vacuum ECD statevector fidelity vs QuTiP `prepare_state`: **1.000000000000** (PASS).
- Joint parameter count n=7 L=4: 37 (production 37).
- n=8 L=4 uses **4 ECD pairs** (2T×2C) vs production **2 pairs** (1T×2C), so 70 params vs 37 at the same depth. Same SPSA gains (`a=0.2, c=0.15`).

## 70-SPSA scoreboard (superseded, for the collapse shape)

### n=7

| L | k/N | success | mean p(GS) | mean ⟨H⟩ |
|---|-----|---------|------------|----------|
| 4 | 168/200 | 0.840 | 0.1926 | 0.933 |
| 5 | 156/200 | 0.780 | 0.1608 | 0.967 |
| 6 | 100/200 | 0.500 | 0.1091 | 1.026 |
| 7 | 76/200 | 0.380 | 0.0746 | 1.053 |
| 8 | 53/200 | 0.265 | 0.0481 | 1.085 |
| 9 | 28/200 | 0.140 | 0.0284 | 1.101 |
| 10 | 16/200 | 0.080 | 0.0208 | 1.110 |
| 11 | 10/200 | 0.050 | 0.0159 | 1.111 |
| 12 | 6/200 | 0.030 | 0.0143 | 1.118 |
| 13 | 7/200 | 0.035 | 0.0141 | 1.125 |
| 14 | 4/200 | 0.020 | 0.0119 | 1.122 |
| 15 | 3/200 | 0.015 | 0.0112 | 1.126 |
| 16 | 5/200 | 0.025 | 0.0118 | 1.126 |
| 17 | 2/200 | 0.010 | 0.0117 | 1.124 |
| 18 | 2/200 | 0.010 | 0.0111 | 1.129 |
| 19 | 1/200 | 0.005 | 0.0109 | 1.122 |
| 20 | 3/200 | 0.015 | 0.0098 | 1.126 |

### n=8

| L | k/N | success | mean p(GS) | mean ⟨H⟩ |
|---|-----|---------|------------|----------|
| 4 | 39/200 | 0.195 | 0.0256 | 1.294 |
| 5 | 12/200 | 0.060 | 0.0096 | 1.313 |
| 6 | 3/200 | 0.015 | 0.0072 | 1.309 |
| 7 | 3/200 | 0.015 | 0.0064 | 1.308 |
| 8 | 1/200 | 0.005 | 0.0058 | 1.309 |
| 9 | 1/200 | 0.005 | 0.0056 | 1.315 |
| 10 | 2/200 | 0.010 | 0.0058 | 1.314 |
| 11 | 2/200 | 0.010 | 0.0054 | 1.315 |
| 12 | 3/200 | 0.015 | 0.0055 | 1.312 |
| 13 | 2/200 | 0.010 | 0.0057 | 1.318 |
| 14 | 0/200 | 0.000 | 0.0050 | 1.314 |
| 15 | 1/200 | 0.005 | 0.0058 | 1.311 |
| 16 | 0/200 | 0.000 | 0.0053 | 1.314 |
| 17 | 1/200 | 0.005 | 0.0050 | 1.318 |
| 18 | 1/200 | 0.005 | 0.0047 | 1.311 |
| 19 | 0/200 | 0.000 | 0.0047 | 1.315 |
| 20 | 1/200 | 0.005 | 0.0047 | 1.319 |

## 4. Random-init success (no SPSA)

Uniform 1/2^n for n=8 is 0.003906.

| L | k/N | success | mean p(GS) | 1/2^n |
|---|-----|---------|------------|-------|
| 4 | 4/2000 | 0.00200 | 0.00331 | 0.00391 |
| 8 | 6/2000 | 0.00300 | 0.00386 | 0.00391 |
| 12 | 4/2000 | 0.00200 | 0.00404 | 0.00391 |

## Root cause

Embedding is not the bug: n=8 planted bitstrings encode→decode, hybrid ground energy is unique, and dim=256=2^8. n=7 decode matches production `bits_from_qnm`. n=7 L=4 ECD statevector matches production QuTiP to numerical precision. Random-init success (L=4 0.0020, L=8 0.0030, L=12 0.0020) is order-1/2^n (0.0039), so argmax is not stuck on a single garbage label.

## Live ladder

Controlled study not finished; do not start the live L-sweep yet.

