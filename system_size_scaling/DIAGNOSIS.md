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

## 3. Controlled n=8 study (200 joint SPSA)

### default_a (outer_iter=200)

| L | k/N | success | mean p(GS) | max p(GS) | mean ⟨H⟩ | mean\|β\| | a | n_params |
|---|-----|---------|------------|-----------|----------|-----------|---|----------|
| 4 | 164/200 | 0.820 | 0.1475 | 0.5187 | 1.148 | 1.963 | 0.2000 | 70 |
| 8 | 8/200 | 0.040 | 0.0079 | 0.0306 | 1.313 | 2.098 | 0.2000 | 134 |
| 12 | 1/200 | 0.005 | 0.0064 | 0.0261 | 1.312 | 2.089 | 0.2000 | 198 |

Mean H0 traces p(GS) vs step at L=4:

| step | mean p(GS) | mean ⟨H⟩ |
|------|------------|----------|
| 0 | 0.0012 | 1.355 |
| 1 | 0.0012 | 1.355 |
| 6 | 0.0038 | 1.327 |
| 11 | 0.0022 | 1.312 |
| 16 | 0.0038 | 1.360 |
| 21 | 0.0081 | 1.314 |
| 26 | 0.0043 | 1.283 |
| 31 | 0.0040 | 1.281 |
| 36 | 0.0130 | 1.321 |
| 41 | 0.0053 | 1.354 |
| 46 | 0.0039 | 1.345 |
| 51 | 0.0037 | 1.292 |
| 196 | 0.1901 | 1.041 |

Mean H0 traces p(GS) vs step at L=8:

| step | mean p(GS) | mean ⟨H⟩ |
|------|------------|----------|
| 0 | 0.0049 | 1.311 |
| 1 | 0.0049 | 1.311 |
| 6 | 0.0025 | 1.337 |
| 11 | 0.0014 | 1.323 |
| 16 | 0.0032 | 1.293 |
| 21 | 0.0041 | 1.332 |
| 26 | 0.0059 | 1.275 |
| 31 | 0.0053 | 1.296 |
| 36 | 0.0040 | 1.306 |
| 41 | 0.0079 | 1.303 |
| 46 | 0.0092 | 1.346 |
| 51 | 0.0039 | 1.362 |
| 196 | 0.0058 | 1.294 |

Mean H0 traces p(GS) vs step at L=12:

| step | mean p(GS) | mean ⟨H⟩ |
|------|------------|----------|
| 0 | 0.0025 | 1.327 |
| 1 | 0.0025 | 1.327 |
| 6 | 0.0035 | 1.341 |
| 11 | 0.0045 | 1.343 |
| 16 | 0.0043 | 1.320 |
| 21 | 0.0030 | 1.301 |
| 26 | 0.0053 | 1.308 |
| 31 | 0.0034 | 1.298 |
| 36 | 0.0024 | 1.333 |
| 41 | 0.0025 | 1.350 |
| 46 | 0.0060 | 1.296 |
| 51 | 0.0038 | 1.304 |
| 196 | 0.0119 | 1.303 |

### scaled_a (outer_iter=200)

| L | k/N | success | mean p(GS) | max p(GS) | mean ⟨H⟩ | mean\|β\| | a | n_params |
|---|-----|---------|------------|-----------|----------|-----------|---|----------|
| 4 | 196/200 | 0.980 | 0.1965 | 0.6891 | 1.077 | 1.672 | 0.1454 | 70 |
| 8 | 84/200 | 0.420 | 0.0322 | 0.3242 | 1.281 | 1.651 | 0.1051 | 134 |
| 12 | 24/200 | 0.120 | 0.0112 | 0.0441 | 1.305 | 1.610 | 0.0865 | 198 |

Mean H0 traces p(GS) vs step at L=4:

| step | mean p(GS) | mean ⟨H⟩ |
|------|------------|----------|
| 0 | 0.0012 | 1.355 |
| 1 | 0.0012 | 1.355 |
| 6 | 0.0038 | 1.360 |
| 11 | 0.0130 | 1.361 |
| 16 | 0.0090 | 1.306 |
| 21 | 0.0049 | 1.356 |
| 26 | 0.0111 | 1.287 |
| 31 | 0.0044 | 1.322 |
| 36 | 0.0115 | 1.252 |
| 41 | 0.0145 | 1.250 |
| 46 | 0.0330 | 1.261 |
| 51 | 0.0313 | 1.310 |
| 196 | 0.2409 | 1.041 |

Mean H0 traces p(GS) vs step at L=8:

| step | mean p(GS) | mean ⟨H⟩ |
|------|------------|----------|
| 0 | 0.0049 | 1.311 |
| 1 | 0.0049 | 1.311 |
| 6 | 0.0062 | 1.309 |
| 11 | 0.0056 | 1.292 |
| 16 | 0.0110 | 1.310 |
| 21 | 0.0098 | 1.366 |
| 26 | 0.0162 | 1.274 |
| 31 | 0.0103 | 1.270 |
| 36 | 0.0127 | 1.274 |
| 41 | 0.0086 | 1.305 |
| 46 | 0.0054 | 1.341 |
| 51 | 0.0075 | 1.336 |
| 196 | 0.0212 | 1.245 |

Mean H0 traces p(GS) vs step at L=12:

| step | mean p(GS) | mean ⟨H⟩ |
|------|------------|----------|
| 0 | 0.0025 | 1.327 |
| 1 | 0.0025 | 1.327 |
| 6 | 0.0066 | 1.295 |
| 11 | 0.0046 | 1.304 |
| 16 | 0.0135 | 1.255 |
| 21 | 0.0097 | 1.276 |
| 26 | 0.0064 | 1.293 |
| 31 | 0.0082 | 1.313 |
| 36 | 0.0087 | 1.327 |
| 41 | 0.0076 | 1.355 |
| 46 | 0.0054 | 1.323 |
| 51 | 0.0095 | 1.386 |
| 196 | 0.0079 | 1.270 |

## Root cause

**Not a circuit / decode bug.** n=8 planted encode→decode is exact, hybrid ground is unique, n=7 ECD matches production QuTiP (fidelity 1.0), and random-init success is ~1/256.

**Two stacked optimizer failures, both about parameter count:**

1. **70 SPSA was far too few** for n=8 L=4 (70 joint params vs production 37). Same seeds at 200 SPSA / a=0.2 jump **39/200 → 164/200**. H0 traces: p(GS) 0.001 → 0.19 over 200 steps (the optimizer is still climbing at the budget).
2. **Production `a=0.2` is too large once n_params > 37.** SPSA RMS step grows ~√d. Scaling `a ← 0.2 × √(37 / n_params)` at 200 steps:

| L | n_params | a=0.2 | scaled a | scaled k/200 |
|---|----------|------:|----------:|-------------:|
| 4 | 70 | 164/200 (82%) | 0.145 | **196/200 (98%)** |
| 8 | 134 | 8/200 (4%) | 0.105 | 84/200 (42%) |
| 12 | 198 | 1/200 (0.5%) | 0.086 | 24/200 (12%) |

Deeper L still loses even with scaled `a` (H0 traces at L=8/12 never leave p(GS)~0.01). Extra layers add parameters that a **fixed** 200-step budget cannot train. Blindly raising L toward 40 with default `a=0.2` would reproduce the old collapse.

The 70-SPSA L=4→20 curve was this same under-training, made worse by the short budget. n=7 showed the identical shape (L4 168/200 → L12 6/200 at 70 steps) despite a correct circuit.

## Live ladder

Sane, and **finished**. With `a = 0.2 * sqrt(37 / n_params)` and 200 joint SPSA, every n hits ≥90% at **L=4** (do not raise L):

| n | L* | k/200 | a |
|---|----|------:|--:|
| 7 | 4 (PR #14) | 186/200 | 0.200 (37 params) |
| 8 | 4 | 196/200 | 0.145 |
| 9 | 4 | 198/200 | 0.119 |
| 10 | 4 | 196/200 | 0.119 |
| 11 | 4 | 188/200 | 0.119 |

n=9 C2 Fock aliasing did not block mode-finding (198/200). Noisy GDR-in-loop remains deferred.

