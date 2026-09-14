from __future__ import annotations

import math

import numpy as np
import pytest

from system_size_scaling.config import N7_L4_NPARAMS, SPSA_A, n_params_for, spsa_a_scaled
from system_size_scaling.diagnose_n12 import (
    A_MODES,
    bits_from_occupations_explicit,
    check_n12_wiring,
    check_roundtrips,
    spsa_a_for_mode,
)
from system_size_scaling.ecd import apply_ecd, prep_to_ket, vacuum_prep
from system_size_scaling.embedding import embedding_for_n, hardware_idle_modes


def test_n12_wiring_check_passes():
    rec = check_n12_wiring()
    assert rec["ok"] is True
    assert rec["ecd_pairs"] == [
        [0, 3],
        [1, 3],
        [2, 3],
        [0, 4],
        [1, 4],
        [2, 4],
        [0, 5],
        [1, 5],
        [2, 5],
    ]
    assert rec["idle"] == ["C3"]
    assert rec["n_params_L4"] == 153


def test_n12_decode_matches_bits_from_qnm_analog():
    emb = embedding_for_n(12)
    for idx in range(emb.dim):
        occ = np.unravel_index(idx, emb.dims)
        ours = list(emb.decode_occupations(occ))
        explicit = bits_from_occupations_explicit(tuple(int(v) for v in occ), 3)
        assert ours == explicit


def test_n12_t2_prep_and_ecd():
    emb = embedding_for_n(12)
    prep = vacuum_prep(emb).copy()
    prep[2] = float(np.pi)
    ket = prep_to_ket(prep, emb)
    occ = tuple(int(v) for v in np.unravel_index(int(np.argmax(np.abs(ket) ** 2)), emb.dims))
    assert occ == (0, 0, 1, 0, 0, 0)
    psi0 = prep_to_ket(vacuum_prep(emb), emb)
    psi = apply_ecd(psi0, emb.dims, 2, 3, 1.0)
    p_t2 = float(np.sum(np.abs(psi.reshape(emb.dims)[:, :, 1, ...]) ** 2))
    assert p_t2 > 0.99


def test_n12_idle_c3_and_param_formula():
    emb = embedding_for_n(12)
    assert hardware_idle_modes(12) == ["C3"]
    assert "C3" not in [m.name for m in emb.modes]
    assert emb.n_pairs == 9
    assert n_params_for(12, 4) == 3 + 2 * 3 + 4 * 4 * 9 == 153


def test_a_modes_at_d153():
    assert spsa_a_for_mode(153, "current_sqrt37") == pytest.approx(spsa_a_scaled(153))
    assert spsa_a_for_mode(153, "unscaled") == pytest.approx(SPSA_A)
    assert spsa_a_for_mode(153, "milder_sqrt70") == pytest.approx(SPSA_A * math.sqrt(70 / 153))
    assert spsa_a_for_mode(153, "fourth_root37") == pytest.approx(SPSA_A * (N7_L4_NPARAMS / 153) ** 0.25)
    assert set(A_MODES) == {"current_sqrt37", "milder_sqrt70", "fourth_root37", "unscaled"}
    # milder and fourth-root sit between current √d and the unscaled production a
    cur = spsa_a_for_mode(153, "current_sqrt37")
    mild = spsa_a_for_mode(153, "milder_sqrt70")
    fourth = spsa_a_for_mode(153, "fourth_root37")
    assert cur < mild < SPSA_A
    assert cur < fourth < SPSA_A


def test_n12_roundtrip_smoke_without_hams():
    # Hamiltonians are on disk in this repo; full check is cheap (4096 ints).
    rec = check_roundtrips(12)
    assert rec["ok"]
    assert rec["exhaustive_logical"] == 4096
    assert rec["hybrid_E0_unique"]
