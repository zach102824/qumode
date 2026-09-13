from __future__ import annotations

import numpy as np
import pytest

from system_size_scaling.config import (
    FOCK_CUTOFF,
    HARDWARE_DIM,
    clause_window,
    ham_dir,
    hardware_plan,
    live_register,
    n_params_for,
    plan_hardware_dim,
)
from system_size_scaling.ecd import hybrid_energy_tensor
from system_size_scaling.embedding import embedding_for_n, hardware_idle_modes
from system_size_scaling.four_sat import load_instance


def test_n11_is_exact_fill():
    emb = embedding_for_n(11)
    assert emb.dim == HARDWARE_DIM == 2048
    assert emb.dims == (2, 2, 8, 8, 8)
    assert [m.name for m in emb.modes] == ["T0", "T1", "C0", "C1", "C2"]
    assert [m.n_bits for m in emb.modes] == [1, 1, 3, 3, 3]
    assert hardware_idle_modes(11) == []


def test_n7_is_production_subspace():
    emb = embedding_for_n(7)
    assert emb.dims == (2, 8, 8)
    assert emb.dim == 128
    assert [m.name for m in emb.modes] == ["T0", "C0", "C1"]
    assert [m.n_bits for m in emb.modes] == [1, 3, 3]
    assert emb.n_pairs == 2
    assert hardware_idle_modes(7) == ["T1", "C2"]


def test_n8_to_10_fit_in_2048():
    assert embedding_for_n(8).dim == 256
    assert embedding_for_n(9).dim == 2048
    assert embedding_for_n(10).dim == 2048
    assert embedding_for_n(8).dims == (2, 2, 8, 8)
    assert embedding_for_n(9).modes[-1].name == "C2"
    assert embedding_for_n(9).modes[-1].n_bits == 1
    assert embedding_for_n(10).modes[-1].n_bits == 2


@pytest.mark.parametrize("n", [7, 8, 9, 10, 11, 12, 13, 14, 15])
def test_encode_decode_roundtrip(n: int):
    emb = embedding_for_n(n)
    rng = np.random.default_rng(n)
    for _ in range(20):
        bits = rng.integers(0, 2, size=n)
        occ = emb.encode_bits(bits)
        got = emb.decode_occupations(occ)
        assert np.array_equal(got, bits)


@pytest.mark.parametrize("n", [8, 9, 12])
def test_exhaustive_logical_roundtrip(n: int):
    emb = embedding_for_n(n)
    for idx in range(1 << n):
        bits = np.array([(idx >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)
        occ = emb.encode_bits(bits)
        assert np.array_equal(emb.decode_occupations(occ), bits)


@pytest.mark.parametrize("n", [13, 14, 15])
def test_sample_logical_roundtrip_n13_to_15(n: int):
    emb = embedding_for_n(n)
    rng = np.random.default_rng(1000 + n)
    # 2^n is 8k–32k; sample rather than exhaust the logical cube.
    for idx in rng.integers(0, 1 << n, size=64):
        bits = np.array([(int(idx) >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)
        occ = emb.encode_bits(bits)
        assert np.array_equal(emb.decode_occupations(occ), bits)
        # Unused Fock bits stay 0 on encode (canonical computational embed).
        for mode, nocc in zip(emb.modes, occ, strict=True):
            if mode.kind == "cavity" and mode.n_bits < 3:
                assert nocc < (1 << mode.n_bits)


@pytest.mark.parametrize("n", [8, 9])
def test_planted_ground_roundtrip_and_unique_energy(n: int):
    emb = embedding_for_n(n)
    paths = sorted(ham_dir(n).glob("four_sat_[0-9][0-9][0-9].npz"))
    assert len(paths) >= 20
    for path in paths:
        inst = load_instance(path)
        bits = np.array([int(c) for c in inst["ground_bitstring"]], dtype=int)
        occ = emb.encode_bits(bits)
        assert np.array_equal(emb.decode_occupations(occ), bits)
        tensor = hybrid_energy_tensor(emb, inst["logical_energies"])
        assert tensor[occ] == pytest.approx(0.0, abs=1e-12)
        n_zero = int(np.sum(np.isclose(tensor, 0.0)))
        if n == 8:
            # Exact fill of T0+T1+C0+C1: unique hybrid ground.
            assert n_zero == 1
        else:
            # Unused C2 Fock bits alias the same logical string.
            assert n_zero >= 1


def test_n7_decode_matches_production_bits_from_qnm():
    """Production Eq. (26): |q, n, m> → 1+3+3 bits, Fock MSB first."""
    emb = embedding_for_n(7)
    for q in range(2):
        for nocc in range(8):
            for m in range(8):
                ours = emb.decode_occupations((q, nocc, m))
                bits = [int(q)]
                for k in range(2, -1, -1):
                    bits.append((int(nocc) >> k) & 1)
                for k in range(2, -1, -1):
                    bits.append((int(m) >> k) & 1)
                assert np.array_equal(ours, bits)


def test_clause_targets_match_spec():
    assert clause_window(7)[1] == 18
    assert clause_window(8)[1] == 21
    assert clause_window(9)[1] == 23
    assert clause_window(10)[1] == 26
    assert clause_window(11)[1] == 28
    assert clause_window(12)[1] == 31
    assert clause_window(13)[1] == 33
    assert clause_window(14)[1] == 36
    assert clause_window(15)[1] == 39


def test_hardware_growth_rule():
    assert hardware_plan(7) == (1, 2)
    assert hardware_plan(8) == (2, 3)
    assert hardware_plan(11) == (2, 3)
    assert hardware_plan(12) == (3, 4)
    assert hardware_plan(15) == (3, 4)
    assert hardware_plan(16) == (4, 5)
    assert hardware_plan(19) == (4, 5)
    assert live_register(12) == (3, 3)
    assert live_register(13) == (3, 4)
    assert live_register(15) == (3, 4)
    assert live_register(16) == (4, 4)
    assert plan_hardware_dim(2, 3) == HARDWARE_DIM == 2048
    assert plan_hardware_dim(3, 4) == 32768
    assert FOCK_CUTOFF == 8


def test_n12_omits_idle_c3():
    emb = embedding_for_n(12)
    assert emb.dims == (2, 2, 2, 8, 8, 8)
    assert emb.dim == 4096
    assert [m.name for m in emb.modes] == ["T0", "T1", "T2", "C0", "C1", "C2"]
    assert [m.n_bits for m in emb.modes] == [1, 1, 1, 3, 3, 3]
    assert emb.n_pairs == 9
    assert hardware_idle_modes(12) == ["C3"]
    # n_params = n_T + 2 n_C + 4 L n_T n_C
    assert n_params_for(12, 4) == 3 + 6 + 4 * 4 * 9 == 153
    assert embedding_for_n(12).n_prep_params == 9


def test_n13_to_15_use_3t4c():
    for n, c3_bits in ((13, 1), (14, 2), (15, 3)):
        emb = embedding_for_n(n)
        assert emb.dims == (2, 2, 2, 8, 8, 8, 8)
        assert emb.dim == 32768
        assert [m.name for m in emb.modes] == ["T0", "T1", "T2", "C0", "C1", "C2", "C3"]
        assert emb.modes[-1].n_bits == c3_bits
        assert emb.n_pairs == 12
        assert hardware_idle_modes(n) == []
        assert n_params_for(n, 4) == 3 + 8 + 4 * 4 * 12 == 203


def test_n16_omits_idle_c4_if_encoded():
    emb = embedding_for_n(16)
    assert emb.n_transmons == 4
    assert emb.n_cavities == 4
    assert emb.dim == 16 * 8**4 == 65536
    assert emb.n_pairs == 16
    assert hardware_idle_modes(16) == ["C4"]
    assert n_params_for(16, 4) == 4 + 8 + 4 * 4 * 16 == 268


def test_n12_exhaustive_hybrid_roundtrip():
    """4096 hybrid labels: encode(decode(occ)) need not be identity (unused Fock
    bits), but logical encode↔decode is. Exact fill, so every Fock 0..7 is used."""
    emb = embedding_for_n(12)
    assert emb.dim == 4096
    for idx in range(emb.dim):
        bits = emb.decode_index(idx)
        occ = emb.encode_bits(bits)
        assert np.array_equal(emb.decode_occupations(occ), bits)
