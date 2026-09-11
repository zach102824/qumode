from __future__ import annotations

import numpy as np
import pytest

from system_size_scaling.config import HARDWARE_DIM, clause_window, ham_dir
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


@pytest.mark.parametrize("n", [7, 8, 9, 10, 11])
def test_encode_decode_roundtrip(n: int):
    emb = embedding_for_n(n)
    rng = np.random.default_rng(n)
    for _ in range(20):
        bits = rng.integers(0, 2, size=n)
        occ = emb.encode_bits(bits)
        got = emb.decode_occupations(occ)
        assert np.array_equal(got, bits)


@pytest.mark.parametrize("n", [8, 9])
def test_exhaustive_logical_roundtrip(n: int):
    emb = embedding_for_n(n)
    for idx in range(1 << n):
        bits = np.array([(idx >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)
        occ = emb.encode_bits(bits)
        assert np.array_equal(emb.decode_occupations(occ), bits)


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
