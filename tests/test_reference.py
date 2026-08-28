"""Stored notebook parameter vector reproduces the published noiseless result."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import TARGET_QNM
from qumode_vqe.vqe import HybridSimulator


def test_reference_metadata(reference):
    assert reference["ndepth"] == 5
    assert reference["nfocks"] == [8, 8]
    assert tuple(reference["target_qnm"]) == TARGET_QNM
    assert reference["xvec"].shape == (40,)
    assert "github.com/CQDMQD/codes_qumode_qubo" in reference["source_url"]
    assert reference["retrieved"] == "2026-08-20"


def test_reference_vector_energy_and_target_peak(reference_xvec):
    sim = HybridSimulator(ndepth=5)
    ev = sim.evaluate(reference_xvec, include_ideal=True)
    # Rounded 8-digit notebook vector; energy should sit very close to -11.996.
    assert ev.energy_physical == pytest.approx(-11.996059486626715, abs=5e-3)
    assert ev.energy_ideal == pytest.approx(ev.energy_physical, abs=1e-8)
    assert ev.most_likely == TARGET_QNM
    assert ev.target_prob_physical > 0.9
    assert ev.most_likely_bitstring == "0110000"
    assert ev.trace == pytest.approx(1.0, abs=1e-10)
    assert ev.purity == pytest.approx(1.0, abs=1e-8)
