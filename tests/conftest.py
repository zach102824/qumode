"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.data import load_reference
from qumode_vqe.hamiltonian import hybrid_hamiltonian
from qumode_vqe.vqe import HybridSimulator


@pytest.fixture(scope="session")
def reference():
    return load_reference()


@pytest.fixture(scope="session")
def reference_xvec(reference):
    return np.asarray(reference["xvec"], dtype=float)


@pytest.fixture
def noiseless_sim():
    return HybridSimulator(ndepth=5)


@pytest.fixture(scope="session")
def hybrid_h():
    return hybrid_hamiltonian()
