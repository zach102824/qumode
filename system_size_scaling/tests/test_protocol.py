from __future__ import annotations

import pytest

from system_size_scaling.config import (
    L_MAX,
    L_START,
    OUTER_ITER,
    PROTOCOL_TAG,
    is_canonical_cell,
)
from system_size_scaling.io_util import row_from_curve, write_conclusion


def test_canonical_protocol_constants():
    assert OUTER_ITER == 200
    assert L_START == 4
    assert L_MAX == 40
    assert PROTOCOL_TAG == "200_joint_spsa_noiseless_a_scaled"


def test_spsa_a_scaled_matches_n7_at_ref():
    from system_size_scaling.config import N7_L4_NPARAMS, SPSA_A, spsa_a_scaled

    assert spsa_a_scaled(N7_L4_NPARAMS) == pytest.approx(SPSA_A)
    # n=8 L=4 is 70 params; scaled a is smaller
    assert spsa_a_scaled(70) < SPSA_A
    assert spsa_a_scaled(70) == pytest.approx(SPSA_A * (37 / 70) ** 0.5)


def test_is_canonical_cell_rejects_70_spsa():
    good = {"outer_iter": 200, "protocol": {"tag": PROTOCOL_TAG, "spsa": {"outer_iter": 200}}}
    old = {"outer_iter": 70, "protocol": {"spsa": {"outer_iter": 70}}}
    assert is_canonical_cell(good)
    assert not is_canonical_cell(old)
    assert not is_canonical_cell({})


def test_row_from_curve_soft_cap_is_l40():
    curve = [{"L": L, "k": 0, "n_total": 200, "success_prob": 0.01, "wall_s": 1.0} for L in range(4, 21)]
    row = row_from_curve(8, curve)
    assert row["status"] == "in_progress"
    assert row["L_star"] is None
    curve40 = curve + [{"L": 40, "k": 2, "n_total": 200, "success_prob": 0.01, "wall_s": 1.0}]
    row40 = row_from_curve(8, curve40)
    assert row40["status"] == "capped_L40_below_threshold"


def test_write_conclusion_marks_superseded_70(tmp_path, monkeypatch):
    import system_size_scaling.io_util as io_util

    monkeypatch.setattr(io_util, "ROOT", tmp_path)
    path = io_util.write_conclusion("unit-test rebuild")
    text = path.read_text(encoding="utf-8")
    assert "200 joint SPSA" in text
    assert "Superseded: 70-SPSA" in text
    assert "39/200" in text
    assert PROTOCOL_TAG in text
