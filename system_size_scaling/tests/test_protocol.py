from __future__ import annotations

import pytest

from system_size_scaling.config import (
    L_MAX,
    L_MAX_HIGHER,
    L_START,
    OUTER_ITER,
    PROTOCOL_TAG,
    is_canonical_cell,
    n_params_for,
    soft_cap_for_n,
    spsa_a_scaled,
)
from system_size_scaling.io_util import row_from_curve, write_conclusion


def test_canonical_protocol_constants():
    assert OUTER_ITER == 200
    assert L_START == 4
    assert L_MAX == 40
    assert L_MAX_HIGHER == 12
    assert soft_cap_for_n(11) == 40
    assert soft_cap_for_n(12) == 12
    assert PROTOCOL_TAG == "200_joint_spsa_noiseless_a_scaled"
    assert n_params_for(7, 4) == 37
    assert n_params_for(12, 4) == 153
    assert n_params_for(13, 4) == 203
    assert spsa_a_scaled(153) == pytest.approx(0.2 * (37 / 153) ** 0.5)


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
    curve12 = [{"L": L, "k": 0, "n_total": 200, "success_prob": 0.01, "wall_s": 1.0} for L in range(4, 13)]
    row12 = row_from_curve(12, curve12)
    assert row12["status"] == "capped_L12_below_threshold"


def test_merged_curve_unions_extra_depths():
    from system_size_scaling.io_util import merged_curve

    extra = [
        {"L": 6, "k": 1, "n_total": 20, "success_prob": 0.05, "wall_s": 1.0, "outer_iter": 200},
        {"L": 4, "k": 2, "n_total": 20, "success_prob": 0.10, "wall_s": 1.0, "outer_iter": 200},
    ]
    # n=99 has no disk cells; extra should sort by L.
    got = merged_curve(99, extra=extra)
    assert [c["L"] for c in got] == [4, 6]


def test_n12_scoreboard_uses_best_full_cell_not_last_scout():
    from system_size_scaling.io_util import cell_stats, curve_from_disk, row_from_curve

    curve = curve_from_disk(12)
    if not curve:
        pytest.skip("n=12 results not on disk")
    row = row_from_curve(12, curve)
    assert row["k"] == 145
    assert row["n_total"] == 200
    assert row["L_star"] is None
    assert row["status"] == "capped_L12_below_threshold"
    assert max(c["success_prob"] for c in curve) == pytest.approx(0.725)
    stats = cell_stats(12)
    assert stats is not None
    assert stats["L"] == 4
    assert stats["k"] == 145
    assert stats["n_total"] == 200


def test_write_conclusion_marks_superseded_70(tmp_path, monkeypatch):
    import system_size_scaling.io_util as io_util

    monkeypatch.setattr(io_util, "ROOT", tmp_path)
    path = io_util.write_conclusion("unit-test rebuild")
    text = path.read_text(encoding="utf-8")
    assert "200 joint SPSA" in text
    assert "Superseded: 70-SPSA" in text
    assert "39/200" in text
    assert PROTOCOL_TAG in text
    higher = (tmp_path / "HIGHER_N.md").read_text(encoding="utf-8")
    assert "n=12+" in higher
    assert "soft cap" in higher.lower() or "L=12" in higher
