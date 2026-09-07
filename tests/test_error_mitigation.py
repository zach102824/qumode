"""Fast checks for Error_mitigation kernels, twins, and readout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Error_mitigation.metrics import total_variation
from Error_mitigation.mitigation import (
    binomial_loss_kernel,
    choose_damp_alpha,
    damp_histogram,
    fit_gdr_interleave,
    fock_kernel,
    holdout_indices,
    observe_histogram,
    select_by_holdout,
    select_kt_method,
    select_research_method,
    choose_mix_alpha,
    classify_opt_quality,
    anneal_identity_weight,
    anneal_prior_theta,
    fisher_twin_weights,
    fit_gdr_afterburn,
    fit_gdr_anneal,
    fit_gdr_band,
    fit_gdr_eta,
    fit_gdr_split,
    mild_residual_allowed,
    oracle_kernels,
    readout_then_zne,
    richardson_lucy,
    run_readout_only,
    thermal_loss_kernel,
    zne_histogram,
    zne_then_readout,
)
from Error_mitigation.twins import designed_twin_plan, designed_twin_plan_grid
from Error_mitigation.noise_models import (
    circuit_noise,
    is_trivial_readout,
    readout_config,
    readout_spec,
)
from Error_mitigation.twins import build_twins, product_coherent_histogram, truncated_poisson
from qumode_vqe.hamiltonian import hybrid_energy_tensor
from qumode_vqe.measurement import is_column_stochastic, nearest_neighbor_fock_confusion
from qumode_vqe.params import random_parameters, random_snap_parameters
from qumode_vqe.vqe import HybridSimulator


def test_kernels_are_column_stochastic():
    for eta in (1.0, 0.9, 0.5):
        b = binomial_loss_kernel(eta, 8)
        t = thermal_loss_kernel(eta, 0.0, 8)
        f = fock_kernel(eta, 0.05, 0.01, 0.01, 0.02, 0.03, 8)
        assert is_column_stochastic(b)
        assert is_column_stochastic(t)
        assert is_column_stochastic(f)


def test_pure_loss_thermal_matches_binomial():
    eta = 0.8
    b = binomial_loss_kernel(eta, 8)
    t = thermal_loss_kernel(eta, 0.0, 8)
    # Truncation at n=L-1 makes the last column differ slightly.
    assert np.allclose(b[:, :7], t[:, :7], atol=0.02)


def test_readout_confusion_column_stochastic():
    for level in ("ideal", "readout_realistic", "readout_strong"):
        spec = readout_spec(level, n_shots=100, seed=0)
        cfg = readout_config(spec, n_fock=8)
        if cfg.qubit_c is not None:
            assert is_column_stochastic(cfg.qubit_c)
            assert is_column_stochastic(cfg.fock1_c)
            assert is_column_stochastic(cfg.fock2_c)
        assert is_column_stochastic(nearest_neighbor_fock_confusion(8, spec.p_nn))


def test_readout_only_skipped_on_ideal():
    spec = readout_spec("ideal", n_shots=100)
    assert is_trivial_readout(spec)
    p = np.zeros((2, 8, 8))
    p[0, 0, 0] = 1.0
    assert run_readout_only(p, spec, (2, 8, 8)) is None


def test_readout_only_reduces_confusion():
    spec = readout_spec("readout_strong", n_shots=0, seed=1)
    rng = np.random.default_rng(0)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    q = observe_histogram(p, spec, (2, 8, 8), seed=1)
    mit = run_readout_only(q, spec, (2, 8, 8))
    assert mit is not None
    assert total_variation(mit.histogram, p) < total_variation(q, p)


def test_rl_inverts_binomial_blur():
    eta = 0.85
    b = binomial_loss_kernel(eta, 8)
    p = np.zeros((2, 8, 8))
    p[1, 3, 2] = 1.0
    q = np.zeros_like(p)
    for n in range(8):
        for m in range(8):
            q[1, n, m] = b[n, 3] * b[m, 2]
    rec = richardson_lucy(q, np.eye(2), b, b, n_iter=40)
    assert rec[1, 3, 2] > 0.8


def test_ecd_gaussian_twins_match_poisson():
    sim = HybridSimulator(ndepth=2, nfocks=(8, 8), ansatz="ecd", energy_tensor=hybrid_energy_tensor((8, 8)))
    rng = np.random.default_rng(2026)
    x = random_parameters(2, rng)
    twins = build_twins(sim, x, rng, n_train=4, n_rank2=0)
    assert len(twins) == 4
    for tw in twins:
        assert tw.t_free == 0
        assert tw.product_tvd is not None
        assert tw.product_tvd < 1e-6
        assert tw.poisson_tvd is not None
        assert tw.poisson_tvd < 1e-6
        q, a1, a2 = tw.qubit, tw.alpha[0], tw.alpha[1]
        poisson = product_coherent_histogram(q, a1, a2, (2, 8, 8))
        # p_analytic is the truncated product evolution, not the Poisson formula.
        assert total_variation(poisson, tw.p_ideal) < 1e-6


def test_snap_gaussian_twins_match_poisson():
    sim = HybridSimulator(ndepth=1, nfocks=(8, 8), ansatz="snap", energy_tensor=hybrid_energy_tensor((8, 8)))
    rng = np.random.default_rng(7)
    x = random_snap_parameters(1, (8, 8), rng)
    twins = build_twins(sim, x, rng, n_train=3, n_rank2=0)
    for tw in twins:
        assert tw.product_tvd < 1e-6
        assert tw.poisson_tvd is not None


def test_oracle_kernels_stochastic():
    cfg = circuit_noise("loss", 0.03)
    spec = readout_spec("readout_realistic", n_shots=100)
    cq, c1, c2 = oracle_kernels(cfg, spec, ndepth=5, dims=(2, 8, 8))
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)


def test_truncated_poisson_vacuum():
    p = truncated_poisson(0.0, 8)
    assert p[0] == pytest.approx(1.0)
    assert p.sum() == pytest.approx(1.0)


def test_holdout_indices_stratified():
    train, hold = holdout_indices(20, 0.25)
    assert train.size + hold.size == 20
    assert hold.size == 5
    assert np.intersect1d(train, hold).size == 0
    # not just the last 25% (those would be all t_free>0 twins)
    assert hold.max() < 19 or hold.min() == 0


def test_damp_histogram_endpoints():
    a = np.zeros((2, 4, 4))
    a[0, 0, 0] = 1.0
    b = np.zeros_like(a)
    b[1, 1, 1] = 1.0
    assert damp_histogram(a, b, 0.0)[0, 0, 0] == pytest.approx(1.0)
    assert damp_histogram(a, b, 1.0)[1, 1, 1] == pytest.approx(1.0)
    mix = damp_histogram(a, b, 0.5)
    assert mix[0, 0, 0] == pytest.approx(0.5)
    assert mix.sum() == pytest.approx(1.0)


def test_choose_damp_alpha_safe_gap_keeps_unfold_when_it_clearly_wins():
    p = np.zeros((2, 4, 4))
    p[0, 0, 0] = 1.0
    q = np.zeros_like(p)
    q[0, 0, 0] = 1.0
    eye2, eye4 = np.eye(2), np.eye(4)
    # identity kernels: unfold = q = p, safe = wrong
    safe = np.zeros_like(p)
    safe[1, 1, 1] = 1.0
    a, info = choose_damp_alpha([p], [q], eye2, eye4, eye4, [safe], alphas=np.linspace(0, 1, 5), slack=0.2, safe_gap=0.01)
    assert a == pytest.approx(0.0)
    assert info["safe_gated"] is True


def test_choose_damp_alpha_slack_picks_safer_mix():
    p = np.zeros((2, 4, 4))
    p[0, 0, 0] = 1.0
    q = np.zeros_like(p)
    q[0, 0, 0] = 0.55
    q[1, 1, 1] = 0.45
    eye2, eye4 = np.eye(2), np.eye(4)
    a0, _ = choose_damp_alpha([p], [q], eye2, eye4, eye4, [p], alphas=np.linspace(0, 1, 5), slack=0.0)
    a_s, info = choose_damp_alpha([p], [q], eye2, eye4, eye4, [p], alphas=np.linspace(0, 1, 5), slack=0.2)
    assert a_s >= a0
    assert info["slack"] == pytest.approx(0.2)


def test_choose_damp_alpha_prefers_safe_when_unfold_is_wrong():
    p = np.zeros((2, 4, 4))
    p[0, 1, 1] = 1.0
    q = np.zeros_like(p)
    q[1, 2, 2] = 1.0
    eye2, eye4 = np.eye(2), np.eye(4)
    alpha, info = choose_damp_alpha([p], [q], eye2, eye4, eye4, [p], alphas=np.linspace(0, 1, 5))
    assert alpha == pytest.approx(1.0)
    assert info["hold_tvd"] == pytest.approx(0.0)


def test_readout_then_zne_beats_raw_zne_under_readout():
    spec = readout_spec("readout_strong", n_shots=0, seed=0)
    rng = np.random.default_rng(1)
    p0 = rng.random((2, 8, 8))
    p0 = p0 / p0.sum()
    eta = 0.9
    b = binomial_loss_kernel(eta, 8)
    b2 = binomial_loss_kernel(eta**2, 8)
    b3 = binomial_loss_kernel(eta**3, 8)

    def apply_b(kernel):
        out = np.zeros_like(p0)
        for q in range(2):
            out[q] = kernel @ p0[q] @ kernel.T
        return out

    phys = {1: apply_b(b), 2: apply_b(b2), 3: apply_b(b3)}
    blurred = {s: observe_histogram(h, spec, (2, 8, 8), seed=10 + s) for s, h in phys.items()}
    raw_zne = zne_histogram(blurred, degree=2)
    hyb = readout_then_zne(blurred, spec, (2, 8, 8), degree=2)
    other = zne_then_readout(blurred, spec, (2, 8, 8), degree=2)
    assert total_variation(hyb, p0) < total_variation(raw_zne, p0)
    assert total_variation(other, p0) <= total_variation(raw_zne, p0) + 1e-12


def test_select_by_holdout_keeps_first_on_tie():
    name, score, ranked = select_by_holdout([("a", 0.2), ("b", 0.1), ("c", 0.1)])
    assert name == "b"
    assert score == pytest.approx(0.1)
    assert len(ranked) == 3


def test_designed_twin_plan_spans_magnitude():
    t_free, scales = designed_twin_plan(12, ndepth=5, n_rank2=3, mag_lo=0.25, mag_hi=1.35)
    assert len(t_free) == 12
    assert t_free.count(0) == 9
    assert t_free.count(2) == 3
    assert min(scales) == pytest.approx(0.25)
    assert max(scales) == pytest.approx(1.35)


def test_select_research_method_picks_residual_on_small_hops():
    name, extra = select_research_method(
        [("safe", 0.08), ("gdr_param", 0.04), ("gdr_damped", 0.035)],
        residual_hops=0.02,
        residual_tfree=0.03,
        gdr_tfree=0.05,
        oracle_tfree=0.04,
    )
    assert name == "gdr_residual"
    assert extra["reason"] == "tfree_residual"


def test_select_research_method_uses_optimized_recipe():
    name, extra = select_research_method(
        [("safe", 0.08), ("gdr_param", 0.04), ("gdr_damped", 0.01)],
        residual_hops=0.2,
        residual_tfree=0.08,
        gdr_tfree=0.04,
        circuit_kind="optimized",
    )
    assert name == "gdr_param"
    assert extra["reason"] == "optimized_gdr"


def test_select_research_method_rejects_large_residual_hops():
    name, extra = select_research_method(
        [("safe", 0.08), ("gdr_param", 0.04), ("gdr_damped", 0.035)],
        residual_hops=0.15,
        residual_tfree=0.01,
        gdr_tfree=0.05,
        oracle_tfree=0.04,
    )
    assert name == "gdr_damped"
    assert extra["reason"] == "holdout"


def test_choose_mix_alpha_picks_better_end():
    p = np.zeros((2, 2, 2))
    p[0, 0, 0] = 1.0
    a = np.zeros_like(p)
    a[1, 1, 1] = 1.0
    alpha, info = choose_mix_alpha([p], [a], [p], alphas=np.linspace(0, 1, 5))
    assert alpha == pytest.approx(1.0)
    assert info["hold_tvd"] == pytest.approx(0.0)


def test_interleave_kernels_column_stochastic():
    cfg = circuit_noise("loss", 0.03)
    spec = readout_spec("ideal", n_shots=200)
    rng = np.random.default_rng(1)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    q = 0.85 * p + 0.15 * rng.random((2, 8, 8))
    q = q / q.sum()
    (cq, c1, c2), info = fit_gdr_interleave([p], [q], cfg, spec, ndepth=5, dims=(2, 8, 8), maxiter=15)
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)
    assert info["kind"] == "gdr_interleave"
    assert 0.15 <= info["eta_early"] <= 1.0
    assert 0.15 <= info["eta_late"] <= 1.0


def test_afterburn_kernels_column_stochastic():
    cfg = circuit_noise("loss", 0.03)
    spec = readout_spec("ideal", n_shots=200)
    rng = np.random.default_rng(0)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    q = 0.9 * p + 0.1 * rng.random((2, 8, 8))
    q = q / q.sum()
    (cq, c1, c2), info = fit_gdr_afterburn([p], [q], cfg, spec, ndepth=5, dims=(2, 8, 8), maxiter=20)
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)
    assert info["kind"] == "gdr_afterburn"
    assert 0.5 <= info["eta_extra"] <= 1.0


def test_split_and_band_kernels_column_stochastic():
    rng = np.random.default_rng(2)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    q = 0.9 * p + 0.1 * rng.random((2, 8, 8))
    q = q / q.sum()
    cq0, c10, c20 = np.eye(2), np.eye(8), np.eye(8)
    (cq, c1, c2), info = fit_gdr_split([p], [q], cq0, c10, c20, 200, (2, 8, 8), maxiter=15)
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)
    assert info["kind"] == "gdr_split"
    assert info["hops"] >= 0.0
    (cq, c1, c2), info = fit_gdr_band([p], [q], cq0, c10, c20, 200, (2, 8, 8), maxiter=15)
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)
    assert info["kind"] == "gdr_band"


def test_classify_opt_quality_default_thresholds():
    h000 = classify_opt_quality(-6.2298, -7.1107, gap=0.7781)
    assert h000["deficit"] == pytest.approx(0.8809, abs=1e-3)
    assert h000["thresh"] == pytest.approx(0.5)
    assert h000["recipe"] == "random"
    h001 = classify_opt_quality(-3.9218, -6.0317, gap=1.1825)
    assert h001["recipe"] == "random"
    tight = classify_opt_quality(-6.2298, -7.1107, gap=0.7781, abs_tol=1.0)
    assert tight["recipe"] == "optimized"
    good = classify_opt_quality(-7.10, -7.1107, gap=0.7781)
    assert good["recipe"] == "optimized"


def test_adaptive_recipe_headlines_on_disk():
    recipe = (ROOT / "Error_mitigation" / "out_research" / "adaptive_recipe.md").read_text()
    paper = (ROOT / "Error_mitigation" / "out_research" / "PAPER_SUMMARY.md").read_text()
    notebook = (ROOT / "Error_mitigation" / "out_research" / "NOTEBOOK.md").read_text()
    plot = (ROOT / "Error_mitigation" / "plot_hard_cells.py").read_text()
    for text in (recipe, paper):
        assert "beats PR #6 `gdr_param`" in text or "**86 / 108**" in text
        assert "**86 / 108**" in text
        assert "**108 / 108**" in text
        assert "**0 / 108**" in text
        assert "**0.203**" in text
        assert "**0.342**" in text
        assert "**0.0369**" in text
        assert "**0.208 ± 0.012**" in text
        assert "**0.314 ± 0.013**" in text
        assert "**0.346 ± 0.008**" in text
        assert "**0.036 ± 0.004**" in text
    assert "near-" in paper.lower() or "H001" in paper
    for token in ("0.203", "0.342", "0.343", "0.0369", "0.012", "0.013", "0.008", "0.004"):
        assert token in plot
    assert "**0.203**" in notebook
    fig = ROOT / "Error_mitigation" / "out_research" / "figures" / "hard_cells_adaptive.png"
    assert fig.is_file() and fig.stat().st_size > 1000


def test_research_smoke_preset_stays_in_out_research():
    from Error_mitigation.run_ablation import (
        DEFAULT_OUT,
        RESEARCH_SMOKE,
        apply_research_smoke,
        parse_args,
    )

    args = parse_args(["--preset", "research_smoke"])
    assert args.preset == "research_smoke"
    args = apply_research_smoke(args)
    assert args.tag == "research_smoke"
    assert args.ansatz == "ecd"
    assert args.params == "optimized"
    assert args.twin_design == "adaptive"
    assert int(args.n_train) == 40
    assert int(args.n_rank2) == 10
    assert args.families == RESEARCH_SMOKE["families"]
    assert DEFAULT_OUT.name == "out_research"
    assert Path(args.outdir).resolve() == DEFAULT_OUT.resolve()


def test_slice_twin_indices_even_subset():
    from Error_mitigation.run_ablation import slice_twin_indices

    full = slice_twin_indices(40, 40)
    assert list(full) == list(range(40))
    keep20 = slice_twin_indices(40, 20)
    assert keep20.size == 20
    assert keep20[0] == 0
    assert keep20[-1] == 39
    keep10 = slice_twin_indices(40, 10)
    assert keep10.size == 10
    assert keep10[0] == 0
    assert keep10[-1] == 39


def test_anneal_prior_pulls_eta_to_one_at_mild_kt():
    cfg = circuit_noise("loss", 0.003)
    spec = readout_spec("ideal", n_shots=100)
    assert anneal_identity_weight(0.003) == pytest.approx(1.0)
    assert anneal_identity_weight(0.1) < 0.2
    theta_mild = anneal_prior_theta(cfg, spec, 5, 0.003)
    theta_high = anneal_prior_theta(cfg, spec, 5, 0.1)
    assert theta_mild[0] == pytest.approx(1.0)
    assert theta_mild[1] == pytest.approx(1.0)
    assert theta_high[0] < theta_mild[0]


def test_fit_gdr_anneal_and_eta_kinds():
    rng = np.random.default_rng(0)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    q = p.copy()
    cfg = circuit_noise("loss", 0.003)
    spec = readout_spec("ideal", n_shots=200)
    theta, info = fit_gdr_anneal([p], [q], cfg, spec, 5, (2, 8, 8), 0.003, maxiter=15)
    assert info["kind"] == "gdr_anneal"
    assert theta.shape == (11,)
    theta_e, info_e = fit_gdr_eta([p], [q], cfg, spec, 5, (2, 8, 8), maxiter=15)
    assert info_e["kind"] == "gdr_eta"
    assert info_e["fitted"]["p_down"] == pytest.approx(0.0)


def test_fisher_twin_weights_positive_and_normalized():
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(4):
        a = rng.random((2, 8, 8))
        ps.append(a / a.sum())
    w = fisher_twin_weights(ps)
    assert w.shape == (4,)
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(4.0)


def test_select_kt_never_residual_on_comprehensive_high():
    name, extra = select_kt_method(
        [("safe", 0.08), ("gdr_param", 0.04), ("gdr_residual", 0.01), ("gdr_damped", 0.03)],
        kappa_tau=0.1,
        family="comprehensive",
        circuit_kind="optimized",
        residual_hops=0.01,
        residual_tfree=0.01,
        gdr_tfree=0.05,
    )
    assert name == "gdr_param"
    assert extra["reason"] == "optimized_gdr"
    name_r, extra_r = select_kt_method(
        [("safe", 0.08), ("gdr_param", 0.04), ("gdr_damped", 0.03)],
        kappa_tau=0.1,
        family="comprehensive",
        circuit_kind="random",
        residual_hops=0.01,
        residual_tfree=0.01,
        gdr_tfree=0.05,
    )
    assert name_r != "gdr_residual"
    assert extra_r["reason"] == "kt_holdout"


def test_mild_residual_gate():
    assert mild_residual_allowed(circuit_kind="optimized", family="loss", kappa_tau=0.003)
    assert not mild_residual_allowed(
        circuit_kind="optimized", family="comprehensive", kappa_tau=0.003
    )
    assert not mild_residual_allowed(circuit_kind="optimized", family="loss", kappa_tau=0.1)
    assert not mild_residual_allowed(circuit_kind="random", family="loss", kappa_tau=0.003)
    name, extra = select_kt_method(
        [("safe", 0.08), ("gdr_param", 0.04), ("gdr_damped", 0.03)],
        kappa_tau=0.003,
        family="loss",
        circuit_kind="optimized",
        residual_hops=0.02,
        residual_tfree=0.02,
        gdr_tfree=0.05,
    )
    assert name == "gdr_residual"
    assert extra["reason"] == "mild_residual"


def test_designed_twin_plan_grid_chebyshev():
    t_free, scales = designed_twin_plan_grid(
        12, ndepth=5, n_rank2=3, mag_lo=0.25, mag_hi=1.35, spacing="chebyshev"
    )
    assert len(t_free) == 12
    assert len(scales) == 12
    assert min(scales) == pytest.approx(0.25)
    assert max(scales) == pytest.approx(1.35)
    log_plan = designed_twin_plan(12, ndepth=5, n_rank2=3, mag_lo=0.25, mag_hi=1.35)[1]
    assert scales != log_plan


def test_round2_dropped_and_ban_list_on_disk():
    dropped = ROOT / "Error_mitigation" / "out_research" / "round2" / "DROPPED.md"
    text = dropped.read_text()
    for token in (
        "gdr_full",
        "gdr_interleave",
        "gdr_split",
        "gdr_band",
        "gdr_afterburn",
        "gdr_blend",
        "Energy-weighted",
        "params=auto",
        "Span twins",
        "gdr_residual",
    ):
        assert token in text
    from Error_mitigation.run_ablation import CHEAP_METHODS, ROUND2_METHODS

    banned = {
        "gdr_full",
        "gdr_interleave",
        "gdr_split",
        "gdr_band",
        "gdr_afterburn",
        "gdr_blend",
        "gdr_energy",
    }
    assert banned.isdisjoint(ROUND2_METHODS)
    assert banned.isdisjoint(CHEAP_METHODS)


def test_round2_hard_cells_have_caches():
    from Error_mitigation.run_round2 import HARD_CELLS

    assert len(HARD_CELLS) == 8
    for cell in HARD_CELLS:
        path = Path(cell["cache"])
        assert path.is_file(), path
        assert path.suffix == ".npz"
