import numpy as np
from scipy.special import softmax

from ham_triage.conformal import aps_scores, coverage_law, lac_scores, mondrian_quantiles, prediction_sets, quantile


def exchangeable(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2, size=(n, 7)) + np.log([0.03, 0.05, 0.11, 0.01, 0.11, 0.67, 0.02])
    p = softmax(z, axis=1)
    y = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return p, y, rng


def test_quantile_index_is_the_finite_sample_one():
    s = np.arange(1, 10) / 10  # n = 9: ceil(10 * 0.9) = 9, the largest score
    assert quantile(s, 0.1) == 0.9
    assert np.isinf(quantile(s[:5], 0.1))  # n = 5: ceil(6 * 0.9) = 6 > 5
    assert quantile(np.arange(1, 101) / 100, 0.1) == 0.91


def test_marginal_coverage_hits_the_target_for_both_scores():
    p_cal, y_cal, rng = exchangeable(3000, 0)
    p_test, y_test, _ = exchangeable(30000, 1)
    for name in ("aps", "lac"):
        if name == "aps":
            s_cal, s_test = aps_scores(p_cal, rng.random(len(y_cal))), aps_scores(p_test, rng.random(len(y_test)))
        else:
            s_cal, s_test = lac_scores(p_cal), lac_scores(p_test)
        q = quantile(s_cal[np.arange(len(y_cal)), y_cal], 0.1)
        sets = prediction_sets(s_test, q)
        cov = sets[np.arange(len(y_test)), y_test].mean()
        assert abs(cov - 0.9) < 0.015, (name, cov)
        assert (~sets.any(axis=1)).mean() < 0.1


def test_mondrian_covers_every_class_and_degenerates_when_small():
    p_cal, y_cal, rng = exchangeable(6000, 2)
    p_test, y_test, _ = exchangeable(30000, 3)
    s_cal, s_test = lac_scores(p_cal), lac_scores(p_test)
    q = mondrian_quantiles(s_cal[np.arange(len(y_cal)), y_cal], y_cal, 0.1)
    sets = prediction_sets(s_test, q)
    for k in range(7):
        m = y_test == k
        assert sets[m, k].mean() > 0.9 - 0.04, k
    # class 3 has ~1% prevalence; with 60 cal points it is fine, with 5 it is the whole space
    q_small = mondrian_quantiles(s_cal[np.arange(len(y_cal)), y_cal][:300], y_cal[:300], 0.1)
    assert np.isinf(q_small[3])


def test_coverage_law_is_centred_near_one_minus_alpha():
    law = coverage_law(838, 0.1)
    assert abs(law.mean() - 0.9) < 0.005
    assert 0.005 < law.std() < 0.02
