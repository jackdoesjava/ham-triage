import numpy as np

from ham_triage.config import CLASSES
from ham_triage.decision import (DEFER, DISCHARGE, MEL, REFER, CostModel, bayes_actions, defer_by_score,
                                 ensemble_mutual_information, expected_cost, expected_miss, prior_shift,
                                 realized_cost, risk_coverage)

NV = CLASSES.index("nv")


def mel_vs_nv(p_mel):
    probs = np.zeros((len(p_mel), len(CLASSES)))
    probs[:, MEL] = p_mel
    probs[:, NV] = 1 - p_mel
    return probs


def test_two_action_rule_is_binary_argmax_when_a_miss_costs_two_referrals():
    probs = mel_vs_nv(np.linspace(0, 1, 100))  # no exact tie at 0.5
    cm = CostModel(miss_mel=2, miss_treat=2, refer=1)
    actions = bayes_actions(probs, cm, defer_knob=np.inf)
    assert np.array_equal(actions == REFER, probs[:, MEL] >= 0.5)
    assert DEFER not in actions


def test_three_regions_are_the_documented_thresholds():
    cm = CostModel(miss_mel=100, miss_treat=10, refer=1, defer=0.3, reader_sensitivity=0.87)
    probs = mel_vs_nv(np.linspace(0, 0.2, 2001))
    s = expected_miss(probs, cm)
    actions = bayes_actions(probs, cm)
    lo, hi = cm.defer / cm.reader_sensitivity, (cm.refer - cm.defer) / (1 - cm.reader_sensitivity)
    assert np.array_equal(actions == DISCHARGE, s < lo)
    assert np.array_equal(actions == DEFER, (s >= lo) & (s < hi))
    assert np.array_equal(actions == REFER, s >= hi)


def test_perfect_reader_never_refers_when_deferral_is_cheaper():
    cm = CostModel(reader_sensitivity=1.0, defer=0.3)
    actions = bayes_actions(mel_vs_nv(np.linspace(0, 1, 50)), cm)
    assert REFER not in actions and DEFER in actions and DISCHARGE in actions


def test_defer_by_score_with_msp_is_chow():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(len(CLASSES)) * 0.3, size=500)
    cm = CostModel()
    t = 0.4
    actions = defer_by_score(1 - probs.max(1), t, probs, cm)
    assert np.array_equal(actions == DEFER, probs.max(1) <= 1 - t)
    rest = actions != DEFER
    assert np.array_equal(actions[rest], bayes_actions(probs, cm, np.inf)[rest])


def test_expected_equals_realized_for_one_hot_posteriors():
    y = np.array([MEL, NV, MEL, CLASSES.index("bcc")])
    probs = np.eye(len(CLASSES))[y]
    cm = CostModel()
    for actions in (np.full(4, DISCHARGE), np.full(4, REFER), np.full(4, DEFER)):
        assert np.allclose(expected_cost(actions, probs, cm), realized_cost(actions, y, cm))
    assert np.isclose(realized_cost(np.array([DEFER]), np.array([MEL]), cm)[0], 0.3 + 0.13 * 100)


def test_prior_shift_and_ensemble_mi():
    probs = mel_vs_nv(np.array([0.1, 0.5]))
    train, deploy = np.full(7, 1 / 7), np.full(7, 1 / 7)
    deploy = deploy.copy()
    deploy[MEL] /= 5
    shifted = prior_shift(probs, train, deploy)
    assert np.allclose(shifted.sum(1), 1) and (shifted[:, MEL] < probs[:, MEL]).all()
    z = np.log(np.clip(probs, 1e-6, 1))
    assert np.allclose(ensemble_mutual_information([z, z, z], [1, 1, 1]), 0, atol=1e-6)
    assert (ensemble_mutual_information([z, z[::-1]], [1, 1]) > 1e-3).all()


def test_risk_coverage_rewards_a_good_ranking():
    error = np.array([0, 0, 0, 1, 1])
    cov, risk, aurc_good = risk_coverage(np.arange(5), error)
    _, _, aurc_bad = risk_coverage(-np.arange(5), error)
    assert cov[-1] == 1 and risk[-1] == 0.4 and aurc_good < aurc_bad
