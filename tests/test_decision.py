import numpy as np

from ham_triage.config import CLASSES
from ham_triage.decision import (DEFER, DISCHARGE, MEL, REFER, CostModel, bayes_actions, cost_matrix, defer_by_score,
                                 ensemble_mutual_information, expected_cost, net_benefit, prior_shift, realized_cost,
                                 risk_coverage)

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


def test_actions_are_contiguous_regions_in_p_mel_and_match_the_matrix():
    cm = CostModel()
    p = np.linspace(0, 1, 5001)
    probs = mel_vs_nv(p)
    actions = bayes_actions(probs, cm)
    assert np.array_equal(actions, (probs @ cost_matrix(cm)).argmin(axis=1))
    changes = np.flatnonzero(np.diff(actions))
    assert list(actions[np.r_[0, changes + 1]]) == [DISCHARGE, DEFER, REFER]
    # a perfect reader does not remove the refer region: a deferred malignant still ends in
    # a referral once flagged, so the read is wasted above p = 1 - defer / refer
    perfect = CostModel(reader_sensitivity=1.0, reader_specificity=1.0)
    a = bayes_actions(probs, perfect)
    assert np.allclose(p[a == REFER].min(), 1 - perfect.defer / perfect.refer, atol=1e-3)
    assert np.allclose(p[a == DISCHARGE].max(), perfect.defer / (perfect.miss_mel - perfect.refer), atol=1e-3)


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
    deferred_mel = cm.defer + cm.reader_sensitivity * cm.refer + (1 - cm.reader_sensitivity) * cm.miss_mel
    deferred_nv = cm.defer + (1 - cm.reader_specificity) * cm.refer
    assert np.isclose(realized_cost(np.array([DEFER]), np.array([MEL]), cm)[0], deferred_mel)
    assert np.isclose(realized_cost(np.array([DEFER]), np.array([NV]), cm)[0], deferred_nv)


def test_net_benefit_endpoints():
    treat = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    p = np.where(treat, 0.9, 0.1)
    nb = net_benefit(p, treat, np.array([0.05, 0.5]))
    assert np.isclose(nb[1], 0.2)  # perfect separation at p_t = 0.5: all TP, no FP
    assert np.isclose(nb[0], 0.2 - 0.8 * 0.05 / 0.95)  # at p_t = 0.05 everyone is flagged
    assert net_benefit(p, treat, np.array([0.95]))[0] == 0  # nobody referred


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
