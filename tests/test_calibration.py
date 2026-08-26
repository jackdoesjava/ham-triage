import numpy as np
from scipy.special import softmax

from ham_triage.calibration import brier, ece, fit_temperature, nll


def synthetic(n=20000, t_true=2.5, seed=0):
    # draw labels from softmax(z / t_true) so that z / t_true is calibrated by construction
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3, size=(n, 7))
    p = softmax(z / t_true, axis=1)
    y = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return z, y


def test_fit_temperature_recovers_generating_temperature():
    z, y = synthetic()
    assert abs(fit_temperature(z, y) - 2.5) < 0.1


def test_ece_small_when_calibrated_and_large_when_overconfident():
    z, y = synthetic()
    p_cal, p_over = softmax(z / 2.5, axis=1), softmax(z, axis=1)
    hit_cal, hit_over = p_cal.argmax(1) == y, p_over.argmax(1) == y
    assert ece(p_cal.max(1), hit_cal) < 0.02
    assert ece(p_over.max(1), hit_over) > 0.1
    assert ece(p_cal.max(1), hit_cal, adaptive=True) < 0.02


def test_nll_and_brier_prefer_the_right_temperature():
    z, y = synthetic()
    assert nll(z, y, 2.5).mean() < nll(z, y, 1.0).mean()
    assert brier(softmax(z / 2.5, axis=1), y).mean() < brier(softmax(z, axis=1), y).mean()
