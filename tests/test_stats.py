import numpy as np

from ham_triage.stats import balanced_accuracy, ci, cluster_bootstrap, recall_per_class


def width(samples):
    lo, hi = ci(samples)
    return hi - lo


def test_cluster_bootstrap_matches_plain_bootstrap_for_singleton_groups():
    x = np.random.default_rng(0).normal(size=2000)
    w = width(cluster_bootstrap(lambda idx: x[idx].mean(), np.arange(len(x)), n_boot=400))
    assert abs(w - 2 * 1.96 / np.sqrt(len(x))) < 0.02


def test_cluster_bootstrap_widens_for_duplicated_images():
    # 1000 lesions with two identical images each: only 1000 independent points
    x = np.repeat(np.random.default_rng(0).normal(size=1000), 2)
    groups = np.repeat(np.arange(1000), 2)
    by_lesion = width(cluster_bootstrap(lambda idx: x[idx].mean(), groups, n_boot=400))
    by_image = width(cluster_bootstrap(lambda idx: x[idx].mean(), np.arange(len(x)), n_boot=400))
    assert abs(by_lesion - 2 * 1.96 / np.sqrt(1000)) < 0.02
    assert by_image < by_lesion / 1.25


def test_recall_and_balanced_accuracy():
    y = np.array([0, 0, 1, 1, 4, 4, 4, 5])
    pred = np.array([0, 1, 1, 1, 4, 4, 5, 5])
    rec = recall_per_class(pred, y)
    assert rec[0] == 0.5 and rec[1] == 1.0 and rec[4] == 2 / 3 and rec[5] == 1.0
    assert np.isnan(rec[2]) and np.isnan(rec[3]) and np.isnan(rec[6])
    assert np.isclose(balanced_accuracy(pred, y), np.mean([0.5, 1.0, 2 / 3, 1.0]))
