import numpy as np
from scipy.stats import beta

from .config import CLASSES


def aps_scores(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """APS nonconformity score for every candidate label, shape [n, K] (Romano et al. 2020).

    For label k the score is the probability mass of the classes ranked above k plus
    (1 - u) times the mass of k itself, u uniform on [0, 1]. The randomisation term
    is what makes coverage exact rather than conservative; without it the sets
    overshoot badly here because nv sits at p ~ 1 and pushes every cumulative sum
    past the quantile. The true-label score is the gather scores[i, y_i]; a set is
    every label whose score is at most the calibrated quantile, so score and set
    are computed by the same expression and the guarantee carries over.
    """
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    sorted_scores = sorted_p.cumsum(axis=1) - u[:, None] * sorted_p
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def lac_scores(probs: np.ndarray) -> np.ndarray:
    # Sadinle et al. 2019: 1 - p_k, the smallest sets of any valid scheme; for a single
    # class this makes "k is in the set" a plain threshold on p_k, which is what the
    # mel-conditional rule needs to be readable as a sensitivity guarantee
    return 1 - probs


def quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest calibration score; inf when that index
    exceeds n, which happens whenever n < 1 / alpha - 1 and means the set is the whole
    label space. Splitting hairs about that index is the entire finite-sample guarantee."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def mondrian_quantiles(scores_true: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.array([quantile(scores_true[y == k], alpha) if (y == k).any() else np.inf for k in range(len(CLASSES))])


def prediction_sets(scores: np.ndarray, q) -> np.ndarray:
    # q is a scalar (marginal) or a per-class vector (Mondrian). Empty sets are allowed
    # on purpose: randomised APS hits exact coverage on a confident nv image by leaving
    # the set empty about alpha of the time, and forcing the top label back in turned
    # 90% marginal coverage into 95% here. Report the empty rate instead of hiding it.
    return scores <= np.asarray(q)


def coverage_law(n_cal: int, alpha: float):
    # Vovk 2012: conditional on the calibration draw, coverage of split conformal is
    # Beta(n + 1 - l, l) with l = floor((n + 1) alpha); this is the reference curve
    # the re-partition histogram is checked against
    l = int(np.floor((n_cal + 1) * alpha))
    return beta(n_cal + 1 - l, l)
