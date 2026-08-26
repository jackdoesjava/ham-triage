import numpy as np
import pandas as pd

from .config import CLASSES


def cluster_bootstrap(fn, groups, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap fn(idx) by resampling whole groups (lesions) with replacement.

    fn receives an integer index array into whatever arrays the caller closed over
    and returns a scalar or 1-D array; the result is [n_boot, ...]. Images of one
    lesion are near-duplicates, so resampling images instead of lesions understates
    the variance, and it does so most on the classes with many images per lesion,
    which is melanoma. Use this on the leaky split too: the clustering is a property
    of the data, not of how the split was drawn.
    """
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(fn(np.concatenate([members[g] for g in pick])))
    return np.asarray(out)


def ci(samples: np.ndarray, level: float = 0.95):
    lo, hi = np.nanpercentile(samples, [50 * (1 - level), 50 * (1 + level)], axis=0)
    return lo, hi


def recall_per_class(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([(pred[y == k] == k).mean() if (y == k).any() else np.nan for k in range(len(CLASSES))])


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.nanmean(recall_per_class(pred, y)))


def fmt(point: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
