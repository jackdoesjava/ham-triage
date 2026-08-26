import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    # Guo et al. 2017. One parameter, so a bounded scalar search on log T is enough and,
    # unlike LBFGS, gives the same answer every time the figures are regenerated.
    def nll(log_t):
        return -log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def nll(logits: np.ndarray, y: np.ndarray, t: float = 1.0) -> np.ndarray:
    return -log_softmax(logits / t, axis=1)[np.arange(len(y)), y]


def brier(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.eye(probs.shape[1])[y]
    return ((probs - onehot) ** 2).sum(axis=1)


def bin_stats(conf: np.ndarray, hit: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> dict:
    # equal-width bins are the textbook ECE; equal-mass bins (adaptive ECE, Nguyen and
    # O'Connor 2015) matter here because two thirds of the test set is nv predicted at
    # confidence ~1, which leaves most equal-width bins nearly empty
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1)) if adaptive else np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_conf = np.bincount(idx, weights=conf, minlength=n_bins) / count
        acc = np.bincount(idx, weights=hit, minlength=n_bins) / count
    return {"edges": edges, "count": count, "conf": mean_conf, "acc": acc}


def ece(conf: np.ndarray, hit: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> float:
    b = bin_stats(conf, hit, n_bins, adaptive)
    ok = b["count"] > 0
    return float(np.sum(b["count"][ok] / len(conf) * np.abs(b["conf"][ok] - b["acc"][ok])))


def calibration_metrics(logits: np.ndarray, y: np.ndarray, t: float = 1.0) -> np.ndarray:
    probs = softmax(logits / t, axis=1)
    conf, hit = probs.max(axis=1), probs.argmax(axis=1) == y
    return np.array([ece(conf, hit), ece(conf, hit, adaptive=True), brier(probs, y).mean(), nll(logits, y, t).mean()])


CALIBRATION_METRICS = ["ece", "adaptive_ece", "brier", "nll"]
