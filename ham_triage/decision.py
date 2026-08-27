from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax

from .config import CLASSES

DISCHARGE, REFER, DEFER = 0, 1, 2
MEL = CLASSES.index("mel")
TREAT2 = np.array([CLASSES.index(c) for c in ("bcc", "akiec")])


@dataclass(frozen=True)
class CostModel:
    # everything in units of one referral. miss_mel is the contested number: a
    # dermatologist number-needed-to-biopsy of 10-30 for melanoma implies clinicians
    # act around p = 3-10%, i.e. miss_mel of 10-30; 100 is the aggressive end and the
    # analysis sweeps it rather than defending it
    miss_mel: float = 100.0
    miss_treat: float = 10.0  # bcc, akiec: needs excision, very rarely lethal
    refer: float = 1.0
    defer: float = 0.3  # a remote read of the image, cheaper than an in-person visit
    # Haenssle et al. 2018, dermatologists reading dermoscopy alone: 86.6% sensitivity,
    # 71.3% specificity. A deferred benign lesion is referred by the reader about a
    # third of the time; a deferred malignant one is missed about one time in eight
    reader_sensitivity: float = 0.87
    reader_specificity: float = 0.71


def miss_costs(cm: CostModel) -> np.ndarray:
    m = np.zeros(len(CLASSES))
    m[MEL] = cm.miss_mel
    m[TREAT2] = cm.miss_treat
    return m


def cost_matrix(cm: CostModel) -> np.ndarray:
    """C[y, a]: the cost of action a when the true class is y (Elkan 2001).

        discharge   the miss cost m_y, zero for benign classes
        refer       one referral, whoever the patient is; the case is resolved
        defer       the read itself, then whatever the reader decides: a benign lesion
                    is referred with probability 1 - specificity, a treated one is
                    referred with probability sensitivity and missed otherwise

    The expected cost of every action is linear in the posterior, so the Bayes rule
    is argmin over probs @ C. Deferring is worth it in the middle: below the read
    price a discharge is cheaper, and once p(treat) exceeds 1 - defer / refer the
    read is wasted because the reader will refer anyway. With defer = inf it is the
    two-action rule, refer iff the expected miss cost exceeds one referral, which
    for miss costs of two referrals is the argmax of the binary problem. Chow 1970
    is not a member of this family (a flat referral price is not a 0-1 loss); it
    is the max-softmax baseline in defer_by_score.
    """
    m = miss_costs(cm)
    treat = m > 0
    deferred = np.where(treat, cm.reader_sensitivity * cm.refer + (1 - cm.reader_sensitivity) * m,
                        (1 - cm.reader_specificity) * cm.refer)
    return np.stack([m, np.full_like(m, cm.refer), cm.defer + deferred], axis=1)


def expected_miss(probs: np.ndarray, cm: CostModel) -> np.ndarray:
    return probs @ miss_costs(cm)


def bayes_actions(probs: np.ndarray, cm: CostModel, defer_knob: float | None = None) -> np.ndarray:
    # defer_knob replaces the defer price in the decision only, to trace the frontier at
    # deferral rates other than the cost-optimal one; costs are always paid at cm.defer
    c = cost_matrix(cm)
    if defer_knob is None:
        return (probs @ c).argmin(axis=1)
    if np.isinf(defer_knob):
        return (probs @ c[:, :DEFER]).argmin(axis=1)  # 0 * inf is nan, so drop the column instead
    c = c.copy()
    c[:, DEFER] += defer_knob - cm.defer
    return (probs @ c).argmin(axis=1)


def realized_cost(actions: np.ndarray, y: np.ndarray, cm: CostModel) -> np.ndarray:
    return cost_matrix(cm)[y, actions]


def expected_cost(actions: np.ndarray, probs: np.ndarray, cm: CostModel) -> np.ndarray:
    # the cost the model itself expects to pay; its gap to realized_cost is the
    # decision-level price of miscalibration
    return (probs @ cost_matrix(cm))[np.arange(len(probs)), actions]


def defer_by_score(score: np.ndarray, threshold: float, probs: np.ndarray, cm: CostModel) -> np.ndarray:
    # defer where score >= threshold, otherwise the two-action Bayes rule; Chow's rule
    # is this with score = 1 - max p, and every other abstention score in the
    # literature is a different choice of score with the same decision on the rest
    actions = bayes_actions(probs, cm, defer_knob=np.inf)
    actions[score >= threshold] = DEFER
    return actions


def melanoma_miss_weight(actions: np.ndarray, cm: CostModel) -> np.ndarray:
    # 1 for a discharged melanoma, the reader's miss probability for a deferred one
    return np.where(actions == DISCHARGE, 1.0, np.where(actions == DEFER, 1 - cm.reader_sensitivity, 0.0))


def net_benefit(p_treat: np.ndarray, treat: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # Vickers and Elkin 2006: benefit of "refer iff p >= p_t" in true positives per
    # patient, with false positives weighted by the odds a clinician at threshold
    # p_t implicitly assigns them. The usual clinical view of the same trade-off.
    n = len(treat)
    out = []
    for t in thresholds:
        flag = p_treat >= t
        out.append((flag & treat).sum() / n - (flag & ~treat).sum() / n * t / (1 - t))
    return np.array(out)


def prior_shift(probs: np.ndarray, train_prior: np.ndarray, deploy_prior: np.ndarray) -> np.ndarray:
    # Saerens et al. 2002: a posterior learned under one class prior is turned into
    # the posterior under another by reweighting and renormalising
    p = probs * (deploy_prior / train_prior)
    return p / p.sum(axis=1, keepdims=True)


def ensemble_mutual_information(logits_by_seed: list[np.ndarray], temperatures: list[float]) -> np.ndarray:
    # the training seeds are a free deep ensemble; MI between the label and the
    # ensemble member is the standard epistemic score (Lakshminarayanan et al. 2017)
    log_p = np.stack([log_softmax(z / t, axis=1) for z, t in zip(logits_by_seed, temperatures)])
    p = np.exp(log_p)
    mean_p = p.mean(axis=0)
    entropy_of_mean = -(mean_p * np.log(mean_p + 1e-12)).sum(axis=1)
    mean_entropy = -(p * log_p).sum(axis=2).mean(axis=0)
    return entropy_of_mean - mean_entropy


def entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)


def risk_coverage(score: np.ndarray, error: np.ndarray):
    # Geifman and El-Yaniv 2017: predict on the most confident fraction (lowest score),
    # selective risk is the error rate among those; AURC is the area under that curve
    order = np.argsort(score, kind="stable")
    risk = np.cumsum(error[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)
    return coverage, risk, float(risk.mean())
