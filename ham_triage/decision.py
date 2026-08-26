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
    reader_sensitivity: float = 0.87  # Haenssle et al. 2018, dermatologists on dermoscopy


def miss_costs(cm: CostModel) -> np.ndarray:
    m = np.zeros(len(CLASSES))
    m[MEL] = cm.miss_mel
    m[TREAT2] = cm.miss_treat
    return m


def expected_miss(probs: np.ndarray, cm: CostModel) -> np.ndarray:
    return probs @ miss_costs(cm)


def bayes_actions(probs: np.ndarray, cm: CostModel, defer_knob: float | None = None) -> np.ndarray:
    """Expected-cost-minimising action among discharge, refer and defer to a human reader.

    With s(x) = sum_y p(y|x) m_y the expected miss cost, the three expected costs are
        discharge: s(x)
        refer:     refer  (the visit is paid whoever the patient is; the case is resolved)
        defer:     defer + (1 - reader_sensitivity) * s(x)
    so every policy in this family is two thresholds on s(x): discharge below
    defer / r, defer up to (refer - defer) / (1 - r), refer above. With a perfect
    reader (r = 1) and defer < refer the refer region vanishes and the rule is the
    rule-out triage of Leibig et al. 2022. With defer_knob = inf it is the two-action
    rule, refer iff s(x) >= refer, which with miss costs of two referrals is the
    argmax of the binary problem. Chow 1970 is not a special case of this family (a
    flat referral cost is not a 0-1 loss); it is the MSP baseline in defer_by_score.

    defer_knob replaces the defer price in the decision only, to trace the frontier
    at deferral rates other than the cost-optimal one; realised and expected costs
    always use the true cm.defer.
    """
    d = cm.defer if defer_knob is None else defer_knob
    s = expected_miss(probs, cm)
    cost = np.stack([s, np.full_like(s, cm.refer), d + (1 - cm.reader_sensitivity) * s], axis=1)
    return cost.argmin(axis=1)


def action_costs(y: np.ndarray, cm: CostModel) -> np.ndarray:
    # [n, 3] cost of each action given the true class; the reader's miss is taken in
    # expectation over their sensitivity rather than simulated
    m = miss_costs(cm)[y]
    return np.stack([m, np.full_like(m, cm.refer), cm.defer + (1 - cm.reader_sensitivity) * m], axis=1)


def realized_cost(actions: np.ndarray, y: np.ndarray, cm: CostModel) -> np.ndarray:
    return action_costs(y, cm)[np.arange(len(y)), actions]


def expected_cost(actions: np.ndarray, probs: np.ndarray, cm: CostModel) -> np.ndarray:
    # the cost the model itself expects to pay; its gap to realized_cost is the
    # decision-level price of miscalibration
    per_class = np.stack([action_costs(np.full(len(probs), k), cm) for k in range(len(CLASSES))], axis=1)
    return np.einsum("nk,nka->na", probs, per_class)[np.arange(len(probs)), actions]


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


def prior_shift(probs: np.ndarray, train_prior: np.ndarray, deploy_prior: np.ndarray) -> np.ndarray:
    # Saerens et al. 2002: a posterior learned under one class prior is turned into
    # the posterior under another by reweighting and renormalising
    p = probs * (deploy_prior / train_prior)
    return p / p.sum(axis=1, keepdims=True)


def ensemble_mutual_information(logits_by_seed: list[np.ndarray], temperatures: list[float]) -> np.ndarray:
    # the three training seeds are a free deep ensemble; MI between the label and the
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
