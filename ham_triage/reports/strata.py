import numpy as np
import pandas as pd
from scipy.special import softmax

from ..calibration import fit_temperature
from ..config import Paths
from ..conformal import aps_scores, prediction_sets, quantile
from ..decision import DEFER, REFER, CostModel, bayes_actions
from ..stats import ci, cluster_bootstrap
from .common import MEL, load_runs, load_split
from .decision import defer_knob_for_rate


def strata(paths: Paths, split: str = "lesion", prefix: str = "full", cm: CostModel = CostModel(), alpha: float = 0.1,
           n_boot: int = 1000, target_defer: float = 0.2, min_n: int = 50) -> dict:
    # Across classes dx_type is class in disguise (every mel, bcc and akiec label is
    # histopathology). Within nv it is not: histo-nv are nevi that looked suspicious
    # enough to excise, follow_up-nv are monitored nevi from one device, consensus-nv
    # are textbook cases. If the triage layer is doing its job it should treat them
    # differently even though they share a label.
    meta, sp = load_split(paths, split)
    test, cal = sp.test.values, sp.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    dx, dx_type = meta.dx.values[test], meta.dx_type.values[test]
    runs = load_runs(paths, prefix)
    seeds = sorted(runs)
    rng = np.random.default_rng(0)
    u_cal, u_test = rng.random(cal.sum()), rng.random(test.sum())

    per_image = []
    for s in seeds:
        t = fit_temperature(runs[s][cal], y_cal)
        p_cal, p_test = softmax(runs[s][cal] / t, axis=1), softmax(runs[s][test] / t, axis=1)
        actions = bayes_actions(p_test, cm, defer_knob_for_rate(p_cal, target_defer, cm))
        sets = prediction_sets(aps_scores(p_test, u_test), quantile(aps_scores(p_cal, u_cal)[np.arange(len(y_cal)), y_cal], alpha))
        flagged = p_test[:, MEL] >= 1 - quantile(1 - p_cal[y_cal == MEL, MEL], alpha)
        per_image.append(np.stack([p_test.argmax(1) != y, p_test[:, MEL], actions == DEFER, actions == REFER,
                                   flagged, sets.sum(1), sets[np.arange(len(y)), y]], axis=1).astype(float))
    per_image = np.stack(per_image)
    metrics = ["error_rate", "p_mel", "defer_rate", "refer_rate", "mel_flag_rate", "aps_set_size", "aps_coverage"]

    groups = [(c, t) for c in ("nv", "bkl") for t in ("histo", "follow_up", "consensus", "confocal")
              if ((dx == c) & (dx_type == t)).sum() >= min_n]
    masks = [(dx == c) & (dx_type == t) for c, t in groups]

    def stat(idx):
        return np.stack([per_image[:, idx][:, m[idx]].mean(axis=1).mean(axis=0) for m in masks])

    point = stat(np.arange(len(y)))
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    lo, hi = ci(boot)
    rows = []
    for g, (c, t) in enumerate(groups):
        for k, m in enumerate(metrics):
            rows.append({"dx": c, "dx_type": t, "n": int(masks[g].sum()), "n_lesions": int(len(np.unique(lesion[masks[g]]))),
                         "metric": m, "point": point[g, k], "lo": lo[g, k], "hi": hi[g, k]})
    table = pd.DataFrame(rows)
    out = {"split": split, "prefix": prefix, "seeds": seeds, "alpha": alpha, "target_defer": target_defer, "min_n": min_n,
           "table": table.round(4).to_dict(orient="records")}
    for g, (c, t) in enumerate(groups):
        line = f"{c}/{t:9s} n={masks[g].sum():4d}"
        for m in ("error_rate", "p_mel", "defer_rate", "refer_rate", "mel_flag_rate", "aps_set_size"):
            j = metrics.index(m)
            line += f"  {m} {point[g, j]:.3f} [{lo[g, j]:.3f}, {hi[g, j]:.3f}]"
        print(line)
    return out
