import numpy as np
import pandas as pd

from ..calibration import fit_temperature, nll
from ..config import CLASSES, Paths
from ..stats import cluster_bootstrap
from .common import METRICS, interval_row, load_runs, load_split, metric_vector


def imbalance(paths: Paths, n_boot: int = 1000) -> dict:
    # a training-procedure comparison, so it stays on the audit split where the
    # class-balanced models were trained against the size-matched clean ones
    meta, split = load_split(paths, "audit")
    test, cal = split.test.values, split.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    train_labels = meta.label.values[split.clean_train.values]
    log_prior = np.log(np.bincount(train_labels, minlength=len(CLASSES)) / len(train_labels))

    ce, cb = load_runs(paths, "clean"), load_runs(paths, "cb")
    seeds = sorted(set(ce) & set(cb))
    # Menon et al. 2021: subtracting the log training prior from CE logits is the
    # Bayes-optimal classifier for balanced error, at zero training cost. It should
    # buy what the class-balanced loss buys in argmax terms, without retraining.
    variants = {
        "ce": [ce[s] for s in seeds],
        "ce_logit_adjusted": [ce[s] - log_prior for s in seeds],
        "cb": [cb[s] for s in seeds],
    }
    pred = {v: np.stack([z[test].argmax(1) for z in zs]) for v, zs in variants.items()}
    # calibration comparisons are only fair after temperature scaling, fitted per variant
    temps = {v: [fit_temperature(z[cal], y_cal) for z in zs] for v, zs in variants.items()}
    nll_raw = {v: np.stack([nll(z[test], y) for z in zs]) for v, zs in variants.items()}
    nll_ts = {v: np.stack([nll(z[test], y, t) for z, t in zip(zs, temps[v])]) for v, zs in variants.items()}
    metrics = METRICS + ["nll_raw", "nll_ts"]

    def stat(idx):
        out = np.empty((len(variants), len(metrics)))
        for j, v in enumerate(variants):
            out[j, :-2] = np.mean([metric_vector(p[idx], y[idx]) for p in pred[v]], axis=0)
            out[j, -2] = nll_raw[v][:, idx].mean()
            out[j, -1] = nll_ts[v][:, idx].mean()
        return out

    point = stat(np.arange(len(y)))
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    per_seed = np.array([[np.concatenate([metric_vector(pred[v][k], y), [nll_raw[v][k].mean(), nll_ts[v][k].mean()]])
                          for v in variants] for k in range(len(seeds))])

    rows = []
    for j, v in enumerate(variants):
        for k, metric in enumerate(metrics):
            rows.append(interval_row(boot[:, j, k], point[j, k], per_seed[:, j, k], variant=v, metric=metric, kind="value"))
            if j:
                rows.append(interval_row(boot[:, j, k] - boot[:, 0, k], point[j, k] - point[0, k],
                                         per_seed[:, j, k] - per_seed[:, 0, k], variant=v, metric=metric, kind="delta_vs_ce"))
    table = pd.DataFrame(rows)
    out = {"seeds": seeds, "n_boot": n_boot, "log_prior": dict(zip(CLASSES, log_prior.round(4).tolist())),
           "temperatures": {v: [round(t, 4) for t in ts] for v, ts in temps.items()},
           "table": table.round(4).to_dict(orient="records")}
    for metric in ("bal_acc", "mel", "nv", "nll_raw", "nll_ts"):
        line = f"{metric:8s}"
        for v in variants:
            r = table.query("variant == @v and metric == @metric and kind == 'value'").iloc[0]
            line += f"  {v} {r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}]"
        print(line)
    return out
