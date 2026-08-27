import numpy as np
import pandas as pd
from scipy.special import softmax

from ..calibration import CALIBRATION_METRICS, bin_stats, calibration_metrics, ece, fit_temperature
from ..config import Paths
from ..stats import cluster_bootstrap
from .common import TREAT, interval_row, load_runs, load_split


def calibration(paths: Paths, split: str = "lesion", prefix: str = "full", n_boot: int = 1000) -> dict:
    meta, sp = load_split(paths, split)
    test, cal = sp.test.values, sp.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    treat = TREAT[y]

    runs = load_runs(paths, prefix)
    seeds = sorted(runs)
    temps = {s: fit_temperature(runs[s][cal], y_cal) for s in seeds}
    stages = {"pre": {s: 1.0 for s in seeds}, "post": temps}
    metrics = CALIBRATION_METRICS + ["ece_treat_adaptive"]

    def p_treat(z, t):
        return softmax(z / t, axis=1)[:, TREAT].sum(axis=1)

    def stat(idx):
        out = np.empty((2, len(metrics)))
        for i, ts in enumerate(stages.values()):
            vals = []
            for s in seeds:
                z = runs[s][test][idx]
                # the discharge decision thresholds p(needs treatment) at a few percent,
                # a region top-label ECE cannot see, so calibrate that probability too
                vals.append(np.append(calibration_metrics(z, y[idx], ts[s]),
                                      ece(p_treat(z, ts[s]), treat[idx], n_bins=10, adaptive=True)))
            out[i] = np.mean(vals, axis=0)
        return out

    point = stat(np.arange(len(y)))
    # ECE is biased upward on a resample (duplicated points make the bins lumpier), so
    # its percentile interval sits high relative to the point estimate; the pre/post
    # delta cancels most of it, and NLL and Brier are plain means with no such issue
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    per_seed = np.array([[np.append(calibration_metrics(runs[s][test], y, ts[s]),
                                    ece(p_treat(runs[s][test], ts[s]), treat, n_bins=10, adaptive=True))
                          for ts in stages.values()] for s in seeds])

    rows = []
    for k, metric in enumerate(metrics):
        for i, stage in enumerate(stages):
            rows.append(interval_row(boot[:, i, k], point[i, k], per_seed[:, i, k], stage=stage, metric=metric))
        rows.append(interval_row(boot[:, 1, k] - boot[:, 0, k], point[1, k] - point[0, k],
                                 per_seed[:, 1, k] - per_seed[:, 0, k], stage="delta", metric=metric))
    table = pd.DataFrame(rows)

    def bins(conf, hit, **kw):
        b = bin_stats(conf, hit, **kw)
        return {k: [None if isinstance(x, float) and np.isnan(x) else float(x) for x in v] for k, v in b.items()}

    # reliability bins pooled over seeds, for the figures
    reliability = {"top_label": {}, "treat": {}}
    for stage, ts in stages.items():
        probs = np.concatenate([softmax(runs[s][test] / ts[s], axis=1) for s in seeds])
        yy = np.tile(y, len(seeds))
        reliability["top_label"][stage] = bins(probs.max(1), probs.argmax(1) == yy)
        reliability["treat"][stage] = bins(probs[:, TREAT].sum(1), TREAT[yy], n_bins=10, adaptive=True)

    out = {"split": split, "prefix": prefix, "seeds": seeds, "n_boot": n_boot, "n_cal": int(cal.sum()),
           "temperatures": {s: round(t, 4) for s, t in temps.items()},
           "table": table.round(4).to_dict(orient="records"), "reliability": reliability}
    print("temperatures " + "  ".join(f"seed {s}: {t:.3f}" for s, t in temps.items()))
    for metric in metrics:
        r = {st: table.query("stage == @st and metric == @metric").iloc[0] for st in ("pre", "post", "delta")}
        print(f"{metric:19s} pre {r['pre'].point:.4f} [{r['pre'].lo:.4f}, {r['pre'].hi:.4f}]  "
              f"post {r['post'].point:.4f} [{r['post'].lo:.4f}, {r['post'].hi:.4f}]  "
              f"delta {r['delta'].point:+.4f} [{r['delta'].lo:+.4f}, {r['delta'].hi:+.4f}]")
    return out
