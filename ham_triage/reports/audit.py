import numpy as np
import pandas as pd

from ..config import CLASSES, Paths
from ..stats import ci, cluster_bootstrap
from .common import MEL, METRICS, interval_row, load_logits, load_runs, load_split, metric_vector

SPLITS = {0: ("audit", "clean", "leaky"), 1: ("audit_s1", "clean_a1", "leaky_a1"), 2: ("audit_s2", "clean_a2", "leaky_a2")}


def one_split(paths: Paths, split_name: str, prefixes, n_boot: int):
    meta, split = load_split(paths, split_name)
    test = split.test.values
    y = meta.label.values[test]
    lesion = meta.lesion_id.values[test]
    leaked = split.sibling_in_leaky_train.values[test]
    subsets = {"all": np.ones(len(y), bool), "leaked": leaked, "unleaked": ~leaked}
    conds = ("clean", "leaky")

    runs = {c: load_runs(paths, p) for c, p in zip(conds, prefixes)}
    seeds = sorted(set(runs["clean"]) & set(runs["leaky"]))
    if not seeds:
        return None
    pred = {c: np.stack([runs[c][s][test].argmax(1) for s in seeds]) for c in conds}

    def stat(idx):
        # [subset, cond, metric] averaged over seeds, all on one lesion resample of the
        # test set, so the subset numbers and the deltas share a single sampling scheme
        out = np.empty((len(subsets), len(conds), len(METRICS)))
        for i, m in enumerate(subsets.values()):
            sel = idx[m[idx]]
            for j, c in enumerate(conds):
                out[i, j] = np.mean([metric_vector(p[sel], y[sel]) for p in pred[c]], axis=0)
        return out

    point = stat(np.arange(len(y)))
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    per_seed = np.array([[[metric_vector(pred[c][k][m], y[m]) for c in conds] for m in subsets.values()]
                         for k in range(len(seeds))])
    rows = []
    for i, name in enumerate(subsets):
        for k, metric in enumerate(METRICS):
            for j, c in enumerate(conds):
                rows.append(interval_row(boot[:, i, j, k], point[i, j, k], per_seed[:, i, j, k],
                                         subset=name, cond=c, metric=metric))
            rows.append(interval_row(boot[:, i, 1, k] - boot[:, i, 0, k], point[i, 1, k] - point[i, 0, k],
                                     per_seed[:, i, 1, k] - per_seed[:, i, 0, k], subset=name, cond="delta", metric=metric))
    table = pd.DataFrame(rows)

    def pick(subset, cond, metric):
        r = table.query("subset == @subset and cond == @cond and metric == @metric").iloc[0]
        return {k: r[k] for k in ("point", "lo", "hi", "seed_min", "seed_max")}

    out = {
        "split": split_name, "seeds": seeds,
        "n_test": int(len(y)), "n_test_lesions": int(len(np.unique(lesion))),
        "n_leaked": int(leaked.sum()), "n_unleaked": int((~leaked).sum()),
        "leak_rate": float(leaked.mean()), "leak_rate_mel": float(leaked[y == MEL].mean()),
        "headline": {
            "bal_acc": {c: pick("all", c, "bal_acc") for c in ("clean", "leaky", "delta")},
            "mel_recall": {c: pick("all", c, "mel") for c in ("clean", "leaky", "delta")},
            "leaked_delta": {m: pick("leaked", "delta", m) for m in ("bal_acc", "mel")},
            "unleaked_delta": {m: pick("unleaked", "delta", m) for m in ("bal_acc", "mel")},
        },
        "table": table.round(4).to_dict(orient="records"),
    }
    # the delta bootstrap draws, kept so splits can be pooled draw by draw
    deltas = boot[:, :, 1, :] - boot[:, :, 0, :]
    return out, deltas, point[:, 1, :] - point[:, 0, :], y, lesion, runs, seeds, test


def audit(paths: Paths, n_boot: int = 1000) -> dict:
    per_split, draws, points = {}, [], []
    first = None
    for seed, (split_name, *prefixes) in SPLITS.items():
        res = one_split(paths, split_name, prefixes, n_boot)
        if res is None:
            continue
        out, deltas, point, y, lesion, runs, seeds, test = res
        per_split[split_name] = out
        draws.append(deltas)
        points.append(point)
        if first is None:
            first = (y, lesion, runs, seeds, test)

    # pooled over test-set draws: the test sets are independent, so averaging the
    # bootstrap draws across splits is a bootstrap of the mean delta; the split range
    # is shown beside it because three splits do not estimate a variance either
    draws, points = np.stack(draws), np.stack(points)
    lo, hi = ci(draws.mean(axis=0))
    subsets = ("all", "leaked", "unleaked")
    pooled = {}
    for i, sub in enumerate(subsets):
        for k, metric in enumerate(METRICS):
            pooled[f"{sub}/{metric}"] = {"point": points[:, i, k].mean(), "lo": lo[i, k], "hi": hi[i, k],
                                        "split_min": points[:, i, k].min(), "split_max": points[:, i, k].max()}

    y, lesion, runs, seeds, test = first
    # the clinical unit is the lesion; average logits over a lesion's test images and
    # check the headline does not move (only about 120 test lesions have more than one image)
    per_lesion = {}
    for c in ("clean", "leaky"):
        vals = []
        for s in seeds:
            frame = pd.DataFrame(runs[c][s][test]).groupby(lesion).mean()
            y_les = pd.Series(y).groupby(lesion).first().loc[frame.index].values
            vals.append(metric_vector(frame.values.argmax(1), y_les))
        per_lesion[c] = dict(zip(METRICS, np.mean(vals, axis=0).round(4).tolist()))

    # what a plain 80/20 image split reports, with all 8012 training images, single seed
    naive = load_logits(paths, "naive_s0")[test].argmax(1)
    nb = cluster_bootstrap(lambda idx: metric_vector(naive[idx], y[idx]), lesion, n_boot=n_boot)
    nlo, nhi = ci(nb)
    naive_out = {m: {"point": v, "lo": l, "hi": h} for m, v, l, h in zip(METRICS, metric_vector(naive, y), nlo, nhi)}

    out = {"n_boot": n_boot, "splits": per_split, "pooled_delta": pooled,
           "naive_image_split": naive_out, "per_lesion": per_lesion}
    for name, s in per_split.items():
        h = s["headline"]
        print(f"{name:9s} leak {s['leak_rate']:.1%} (mel {s['leak_rate_mel']:.1%})  seeds {s['seeds']}  "
              f"bal_acc {h['bal_acc']['clean']['point']:.3f} -> {h['bal_acc']['leaky']['point']:.3f} "
              f"({h['bal_acc']['delta']['point']:+.3f} [{h['bal_acc']['delta']['lo']:+.3f}, {h['bal_acc']['delta']['hi']:+.3f}])  "
              f"mel {h['mel_recall']['delta']['point']:+.3f}  unleaked bal {h['unleaked_delta']['bal_acc']['point']:+.3f}")
    for key in ("all/bal_acc", "all/mel", "leaked/mel", "unleaked/bal_acc", "unleaked/mel"):
        p = pooled[key]
        print(f"pooled {key:16s} {p['point']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]  splits [{p['split_min']:+.3f}, {p['split_max']:+.3f}]")
    print("per lesion  " + "  ".join(f"{c}: bal {v['bal_acc']:.3f} mel {v['mel']:.3f}" for c, v in per_lesion.items()))
    n = naive_out
    print(f"naive 80/20 image split: bal {n['bal_acc']['point']:.3f} [{n['bal_acc']['lo']:.3f}, {n['bal_acc']['hi']:.3f}]  mel {n['mel']['point']:.3f}")
    return out
