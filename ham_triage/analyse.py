import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, Paths
from .stats import ci, cluster_bootstrap, recall_per_class

MEL = CLASSES.index("mel")
METRICS = ["acc", "bal_acc"] + list(CLASSES)


def load_logits(paths: Paths, run_id: str) -> np.ndarray:
    return pd.read_parquet(paths.results / "runs" / run_id / "logits.parquet").filter(like="logit_").values


def load_runs(paths: Paths, prefix: str) -> dict[int, np.ndarray]:
    runs = {}
    for d in sorted((paths.results / "runs").glob(f"{prefix}_s*")):
        info = json.loads((d / "run.json").read_text())
        runs[info["seed"]] = load_logits(paths, d.name)
    return runs


def metric_vector(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    rec = recall_per_class(pred, y)
    return np.concatenate([[(pred == y).mean(), np.nanmean(rec)], rec])


def audit(paths: Paths, n_boot: int = 1000) -> dict:
    meta = pd.read_parquet(paths.meta)
    split = pd.read_parquet(paths.results / "splits" / "audit.parquet")
    test = split.test.values
    y = meta.label.values[test]
    lesion = meta.lesion_id.values[test]
    leaked = split.sibling_in_leaky_train.values[test]
    subsets = {"all": np.ones(len(y), bool), "leaked": leaked, "unleaked": ~leaked}
    conds = ("clean", "leaky")

    runs = {c: load_runs(paths, c) for c in conds}
    seeds = sorted(set(runs["clean"]) & set(runs["leaky"]))
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
    lo, hi = ci(boot)
    dlo, dhi = ci(boot[:, :, 1] - boot[:, :, 0])
    per_seed = np.array([[[metric_vector(pred[c][k][m], y[m]) for c in conds] for m in subsets.values()]
                         for k in range(len(seeds))])

    rows = []
    for i, name in enumerate(subsets):
        for k, metric in enumerate(METRICS):
            for j, c in enumerate(conds):
                rows.append({"subset": name, "cond": c, "metric": metric, "point": point[i, j, k],
                             "lo": lo[i, j, k], "hi": hi[i, j, k],
                             "seed_min": per_seed[:, i, j, k].min(), "seed_max": per_seed[:, i, j, k].max()})
            d = per_seed[:, i, 1, k] - per_seed[:, i, 0, k]
            rows.append({"subset": name, "cond": "delta", "metric": metric, "point": point[i, 1, k] - point[i, 0, k],
                         "lo": dlo[i, k], "hi": dhi[i, k], "seed_min": d.min(), "seed_max": d.max()})
    table = pd.DataFrame(rows)

    # the clinical unit is the lesion; average logits over a lesion's test images and
    # check the headline does not move (only 118 test lesions have more than one image)
    per_lesion = {}
    for c in conds:
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

    def pick(subset, cond, metric):
        r = table.query("subset == @subset and cond == @cond and metric == @metric").iloc[0]
        return {"point": r.point, "lo": r.lo, "hi": r.hi, "seed_min": r.seed_min, "seed_max": r.seed_max}

    out = {
        "seeds": seeds, "n_boot": n_boot,
        "n_test": int(len(y)), "n_test_lesions": int(len(np.unique(lesion))),
        "n_leaked": int(leaked.sum()), "n_unleaked": int((~leaked).sum()),
        "leak_rate": float(leaked.mean()), "leak_rate_mel": float(leaked[y == MEL].mean()),
        "headline": {
            "bal_acc": {c: pick("all", c, "bal_acc") for c in ("clean", "leaky", "delta")},
            "mel_recall": {c: pick("all", c, "mel") for c in ("clean", "leaky", "delta")},
            "leaked_delta": {m: pick("leaked", "delta", m) for m in ("bal_acc", "mel")},
            "unleaked_delta": {m: pick("unleaked", "delta", m) for m in ("bal_acc", "mel")},
        },
        "naive_image_split": naive_out,
        "per_lesion": per_lesion,
        "table": table.round(4).to_dict(orient="records"),
    }
    return out


def dump(paths: Paths, name: str, obj: dict) -> None:
    (paths.results / "derived").mkdir(parents=True, exist_ok=True)
    with open(paths.results / "derived" / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=float)


if __name__ == "__main__":
    paths = Paths()
    a = audit(paths)
    dump(paths, "audit", a)
    h = a["headline"]
    print(f"leak rate {a['leak_rate']:.1%} (mel {a['leak_rate_mel']:.1%}), seeds {a['seeds']}")
    for k in ("bal_acc", "mel_recall"):
        d = h[k]["delta"]
        print(f"{k:10s} clean {h[k]['clean']['point']:.3f}  leaky {h[k]['leaky']['point']:.3f}  "
              f"delta {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  seeds [{d['seed_min']:+.3f}, {d['seed_max']:+.3f}]")
    for sub in ("leaked", "unleaked"):
        d = h[f"{sub}_delta"]
        print(f"{sub:10s} delta bal_acc {d['bal_acc']['point']:+.3f} [{d['bal_acc']['lo']:+.3f}, {d['bal_acc']['hi']:+.3f}]  "
              f"mel {d['mel']['point']:+.3f} [{d['mel']['lo']:+.3f}, {d['mel']['hi']:+.3f}]")
    print("per lesion  " + "  ".join(f"{c}: bal {v['bal_acc']:.3f} mel {v['mel']:.3f}" for c, v in a["per_lesion"].items()))
    n = a["naive_image_split"]
    print(f"naive 80/20 image split: bal {n['bal_acc']['point']:.3f} [{n['bal_acc']['lo']:.3f}, {n['bal_acc']['hi']:.3f}]  mel {n['mel']['point']:.3f}")
