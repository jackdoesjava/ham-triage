import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from .calibration import CALIBRATION_METRICS, bin_stats, calibration_metrics, ece, fit_temperature, nll
from .config import CLASSES, Paths
from .conformal import aps_scores, coverage_law, lac_scores, mondrian_quantiles, prediction_sets, quantile
from .stats import ci, cluster_bootstrap, recall_per_class

MEL = CLASSES.index("mel")
TREAT = np.isin(np.arange(len(CLASSES)), [CLASSES.index(c) for c in ("mel", "bcc", "akiec")])
METRICS = ["acc", "bal_acc"] + list(CLASSES)


def load_logits(paths: Paths, run_id: str) -> np.ndarray:
    return pd.read_parquet(paths.results / "runs" / run_id / "logits.parquet").filter(like="logit_").values


def load_runs(paths: Paths, prefix: str) -> dict[int, np.ndarray]:
    runs = {}
    for d in sorted((paths.results / "runs").glob(f"{prefix}_s*")):
        info = json.loads((d / "run.json").read_text())
        runs[info["seed"]] = load_logits(paths, d.name)
    return runs


def load_test(paths: Paths):
    meta = pd.read_parquet(paths.meta)
    split = pd.read_parquet(paths.results / "splits" / "audit.parquet")
    return meta, split, split.test.values


def metric_vector(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    rec = recall_per_class(pred, y)
    return np.concatenate([[(pred == y).mean(), np.nanmean(rec)], rec])


def interval_row(samples: np.ndarray, point: float, per_seed: np.ndarray, **keys) -> dict:
    lo, hi = ci(samples)
    return {**keys, "point": point, "lo": lo, "hi": hi, "seed_min": per_seed.min(), "seed_max": per_seed.max()}


def audit(paths: Paths, n_boot: int = 1000) -> dict:
    meta, split, test = load_test(paths)
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
        return {k: r[k] for k in ("point", "lo", "hi", "seed_min", "seed_max")}

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
    h = out["headline"]
    print(f"leak rate {out['leak_rate']:.1%} (mel {out['leak_rate_mel']:.1%}), seeds {seeds}")
    for k in ("bal_acc", "mel_recall"):
        d = h[k]["delta"]
        print(f"{k:10s} clean {h[k]['clean']['point']:.3f}  leaky {h[k]['leaky']['point']:.3f}  "
              f"delta {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  seeds [{d['seed_min']:+.3f}, {d['seed_max']:+.3f}]")
    for sub in ("leaked", "unleaked"):
        d = h[f"{sub}_delta"]
        print(f"{sub:10s} delta bal_acc {d['bal_acc']['point']:+.3f} [{d['bal_acc']['lo']:+.3f}, {d['bal_acc']['hi']:+.3f}]  "
              f"mel {d['mel']['point']:+.3f} [{d['mel']['lo']:+.3f}, {d['mel']['hi']:+.3f}]")
    print("per lesion  " + "  ".join(f"{c}: bal {v['bal_acc']:.3f} mel {v['mel']:.3f}" for c, v in per_lesion.items()))
    n = naive_out
    print(f"naive 80/20 image split: bal {n['bal_acc']['point']:.3f} [{n['bal_acc']['lo']:.3f}, {n['bal_acc']['hi']:.3f}]  mel {n['mel']['point']:.3f}")
    return out


def imbalance(paths: Paths, n_boot: int = 1000) -> dict:
    meta, split, test = load_test(paths)
    cal = split.cal.values
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


def calibration(paths: Paths, n_boot: int = 1000) -> dict:
    meta, split, test = load_test(paths)
    cal = split.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    treat = TREAT[y]

    ce = load_runs(paths, "clean")
    seeds = sorted(ce)
    temps = {s: fit_temperature(ce[s][cal], y_cal) for s in seeds}
    stages = {"pre": {s: 1.0 for s in seeds}, "post": temps}
    metrics = CALIBRATION_METRICS + ["ece_treat_adaptive"]

    def p_treat(z, t):
        return softmax(z / t, axis=1)[:, TREAT].sum(axis=1)

    def stat(idx):
        out = np.empty((2, len(metrics)))
        for i, ts in enumerate(stages.values()):
            vals = []
            for s in seeds:
                z = ce[s][test][idx]
                # the discharge decision later thresholds p(needs treatment) at a few percent,
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
    per_seed = np.array([[np.append(calibration_metrics(ce[s][test], y, ts[s]),
                                    ece(p_treat(ce[s][test], ts[s]), treat, n_bins=10, adaptive=True))
                          for ts in stages.values()] for s in seeds])

    rows = []
    for k, metric in enumerate(metrics):
        for i, stage in enumerate(stages):
            rows.append(interval_row(boot[:, i, k], point[i, k], per_seed[:, i, k], stage=stage, metric=metric))
        rows.append(interval_row(boot[:, 1, k] - boot[:, 0, k], point[1, k] - point[0, k],
                                 per_seed[:, 1, k] - per_seed[:, 0, k], stage="delta", metric=metric))
    table = pd.DataFrame(rows)

    # reliability bins pooled over seeds, for the figures
    def bins(conf, hit, **kw):
        b = bin_stats(conf, hit, **kw)
        return {k: [None if isinstance(x, float) and np.isnan(x) else float(x) for x in v] for k, v in b.items()}

    reliability = {"top_label": {}, "treat": {}}
    for stage, ts in stages.items():
        probs = np.concatenate([softmax(ce[s][test] / ts[s], axis=1) for s in seeds])
        yy = np.tile(y, len(seeds))
        reliability["top_label"][stage] = bins(probs.max(1), probs.argmax(1) == yy)
        reliability["treat"][stage] = bins(probs[:, TREAT].sum(1), TREAT[yy], n_bins=10, adaptive=True)

    out = {"seeds": seeds, "n_boot": n_boot, "n_cal": int(cal.sum()),
           "temperatures": {s: round(t, 4) for s, t in temps.items()},
           "table": table.round(4).to_dict(orient="records"), "reliability": reliability}
    print("temperatures " + "  ".join(f"seed {s}: {t:.3f}" for s, t in temps.items()))
    for metric in metrics:
        r = {st: table.query("stage == @st and metric == @metric").iloc[0] for st in ("pre", "post", "delta")}
        print(f"{metric:19s} pre {r['pre'].point:.4f} [{r['pre'].lo:.4f}, {r['pre'].hi:.4f}]  "
              f"post {r['post'].point:.4f} [{r['post'].lo:.4f}, {r['post'].hi:.4f}]  "
              f"delta {r['delta'].point:+.4f} [{r['delta'].lo:+.4f}, {r['delta'].hi:+.4f}]")
    return out


def conformal(paths: Paths, alpha: float = 0.1, n_boot: int = 1000, n_repartitions: int = 200) -> dict:
    meta, split, test = load_test(paths)
    cal = split.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    ce = load_runs(paths, "clean")
    seeds = sorted(ce)
    rng = np.random.default_rng(0)

    # one temperature and one set of scores per seed; the same u per image is reused
    # across seeds so seed averaging does not also average over the randomisation
    temps = {s: fit_temperature(ce[s][cal], y_cal) for s in seeds}
    probs = {s: (softmax(ce[s][cal] / temps[s], axis=1), softmax(ce[s][test] / temps[s], axis=1)) for s in seeds}
    u_cal, u_test = rng.random(cal.sum()), rng.random(test.sum())
    ar_cal, ar_test = np.arange(cal.sum()), np.arange(test.sum())

    marginal, marginal_lac, mondrian, mel_rule = {}, {}, {}, {}
    for s in seeds:
        p_cal, p_test = probs[s]
        s_cal, s_test = aps_scores(p_cal, u_cal), aps_scores(p_test, u_test)
        marginal[s] = prediction_sets(s_test, quantile(s_cal[ar_cal, y_cal], alpha))
        l_cal, l_test = lac_scores(p_cal), lac_scores(p_test)
        marginal_lac[s] = prediction_sets(l_test, quantile(l_cal[ar_cal, y_cal], alpha))
        q = mondrian_quantiles(l_cal[ar_cal, y_cal], y_cal, alpha)
        mondrian[s] = (q, prediction_sets(l_test, q))
        # "flag if mel is in the set" with the quantile from mel cal images only: a
        # threshold on p(mel) chosen so that P(flagged | mel) >= 1 - a over cal draws
        mel_rule[s] = {a: l_test[:, MEL] <= quantile(l_cal[y_cal == MEL, MEL], a) for a in (0.1, 0.05)}

    def stat(idx):
        yy = y[idx]
        out = []
        for s in seeds:
            m, ml, md = marginal[s][idx], marginal_lac[s][idx], mondrian[s][1][idx]
            row = [m[np.arange(len(idx)), yy].mean(), m.sum(1).mean(), (~m.any(1)).mean()]
            row += [m[yy == k, k].mean() for k in range(len(CLASSES))]
            row += [ml[np.arange(len(idx)), yy].mean(), ml.sum(1).mean(), (~ml.any(1)).mean()]
            row += [md[np.arange(len(idx)), yy].mean(), md.sum(1).mean()]
            row += [md[yy == k, k].mean() for k in range(len(CLASSES))]
            for a in (0.1, 0.05):
                f = mel_rule[s][a][idx]
                row += [f[yy == MEL].mean(), f[yy != MEL].mean(), f[TREAT[yy] & (yy != MEL)].mean()]
            out.append(row)
        return np.mean(out, axis=0)

    names = (["marginal_coverage", "marginal_set_size", "marginal_empty_rate"] + [f"marginal_cov_{c}" for c in CLASSES]
             + ["lac_coverage", "lac_set_size", "lac_empty_rate"]
             + ["mondrian_coverage", "mondrian_set_size"] + [f"mondrian_cov_{c}" for c in CLASSES]
             + [f"mel_rule_{a}_{k}" for a in (0.1, 0.05) for k in ("sensitivity", "flag_rate_non_mel", "flag_rate_bcc_akiec")])
    point = stat(np.arange(len(y)))
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    lo, hi = ci(boot)
    table = [{"metric": n, "point": p, "lo": l, "hi": h} for n, p, l, h in zip(names, point, lo, hi)]

    n_cal_class = np.bincount(y_cal, minlength=len(CLASSES))
    per_class = [{"class": c, "n_cal": int(n), "alpha_min": round(1 / (n + 1), 3),
                  "degenerate_at_alpha": bool(np.isinf(np.mean([mondrian[s][0][k] for s in seeds]))),
                  "p_threshold_mean": None if np.isinf(np.mean([mondrian[s][0][k] for s in seeds]))
                  else float(1 - np.mean([mondrian[s][0][k] for s in seeds]))}
                 for k, (c, n) in enumerate(zip(CLASSES, n_cal_class))]

    # coverage over lesion-grouped re-partitions of the pooled cal + test images, seed 0,
    # with the temperature refit each time so cal and test stay exchangeable given T;
    # compared against Beta(n + 1 - l, l) with n = number of cal lesions, not images
    pool = cal | test
    z = ce[seeds[0]][pool]
    y_pool, les_pool = meta.label.values[pool], meta.lesion_id.values[pool]
    uniq_les = np.unique(les_pool)
    n_cal_lesions = int(cal.sum())
    rep = {"marginal": [], "mel": []}
    for _ in range(n_repartitions):
        chosen = set(rng.choice(uniq_les, n_cal_lesions, replace=False))
        in_cal_les = np.array([l in chosen for l in les_pool])
        order = rng.permutation(np.flatnonzero(in_cal_les))
        cal_idx = pd.Series(order).groupby(les_pool[order]).first().values
        test_idx = np.flatnonzero(~in_cal_les)
        t = fit_temperature(z[cal_idx], y_pool[cal_idx])
        pc, pt = softmax(z[cal_idx] / t, axis=1), softmax(z[test_idx] / t, axis=1)
        sc, st = aps_scores(pc, rng.random(len(cal_idx))), aps_scores(pt, rng.random(len(test_idx)))
        sets = prediction_sets(st, quantile(sc[np.arange(len(cal_idx)), y_pool[cal_idx]], alpha))
        rep["marginal"].append(sets[np.arange(len(test_idx)), y_pool[test_idx]].mean())
        q_mel = quantile(1 - pc[y_pool[cal_idx] == MEL, MEL], alpha)
        rep["mel"].append((1 - pt[y_pool[test_idx] == MEL, MEL] <= q_mel).mean())
    n_mel_cal = int((y_cal == MEL).sum())
    laws = {"marginal": coverage_law(n_cal_lesions, alpha), "mel": coverage_law(n_mel_cal, alpha)}

    # the test set was drawn at the image level, which over-samples multi-image lesions
    # relative to the lesion-level cal draw; the class mix is stored so the write-up can
    # show why marginal coverage on this fixed split sits below the re-partitioned value
    class_mix = {"cal": (n_cal_class / n_cal_class.sum()).round(4).tolist(),
                 "test": (np.bincount(y, minlength=len(CLASSES)) / len(y)).round(4).tolist()}
    out = {"alpha": alpha, "seeds": seeds, "n_boot": n_boot, "n_cal": n_cal_lesions, "n_mel_cal": n_mel_cal,
           "class_mix": class_mix,
           "temperatures": {s: round(t, 4) for s, t in temps.items()},
           "table": table, "per_class": per_class,
           "repartitions": {k: {"coverage": np.round(v, 4).tolist(), "beta_a": laws[k].args[0], "beta_b": laws[k].args[1],
                                "beta_mean": laws[k].mean(), "beta_std": laws[k].std(),
                                "empirical_mean": float(np.mean(v)), "empirical_std": float(np.std(v))}
                            for k, v in rep.items()}}
    show = {r["metric"]: r for r in table}
    print(f"alpha {alpha}: marginal APS coverage {show['marginal_coverage']['point']:.3f} "
          f"[{show['marginal_coverage']['lo']:.3f}, {show['marginal_coverage']['hi']:.3f}]  set size {show['marginal_set_size']['point']:.2f}  "
          f"empty {show['marginal_empty_rate']['point']:.1%}")
    print(f"  LAC marginal coverage {show['lac_coverage']['point']:.3f}  set size {show['lac_set_size']['point']:.2f}  "
          f"empty {show['lac_empty_rate']['point']:.1%}")
    print("  per-class coverage  " + "  ".join(f"{c} {show[f'marginal_cov_{c}']['point']:.2f}" for c in CLASSES))
    print(f"mondrian LAC coverage {show['mondrian_coverage']['point']:.3f}  set size {show['mondrian_set_size']['point']:.2f}")
    print("  per-class coverage  " + "  ".join(f"{c} {show[f'mondrian_cov_{c}']['point']:.2f}" for c in CLASSES))
    print("  n_cal per class     " + "  ".join(f"{r['class']} {r['n_cal']}{'*' if r['degenerate_at_alpha'] else ''}" for r in per_class))
    for a in (0.1, 0.05):
        s, f = show[f"mel_rule_{a}_sensitivity"], show[f"mel_rule_{a}_flag_rate_non_mel"]
        print(f"mel rule alpha {a}: sensitivity {s['point']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  "
              f"flags {f['point']:.1%} [{f['lo']:.1%}, {f['hi']:.1%}] of non-mel")
    for k, v in out["repartitions"].items():
        print(f"repartition {k:8s} coverage mean {v['empirical_mean']:.4f} sd {v['empirical_std']:.4f}   "
              f"beta mean {v['beta_mean']:.4f} sd {v['beta_std']:.4f}")
    return out


def dump(paths: Paths, name: str, obj: dict) -> None:
    (paths.results / "derived").mkdir(parents=True, exist_ok=True)
    with open(paths.results / "derived" / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=float)


if __name__ == "__main__":
    paths = Paths()
    for name, section in (("audit", audit), ("imbalance", imbalance), ("calibration", calibration), ("conformal", conformal)):
        print(f"--- {name} ---")
        dump(paths, name, section(paths))
