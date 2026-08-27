import numpy as np
import pandas as pd
from scipy.special import softmax

from ..calibration import fit_temperature
from ..config import CLASSES, Paths
from ..conformal import aps_scores, coverage_law, lac_scores, mondrian_quantiles, prediction_sets, quantile
from ..stats import ci, cluster_bootstrap
from .common import MEL, TREAT, load_runs, load_split


def conformal(paths: Paths, split: str = "lesion", prefix: str = "full", alpha: float = 0.1,
              n_boot: int = 1000, n_repartitions: int = 200, n_draws: int = 10) -> dict:
    meta, sp = load_split(paths, split)
    test, cal = sp.test.values, sp.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    # Calibration points are one image per lesion; the test set keeps every image, and
    # multi-image lesions are both over-represented per image and much harder. The
    # guarantee is per exchangeable unit, so coverage is measured per lesion (each
    # image weighted by 1 / images of its lesion) and the per-image number is kept
    # beside it to show the gap.
    w = 1.0 / pd.Series(lesion).map(pd.Series(lesion).value_counts()).values
    runs = load_runs(paths, prefix)
    seeds = sorted(runs)
    rng = np.random.default_rng(0)

    temps = {s: fit_temperature(runs[s][cal], y_cal) for s in seeds}
    probs = {s: (softmax(runs[s][cal] / temps[s], axis=1), softmax(runs[s][test] / temps[s], axis=1)) for s in seeds}
    ar_cal, ar_test = np.arange(cal.sum()), np.arange(test.sum())

    # APS sets are averaged over several randomisation draws per seed; a single draw
    # shared across seeds froze one realisation and moved coverage by two sd
    marginal, marginal_lac, mondrian, mel_rule = {}, {}, {}, {}
    for s in seeds:
        p_cal, p_test = probs[s]
        marginal[s] = np.stack([
            prediction_sets(aps_scores(p_test, rng.random(len(y))),
                            quantile(aps_scores(p_cal, rng.random(len(y_cal)))[ar_cal, y_cal], alpha))
            for _ in range(n_draws)])
        l_cal, l_test = lac_scores(p_cal), lac_scores(p_test)
        marginal_lac[s] = prediction_sets(l_test, quantile(l_cal[ar_cal, y_cal], alpha))
        q = mondrian_quantiles(l_cal[ar_cal, y_cal], y_cal, alpha)
        mondrian[s] = (q, prediction_sets(l_test, q))
        # "flag if mel is in the set" with the quantile from mel cal images only: a
        # threshold on p(mel) chosen so that P(flagged | mel) >= 1 - a over cal draws
        mel_rule[s] = {a: l_test[:, MEL] <= quantile(l_cal[y_cal == MEL, MEL], a) for a in (0.1, 0.05)}

    def wmean(x, ww):
        return float(np.average(x, weights=ww)) if len(x) else np.nan

    def stat(idx):
        yy, ww = y[idx], w[idx]
        out = []
        for s in seeds:
            hit = marginal[s][:, idx][:, np.arange(len(idx)), yy].mean(axis=0)
            size = marginal[s][:, idx].sum(axis=2).mean(axis=0)
            empty = (~marginal[s][:, idx].any(axis=2)).mean(axis=0)
            ml, md = marginal_lac[s][idx], mondrian[s][1][idx]
            row = [wmean(hit, ww), hit.mean(), wmean(size, ww), empty.mean()]
            row += [wmean(hit[yy == k], ww[yy == k]) for k in range(len(CLASSES))]
            lhit = ml[np.arange(len(idx)), yy]
            row += [wmean(lhit, ww), lhit.mean(), wmean(ml.sum(1), ww), (~ml.any(1)).mean()]
            mhit = md[np.arange(len(idx)), yy]
            row += [wmean(mhit, ww), wmean(md.sum(1), ww)]
            row += [wmean(mhit[yy == k], ww[yy == k]) for k in range(len(CLASSES))]
            for a in (0.1, 0.05):
                f = mel_rule[s][a][idx]
                row += [wmean(f[yy == MEL], ww[yy == MEL]), f[yy == MEL].mean(), wmean(f[yy != MEL], ww[yy != MEL]),
                        wmean(f[TREAT[yy] & (yy != MEL)], ww[TREAT[yy] & (yy != MEL)])]
            out.append(row)
        return np.mean(out, axis=0)

    names = (["marginal_coverage", "marginal_coverage_per_image", "marginal_set_size", "marginal_empty_rate"]
             + [f"marginal_cov_{c}" for c in CLASSES]
             + ["lac_coverage", "lac_coverage_per_image", "lac_set_size", "lac_empty_rate"]
             + ["mondrian_coverage", "mondrian_set_size"] + [f"mondrian_cov_{c}" for c in CLASSES]
             + [f"mel_rule_{a}_{k}" for a in (0.1, 0.05)
                for k in ("sensitivity", "sensitivity_per_image", "flag_rate_non_mel", "flag_rate_bcc_akiec")])
    point = stat(np.arange(len(y)))
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    lo, hi = ci(boot)
    table = [{"metric": n, "point": p, "lo": l, "hi": h} for n, p, l, h in zip(names, point, lo, hi)]

    n_cal_class = np.bincount(y_cal, minlength=len(CLASSES))
    per_class = []
    for k, (c, n) in enumerate(zip(CLASSES, n_cal_class)):
        q_mean = np.mean([mondrian[s][0][k] for s in seeds])
        per_class.append({"class": c, "n_cal": int(n), "alpha_min": round(1 / (n + 1), 3),
                          "degenerate_at_alpha": bool(np.isinf(q_mean)),
                          "p_threshold_mean": None if np.isinf(q_mean) else float(1 - q_mean)})

    # coverage over lesion-grouped re-partitions of the pooled cal + test images, seed 0,
    # with the temperature refit each time so cal and test stay exchangeable given T;
    # compared against Beta(n + 1 - l, l) with n = number of cal lesions, not images
    pool = cal | test
    z = runs[seeds[0]][pool]
    y_pool, les_pool = meta.label.values[pool], meta.lesion_id.values[pool]
    w_pool = 1.0 / pd.Series(les_pool).map(pd.Series(les_pool).value_counts()).values
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
        rep["marginal"].append(np.average(sets[np.arange(len(test_idx)), y_pool[test_idx]], weights=w_pool[test_idx]))
        q_mel = quantile(1 - pc[y_pool[cal_idx] == MEL, MEL], alpha)
        is_mel = y_pool[test_idx] == MEL
        rep["mel"].append(np.average(1 - pt[is_mel, MEL] <= q_mel, weights=w_pool[test_idx][is_mel]))
    n_mel_cal = int((y_cal == MEL).sum())
    laws = {"marginal": coverage_law(n_cal_lesions, alpha), "mel": coverage_law(n_mel_cal, alpha)}

    class_mix = {"cal": (n_cal_class / n_cal_class.sum()).round(4).tolist(),
                 "test": (np.bincount(y, minlength=len(CLASSES)) / len(y)).round(4).tolist()}
    out = {"split": split, "prefix": prefix, "alpha": alpha, "seeds": seeds, "n_boot": n_boot, "n_draws": n_draws,
           "n_cal": n_cal_lesions, "n_mel_cal": n_mel_cal, "class_mix": class_mix,
           "multi_image_share": {"cal": float(np.mean(pd.Series(meta.lesion_id.values[cal]).map(
               meta.lesion_id.value_counts()).values > 1)), "test": float(np.mean(w < 1))},
           "temperatures": {s: round(t, 4) for s, t in temps.items()},
           "table": table, "per_class": per_class,
           "repartitions": {k: {"coverage": np.round(v, 4).tolist(), "beta_a": laws[k].args[0], "beta_b": laws[k].args[1],
                                "beta_mean": laws[k].mean(), "beta_std": laws[k].std(),
                                "empirical_mean": float(np.mean(v)), "empirical_std": float(np.std(v))}
                            for k, v in rep.items()}}
    show = {r["metric"]: r for r in table}
    print(f"alpha {alpha}: marginal APS coverage per lesion {show['marginal_coverage']['point']:.3f} "
          f"[{show['marginal_coverage']['lo']:.3f}, {show['marginal_coverage']['hi']:.3f}]  per image {show['marginal_coverage_per_image']['point']:.3f}  "
          f"set size {show['marginal_set_size']['point']:.2f}  empty {show['marginal_empty_rate']['point']:.1%}")
    print(f"  LAC marginal coverage per lesion {show['lac_coverage']['point']:.3f}  per image {show['lac_coverage_per_image']['point']:.3f}  "
          f"set size {show['lac_set_size']['point']:.2f}")
    print("  per-class coverage  " + "  ".join(f"{c} {show[f'marginal_cov_{c}']['point']:.2f}" for c in CLASSES))
    print(f"mondrian LAC coverage {show['mondrian_coverage']['point']:.3f}  set size {show['mondrian_set_size']['point']:.2f}")
    print("  per-class coverage  " + "  ".join(f"{c} {show[f'mondrian_cov_{c}']['point']:.2f}" for c in CLASSES))
    print("  n_cal per class     " + "  ".join(f"{r['class']} {r['n_cal']}{'*' if r['degenerate_at_alpha'] else ''}" for r in per_class))
    for a in (0.1, 0.05):
        s, f = show[f"mel_rule_{a}_sensitivity"], show[f"mel_rule_{a}_flag_rate_non_mel"]
        print(f"mel rule alpha {a}: sensitivity per lesion {s['point']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  "
              f"per image {show[f'mel_rule_{a}_sensitivity_per_image']['point']:.3f}  flags {f['point']:.1%} of non-mel")
    for k, v in out["repartitions"].items():
        print(f"repartition {k:8s} coverage mean {v['empirical_mean']:.4f} sd {v['empirical_std']:.4f}   "
              f"beta mean {v['beta_mean']:.4f} sd {v['beta_std']:.4f}")
    return out
