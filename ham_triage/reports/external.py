import numpy as np
import pandas as pd
from scipy.special import softmax

from ..calibration import fit_temperature, nll
from ..config import CLASSES, Paths
from ..conformal import aps_scores, prediction_sets, quantile
from ..decision import DEFER, REFER, CostModel, bayes_actions, melanoma_miss_weight, realized_cost
from ..stats import ci, cluster_bootstrap
from .common import MEL, METRICS, TREAT, interval_row, load_runs, load_split, metric_vector
from .decision import defer_knob_for_rate

# which HAM10000 split supplied each model family's calibration images; the audit cal
# lesions were excluded from both the clean and the leaky training sets, so both may use it
GROUPS = {"leaky": ("leaky", "audit"), "clean_size_matched": ("clean", "audit"), "full": ("full", "lesion")}


def external(paths: Paths, cm: CostModel = CostModel(), alpha: float = 0.1, n_boot: int = 1000,
             target_defer: float = 0.2) -> dict:
    # The ISIC 2018 task 3 test set has no lesion ids in its public files, so the
    # bootstrap is by image and the intervals are optimistic by however many
    # multi-image lesions it contains. It is the only evaluation here on images the
    # models never saw a sibling of, from a collection that includes other centres.
    ext = pd.read_parquet(paths.data / "isic2018_meta.parquet")
    y_ext = ext.label.values
    mel, benign = y_ext == MEL, ~TREAT[y_ext]
    meta = pd.read_parquet(paths.meta)
    rng = np.random.default_rng(0)
    u_ext = rng.random(len(y_ext))

    per_image, meta_out, internal = {}, {}, {}
    for group, (prefix, split_name) in GROUPS.items():
        ham = load_runs(paths, prefix)
        isic = load_runs(paths, prefix, name="isic2018")
        _, sp = load_split(paths, split_name)
        cal, test = sp.cal.values, sp.test.values
        y_cal = meta.label.values[cal]
        seeds = sorted(set(ham) & set(isic))
        if not seeds:
            continue
        # the same models on their own HAM test split, lesion-bootstrapped, so the
        # internal and external numbers sit side by side
        y_int, les_int = meta.label.values[test], meta.lesion_id.values[test]
        preds = [ham[s][test].argmax(1) for s in seeds]
        ib = cluster_bootstrap(lambda idx: np.mean([metric_vector(pr[idx], y_int[idx]) for pr in preds], axis=0), les_int, n_boot=n_boot)
        ilo, ihi = ci(ib)
        ipoint = np.mean([metric_vector(pr, y_int) for pr in preds], axis=0)
        internal[group] = [{"metric": m, "point": v, "lo": l, "hi": h} for m, v, l, h in zip(METRICS, ipoint, ilo, ihi)]
        rows = []
        for s in seeds:
            t = fit_temperature(ham[s][cal], y_cal)
            p_cal, p = softmax(ham[s][cal] / t, axis=1), softmax(isic[s] / t, axis=1)
            pred = p.argmax(1)
            # the same HAM-calibrated conformal thresholds and the same cal-chosen deferral
            # price, applied to data the calibration set is not exchangeable with
            u_cal = rng.random(cal.sum())
            sets = prediction_sets(aps_scores(p, u_ext), quantile(aps_scores(p_cal, u_cal)[np.arange(len(y_cal)), y_cal], alpha))
            flagged = p[:, MEL] >= 1 - quantile(1 - p_cal[y_cal == MEL, MEL], alpha)
            actions = bayes_actions(p, cm, defer_knob_for_rate(p_cal, target_defer, cm))
            rows.append(np.stack([pred == y_ext, pred, nll(isic[s], y_ext, t), sets[np.arange(len(y_ext)), y_ext],
                                  sets.sum(1), flagged, actions == DEFER, actions == REFER,
                                  melanoma_miss_weight(actions, cm), realized_cost(actions, y_ext, cm)], axis=1))
        per_image[group] = np.stack(rows)
        meta_out[group] = {"seeds": seeds, "n_cal": int(cal.sum())}

    groups = list(per_image)
    names = METRICS + ["nll_ts", "aps_coverage", "aps_set_size", "aps_cov_mel", "mel_rule_sensitivity",
                       "mel_rule_flag_rate_non_mel", "defer_rate", "refer_rate", "mel_miss_rate", "cost"]

    def summarize(rows, idx):
        out = []
        for r in rows:
            sub = r[idx]
            yy = y_ext[idx]
            hit, pred = sub[:, 0], sub[:, 1].astype(int)
            out.append(np.concatenate([metric_vector(pred, yy), [
                sub[:, 2].mean(), sub[:, 3].mean(), sub[:, 4].mean(), sub[mel[idx], 3].mean(),
                sub[mel[idx], 5].mean(), sub[~mel[idx], 5].mean(), sub[:, 6].mean(), sub[:, 7].mean(),
                sub[mel[idx], 8].mean(), sub[:, 9].mean()]]))
        return np.mean(out, axis=0)

    def stat(idx):
        return np.stack([summarize(per_image[g], idx) for g in groups])

    point = stat(np.arange(len(y_ext)))
    boot = cluster_bootstrap(stat, np.arange(len(y_ext)), n_boot=n_boot)
    per_seed = {g: np.array([summarize(per_image[g][k:k + 1], np.arange(len(y_ext))) for k in range(len(per_image[g]))]) for g in groups}
    table = []
    for i, g in enumerate(groups):
        for k, metric in enumerate(names):
            table.append(interval_row(boot[:, i, k], point[i, k], per_seed[g][:, k], group=g, metric=metric, kind="value"))
    base = groups.index("clean_size_matched") if "clean_size_matched" in groups else 0
    for i, g in enumerate(groups):
        if i == base:
            continue
        for k, metric in enumerate(names):
            d = boot[:, i, k] - boot[:, base, k]
            lo, hi = ci(d)
            table.append({"group": g, "metric": metric, "kind": f"delta_vs_{groups[base]}", "point": point[i, k] - point[base, k],
                          "lo": lo, "hi": hi, "seed_min": np.nan, "seed_max": np.nan})

    out = {"n_images": int(len(y_ext)), "class_counts": dict(zip(CLASSES, np.bincount(y_ext, minlength=len(CLASSES)).tolist())),
           "alpha": alpha, "target_defer": target_defer, "cost_model": cm.__dict__, "n_boot": n_boot,
           "bootstrap_unit": "image (no lesion ids published)", "groups": meta_out,
           "internal": {g: pd.DataFrame(v).round(4).to_dict(orient="records") for g, v in internal.items()},
           "table": pd.DataFrame(table).round(4).to_dict(orient="records")}
    tbl = pd.DataFrame(table)
    for g in groups:
        v = {m: tbl.query("group == @g and metric == @m and kind == 'value'").iloc[0] for m in
             ("acc", "bal_acc", "mel", "nll_ts", "aps_coverage", "aps_cov_mel", "mel_rule_sensitivity", "mel_rule_flag_rate_non_mel", "mel_miss_rate", "cost")}
        print(f"{g:19s} acc {v['acc'].point:.3f}  bal {v['bal_acc'].point:.3f} [{v['bal_acc'].lo:.3f}, {v['bal_acc'].hi:.3f}]  "
              f"mel {v['mel'].point:.3f}  nll {v['nll_ts'].point:.3f}  APS cov {v['aps_coverage'].point:.3f} (mel {v['aps_cov_mel'].point:.2f})  "
              f"mel rule sens {v['mel_rule_sensitivity'].point:.3f} flags {v['mel_rule_flag_rate_non_mel'].point:.2f}  "
              f"triage miss {v['mel_miss_rate'].point:.3f} cost {v['cost'].point:.3f}")
    for g in groups:
        if g == groups[base]:
            continue
        v = {m: tbl.query(f"group == @g and metric == @m and kind == 'delta_vs_{groups[base]}'").iloc[0] for m in ("bal_acc", "mel")}
        print(f"{g} - {groups[base]}: bal {v['bal_acc'].point:+.3f} [{v['bal_acc'].lo:+.3f}, {v['bal_acc'].hi:+.3f}]  "
              f"mel {v['mel'].point:+.3f} [{v['mel'].lo:+.3f}, {v['mel'].hi:+.3f}]")
    return out
