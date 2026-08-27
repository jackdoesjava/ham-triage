from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.special import softmax

from ..calibration import fit_temperature
from ..config import CLASSES, Paths
from ..conformal import quantile
from ..decision import (DEFER, DISCHARGE, REFER, CostModel, bayes_actions, defer_by_score, ensemble_mutual_information,
                        entropy, expected_cost, expected_miss, melanoma_miss_weight, net_benefit, prior_shift,
                        realized_cost, risk_coverage)
from ..stats import cluster_bootstrap
from .common import MEL, TREAT, interval_row, load_runs, load_split

COLS = ["cost", "expected_cost", "defer_rate", "refer_rate", "mel_miss_rate", "benign_human_rate"]


def defer_knob_for_rate(p: np.ndarray, rate: float, cm: CostModel) -> float:
    # deferral rate is monotone in the defer price, so bisect on its log against the cal set
    lo, hi = np.log(1e-4), np.log(cm.refer)
    for _ in range(40):
        mid = (lo + hi) / 2
        if (bayes_actions(p, cm, np.exp(mid)) == DEFER).mean() > rate:
            lo = mid
        else:
            hi = mid
    return float(np.exp((lo + hi) / 2))


def decision(paths: Paths, split: str = "lesion", prefix: str = "full", cm: CostModel = CostModel(),
             n_boot: int = 1000, target_defer: float = 0.2) -> dict:
    meta, sp = load_split(paths, split)
    test, cal = sp.test.values, sp.cal.values
    y, y_cal = meta.label.values[test], meta.label.values[cal]
    lesion = meta.lesion_id.values[test]
    runs = load_runs(paths, prefix)
    seeds = sorted(runs)
    temps = [fit_temperature(runs[s][cal], y_cal) for s in seeds]
    p_cal = [softmax(runs[s][cal] / t, axis=1) for s, t in zip(seeds, temps)]
    p_test = [softmax(runs[s][test] / t, axis=1) for s, t in zip(seeds, temps)]
    p_raw = [softmax(runs[s][test], axis=1) for s in seeds]
    mi_cal = ensemble_mutual_information([runs[s][cal] for s in seeds], temps)
    mi_test = ensemble_mutual_information([runs[s][test] for s in seeds], temps)
    rng = np.random.default_rng(0)
    mel, benign = y == MEL, ~TREAT[y]

    def per_image(actions, p, c=cm):
        # [n, 6]: realized cost, expected cost, deferred, referred, mel-miss weight, benign sent to a human
        return np.stack([realized_cost(actions, y, c), expected_cost(actions, p, c), actions == DEFER,
                         actions == REFER, melanoma_miss_weight(actions, c) * mel, benign & (actions != DISCHARGE)], axis=1)

    def summarize(rows, idx):
        # mean over idx, except the mel miss rate which conditions on melanoma
        sub = rows[idx]
        out = sub.mean(axis=0)
        out[4] = sub[mel[idx], 4].mean()
        out[5] = sub[benign[idx], 5].mean()
        return out

    def score_sets(p, m):
        return {"msp": 1 - p.max(1), "entropy": entropy(p), "ensemble_mi": m}

    # operating point: every deferral policy at target_defer chosen on cal, plus fixed rules
    policies = {}
    for k, s in enumerate(seeds):
        d = defer_knob_for_rate(p_cal[k], target_defer, cm)
        acts = {"bayes": bayes_actions(p_test[k], cm, d),
                "bayes_cost_optimal": bayes_actions(p_test[k], cm),
                "bayes_raw_posteriors": bayes_actions(p_raw[k], cm, defer_knob_for_rate(softmax(runs[s][cal], axis=1), target_defer, cm))}
        sc_cal, sc_test = score_sets(p_cal[k], mi_cal), score_sets(p_test[k], mi_test)
        for name in sc_cal:
            acts[name] = defer_by_score(sc_test[name], np.quantile(sc_cal[name], 1 - target_defer), p_test[k], cm)
        acts["random"] = defer_by_score(rng.random(len(y)), 1 - target_defer, p_test[k], cm)
        acts["always_predict"] = bayes_actions(p_test[k], cm, np.inf)
        acts["argmax_7class"] = np.where(TREAT[p_test[k].argmax(1)], REFER, DISCHARGE)
        acts["refer_all"] = np.full(len(y), REFER)
        acts["discharge_all"] = np.full(len(y), DISCHARGE)
        q_mel = quantile(1 - p_cal[k][y_cal == MEL, MEL], 0.1)
        acts["conformal_mel_0.1"] = np.where(p_test[k][:, MEL] >= 1 - q_mel, REFER, DISCHARGE)
        for name, a in acts.items():
            policies.setdefault(name, []).append(per_image(a, p_raw[k] if name == "bayes_raw_posteriors" else p_test[k]))
    names = list(policies)
    rows_by_policy = {n: np.stack(v) for n, v in policies.items()}

    def stat(idx):
        return np.stack([np.mean([summarize(r, idx) for r in rows_by_policy[n]], axis=0) for n in names])

    point = stat(np.arange(len(y)))
    boot = cluster_bootstrap(stat, lesion, n_boot=n_boot)
    per_seed = np.array([[summarize(rows_by_policy[n][k], np.arange(len(y))) for n in names] for k in range(len(seeds))])
    b = names.index("bayes")
    table = []
    for i, n in enumerate(names):
        for j, c in enumerate(COLS):
            table.append(interval_row(boot[:, i, j], point[i, j], per_seed[:, i, j], policy=n, metric=c, kind="value"))
            if n != "bayes":
                table.append(interval_row(boot[:, i, j] - boot[:, b, j], point[i, j] - point[b, j],
                                          per_seed[:, i, j] - per_seed[:, b, j], policy=n, metric=c, kind="delta_vs_bayes"))

    # frontier curves, seed-averaged point estimates; thresholds come from cal at each target rate
    grid = np.linspace(0, 0.9, 19)
    curves = {}
    for k in range(len(seeds)):
        sc_cal, sc_test = score_sets(p_cal[k], mi_cal), score_sets(p_test[k], mi_test)
        for f in grid:
            d = defer_knob_for_rate(p_cal[k], f, cm) if f > 0 else np.inf
            pts = {"bayes": (bayes_actions(p_test[k], cm, d), p_test[k]),
                   "bayes_raw_posteriors": (bayes_actions(p_raw[k], cm, d), p_raw[k])}
            for name in sc_cal:
                pts[name] = (defer_by_score(sc_test[name], np.quantile(sc_cal[name], 1 - f), p_test[k], cm), p_test[k])
            for name, (a, p) in pts.items():
                curves.setdefault(name, []).append(summarize(per_image(a, p), np.arange(len(y))))
    deferral_curves = {n: np.mean(np.array(v).reshape(len(seeds), len(grid), 6), axis=0).round(4).tolist() for n, v in curves.items()}
    # random deferral of a fraction f is the straight line between these two points
    ends = [summarize(per_image(a, p_test[0]), np.arange(len(y))) for a in
            (bayes_actions(p_test[0], cm, np.inf), np.full(len(y), DEFER))]
    deferral_curves["random_endpoints"] = np.round(ends, 4).tolist()

    # referral panel: no deferral, refer iff a score exceeds a threshold; the conformal mel
    # rule is a point on the p(mel) curve whose threshold was picked for coverage instead
    referral = {}
    rate_grid = np.linspace(0.02, 0.98, 49)
    for k in range(len(seeds)):
        s_cal, s_test = expected_miss(p_cal[k], cm), expected_miss(p_test[k], cm)
        for name, (c_, t_) in {"expected_miss": (s_cal, s_test), "p_mel": (p_cal[k][:, MEL], p_test[k][:, MEL])}.items():
            for f in rate_grid:
                a = np.where(t_ >= np.quantile(c_, 1 - f), REFER, DISCHARGE)
                referral.setdefault(name, []).append([(a == REFER).mean(), (a[mel] == DISCHARGE).mean(), (a[benign] == REFER).mean()])
        for alpha in (0.2, 0.1, 0.05, 0.02):
            q = quantile(1 - p_cal[k][y_cal == MEL, MEL], alpha)
            a = np.where(p_test[k][:, MEL] >= 1 - q, REFER, DISCHARGE)
            referral.setdefault(f"conformal_{alpha}", []).append([(a == REFER).mean(), (a[mel] == DISCHARGE).mean(), (a[benign] == REFER).mean()])
    referral_curves = {n: np.mean(np.array(v).reshape(len(seeds), -1, 3), axis=0).round(4).tolist() for n, v in referral.items()}

    # decision curve (Vickers and Elkin 2006) for "refer iff p(needs treatment) >= p_t",
    # against refer-all and refer-none, seed-averaged; the clinical view of the same trade-off
    thresholds = np.linspace(0.01, 0.5, 50)
    treat = TREAT[y]
    nb_model = np.mean([net_benefit(p[:, TREAT].sum(1), treat, thresholds) for p in p_test], axis=0)
    nb_raw = np.mean([net_benefit(p[:, TREAT].sum(1), treat, thresholds) for p in p_raw], axis=0)
    prevalence = treat.mean()
    nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    decision_curve = {"threshold": thresholds.round(3).tolist(), "calibrated": nb_model.round(4).tolist(),
                      "raw": nb_raw.round(4).tolist(), "refer_all": nb_all.round(4).tolist(), "prevalence": float(prevalence)}

    # does the ordering of deferral policies survive the invented constants
    sensitivity = []
    for miss_mel in (10, 30, 100, 300):
        for r in (1.0, 0.87, 0.8):
            for spec in (1.0, 0.71):
                c = CostModel(miss_mel=miss_mel, miss_treat=cm.miss_treat, refer=cm.refer, defer=cm.defer,
                              reader_sensitivity=r, reader_specificity=spec)
                res = {}
                for k in range(len(seeds)):
                    d = defer_knob_for_rate(p_cal[k], target_defer, c)
                    sc_cal, sc_test = score_sets(p_cal[k], mi_cal), score_sets(p_test[k], mi_test)
                    acts = {"bayes": bayes_actions(p_test[k], c, d),
                            "random": defer_by_score(rng.random(len(y)), 1 - target_defer, p_test[k], c)}
                    for name in sc_cal:
                        acts[name] = defer_by_score(sc_test[name], np.quantile(sc_cal[name], 1 - target_defer), p_test[k], c)
                    for name, a in acts.items():
                        res.setdefault(name, []).append([realized_cost(a, y, c).mean(), melanoma_miss_weight(a, c)[mel].mean()])
                row = {"miss_mel": miss_mel, "reader_sensitivity": r, "reader_specificity": spec}
                for name, v in res.items():
                    row[f"{name}_cost"], row[f"{name}_mel_miss"] = np.mean(v, axis=0).round(4).tolist()
                sensitivity.append(row)

    # deployment prevalence: HAM is excision-enriched (11% melanoma by image). Shift the
    # posterior (Saerens et al. 2002) and importance-weight the test set to a prevalence
    # with the treated classes five times rarer, and see what the same rules do
    train_col = "train" if "train" in sp else "clean_train"
    train_prior = np.bincount(meta.label.values[sp[train_col].values], minlength=len(CLASSES)) / sp[train_col].sum()
    deploy_prior = train_prior.copy()
    deploy_prior[TREAT] /= 5
    deploy_prior /= deploy_prior.sum()
    test_prior = np.bincount(y, minlength=len(CLASSES)) / len(y)
    w = (deploy_prior / test_prior)[y]
    shift_rows = []
    for k in range(len(seeds)):
        ps_cal, ps_test = prior_shift(p_cal[k], train_prior, deploy_prior), prior_shift(p_test[k], train_prior, deploy_prior)
        for name, p_c, p_t in (("ham_prevalence", p_cal[k], p_test[k]), ("screening_prevalence", ps_cal, ps_test)):
            d = defer_knob_for_rate(p_c, target_defer, cm)
            a = bayes_actions(p_t, cm, d)
            ww = w if name == "screening_prevalence" else np.ones(len(y))
            miss = melanoma_miss_weight(a, cm)
            shift_rows.append({"scenario": name, "seed": seeds[k], "defer_knob": d,
                               "cost": float(np.average(realized_cost(a, y, cm), weights=ww)),
                               "defer_rate": float(np.average(a == DEFER, weights=ww)),
                               "refer_rate": float(np.average(a == REFER, weights=ww)),
                               "mel_miss_rate": float(miss[mel].mean()),
                               "benign_human_rate": float(np.average((a != DISCHARGE)[benign], weights=ww[benign]))})
    shift = pd.DataFrame(shift_rows).groupby("scenario").mean(numeric_only=True).drop(columns="seed").round(4)

    # the 0-1 slice: risk-coverage of the 7-class argmax under each abstention score
    rc = {}
    for k in range(len(seeds)):
        err = p_test[k].argmax(1) != y
        for name, s in score_sets(p_test[k], mi_test).items():
            rc.setdefault(name, []).append((s, err))

    def aurc_stat(idx):
        return np.array([np.mean([risk_coverage(s[idx], e[idx])[2] for s, e in rc[n]]) for n in rc])

    aurc_point = aurc_stat(np.arange(len(y)))
    aurc_boot = cluster_bootstrap(aurc_stat, lesion, n_boot=n_boot)
    aurc = []
    for i, n in enumerate(rc):
        aurc.append(interval_row(aurc_boot[:, i], aurc_point[i], np.array([risk_coverage(s, e)[2] for s, e in rc[n]]), score=n, kind="value"))
        if n != "msp":
            aurc.append(interval_row(aurc_boot[:, i] - aurc_boot[:, 0], aurc_point[i] - aurc_point[0],
                                     np.array([risk_coverage(s, e)[2] - risk_coverage(s0, e0)[2] for (s, e), (s0, e0) in zip(rc[n], rc["msp"])]),
                                     score=n, kind="delta_vs_msp"))
    rc_curves = {}
    for n in rc:
        cov, risk, _ = risk_coverage(*rc[n][0])
        pick = np.linspace(0, len(cov) - 1, 100).astype(int)
        rc_curves[n] = {"coverage": cov[pick].round(4).tolist(), "risk": risk[pick].round(4).tolist()}

    out = {"split": split, "prefix": prefix, "cost_model": asdict(cm), "target_defer": target_defer, "seeds": seeds,
           "n_boot": n_boot, "temperatures": [round(t, 4) for t in temps],
           "operating_point": pd.DataFrame(table).round(4).to_dict(orient="records"),
           "deferral_curves": {"grid": grid.round(3).tolist(), "columns": COLS, "curves": deferral_curves},
           "referral_curves": {"columns": ["refer_rate", "mel_miss_rate", "benign_refer_rate"], "curves": referral_curves},
           "decision_curve": decision_curve,
           "sensitivity": sensitivity,
           "prior_shift": {"deploy_prior": dict(zip(CLASSES, deploy_prior.round(4).tolist())), "table": shift.to_dict(orient="index")},
           "risk_coverage": {"aurc": pd.DataFrame(aurc).round(4).to_dict(orient="records"), "curves_seed0": rc_curves}}

    tbl = pd.DataFrame(table)
    print(f"cost model {asdict(cm)}, {target_defer:.0%} deferral chosen on cal")
    for n in names:
        v = {c: tbl.query("policy == @n and metric == @c and kind == 'value'").iloc[0] for c in COLS}
        print(f"{n:22s} cost {v['cost'].point:6.3f} [{v['cost'].lo:.3f}, {v['cost'].hi:.3f}]  expected {v['expected_cost'].point:6.3f}  "
              f"defer {v['defer_rate'].point:.2f} refer {v['refer_rate'].point:.2f}  mel miss {v['mel_miss_rate'].point:.3f} "
              f"[{v['mel_miss_rate'].lo:.3f}, {v['mel_miss_rate'].hi:.3f}]  benign->human {v['benign_human_rate'].point:.2f}")
    print("sensitivity (cost | mel miss) at 20% deferral, reader specificity 0.71 rows:")
    for r in sensitivity:
        if r["reader_specificity"] == 0.71:
            print(f"  R={r['miss_mel']:>3} r={r['reader_sensitivity']:.2f}  " + "  ".join(
                f"{n} {r[f'{n}_cost']:.3f}|{r[f'{n}_mel_miss']:.3f}" for n in ("bayes", "msp", "entropy", "ensemble_mi", "random")))
    print("prior shift:\n" + shift.to_string())
    print("AURC: " + "  ".join(f"{r['score']} {r['point']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]" for r in aurc if r["kind"] == "value"))
    return out
