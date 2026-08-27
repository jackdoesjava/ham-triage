import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from .config import CLASSES, Paths
from .reports.common import load_runs, load_split

# one fixed colour per policy, never assigned by rank; the five were checked for
# colour-vision-deficiency separation as a set
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#898781"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
POLICY = {"bayes": (BLUE, "expected-cost rule"), "msp": (ORANGE, "max softmax (Chow)"),
          "entropy": (AQUA, "entropy"), "ensemble_mi": (VIOLET, "seed-ensemble MI")}
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.8,
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load(paths: Paths, name: str) -> dict:
    return json.loads((paths.results / "derived" / f"{name}.json").read_text())


def clean(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def reliability(paths: Paths, out):
    c = load(paths, "calibration")
    temps = np.mean(list(c["temperatures"].values()))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, key, title in zip(axes, ("top_label", "treat"), ("top-label confidence", "p(needs treatment)")):
        for stage, color, label in (("pre", GREY, "raw softmax"), ("post", BLUE, f"temperature scaled (T={temps:.2f})")):
            b = c["reliability"][key][stage]
            conf, acc, n = clean(b["conf"]), clean(b["acc"]), clean(b["count"])
            ok = ~np.isnan(conf) & (n >= 20)  # a bin with a handful of images is noise, not a point
            ax.plot(conf[ok], acc[ok], "o-", color=color, ms=4, label=label)
        lo = 1e-3 if key == "treat" else 0
        ax.plot([lo, 1], [lo, 1], color=AXIS, lw=0.8, zorder=0)
        if key == "treat":
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-3, 1)
            ax.set_ylim(1e-3, 1)
        ax.set_title(title)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
    axes[0].legend(loc="upper left")
    axes[1].text(0.02, 0.98, "10 equal-mass bins; the discharge\nthreshold sits at a few percent",
                 transform=axes[1].transAxes, va="top", fontsize=7, color=MUTED)
    fig.savefig(out / "reliability.png")
    plt.close(fig)


def risk_coverage(paths: Paths, out):
    d = load(paths, "decision")["risk_coverage"]
    aurc = {r["score"]: r for r in d["aurc"] if r["kind"] == "value"}
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, (color, label) in POLICY.items():
        if name == "bayes":
            continue
        cv = d["curves_seed0"][name]
        a = aurc[name]
        ax.plot(cv["coverage"], cv["risk"], color=color, label=f"{label}, AURC {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")
    ax.set_xlabel("coverage (fraction not abstained)")
    ax.set_ylabel("selective risk (7-class error)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.savefig(out / "risk_coverage.png")
    plt.close(fig)


def frontier(paths: Paths, out):
    d = load(paths, "decision")
    cols = d["deferral_curves"]["columns"]
    cv = {k: np.array(v) for k, v in d["deferral_curves"]["curves"].items()}
    i_cost, i_exp, i_def = cols.index("cost"), cols.index("expected_cost"), cols.index("defer_rate")
    ops = {(r["policy"], r["metric"]): r for r in d["operating_point"] if r["kind"] == "value"}
    cm = d["cost_model"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))

    ax = axes[0]
    for name, (color, label) in POLICY.items():
        ax.plot(cv[name][:, i_def], cv[name][:, i_cost], color=color, label=label)
    ends = cv["random_endpoints"]
    ax.plot(ends[:, i_def], ends[:, i_cost], color=GREY, ls="--", lw=1.2, label="random deferral")
    ax.axhline(cm["refer"], color=AXIS, lw=0.8, zorder=0)
    ax.text(0.60, cm["refer"] + 0.02, "refer everyone", ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("fraction deferred to a human reader")
    ax.set_ylabel("realised cost per image (referrals)")
    ax.set_title(f"miss mel = {cm['miss_mel']:.0f}, reader {cm['reader_sensitivity']} sens / {cm['reader_specificity']} spec")
    ax.set_xlim(0, 0.9)
    ax.legend(loc="upper left")
    ax.text(0.98, 0.08, "the cost rule defers only where a read\nbeats both discharge and referral,\nso its curve stops early",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)

    ax = axes[1]
    for name, color, label in (("bayes", BLUE, "calibrated"), ("bayes_raw_posteriors", ORANGE, "raw softmax")):
        ax.plot(cv[name][:, i_def], cv[name][:, i_cost], color=color, label=f"{label}, realised")
        ax.plot(cv[name][:, i_def], cv[name][:, i_exp], color=color, ls=":", lw=1.3, label=f"{label}, expected")
    ax.set_xlabel("fraction deferred to a human reader")
    ax.set_ylabel("cost per image (referrals)")
    ax.set_title("expected-cost rule: what it thinks it pays")
    ax.set_xlim(0, 0.9)
    ax.legend(loc="upper right")

    ax = axes[2]
    r = {k: np.array(v) for k, v in d["referral_curves"]["curves"].items()}
    ax.plot(r["expected_miss"][:, 0], r["expected_miss"][:, 1], color=BLUE, label="threshold on expected miss cost")
    ax.plot(r["p_mel"][:, 0], r["p_mel"][:, 1], color=ORANGE, label="threshold on p(mel)")
    for alpha in (0.2, 0.1, 0.05, 0.02):
        pt = r[f"conformal_{alpha}"][0]
        ax.plot(pt[0], pt[1], "o", color=ORANGE, ms=6, mec="white", mew=1)
        ax.annotate(f"conformal, {alpha}", (pt[0], pt[1]), xytext=(6, 2), textcoords="offset points", fontsize=7, color=MUTED)
    am = (ops[("argmax_7class", "refer_rate")]["point"], ops[("argmax_7class", "mel_miss_rate")]["point"])
    ax.plot(*am, "s", color=INK, ms=5)
    ax.annotate("7-class argmax", am, xytext=(6, 2), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlabel("fraction referred (no deferral)")
    ax.set_ylabel("melanomas discharged (fraction)")
    ax.set_title("sensitivity against workload")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    fig.savefig(out / "frontier.png")
    plt.close(fig)


def decision_curve(paths: Paths, out):
    dc = load(paths, "decision")["decision_curve"]
    t = np.array(dc["threshold"])
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(t, dc["calibrated"], color=BLUE, label="refer if p(needs treatment) >= threshold, calibrated")
    ax.plot(t, dc["raw"], color=ORANGE, lw=1.3, label="same rule on raw softmax")
    ax.plot(t, dc["refer_all"], color=GREY, ls="--", lw=1.2, label="refer everyone")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.text(0.49, 0.005, "refer no one", ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit (true referrals per image)")
    ax.set_title(f"decision curve, prevalence {dc['prevalence']:.2f}")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.02, None)
    ax.legend(loc="upper right")
    fig.savefig(out / "decision_curve.png")
    plt.close(fig)


def confusion(paths: Paths, out):
    meta, split = load_split(paths, "audit")
    test = split.test.values
    y = meta.label.values[test]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7))
    for ax, cond, title in zip(axes, ("clean", "leaky"), ("lesion-disjoint training", "sibling images in training")):
        counts = np.zeros((len(CLASSES), len(CLASSES)))
        for z in load_runs(paths, cond).values():
            np.add.at(counts, (y, z[test].argmax(1)), 1)
        recall = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(recall, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{recall[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if recall[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(length=0)
    fig.suptitle("row-normalised confusion on the same 2003 test images, three seeds pooled", fontsize=9)
    fig.savefig(out / "confusion.png")
    plt.close(fig)


def coverage(paths: Paths, out):
    c = load(paths, "conformal")
    fixed = {r["metric"]: r["point"] for r in c["table"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, title, fixed_key in zip(axes, ("marginal", "mel"), ("marginal, APS", "melanoma-conditional, LAC"),
                                         ("marginal_coverage", "mel_rule_0.1_sensitivity")):
        r = c["repartitions"][key]
        cov = np.array(r["coverage"])
        ax.hist(cov, bins=18, density=True, color=BLUE, alpha=0.55, label=f"{len(cov)} lesion-grouped re-partitions of cal + test")
        xs = np.linspace(cov.min() - 0.03, min(1, cov.max() + 0.03), 300)
        pdf = beta(r["beta_a"], r["beta_b"]).pdf(xs)
        ax.plot(xs, pdf, color=INK, lw=1.2, label="Beta(n + 1 - l, l) given the calibration draw")
        ax.axvline(fixed[fixed_key], color=ORANGE, lw=1.4, label="the fixed split used everywhere else")
        ax.axvline(1 - c["alpha"], color=AXIS, lw=0.8, zorder=0)
        ax.text(0.02, 0.97, f"Beta({r['beta_a']:.0f}, {r['beta_b']:.0f})\nn = {r['beta_a'] + r['beta_b'] - 1:.0f} cal lesions",
                transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
        ax.set_ylim(0, pdf.max() * 1.25)
        ax.set_title(title)
        ax.set_xlabel(f"test coverage at alpha = {c['alpha']}")
        ax.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(out / "coverage.png")
    plt.close(fig)


def external(paths: Paths, out):
    e = load(paths, "external")
    groups = [g for g in ("leaky", "clean_size_matched", "full") if g in e["internal"]]
    labels = {"leaky": "sibling images\nin training", "clean_size_matched": "lesion-disjoint,\nsize-matched", "full": "lesion-disjoint,\nall data"}
    ext = {(r["group"], r["metric"]): r for r in e["table"] if r["kind"] == "value"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, metric, title in zip(axes, ("bal_acc", "mel"), ("balanced accuracy", "melanoma recall")):
        for i, g in enumerate(groups):
            r_int = next(r for r in e["internal"][g] if r["metric"] == metric)
            r_ext = ext[(g, metric)]
            for dx, r, color, mfc, label in ((-0.12, r_int, BLUE, BLUE, "own HAM10000 test split"),
                                             (0.12, r_ext, ORANGE, "white", "ISIC 2018 test set (external)")):
                ax.errorbar(i + dx, r["point"], yerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]], fmt="o", color=color,
                            mfc=mfc, ms=6, mew=1.5, capsize=3, lw=1.2, label=label if i == 0 else None)
        ax.set_xticks(range(len(groups)), [labels[g] for g in groups])
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    fig.savefig(out / "external.png")
    plt.close(fig)


if __name__ == "__main__":
    paths = Paths()
    out = paths.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for make in (reliability, risk_coverage, frontier, decision_curve, confusion, coverage, external):
        make(paths, out)
        print("wrote", make.__name__)
