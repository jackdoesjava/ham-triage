# ham-triage

![frontier](results/figures/frontier.png)

A skin-lesion classifier that outputs an argmax is not a clinical tool, and on
HAM10000 its headline accuracy is not even a measurement of the classifier. This
repo treats the interesting object as a triage policy: posteriors calibrated where
the decision is made, prediction sets with a finite-sample coverage guarantee for
the class that matters, and a deferral rule derived from an explicit cost model in
which the human reader has a stated sensitivity and specificity. The backbone is a
stock EfficientNet-B0 from timm and is not the point. Every split is grouped by
`lesion_id`, every number carries a lesion-clustered bootstrap interval, every
artifact regenerates byte-for-byte from committed logits (there is a test for
that), and the leakage result is checked on an external test set.

## Three numbers

1. **Leakage.** HAM10000 has several images of the same lesion, and a random
   image-level split puts a sibling of about 40% of the test images (70% of the
   melanomas) on the training side. Trained on the same images with only that
   difference and scored on the same test images, the leaky model is ahead by
   **0.104 [0.086, 0.123] balanced accuracy and 0.148 [0.117, 0.177] melanoma
   recall**, pooled over three test-set draws and three training seeds each
   (per draw +0.081 to +0.121); on images without a training sibling the gap is
   -0.007 [-0.027, +0.012] balanced accuracy and -0.027 [-0.069, +0.012] melanoma
   recall. On the ISIC 2018 challenge test set, which shares no lesion with
   HAM10000, the same leaky and clean models are indistinguishable (balanced
   accuracy **-0.017 [-0.043, +0.010]**): the model did not get better, its test
   set got easier.
2. **Deferral.** With a missed melanoma costing 100 referrals and a reader at 87%
   sensitivity and 71% specificity, the expected-cost rule at 20% deferral
   (threshold chosen on the calibration split) realises **0.65 [0.59, 0.73] per
   image and misses 1.1% [0.6%, 1.6%] of melanomas**, against 0.68 for never
   deferring, 1.01 for random deferral and 1.56 with 8.8% missed for max-softmax
   deferral, which loses to random whenever the reader is imperfect because it
   concentrates the melanomas in the reader's queue. The same rule on uncalibrated
   posteriors realises 1.41 while expecting 0.38. On the external test set the
   calibration-split threshold misses 1.7% [0.8%, 2.6%] of melanomas at 0.78 per
   image.
3. **Guarantee.** A melanoma-conditional conformal threshold on p(mel) at
   alpha = 0.1 realises **0.882 [0.843, 0.920] sensitivity while flagging 14.8% of
   non-melanomas** (alpha = 0.05: 0.923 at 19.7%); over 200 lesion-grouped
   re-partitions its coverage is 0.904 +/- 0.042 against the analytic Beta(58, 6)
   law's 0.906 +/- 0.036, and on the external test set it realises 0.891 [0.852,
   0.928]. The marginal set at the same alpha covers 0.897 [0.889, 0.903] of
   lesions overall but nv at 0.91 and melanoma at 0.81, which is what marginal
   coverage is worth when two thirds of the data is one class.

## What else is in here

Temperature scaling brings the temperature to about 2.1 on every seed and NLL from
0.58 to 0.42. The second panel of `results/figures/reliability.png` is the
reliability of p(needs treatment) on log axes, where the discharge decision sits:
lesions the raw softmax scores at 0.1% need treatment 6% of the time, and after
scaling the lesions scored at 0.3% and 1.3% need it 0.2% and 2.6% of the time.
The class-imbalance comparison is made on calibrated posteriors, three seeds each:
subtracting the log training prior from the plain model's logits (Menon et al.
2021) lifts balanced accuracy by 0.037 [0.021, 0.054] and melanoma recall by
0.069 [0.048, 0.093] at zero training cost, while a retrained class-balanced loss
(Cui et al. 2019) gets 0.018 [-0.003, 0.039] and 0.084 [0.050, 0.120]. Both pay
in NLL after temperature scaling, +0.071 and +0.142 [0.059, 0.296] on a base of
0.467, because the posterior is now calibrated to a prior the test population
does not have; the decision layer can multiply in whatever prior it needs, the
loss does not have to.

Coverage is counted per lesion, because that is the unit the calibration set is
exchangeable in: calibration holds one image per lesion, the test set keeps every
image, and multi-image lesions are 45% of test images against 26% of calibration
ones and are far harder (accuracy 0.77 against 0.93 on single-image lesions). The
same LAC sets cover 0.885 of lesions and 0.857 of images; both numbers are in
`results/derived/conformal.json`. Per-class Mondrian sets for df and vasc rest on
9 and 11 calibration lesions and are near-degenerate by arithmetic (the quantile
index is the sample maximum below 20). The risk-coverage curve under 0-1 loss is
the uniform-cost slice of the frontier; max softmax has the best AURC (0.032) and
the seed-ensemble mutual information the worst, which is the expected
in-distribution null. The sensitivity grid in `results/derived/decision.json`
sweeps the melanoma miss cost over 10 to 300, the reader sensitivity over 0.8 to
1.0 and the specificity over 0.71 and 1.0: the ordering of deferral policies is
stable whenever the reader is imperfect. A prior-shift scenario with the treated
classes five times rarer and the decision curve
(`results/figures/decision_curve.png`) are in the same file.

`dx_type` records how each label was established, and across classes it is class
in disguise (every malignant label is histopathology), which is why the planned
experiment on down-weighting low-confidence labels was dropped. Within nv it is
not: histo-nv are nevi that looked suspicious enough to excise, follow_up-nv are
monitored nevi from one device. The triage layer treats them as the different
populations they are: excised nevi get p(mel) of 0.098 [0.083, 0.114], go to a
human 81% of the time and are flagged by the melanoma rule 31% of the time,
against 0.006, 15% and 1.8% for followed-up nevi, with argmax error rates of 11%
and 0.4% (`results/derived/strata.json`). Whether that is the model reading
morphology or reading the MoleMax device is not something HAM10000 can settle.

`results/figures/external.png` puts every model's own test split next to the ISIC
2018 test set. The full-data lesion-disjoint models score 0.689 [0.645, 0.736]
balanced accuracy at home and 0.654 [0.622, 0.687] abroad; the leaky models
score 0.771 at home and 0.635 abroad. Fifteen percent more training data is
worth +0.003 [-0.025, +0.031] externally, so the recipe, not the data, is the
ceiling here.

## Reproduce

```
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe -e .
python -m ham_triage.data          # kagglehub download, 256px uint8 cache, ~2 min
python -m ham_triage.splits        # image-level, paired audit (3 seeds) and lesion-level splits
python -m ham_triage.train --split lesion   --train-col train       --seed 0 --run-id full_s0
python -m ham_triage.train --split audit    --train-col clean_train --seed 0 --run-id clean_s0
python -m ham_triage.train --split audit    --train-col leaky_train --seed 0 --run-id leaky_s0
python -m ham_triage.train --split audit_s1 --train-col clean_train --seed 0 --run-id clean_a1_s0
python -m ham_triage.train --split audit    --train-col clean_train --loss cb --seed 0 --run-id cb_s0
python -m ham_triage.train --split image_level --seed 0 --run-id naive_s0
python -m ham_triage.external      # ISIC 2018 task 3 test set: download, cache, score every checkpoint
python -m ham_triage.analyse       # everything post hoc, results/runs to results/derived, about two minutes
python -m ham_triage.figures
pytest                             # includes regenerating every artifact and diffing it
```

Seeds 0 to 4 for the full-data models, 0 to 2 for everything else, and the
audit_s2 split alongside audit_s1: 28 runs in all. Each run is about 20 minutes on
a GTX 1050 Ti at 224px with fp16 autocast and checkpoints every epoch, so a killed
run resumes exactly where it stopped. The logits of every run are committed, so
`analyse`, `figures` and the regression test work without training or the image
cache. The torch build has to come from the cu126 index on Windows, which
`pyproject.toml` pins.

## Caveats

The audit measures what sibling images of the test lesions do to the score of the
same model on the same images, over three independent test draws; its clean
number is a size-matched model on 5914 images and is not the deployable one. The
deployable models reach 0.69 balanced accuracy, below most published HAM10000
numbers, most of which are leaky; the external test says more data would not
have fixed that, and nothing here was tuned.

The cost constants are invented. Only the ordering of policies across the
sensitivity grid is a claim; the cost values are not. The reader is modelled by
two fixed numbers from one reader study, not estimated here, because HAM10000
labels are the reference standard and carry no reader errors to learn from.
HAM10000 is an excision-enriched specialist collection with 11% melanoma by image,
so the operating points do not transfer to a screening population without the
prior shift, and the prior-shift scenario reuses the same test images with
importance weights rather than a real screening sample.

The external test set is the challenge organisers' held-out set from the same
sources plus four other centres; it ships no lesion ids, so its intervals are
image bootstraps and its disjointness from HAM10000 rests on the organisers'
statement. Temperature and conformal quantiles are fitted on the same
calibration split, a one-parameter double dip that is refit inside every
re-partition. GPU training is seeded but not bitwise reproducible. Bootstrap
intervals for ECE sit high relative to the point estimate because ECE is biased
upward on a resample; NLL and Brier are plain means and are the calibration
numbers to trust.
