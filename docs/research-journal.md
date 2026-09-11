# Research journal

What has been tried on BEETLE, what it showed, and what remains open. Add an entry when an experiment concludes, whether or not it is submitted. Every number below cites the file it comes from.

## Verdict rule

Comparisons pair folds on identical splits and use **test-fold** scores: model k, selected on fold k+1, scored on fold k with `python -m beetle score-test`. An experiment shows a **signal** against a comparator when its paired mean test-fold delta is at least +0.002 **and** it improves at least 4 of 5 folds. Anything less is **no detectable effect**. The patient-bootstrap 95% interval of the pooled macro Dice difference (527 patients, 10,000 draws) is reported beside the verdict and does not decide it. `python -m beetle.comparison --split test` produces both.

Changed 2026-09-11: verdicts were first decided on tune-fold scores, which checkpoint selection inflates ([#7](https://github.com/clemsgrs/beetle-solution/issues/7)). Tune scores stay as a secondary column.

## Current beliefs

- **Decoder capacity is closed.** On untouched test folds both heavier decoders score below Attempt 01's lightweight decoder, and checkpoint selection inflated their tune scores more: tune-to-test drops of −0.010, −0.018, and −0.025 for Attempts 01–03. New experiments keep Attempt 01's decoder as their baseline; retest capacity only with a finer-grained encoder.
- **Tune-fold scores mislead; test-fold scores resolve small differences.** Selection reversed the ranking of Attempts 01–03, and on test folds a −0.006 paired difference already has a bootstrap interval that excludes zero.
- **The biggest measured weakness is external non-invasive epithelium**: 0.893 on development test folds versus 0.748 on the leaderboard (finding 3).

## Experiments

Test and tune Dice are the equal-weight mean ± sample SD of five fold dataset-global four-class macro Dice scores. Deltas and bootstrap intervals are test-fold and paired against Attempt 01. Source: `provenance/evaluations/test-folds/report.json`.

| Experiment | Axis | Change vs Attempt 01 | Test Dice | Tune Dice | Paired Δ, folds improved | Bootstrap Δ (95% CI) | Verdict |
|---|---|---|---|---|---|---|---|
| Attempt 01 | baseline | Frozen Virchow2, `lightweight_conv` (2 bilinear blocks, 1.51M params), uniform ROI sampling, no augmentation | 0.8758 ± 0.0396 | 0.8861 ± 0.0285 | — | — | Submitted; leaderboard overall Dice 0.9063 |
| Class-conditioned sampler | sampling | Equal class request ratios | not scored | 0.8822 ± 0.0260 | tune only: −0.0039, 1/5 | not recorded | No detectable effect on tune (worse) |
| Attempt 02 | decoder capacity | 4 bilinear blocks (2.69M params), 592×592 logits | 0.8705 ± 0.0398 | 0.8880 ± 0.0287 | −0.0053, 0/5 | −0.0063 [−0.0125, −0.0001] | No detectable effect (worse) |
| Attempt 03 | decoder capacity | `heavy_conv`: pyramid pooling, 2 learned transposed-conv blocks (5.25M params) | 0.8642 ± 0.0520 | 0.8894 ± 0.0286 | −0.0116, 1/5 | −0.0123 [−0.0206, −0.0051] | No detectable effect (worse); vs A02 −0.0063, 2/5 |

- **Class-conditioned sampler** (development arm before Attempt 01, not re-scored on test folds): raised necrosis to 19.9% of annotated pixels in drawn ROIs (3.0% under uniform draws). Evidence: `git show c68e6c1^:provenance/attempts/uniform/arm_selection.json`; exposure audits under `/maindisk/clement/soma-beetle-campaign-20260827/data/beetle/handoff/sampler_audits/`.
- **Attempt 02**: [#2](https://github.com/clemsgrs/beetle-solution/issues/2), `provenance/attempts/attempt-02/`.
- **Attempt 03**: [#5](https://github.com/clemsgrs/beetle-solution/issues/5), `provenance/attempts/attempt-03/`. Its tune-fold signal (+0.0033 vs Attempt 01 on 4/5 folds) did not survive test folds. The transposed blocks left a sub-visible boundary bias: predicted label transitions favour one phase of a 7-px period, with a max/min share ratio of 1.28 against 1.03 for Attempt 02's bilinear decoder, and no visible checkerboard. Prefer resize-convolution if learned upsampling returns.

## Findings

These bound every comparison above.

1. **Checkpoint selection inflates tune-fold scores.** Curation assigns test = fold k, tune = fold k+1, train = the other three, and with `holdout_test: true` Soma drops the test fold, so training reports the fold that selected each checkpoint. Attempts 01–03 lose 0.010–0.025 from tune to test, and bigger decoders lose more. Sources: `beetle/curate.py`, `configs/base.yaml`, `provenance/evaluations/test-folds/report.json`.
2. **Leaderboard overall Dice is a different aggregation.** Attempt 01's four external class Dice average 0.8233, while its overall Dice is 0.9063. The official formula is unpublished, so development class-macro Dice and leaderboard overall Dice are not comparable. Source: `provenance/attempts/attempt-01/leaderboard.json`.
3. **The external drop concentrates in non-invasive epithelium.** Attempt 01 development test-fold pooled → external class Dice: other 0.9762 → 0.9518, non-invasive 0.8932 → 0.7476, invasive 0.8561 → 0.8392, necrosis 0.7831 → 0.7546. External centres overall: Biopticka 0.9061, SCDC 0.8951, UW 0.9185. ILC invasive 0.8432, NST invasive 0.8377. Sources: `provenance/evaluations/test-folds/report.json`, `provenance/attempts/attempt-01/leaderboard.json`.
4. **One source dominates training exposure.** RUMC contributes 92,138 of 124,697 ROIs (73.9%); SCH 11.2%, TCGA 7.7%, NKI 6.8%, JB 0.4%. Attempt 01 test-fold macro Dice by source: NKI 0.915, SCH 0.896, RUMC 0.864, JB 0.783, TCGA 0.775 (TCGA non-invasive 0.496). Per-source scores are descriptive; source confounds case mix. Sources: `handoff/beetle-attempt-01-paper/cross_validation/confusion_matrix_per_roi.csv`, `provenance/evaluations/test-folds/report.json`.
5. **Errors concentrate where classes meet.** Attempt 01 test-fold pixel accuracy is 0.964 on single-class ROIs and 0.895, 0.837, 0.723 on two-, three-, and four-class ROIs (the last group has only 44 ROIs). This is not yet a distance-to-boundary measurement. Source: `handoff/beetle-attempt-01-paper/cross_validation/confusion_matrix_per_roi.csv`, which holds test-fold scores since 2026-09-11.
6. **Training sees fixed crops and no augmentation.** ROI tiles are cut once (512 px, no overlap, no jitter) and cached as features; only the draw order changes between epochs. Every augmentation setting is 0, and Soma rejects augmentation on cached features. Sources: `configs/base.yaml`, `provenance/attempts/attempt-03/config.yaml`.
7. **Models overfit early and unevenly.** Selected epochs range from 2 to 24 across folds and attempts, and tune loss ends above its minimum in all 15 fold runs of Attempts 01–03. Selecting on that same fold then overstates the result (finding 1). Sources: each attempt's fold `training_history.json`.

## Open directions

Unranked. New experiments keep Attempt 01's decoder as their baseline and are judged on test folds. Finding 2 still cuts across all of them: until the leaderboard formula is confirmed, development gains may not carry over one-to-one. Final submission members could train on 4/5 of patients; how they select a checkpoint is decided at the first submission run.

- **Encoder.** Known: all attempts decode the same frozen Virchow2 final-layer grid (14-px tokens at 0.5 µm/px, 224-px windows). Test: change the representation with Attempt 01's decoder held fixed, by adapting Virchow2 or using a finer-stride encoder. Both need live encoding or a new cache.
- **Augmentation.** Known: none is used (finding 6), and the largest external loss is non-invasive epithelium at unseen centres (finding 3). Test: stain and geometric augmentation applied before the encoder, via an augmented feature cache or live encoding.
- **Sampling.** Known: RUMC dominates exposure (finding 4), equal class balancing lost Dice on tune folds, and mixed-class ROIs are hardest (finding 5). Test: a sampler that tempers source exposure or mixes in interface ROIs while most draws stay uniform, logging realized exposure.
- **Loss.** Known: unweighted cross-entropy plus soft Dice; necrosis is the weakest development class (0.78 on test folds). Test: one mild change at a time, once the leaderboard aggregation is known.
- **Context-conditioned prediction.** Known: errors rise with the number of classes in an ROI (finding 5), and the model sees one 512-px ROI through 224-px encoder windows. Test: condition predictions on lower-magnification context around each ROI, after measuring interior versus boundary errors.
