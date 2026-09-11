# Research journal

What has been tried on BEETLE, what it showed, and what remains open. Add an entry when an experiment concludes, whether or not it is submitted. Every number below cites the file it comes from.

## Verdict rule

Comparisons pair folds on identical splits. An experiment shows a **signal** against a comparator when its paired mean fold delta is at least +0.002 **and** it improves at least 4 of 5 folds. Anything less is **no detectable effect**. The patient-bootstrap 95% interval of the pooled macro Dice difference (527 patients, 10,000 draws) is reported beside the verdict and does not decide it. `python -m beetle.comparison` produces both.

## Current beliefs

- **Decoder capacity is a low-yield axis.** Two larger decoders on the same frozen Virchow2 37×37 token grid moved development Dice by at most +0.0033, and both bootstrap intervals include zero. Larger gains need new information (representation, data exposure, augmentation, context), not more decoding of the same tokens.
- **Small effects are below the resolution of the current protocol.** Fold-to-fold SD is ~0.029, and every paired bootstrap interval so far spans zero (findings 1–2).
- **The biggest measured weakness is external non-invasive epithelium**: 0.907 internal versus 0.748 on the leaderboard (finding 3).

## Experiments

Dev Dice is the equal-weight mean ± sample SD of five fold dataset-global four-class macro Dice scores. Deltas are paired against Attempt 01. Fold deltas and bootstrap intervals for Attempts 01–03 come from `comparison_report.json` in the Attempt 03 release evidence.

| Experiment | Axis | Change vs Attempt 01 | Dev Dice | Paired Δ, folds improved | Bootstrap Δ (95% CI) | Verdict |
|---|---|---|---|---|---|---|
| Attempt 01 | baseline | Frozen Virchow2, `lightweight_conv` (2 bilinear blocks, 1.51M params), uniform ROI sampling, no augmentation | 0.8861 ± 0.0285 | — | — | Submitted; leaderboard overall Dice 0.9063 |
| Class-conditioned sampler | sampling | Equal class request ratios | 0.8822 ± 0.0260 | −0.0039, 1/5 | not recorded | No detectable effect (negative) |
| Attempt 02 | decoder capacity | 4 bilinear blocks (2.69M params), 592×592 logits | 0.8880 ± 0.0287 | +0.0019, 4/5 | +0.0026 [−0.0020, +0.0071] | No detectable effect |
| Attempt 03 | decoder capacity | `heavy_conv`: pyramid pooling, 2 learned transposed-conv blocks (5.25M params) | 0.8894 ± 0.0286 | +0.0033, 4/5 | +0.0046 [−0.0022, +0.0116] | Small signal vs A01; vs A02 +0.0014, 3/5, no detectable effect |

- **Class-conditioned sampler** (development arm before Attempt 01): raised necrosis to 19.9% of annotated pixels in drawn ROIs (3.0% under uniform draws). Evidence: `git show c68e6c1^:provenance/attempts/uniform/arm_selection.json`; exposure audits under `/maindisk/clement/soma-beetle-campaign-20260827/data/beetle/handoff/sampler_audits/`.
- **Attempt 02**: [#2](https://github.com/clemsgrs/beetle-solution/issues/2), `provenance/attempts/attempt-02/`.
- **Attempt 03**: [#5](https://github.com/clemsgrs/beetle-solution/issues/5), `provenance/attempts/attempt-03/`. Largest pooled class gain is non-invasive epithelium (0.9133 vs 0.9067). The transposed blocks left a sub-visible boundary bias: predicted label transitions favour one phase of a 7-px period (two logit cells) with a max/min share ratio of 1.28, against 1.03 for Attempt 02's bilinear decoder. Masks show no visible checkerboard. Prefer resize-convolution if learned upsampling returns.

## Findings

These bound every comparison above.

1. **Headline scores are tune-fold scores.** Curation assigns test = fold k, tune = fold k+1, train = the other three. With `holdout_test: true`, Soma drops the test fold, so each model trains on 3/5 of patients and is scored on the fold that selected its checkpoint. Sources: `beetle/curate.py`, `configs/base.yaml`.
2. **Leaderboard overall Dice is a different aggregation.** Attempt 01's four external class Dice average 0.8233, while its overall Dice is 0.9063. The official formula is unpublished, so development class-macro Dice and leaderboard overall Dice are not comparable. Source: `provenance/attempts/attempt-01/leaderboard.json`.
3. **The external drop concentrates in non-invasive epithelium.** Attempt 01 internal pooled → external class Dice: other 0.9784 → 0.9518, non-invasive 0.9067 → 0.7476, invasive 0.8768 → 0.8392, necrosis 0.7825 → 0.7546. External centres overall: Biopticka 0.9061, SCDC 0.8951, UW 0.9185. ILC invasive 0.8432, NST invasive 0.8377. Sources: `artifacts/releases/attempt-01/development/cv_results.json`, `provenance/attempts/attempt-01/leaderboard.json`.
4. **One source dominates training exposure.** RUMC contributes 92,138 of 124,697 ROIs (73.9%); SCH 11.2%, TCGA 7.7%, NKI 6.8%, JB 0.4%. Attempt 01 macro Dice by source: NKI 0.917, SCH 0.901, RUMC 0.876, JB 0.778, TCGA 0.775 (TCGA non-invasive 0.457). Source: `handoff/beetle-attempt-01-paper/cross_validation/confusion_matrix_per_roi.csv`.
5. **Errors concentrate where classes meet.** Attempt 01 pixel accuracy is 0.969 on single-class ROIs and 0.905, 0.835, 0.707 on two-, three-, and four-class ROIs. This is not yet a distance-to-boundary measurement. Source: as finding 4.
6. **Training sees fixed crops and no augmentation.** ROI tiles are cut once (512 px, no overlap, no jitter) and cached as features; only the draw order changes between epochs. Every augmentation setting is 0, and Soma rejects augmentation on cached features. Sources: `configs/base.yaml`, `provenance/attempts/attempt-03/config.yaml`.
7. **Models overfit early and unevenly.** Selected epochs range from 2 to 24 across folds and attempts, and tune loss ends above its minimum in all 15 fold runs of Attempts 01–03. Sources: each attempt's fold `training_history.json`.

## Open directions

Unranked. Findings 1–2 cut across all of them: while scores are tune-selected and the leaderboard aggregation is unconfirmed, an effect of +0.003 is indistinguishable from noise. Scoring the unused test folds out of fold, training on 4/5 of patients, and confirming the leaderboard formula would sharpen every axis below.

- **Encoder.** Known: all attempts decode the same frozen Virchow2 final-layer grid (14-px tokens at 0.5 µm/px, 224-px windows). Test: change the representation with the decoder held fixed, by adapting Virchow2 or using a finer-stride encoder. Both need live encoding or a new cache.
- **Augmentation.** Known: none is used (finding 6), and the largest external loss is non-invasive epithelium at unseen centres (finding 3). Test: stain and geometric augmentation applied before the encoder, via an augmented feature cache or live encoding.
- **Sampling.** Known: RUMC dominates exposure (finding 4), equal class balancing lost Dice, and mixed-class ROIs are hardest (finding 5). Test: a sampler that tempers source exposure or mixes in interface ROIs while most draws stay uniform, logging realized exposure.
- **Loss.** Known: unweighted cross-entropy plus soft Dice; necrosis is the weakest internal class (0.78–0.79). Test: one mild change at a time, once the leaderboard aggregation is known.
- **Context-conditioned prediction.** Known: errors rise with the number of classes in an ROI (finding 5), and the model sees one 512-px ROI through 224-px encoder windows. Test: condition predictions on lower-magnification context around each ROI, after measuring interior versus boundary errors.
