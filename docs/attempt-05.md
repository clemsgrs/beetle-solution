# Attempt 05: frozen GenBio-PathFM

GenBio-PathFM was among the best encoders on the user's detection benchmark; Attempt 05
asks whether that transfers to BEETLE dense segmentation. It is independent of Attempt 04.

Attempt 05 keeps Attempt 01's lightweight two-block decoder, ROI population, patient
folds, uniform sampling, losses, optimizer, schedule, effective batch 64, and no
augmentation. The encoder is `genbio-ai/genbio-pathfm` at
`6eab2a4ea6fbaee16e5f193187952042ebe5d3ec`, run at its native settings through slide2vec
5.8.2: FP32, 0.5 µm/px, 224-px windows with overlap 0.5, patch 16. The cache stays FP16.

Two things change with the representation, and both are accepted as part of it:

- The backbone is single-channel. Each RGB channel is encoded separately and the three
  1,536-d patch tokens are concatenated, so the decoder input projection grows from 1,280
  to 4,608 channels and the decoder from 1.51M to 2,362,628 parameters. No projection is
  added to equalise it.
- Patch 16 gives a 32×32 feature grid per 512-px ROI instead of 37×37.

slide2vec loads the repository's `main` ref without a revision. `smoke` and `extract`
therefore fail if `huggingface/hub/models--genbio-ai--genbio-pathfm/refs/main` differs from
the pinned revision. The model's remote code was read at that revision: it imports only
`torch` and `transformers`, and its RoPE coordinate jitter applies in training mode only.

The run directory is `/maindisk/clement/beetle-attempt-05-20260919` (on the 27 TB
`/maindisk/clement` pool; the full cache is about 1.2 TB). Its layout matches Attempt 04's
(`docs/attempt-04.md`), with `checkout/` holding the running copy of `beetle/` and
`configs/`. Any code fix goes into both `checkout/` and this branch.

## Phases

`python -m beetle.attempt_05 <phase>`, run from `checkout/` with `venv/bin/python`,
`PYTHONPATH=checkout`, `HF_HOME=<run dir>/huggingface`, and `HF_HUB_OFFLINE` unset
(slide2vec calls the hub on load).

1. `prepare`: verifies the original slide/split hashes, replays the exact 124,697 ROI
   coordinates into the new cache, checks the replayed ROI files byte-for-byte, and records
   the weight hash in `evidence/preparation.json`.
2. `smoke`: encodes one ROI, checks the `(4608, 32, 32)` FP16 grid, and runs one batch-64
   decoder step. Result in `evidence/smoke.json`.
3. `extract`: fills the cache on all four GPUs (`execution.num_gpus: 4`), validates
   coverage, feature dimension and grid, and writes `evidence/cache.json`. Extraction
   resumes per slide, so relaunch the same command after a failure. Logs are in
   `jobs/extract-NN/run.log`.
4. `train --fold K`: trains one fold on the GPU selected by `CUDA_VISIBLE_DEVICES`. Every
   launch pins Soma run `attempt-05-genbio-pathfm-20260919` and passes `run.folds: [K]`, so
   launches given different folds can run side by side, one per GPU. A fold with
   `metrics.json` is skipped; an interrupted fold restarts from epoch 0. Two launches given
   the same fold would both train it. Logs are in `jobs/train-foldK-NN/run.log`.
5. `finish`: run once all five folds have `metrics.json`. It validates the cache, lets
   Soma write the run summary (training any fold still pending), records the attempt,
   checks the training ROI files against the baseline replay, scores the test folds, and
   compares against Attempt 01 with Attempt 04 alongside (`provenance/comparison.json`).

`run.folds` comes from Soma issue clemsgrs/soma#478. Attempts 01–04 ran on Soma 1.11.2 at
`4a3d6c8`; the run venv now holds the backport of that one change to the same commit
(`eb5470b`, branch `backport/run-folds-1.11.2`). Folds seed independently, so splitting
them across launches does not change any fold's result. `evidence/preparation.json` still
names `4a3d6c8`: it was written before the backport was installed.

## GPU guard waiver

No phase runs under `gpu_guard.py`. The user waived CLAUDE.md's shared-GPU rule for
Attempt 05 because the node was reserved for them for a few days: extraction on
2026-09-19, training on 2026-09-20. The waiver does not cover any later attempt.

## Verdict

All five folds, paired test-fold scores against Attempt 01 under the research journal's
rule, with Attempt 04 as a secondary column. Add the result and verdict to the research
journal once the experiment concludes.
