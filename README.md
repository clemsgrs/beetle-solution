# Virchow2-based BEETLE solution

This repository holds our submissions to the [BEETLE challenge](https://beetle.grand-challenge.org/). The method is simple: freeze [Virchow2](https://huggingface.co/paige-ai/Virchow2), extract dense 2D feature grids, and train a lightweight convolutional decoder to predict the four segmentation classes. All general machinery lives in [Soma](https://github.com/clemsgrs/soma). This repository only holds BEETLE-specific glue and configuration.

One attempt is one config overlay in `configs/attempts/`. See [METHOD.md](METHOD.md) for the algorithmic specification of the submitted attempt.

## Install

The reference runtime is Python 3.11, PyTorch 2.7.1+cu128, and CUDA 12.8. Soma is pinned to an exact commit in `pyproject.toml`.

```bash
uv sync
uv run pytest -q
```

Virchow2 is gated. Request access on its model page. Then download the exact revision named in `configs/base.yaml`.

## Run

The pipeline has four commands. Each command is thin: Soma does the work.

```bash
# 1. Build the development slide manifest from the raw BEETLE data.
python -m beetle curate --beetle-root /path/to/beetle

# 2. Fill the dense feature cache. This step does not train decoders.
python -m beetle extract --work-dir /path/to/work

# 3. Train the five fold decoders for one attempt.
python -m beetle train --attempt configs/attempts/uniform.yaml

# 4. Predict the External ROIs, validate the contract, and write the ZIP.
python -m beetle infer \
  --run-dir /path/to/run \
  --roi-dir /path/to/External/ROIs \
  --roi-sidecar /path/to/roi_sidecar.json \
  --output-dir /path/to/predictions \
  --zip submission.zip \
  --attempt uniform
```

`train` and `infer` write a record of the attempt to `provenance/attempts/<name>/`. A recording failure does not stop a run.

## Attempts

| Attempt | Config | Dev Dice | Leaderboard | Tag | Notes |
|---|---|---|---|---|---|
| uniform | `configs/attempts/uniform.yaml` | 0.8861 | rank 16 | — | Submitted. Uniform ROI sampling. |
| class_conditioned | — | 0.8822 | — | — | Development arm, not submitted. Evidence in `provenance/attempts/uniform/arm_selection.json`. |

Dev Dice is the mean fold-macro dataset-global mean class Dice over the five development folds. For each submitted attempt: tag the commit (`attempt-NN`), attach the five decoder checkpoints to a GitHub release, and add a row here.

Evidence for each attempt is in `provenance/attempts/<name>/`.

## Scope and licensing

Code is Apache-2.0. See `THIRD_PARTY_NOTICES.md`. Obtain BEETLE and Virchow2 under their own terms. This repository is for research only, not for clinical use.
