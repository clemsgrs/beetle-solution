# BEETLE solutions

This repository holds my solutions to the [BEETLE challenge](https://beetle.grand-challenge.org/) (breast-cancer segmentation in whole-slide images). Each attempt is one config overlay in `configs/attempts/` plus one tagged release with the trained decoder weights. All general machinery lives in [soma](https://github.com/clemsgrs/soma). This repository only holds BEETLE-specific glue and configuration.

Attempt 01 uses a [Virchow2](https://huggingface.co/paige-ai/Virchow2) encoder with a lightweight segmentation decoder and uniform ROI sampling. Later attempts can use a different method. See [METHOD.md](METHOD.md) for its algorithmic specification.

## Attempts

| Attempt | Config | Dev Dice | Leaderboard overall Dice | Tag | Weights |
|---|---|---|---|---|---|
| attempt-01 | `configs/attempts/attempt-01.yaml` | 0.8861 | 0.9063 | `attempt-01` | [download](https://github.com/clemsgrs/beetle-solution/releases/download/attempt-01/beetle-attempt-01-weights.zip) |
| attempt-02 | `configs/attempts/attempt-02.yaml` | 0.8880 | — | `attempt-02` | [download](https://github.com/clemsgrs/beetle-solution/releases/download/attempt-02/beetle-attempt-02-weights.zip) |

Dev Dice is the mean dataset-global mean class Dice over the five development folds. The leaderboard score is the official `overall_dice` from the [challenge leaderboard](https://beetle.grand-challenge.org/evaluation/beetle/leaderboard/). Attempt 01's evidence is in `provenance/attempts/attempt-01/`.

The Attempt 01 release also provides the five-fold CV evidence and the submitted External prediction ZIP as separate assets. Attempt 02 is a development-only four-block decoder-depth ablation; its release provides five-fold evidence and does not replace the Attempt 01 External model.

## Install

The reference runtime is Python 3.11, PyTorch 2.7.1+cu128, and CUDA 12.8. soma is pinned to an exact commit in `pyproject.toml`.

```bash
uv sync
uv run pytest -q
```

Virchow2 is gated. Request access on its model page. Then download the exact revision named in `configs/base.yaml`.

The `curate`, `extract`, and `train` commands run from a repository checkout: they read `configs/` and write `provenance/`. `infer` only needs a run directory, so it also works from a plain package install.

## Inference

To reproduce a submission, you do not need to train. Download the attempt's release assets and arrange them as a run directory:

```
run/
  config.yaml            # resolved attempt config, from the release
  fold_0/best_model.pt   # one decoder checkpoint per fold
  ...
  fold_4/best_model.pt
```

Then predict the External ROIs, validate the submission contract, and write the ZIP:

```bash
python -m beetle infer \
  --run-dir run \
  --roi-dir /path/to/External/ROIs \
  --roi-sidecar /path/to/roi_sidecar.json \
  --output-dir /path/to/predictions \
  --zip submission.zip \
  --attempt attempt-01
```

`infer` averages the five fold softmax outputs over Hann-blended sliding tiles, checks every PNG against the BEETLE contract, and writes a deterministic flat ZIP.

## Training

To train an attempt from scratch, run the three remaining commands. Each command is thin: soma does the work.

```bash
# 1. Build the development slide manifest from the raw BEETLE data.
python -m beetle curate --beetle-root /path/to/beetle

# 2. Fill the dense feature cache. This step does not train decoders.
python -m beetle extract --work-dir /path/to/work

# 3. Train the five fold decoders for one attempt.
python -m beetle train --attempt configs/attempts/attempt-01.yaml
```

`train` and `infer` write a record of the attempt to `provenance/attempts/<name>/`. A recording failure does not stop a run.

For each submitted attempt: tag the commit (`attempt-NN`), attach the five decoder checkpoints and the resolved config to a GitHub release, and add a row to the attempts table.

## Scope and licensing

Code is Apache-2.0. See `THIRD_PARTY_NOTICES.md`. Obtain BEETLE and Virchow2 under their own terms. This repository is for research only, not for clinical use.
