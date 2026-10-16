# Segmentation on new slides

Use `slide-predict` to run the released Virchow2-based Attempt 01 model on your own
breast slides. It ensembles the five trained decoders and predicts inside a
supplied tissue or ROI mask. No retraining or BEETLE dataset is needed.

## Install

Use a Linux machine with an NVIDIA GPU for the reference runtime: Python 3.11,
PyTorch 2.7.1, and CUDA 12.8. With Git, curl, and
[uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run:

```bash
git clone https://github.com/clemsgrs/beetle-solution.git
cd beetle-solution
uv sync --python 3.11
```

Run all remaining commands from this directory. Use `uv run` to keep inference in
the repository's environment; newer hs2p and slide2vec versions have changed APIs
that this code uses.

## Download the decoder weights

The [Attempt 01 release](https://github.com/clemsgrs/beetle-solution/releases/tag/attempt-01)
contains the resolved configuration and five decoder checkpoints:

```bash
mkdir -p artifacts/releases artifacts/checkpoints/attempt-01

curl -fL \
  https://github.com/clemsgrs/beetle-solution/releases/download/attempt-01/beetle-attempt-01-weights.zip \
  -o artifacts/releases/beetle-attempt-01-weights.zip

uv run python -m zipfile -e \
  artifacts/releases/beetle-attempt-01-weights.zip \
  artifacts/checkpoints/attempt-01
```

The run directory should contain:

```text
artifacts/checkpoints/attempt-01/
├── config.yaml
├── fold_0/best_model.pt
├── fold_1/best_model.pt
├── fold_2/best_model.pt
├── fold_3/best_model.pt
└── fold_4/best_model.pt
```

## Virchow2 access

Obtain access and accept the terms on the
[Virchow2 model page](https://huggingface.co/paige-ai/Virchow2), then authenticate
on the inference machine with a token from that account:

```bash
uv run python -c "from huggingface_hub import login; login()"
```

The loader downloads the encoder into the Hugging Face cache on first use. Those
weights are separate from the decoder release ZIP.

For exact reproduction, check the encoder against the
[recorded revision and weight checksum](../provenance/attempts/attempt-01/executions.json).
The current loader requests Hugging Face's default revision and does not enforce
the recorded revision. Downloading a snapshot to an arbitrary local directory
does not redirect the loader to it.

## Prepare the slide masks and CSV

Each slide needs a binary tissue or ROI mask, with nonzero pixels where inference
should run and zero elsewhere. Ground-truth class annotations are unnecessary.
The command requires an existing mask; it does not generate one from the slide.

Use an OpenSlide-readable tiled TIFF mask with correct pixel-spacing metadata.
It must share the slide's physical origin and cover its full extent; it may have
a different pixel spacing. A cropped ROI mask with an offset is not supported.
For full-tissue inference, mark all tissue. For selected ROIs, mark those regions
within the same slide-aligned canvas and leave the rest zero.

For pyramidal masks, foreground must remain visible in the overview levels used
to select chunks (up to 16× downsampling). Small or thin ROIs that disappear there
can be skipped. Use a single-resolution tiled mask for very small ROIs, or build
pyramid levels that preserve their foreground.

The output inherits the mask's dimensions and spacing. A mask at 0.5 µm/px gives
an output at 0.5 µm/px; a coarse thumbnail mask gives a coarse output. This controls
the saved grid, while image reading follows the run's preprocessing configuration.

Create `slides.csv` in the repository directory, replacing the example paths with
absolute paths to your files:

```csv
sample_id,image_path,label_mask_path
slide01,/path/to/slide01.svs,/path/to/slide01_tissue_mask.tif
slide02,/path/to/slide02.svs,/path/to/slide02_tissue_mask.tif
```

Use a unique `sample_id` for each slide; it becomes the output filename. If omitted,
the slide filename without its extension is used. Leave out `validation_fold` for
new slides so all five decoders are ensembled on every slide. That column is for
the [out-of-fold development export](../README.md#whole-slide-masks).

## Run inference

Replace the scratch path with a directory on a local disk with enough free space:

```bash
uv run python -m beetle slide-predict \
  --run-dir artifacts/checkpoints/attempt-01 \
  --slides-csv slides.csv \
  --output-dir artifacts/predictions/new-slides \
  --scratch-dir /path/to/local/scratch/beetle
```

The output and scratch directories are created if needed. The temporary label
canvas uses one byte per mask pixel: a 50,000 × 50,000 mask needs about 2.5 GB of
scratch space, in addition to the final output. Scratch holds one slide's canvas
at a time; the canvas is disk-backed and filled in chunks. Empty chunks are skipped
during prediction, but the canvas still spans the full mask. Keep scratch off
network shares.

## Outputs

With the example above, results appear under the repository directory:

```text
artifacts/predictions/new-slides/
├── slide01.tif
├── slide02.tif
└── summary.csv
```

Each TIFF contains a single-channel `uint8` label mask:

| Value | Meaning |
|---|---|
| 0 | Outside the inference mask; not predicted |
| 1 | Other tissue |
| 2 | Non-invasive epithelium |
| 3 | Invasive epithelium |
| 4 | Necrosis |

The TIFF keeps the supplied mask's full dimensions, even if only a few ROIs were
selected. Predictions occupy those regions and everything else stays zero; no
separate cropped ROI files are written.

Outputs use 512 × 512 tiles, LZW compression (Deflate if imagecodecs is unavailable),
and progressively downsampled pyramid levels with pixel-spacing metadata. Large
canvases use BigTIFF with the same `.tif` extension. For example, a 50,000 × 50,000
mask at 0.5 µm/px produces a BigTIFF with that base grid plus its pyramid. Final
file size depends on compression; the 2.5 GB scratch estimate is uncompressed.

`summary.csv` records each slide's output path, geometry, read spacing, masked
pixel count (`annotated_pixels`), and per-class predicted pixel counts. The
`annotated_*` columns count input-mask values; with a binary tissue mask they do
not represent ground-truth tissue classes.

Rerunning the command skips existing output TIFFs, allowing an interrupted export
to resume. Use a fresh output directory when changing the masks or model;
`--overwrite` replaces TIFFs but appends new rows to the existing summary.
