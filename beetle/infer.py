"""Run the five-fold ensemble on the External ROIs, validate, and write submission.zip.

Averages the five fold softmax tensors over Hann-blended 512-pixel sliding tiles,
maps class indices to submission labels 1-4, checks every PNG against the BEETLE
contract, and bundles the deterministic flat ZIP.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from beetle.contract import (
    MODEL_INDEX_TO_SUBMISSION_LABEL,
    NUM_CLASSES,
    NUM_FOLDS,
    load_roi_sidecar,
    validate_roi_inputs,
    validate_submission_pngs,
    write_flat_submission_zip,
)
from beetle.record import record_inference


def load_fold_predictor(run_dir: Path):
    """Load the run's recipe, the frozen encoder, and all five fold decoders."""
    from soma.config import load_config
    from soma.dense.predict import (
        SlidingWindowSegmentationPredictor,
        build_live_segmentation_models,
    )
    from soma.pipeline import Pipeline

    config = load_config(run_dir / "config.yaml")
    if config.decoder is None or config.task is None or config.encoder is None:
        raise ValueError("Inference requires encoder, decoder, and task config")
    checkpoints = tuple(
        run_dir / f"fold_{fold}" / "best_model.pt" for fold in range(NUM_FOLDS)
    )
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fold checkpoints: {missing}")
    source = Pipeline(config)._build_live_segmentation_source()
    models = build_live_segmentation_models(
        source,
        decoder_name=config.decoder.name,
        decoder_params=config.decoder.params,
        num_classes=int(config.task.params["num_classes"]),
        ckpt_paths=checkpoints,
        normalization=config.normalization,
        projection=config.projection,
        encoder_identity=config.encoder.name,
    )
    return SlidingWindowSegmentationPredictor.from_source(source, models)


def run_inference(
    *,
    run_dir: str | Path,
    roi_dir: str | Path,
    roi_sidecar: str | Path,
    output_dir: str | Path,
    zip_path: str | Path,
    attempt: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    roi_dir = Path(roi_dir)
    output_dir = Path(output_dir)
    records = load_roi_sidecar(roi_sidecar)
    validate_roi_inputs(roi_dir, records)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Submission output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_fold_predictor(run_dir)
    for record in records:
        result = predictor.predict_image(
            roi_dir / record.roi_filename,
            native_spacing_um=record.native_spacing_um,
            allow_upsample=False,
            return_probs=False,
        )
        expected_shape = (record.height, record.width)
        if result.labels.shape != expected_shape:
            raise ValueError(
                f"Prediction {record.roi_filename!r} shape {result.labels.shape} "
                f"does not match input {expected_shape}"
            )
        model_indices = np.asarray(result.labels)
        if np.any(model_indices < 0) or np.any(model_indices >= NUM_CLASSES):
            raise ValueError(
                f"Prediction {record.roi_filename!r} has a class index outside "
                "the four-class vocabulary"
            )
        labels = MODEL_INDEX_TO_SUBMISSION_LABEL[model_indices]
        Image.fromarray(labels, mode="L").save(output_dir / record.roi_filename)

    paths = validate_submission_pngs(output_dir, records)
    archive = write_flat_submission_zip(paths, zip_path)
    print(f"Validated {len(paths)} predictions; wrote {archive}")
    record_inference(attempt or run_dir.name, run_dir, archive, len(paths))
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beetle infer", description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="trained run directory (config.yaml + fold_*/best_model.pt)",
    )
    parser.add_argument("--roi-dir", type=Path, required=True)
    parser.add_argument("--roi-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument(
        "--attempt", default=None, help="attempt name for the provenance record"
    )
    args = parser.parse_args(argv)
    run_inference(
        run_dir=args.run_dir,
        roi_dir=args.roi_dir,
        roi_sidecar=args.roi_sidecar,
        output_dir=args.output_dir,
        zip_path=args.zip,
        attempt=args.attempt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
