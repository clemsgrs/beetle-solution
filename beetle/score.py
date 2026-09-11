"""Score each fold's selected checkpoint on its held-out ROIs from cached features.

Training scores every checkpoint on the tune fold that selected it and drops the
test fold (organizer fold k for model k). ``score-test`` loads each fold's
``best_model.pt``, predicts that fold's ROIs of the requested split from the cached
grids, and writes ``<output>/<attempt>/fold_k/confusion_evidence_<split>.json`` in
Soma's schema, which ``python -m beetle.comparison --split`` reads.

A fold is finished once its ``metrics_<split>.json`` exists, so rerunning resumes.
Every attempt runs in one process: a shared-GPU guard then sees one GPU client.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

FOLD_IDS = tuple(range(5))
SPLITS = ("tune", "test")
PATCH_SIZE = 14  # Virchow2 ViT-H/14, the encoder behind every cached grid here.


def read_split_sample_ids(roi_splits_csv: str | Path, *, split: str) -> dict[int, list[str]]:
    """ROI ids each model holds out as ``split``, keyed by fold, in file order."""
    by_fold: dict[int, list[str]] = {fold: [] for fold in FOLD_IDS}
    with Path(roi_splits_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split:
                by_fold.setdefault(int(row["fold"]), []).append(row["sample_id"])
    return by_fold


def load_scoring_context(roi_manifest: Path, feature_dir: Path):
    """Read the ROI manifest and index the cached grids once for every attempt."""
    from soma.dataset import SegmentationManifest
    from soma.dense import DenseFeatureStore

    manifest = SegmentationManifest(roi_manifest)
    stems = {}
    for record in manifest.samples.values():
        if record.region is None or record.slide_id is None:
            raise ValueError(
                f"ROI '{record.sample_id}' lacks the slide and region that address its grid"
            )
        x, y = record.region
        # slide2vec namespaces ROI grids per slide; address them from the manifest row.
        stems[record.sample_id] = f"{record.slide_id}/{int(x)}_{int(y)}"
    return manifest.samples, DenseFeatureStore(feature_dir, payload_stems=stems)


def score_fold(
    *,
    run_dir: Path,
    fold: int,
    split: str,
    records: list,
    feature_store,
    fold_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Predict one fold's held-out ROIs with its selected checkpoint and write evidence."""
    import torch
    from torch.utils.data import DataLoader

    from soma.config import load_config
    from soma.decoders.registry import build_decoder_for_grid
    from soma.dense.geometry import compute_dense_geometry

    # Private but pinned: Soma 1.11.2 has no evaluation-only entry point, and these are
    # the exact head, metric, and evidence paths its training run used.
    from soma.pipeline import (
        _build_segmentation_head,
        _evaluate_segmentation,
        _segmentation_class_vocabulary,
    )
    from soma.training.model import SegmentationModel
    from soma.training.segmentation_dataset import (
        SegmentationDataset,
        segmentation_collate_fn,
    )

    config = load_config(run_dir / "config.yaml")
    masks = config.preprocessing.masks
    geometry = compute_dense_geometry(
        target_size=int(config.preprocessing.requested_tile_size_px), patch_size=PATCH_SIZE
    )
    grid = feature_store.load(records[0].sample_id)
    if tuple(grid.shape[-2:]) != tuple(geometry.grid_shape):
        raise ValueError(
            f"Cached grid {tuple(grid.shape)} does not match geometry {geometry.grid_shape}"
        )
    head = _build_segmentation_head(
        task=config.task,
        evaluation=config.evaluation,
        preprocessing=config.preprocessing,
        masks=masks,
        geometry=geometry,
    )
    decoder = build_decoder_for_grid(
        config.decoder.name,
        config.decoder.params,
        geometry=geometry,
        input_dim=int(grid.shape[0]),
        num_classes=head.num_classes,
    )
    model = SegmentationModel(decoder=decoder, task_head=head)
    checkpoint = torch.load(
        run_dir / f"fold_{fold}" / "best_model.pt", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    loader = DataLoader(
        SegmentationDataset(records, feature_store, head.extract_targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        collate_fn=functools.partial(segmentation_collate_fn, target_dtypes=head.target_dtypes),
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        report = _evaluate_segmentation(
            model,
            loader,
            split,
            resolved_device,
            output_dir=fold_dir,
            save_segmentation_overlays=False,
            save_segmentation_probabilities=False,
            save_confusion_evidence=True,
            write_dense_artifacts=False,
            fold=fold,
            class_vocabulary=_segmentation_class_vocabulary(masks, head.num_classes),
        )
    return {name: _plain(value) for name, value in report.metrics.items()}


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _soma_version() -> str | None:
    try:
        return version("soma-pathology")
    except PackageNotFoundError:
        return None


def score_attempts(
    *,
    attempts: Sequence[tuple[str, Path]],
    roi_manifest: str | Path,
    roi_splits: str | Path,
    feature_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    folds: Sequence[int] = FOLD_IDS,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 8,
    context_builder: Callable[[Path, Path], tuple] | None = None,
    scorer: Callable[..., dict] | None = None,
) -> dict:
    """Score every requested fold of every attempt, skipping folds already finished."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}: {split!r}")
    output_dir = Path(output_dir)
    attempts = [(name, Path(run_dir)) for name, run_dir in attempts]
    pending, skipped = [], []
    for name, run_dir in attempts:
        for fold in folds:
            done = output_dir / name / f"fold_{fold}" / f"metrics_{split}.json"
            (skipped if done.is_file() else pending).append((name, run_dir, fold))
    summary = {
        "split": split,
        "skipped": [{"attempt": name, "fold": fold} for name, _, fold in skipped],
        "scored": [],
    }
    if not pending:
        return summary

    samples, feature_store = (context_builder or load_scoring_context)(
        Path(roi_manifest), Path(feature_dir)
    )
    ids_by_fold = read_split_sample_ids(roi_splits, split=split)
    for name, run_dir, fold in pending:
        sample_ids = ids_by_fold.get(fold, [])
        if not sample_ids:
            raise ValueError(f"{name}: fold {fold} has no {split} ROIs in {roi_splits}")
        missing = [sample_id for sample_id in sample_ids if sample_id not in samples]
        if missing:
            raise ValueError(
                f"{name}: {len(missing)} {split} ROIs of fold {fold} are absent from the "
                f"ROI manifest, e.g. {missing[0]}"
            )
        checkpoint = run_dir / f"fold_{fold}" / "best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name}: missing checkpoint {checkpoint}")
        fold_dir = output_dir / name / f"fold_{fold}"
        print(f"scoring {name} fold {fold}: {len(sample_ids)} {split} ROIs", flush=True)
        started = perf_counter()
        metrics = (scorer or score_fold)(
            run_dir=run_dir,
            fold=fold,
            split=split,
            records=[samples[sample_id] for sample_id in sample_ids],
            feature_store=feature_store,
            fold_dir=fold_dir,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            fold_dir / f"metrics_{split}.json",
            {
                "attempt_id": name,
                "fold": fold,
                "split": split,
                "roi_count": len(sample_ids),
                "run_dir": str(run_dir.resolve()),
                "checkpoint": {"bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)},
                "metrics": metrics,
                "soma_version": _soma_version(),
                "elapsed_seconds": round(perf_counter() - started, 1),
                "scored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            },
        )
        summary["scored"].append({"attempt": name, "fold": fold})
        print(
            f"scored {name} fold {fold} in {perf_counter() - started:.0f}s: "
            f"dataset_global_mean_dice={metrics.get('dataset_global_mean_dice')}",
            flush=True,
        )
    return summary


def _named_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"--attempt expects NAME=RUN_DIR, got {value!r}")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beetle score-test", description=__doc__)
    parser.add_argument(
        "--attempt",
        dest="attempts",
        type=_named_run,
        action="append",
        required=True,
        help="NAME=RUN_DIR holding config.yaml and fold_k/best_model.pt, repeated",
    )
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--roi-splits", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--folds", type=int, nargs="+", choices=FOLD_IDS, default=list(FOLD_IDS))
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args(argv)
    names = [name for name, _ in args.attempts]
    if len(names) != len(set(names)):
        parser.error("--attempt names must be unique")
    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = score_attempts(
        attempts=args.attempts,
        roi_manifest=args.roi_manifest,
        roi_splits=args.roi_splits,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        split=args.split,
        folds=args.folds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
