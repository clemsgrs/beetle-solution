"""Export out-of-fold whole-slide segmentation masks for the development cohort.

For each fold ``k``, load only fold ``k``'s decoder and predict every slide that fold
holds out (``split == test`` in ``splits.csv``). The Zenodo annotation raster of each
slide is the inference mask: only annotated pixels (raw value ``> 0``) are predicted,
everything else is written as ``0``. Each output is a pyramidal tiled TIFF with the same
level-0 dimensions and spacing as the annotation raster, holding submission labels 1-4,
so a collaborator can score it against the annotation pixel for pixel, with no
patch-sampling coverage rule in between.

Layout of ``--output-dir``::

    fold_0/<sample_id>.tif       one mask per held-out slide
    fold_0/summary.csv           per-slide annotated / predicted pixel counts
    ...
    fold_4/

Slides whose mask already exists are skipped, so an interrupted export resumes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import numpy as np

from beetle.contract import (
    MODEL_INDEX_TO_SUBMISSION_LABEL,
    NUM_CLASSES,
    NUM_FOLDS,
    PIXEL_MAPPING,
)

CLASS_NAMES = tuple(name for name, value in PIXEL_MAPPING.items() if value != 0)
SUMMARY_COLUMNS = (
    "sample_id",
    "patient_id",
    "fold",
    "output_path",
    "mask_width",
    "mask_height",
    "mask_spacing_um",
    "read_spacing_um",
    "native_spacing_exception",
    "annotated_pixels",
    *(f"annotated_{name}" for name in CLASS_NAMES),
    *(f"predicted_{name}" for name in CLASS_NAMES),
)


@dataclass(frozen=True)
class SlideRecord:
    sample_id: str
    patient_id: str
    image_path: Path
    label_mask_path: Path
    spacing_at_level_0: float | None


@dataclass
class SlideRasters:
    """Spacing-aware region readers for one slide and its annotation raster.

    ``read_image(location_xy, spacing_um, size_wh)`` returns ``(h, w, 3)`` uint8, with
    ``location_xy`` in slide level-0 pixels. ``read_mask(location_xy, size_wh)`` returns
    the annotation region at the mask's own level-0 spacing, ``location_xy`` in mask
    level-0 pixels.

    ``request_spacing_um`` is what to ask the reader for (the training request, or the
    native spacing when the slide is coarser); ``read_spacing_um`` is the pixel pitch
    that request actually yields. They differ when a pyramid level lies within
    tolerance of the request: hs2p then reads that level unresampled, exactly as
    training did, so the chunk geometry must be derived from the level's own pitch.
    """

    image_spacing_um: float
    image_size_wh: tuple[int, int]
    mask_spacing_um: float
    mask_size_wh: tuple[int, int]
    request_spacing_um: float
    read_spacing_um: float
    read_image: Callable[[tuple[int, int], float, tuple[int, int]], np.ndarray]
    read_mask: Callable[[tuple[int, int], tuple[int, int]], np.ndarray]
    # Optional ``() -> (overview, downsample)``: a coarse pyramid level of the whole
    # annotation raster, used only to decide which chunks can hold annotation. Reading
    # every level-0 chunk of a 20-gigapixel mask takes ~20 min; the overview takes seconds.
    read_mask_overview: Callable[[], tuple[np.ndarray, float]] | None = None


# -- manifest -------------------------------------------------------------------------


def load_fold_slides(
    dataset_csv: str | Path, splits_csv: str | Path, fold: int
) -> tuple[SlideRecord, ...]:
    """Slides held out (``test``) by ``fold``, in manifest order."""
    with Path(dataset_csv).open(newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with Path(splits_csv).open(newline="") as handle:
        held_out = [
            row["sample_id"]
            for row in csv.DictReader(handle)
            if int(row["fold"]) == fold and row["split"] == "test"
        ]
    if not held_out:
        raise ValueError(f"No held-out slides for fold {fold} in {splits_csv}")
    missing = [sample_id for sample_id in held_out if sample_id not in rows]
    if missing:
        raise ValueError(f"splits.csv names slides absent from dataset.csv: {missing}")
    records = []
    for sample_id in held_out:
        row = rows[sample_id]
        declared = str(row.get("spacing_at_level_0", "")).strip()
        records.append(
            SlideRecord(
                sample_id=sample_id,
                patient_id=row["patient_id"],
                image_path=Path(row["image_path"]),
                label_mask_path=Path(row["label_mask_path"]),
                spacing_at_level_0=float(declared) if declared else None,
            )
        )
    return tuple(records)


# -- readers ----------------------------------------------------------------------------


def open_slide_rasters(
    record: SlideRecord,
    *,
    backend: str,
    mask_backend: str,
    requested_spacing_um: float,
    tolerance: float,
) -> SlideRasters:
    """Open the slide and its annotation raster through hs2p, mirroring training reads."""
    from hs2p.wsi.masks import read_label_region_at_spacing
    from hs2p.wsi.wsi import WSI

    slide = WSI(
        record.image_path, backend=backend, spacing_at_level_0=record.spacing_at_level_0
    )
    mask = WSI(record.label_mask_path, backend=mask_backend)
    # Spacing the mask file declares (what hs2p needs to pick its level 0 unresampled)
    # and the spacing the output is stamped with. They coincide unless the slide's own
    # metadata had to be overridden (the three TCGA slides at ~0.657 um): the annotation
    # raster shares that slide's level-0 grid, so its true pitch is the override, not
    # whatever the mask file happens to declare.
    mask_declared_spacing = float(mask.get_level_spacing(0))
    mask_spacing = mask_declared_spacing
    if record.spacing_at_level_0 is not None and tuple(mask.level_dimensions[0]) == tuple(
        slide.level_dimensions[0]
    ):
        mask_spacing = float(record.spacing_at_level_0)
    request, effective = plan_read_spacing(
        level0_spacing_um=float(slide.get_level_spacing(0)),
        level_downsamples=list(slide.level_downsamples),
        requested_spacing_um=requested_spacing_um,
        tolerance=tolerance,
    )

    def read_image(location, spacing_um, size):
        arr = slide.read_region_at_spacing(
            (int(location[0]), int(location[1])),
            float(spacing_um),
            (int(size[0]), int(size[1])),
            tolerance=tolerance,
            interpolation="area",
        )
        return np.ascontiguousarray(np.asarray(arr)[..., :3])

    def read_mask(location, size):
        return read_label_region_at_spacing(
            mask,
            (int(location[0]), int(location[1])),
            mask_declared_spacing,
            (int(size[0]), int(size[1])),
            tolerance=tolerance,
        )

    overview_level = _overview_level(mask.level_downsamples, MAX_OVERVIEW_DOWNSAMPLE)

    def read_mask_overview():
        downsample = _downsample_of(mask.level_downsamples[overview_level])
        width, height = (int(v) for v in mask.level_dimensions[overview_level])
        overview = read_label_region_at_spacing(
            mask, (0, 0), mask_declared_spacing * downsample, (width, height), tolerance=tolerance
        )
        return np.asarray(overview), downsample

    return SlideRasters(
        image_spacing_um=float(slide.get_level_spacing(0)),
        image_size_wh=tuple(int(v) for v in slide.level_dimensions[0]),
        mask_spacing_um=mask_spacing,
        mask_size_wh=tuple(int(v) for v in mask.level_dimensions[0]),
        request_spacing_um=request,
        read_spacing_um=effective,
        read_image=read_image,
        read_mask=read_mask,
        read_mask_overview=read_mask_overview if overview_level > 0 else None,
    )


# Coarsest mask pyramid level used to locate annotated chunks. Pyramid levels are
# subsampled, so an annotation narrower than this many level-0 pixels could vanish from
# the overview; BEETLE annotations are region polygons far wider than 16 px.
MAX_OVERVIEW_DOWNSAMPLE = 16.0


def _downsample_of(value) -> float:
    if isinstance(value, (tuple, list)):
        return float(value[0])
    return float(value)


def _overview_level(level_downsamples, max_downsample: float) -> int:
    """Index of the coarsest level whose downsample is <= ``max_downsample`` (0 if none)."""
    best = 0
    for index, value in enumerate(level_downsamples):
        if 1.0 < _downsample_of(value) <= max_downsample:
            best = index
    return best


def candidate_chunks(
    overview: np.ndarray, downsample: float, *, mask_size_wh: tuple[int, int], chunk_px: int
) -> set[tuple[int, int]]:
    """Chunk origins ``(x0, y0)`` that may contain annotation, from a coarse overview.

    Every non-zero overview pixel is dilated by one overview pixel before mapping to
    chunks so that subsampling jitter at chunk borders cannot drop a chunk.
    """
    mask_w, mask_h = mask_size_wh
    ys, xs = np.nonzero(np.asarray(overview) > 0)
    if len(ys) == 0:
        return set()
    chunks: set[tuple[int, int]] = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x_lvl0 = np.clip((xs + dx) * downsample, 0, mask_w - 1)
            y_lvl0 = np.clip((ys + dy) * downsample, 0, mask_h - 1)
            cx = (x_lvl0 // chunk_px).astype(np.int64) * chunk_px
            cy = (y_lvl0 // chunk_px).astype(np.int64) * chunk_px
            chunks.update(zip(cx.tolist(), cy.tolist()))
    return chunks


def plan_read_spacing(
    *,
    level0_spacing_um: float,
    level_downsamples: Sequence[tuple[float, float]],
    requested_spacing_um: float,
    tolerance: float,
) -> tuple[float, float]:
    """``(request, effective)`` spacings for reading a slide the way training read it.

    ``native_if_coarser``: a slide coarser than the request (beyond tolerance) is read
    at its own level-0 spacing. Then, when hs2p finds a pyramid level within tolerance
    of the request, it reads that level without resampling, so the effective pixel
    pitch is the level's spacing rather than the request.
    """
    from hs2p.wsi.geometry import select_level_for_spacing_read

    request = float(requested_spacing_um)
    if level0_spacing_um > request * (1.0 + tolerance):
        request = float(level0_spacing_um)
    selection = select_level_for_spacing_read(
        requested_spacing_um=request,
        level0_spacing_um=float(level0_spacing_um),
        level_downsamples=list(level_downsamples),
        tolerance=float(tolerance),
        content_kind="image",
    )
    effective = float(selection.read_spacing_um) if selection.is_within_tolerance else request
    return request, effective


# -- prediction -------------------------------------------------------------------------


def _resize_nearest(labels: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if labels.shape == (size_wh[1], size_wh[0]):
        return labels
    import cv2

    return cv2.resize(labels, (int(size_wh[0]), int(size_wh[1])), interpolation=cv2.INTER_NEAREST)


def predict_slide_mask(
    predictor,
    rasters: SlideRasters,
    *,
    canvas: np.ndarray,
    chunk_px: int = 4096,
    overlap: float = 0.5,
    batch_size: int = 8,
) -> dict[str, int | float]:
    """Fill ``canvas`` (mask level-0 geometry, uint8) with submission labels on annotated pixels.

    The slide is walked in ``chunk_px`` squares of mask pixels. Chunks with no annotation
    are skipped; when the mask exposes a coarse overview, chunks the overview shows as
    empty are not even read at level 0. Each predicted chunk is read with a half-tile halo so the Hann-blended
    sliding window sees full context at chunk borders, then cropped back.
    """
    mask_w, mask_h = rasters.mask_size_wh
    if canvas.shape != (mask_h, mask_w):
        raise ValueError(f"canvas {canvas.shape} must match mask geometry {(mask_h, mask_w)}")
    read_spacing = rasters.read_spacing_um
    # Scale factors: mask level-0 px -> read px, and read px -> slide level-0 px.
    mask_to_read = rasters.mask_spacing_um / read_spacing
    read_to_slide = read_spacing / rasters.image_spacing_um
    slide_w, slide_h = rasters.image_size_wh
    read_w = int(round(slide_w / read_to_slide))
    read_h = int(round(slide_h / read_to_slide))
    tile_h, tile_w = (int(v) for v in predictor.geometry.target_size)
    halo = max(tile_h, tile_w) // 2

    annotated = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    predicted = np.zeros(NUM_CLASSES + 1, dtype=np.int64)

    candidates: set[tuple[int, int]] | None = None
    if rasters.read_mask_overview is not None:
        overview, downsample = rasters.read_mask_overview()
        candidates = candidate_chunks(
            overview, downsample, mask_size_wh=(mask_w, mask_h), chunk_px=chunk_px
        )

    for y0 in range(0, mask_h, chunk_px):
        for x0 in range(0, mask_w, chunk_px):
            if candidates is not None and (x0, y0) not in candidates:
                continue
            w = min(chunk_px, mask_w - x0)
            h = min(chunk_px, mask_h - y0)
            mask_chunk = np.asarray(rasters.read_mask((x0, y0), (w, h)))
            if mask_chunk.shape != (h, w):
                raise ValueError(
                    f"mask region {(x0, y0, w, h)} came back as {mask_chunk.shape}"
                )
            annotated += np.bincount(
                np.clip(mask_chunk, 0, NUM_CLASSES).ravel(), minlength=NUM_CLASSES + 1
            )
            inside = mask_chunk > 0
            if not inside.any():
                continue

            # Chunk footprint in read px, padded by the halo and clamped to the slide.
            rx0 = int(np.floor(x0 * mask_to_read))
            ry0 = int(np.floor(y0 * mask_to_read))
            rx1 = int(np.ceil((x0 + w) * mask_to_read))
            ry1 = int(np.ceil((y0 + h) * mask_to_read))
            px0, py0 = max(0, rx0 - halo), max(0, ry0 - halo)
            px1, py1 = min(read_w, rx1 + halo), min(read_h, ry1 + halo)
            rgb = rasters.read_image(
                (int(round(px0 * read_to_slide)), int(round(py0 * read_to_slide))),
                rasters.request_spacing_um,
                (px1 - px0, py1 - py0),
            )
            if rgb.shape[:2] != (py1 - py0, px1 - px0):
                raise ValueError(
                    f"image region came back as {rgb.shape[:2]}, expected {(py1 - py0, px1 - px0)}"
                )
            result = predictor.predict_array(
                rgb, overlap=overlap, batch_size=batch_size, return_probs=False
            )
            labels = np.asarray(result.labels)
            labels = labels[ry0 - py0 : ry1 - py0, rx0 - px0 : rx1 - px0]
            labels = _resize_nearest(labels.astype(np.uint8), (w, h))
            if labels.min() < 0 or labels.max() >= NUM_CLASSES:
                raise ValueError("prediction has a class index outside the four-class vocabulary")
            out = np.where(inside, MODEL_INDEX_TO_SUBMISSION_LABEL[labels], 0).astype(np.uint8)
            predicted += np.bincount(out.ravel(), minlength=NUM_CLASSES + 1)
            canvas[y0 : y0 + h, x0 : x0 + w] = out

    stats: dict[str, int | float] = {
        "read_spacing_um": read_spacing,
        "annotated_pixels": int(annotated[1:].sum()),
    }
    for index, name in enumerate(CLASS_NAMES, start=1):
        stats[f"annotated_{name}"] = int(annotated[index])
        stats[f"predicted_{name}"] = int(predicted[index])
    return stats


# -- output ----------------------------------------------------------------------------


def _tiff_compression() -> str:
    """LZW when imagecodecs is installed (matches the Zenodo rasters), else deflate.

    tifffile writes deflate with the standard library alone; LZW needs imagecodecs.
    Both are tiled TIFF variants that OpenSlide and ASAP read.
    """
    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        return "zlib"
    return "lzw"


def write_pyramidal_mask(
    canvas: np.ndarray,
    path: Path,
    *,
    spacing_um: float,
    tile: int = 512,
    compression: str | None = None,
) -> None:
    """Write a uint8 label pyramid (tiled, /2 nearest levels, resolution tags)."""
    import tifffile

    if canvas.dtype != np.uint8 or canvas.ndim != 2:
        raise ValueError("mask canvas must be a 2-D uint8 array")
    compression = compression or _tiff_compression()
    levels = [canvas]
    while min(levels[-1].shape) > tile:
        levels.append(levels[-1][::2, ::2])
    pixels_per_cm = 10_000.0 / float(spacing_um)
    tmp = path.with_name(path.name + ".partial")
    with tifffile.TiffWriter(tmp, bigtiff=canvas.nbytes > 2**31) as writer:
        for index, level in enumerate(levels):
            scale = 2**index
            writer.write(
                level,
                tile=(tile, tile),
                compression=compression,
                photometric="minisblack",
                resolution=(pixels_per_cm / scale, pixels_per_cm / scale),
                resolutionunit="CENTIMETER",
                subifds=len(levels) - 1 if index == 0 else None,
                subfiletype=0 if index == 0 else 1,
            )
    tmp.replace(path)


def _append_summary(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


# -- driver ----------------------------------------------------------------------------


def export_slides(
    *,
    predictor,
    records: Sequence[SlideRecord],
    output_dir: Path,
    open_rasters: Callable[[SlideRecord], SlideRasters],
    chunk_px: int,
    overwrite: bool = False,
    fold: int | str = "ensemble",
    scratch_dir: Path | None = None,
) -> list[Path]:
    """Predict every record inside its mask and write ``<sample_id>.tif`` + ``summary.csv``.

    The full-resolution canvas is a memmap in ``scratch_dir`` (system temp by default):
    it can reach 20 GB for the largest slides, so keep it on a local disk rather than
    the network share that usually holds ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir) if scratch_dir is not None else Path(tempfile.gettempdir())
    scratch_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in records:
        target = output_dir / f"{record.sample_id}.tif"
        if target.exists() and not overwrite:
            print(f"[{fold}] {record.sample_id}: exists, skipping")
            written.append(target)
            continue
        rasters = open_rasters(record)
        mask_w, mask_h = rasters.mask_size_wh
        with tempfile.NamedTemporaryFile(dir=scratch_dir, prefix=".canvas-", delete=False) as scratch:
            scratch_path = Path(scratch.name)
        try:
            canvas = np.memmap(scratch_path, dtype=np.uint8, mode="w+", shape=(mask_h, mask_w))
            stats = predict_slide_mask(predictor, rasters, canvas=canvas, chunk_px=chunk_px)
            write_pyramidal_mask(canvas, target, spacing_um=rasters.mask_spacing_um)
            del canvas
        finally:
            scratch_path.unlink(missing_ok=True)
        _append_summary(
            output_dir / "summary.csv",
            {
                "sample_id": record.sample_id,
                "patient_id": record.patient_id,
                "fold": fold,
                "output_path": str(target),
                "mask_width": mask_w,
                "mask_height": mask_h,
                "mask_spacing_um": rasters.mask_spacing_um,
                "native_spacing_exception": record.spacing_at_level_0 is not None,
                **stats,
            },
        )
        print(
            f"[{fold}] {record.sample_id}: {stats['annotated_pixels']} masked px "
            f"at {stats['read_spacing_um']} um/px -> {target.name}"
        )
        written.append(target)
    return written


def export_fold(
    *,
    predictor,
    records: Sequence[SlideRecord],
    fold: int,
    output_dir: Path,
    open_rasters: Callable[[SlideRecord], SlideRasters],
    chunk_px: int,
    overwrite: bool = False,
    scratch_dir: Path | None = None,
) -> list[Path]:
    return export_slides(
        predictor=predictor,
        records=records,
        output_dir=output_dir / f"fold_{fold}",
        open_rasters=open_rasters,
        chunk_px=chunk_px,
        overwrite=overwrite,
        fold=fold,
        scratch_dir=scratch_dir,
    )


def load_slide_csv(path: str | Path) -> tuple[SlideRecord, ...]:
    """Records from a ``wsi_path,roi_mask_path[,sample_id][,patient_id]`` CSV."""
    records = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            image = Path(row["wsi_path"].strip())
            mask = Path(row["roi_mask_path"].strip())
            sample_id = str(row.get("sample_id") or "").strip() or image.stem
            records.append(
                SlideRecord(
                    sample_id=sample_id,
                    patient_id=str(row.get("patient_id") or "").strip(),
                    image_path=image,
                    label_mask_path=mask,
                    spacing_at_level_0=None,
                )
            )
    if not records:
        raise ValueError(f"No slides listed in {path}")
    if len({r.sample_id for r in records}) != len(records):
        raise ValueError(f"Duplicate sample ids in {path}")
    return tuple(records)


def _raster_opener(config) -> Callable[[SlideRecord], SlideRasters]:
    pre = config.preprocessing

    def open_rasters(record: SlideRecord) -> SlideRasters:
        return open_slide_rasters(
            record,
            backend=str(pre.backend),
            mask_backend=str(pre.mask_backend),
            requested_spacing_um=float(pre.requested_spacing_um),
            tolerance=float(pre.tolerance),
        )

    return open_rasters


def run_slide_predict(
    *,
    run_dir: str | Path,
    slides_csv: str | Path,
    output_dir: str | Path,
    folds: Sequence[int] | None = None,
    chunk_px: int = 4096,
    overwrite: bool = False,
    scratch_dir: str | Path | None = None,
) -> list[Path]:
    """Ensemble (all folds by default) inside each slide's ROI mask, for external sets."""
    from soma.config import load_config

    from beetle.infer import load_fold_predictor

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    records = load_slide_csv(slides_csv)
    predictor = load_fold_predictor(run_dir, folds=folds)
    written = export_slides(
        predictor=predictor,
        records=records,
        output_dir=Path(output_dir),
        open_rasters=_raster_opener(config),
        chunk_px=chunk_px,
        overwrite=overwrite,
        scratch_dir=Path(scratch_dir) if scratch_dir else None,
    )
    print(f"Wrote {len(written)} slide masks under {output_dir}")
    return written


def run_cv_predict(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    dataset_csv: str | Path | None = None,
    splits_csv: str | Path | None = None,
    folds: Sequence[int] | None = None,
    chunk_px: int = 4096,
    overwrite: bool = False,
    scratch_dir: str | Path | None = None,
) -> list[Path]:
    from soma.config import load_config

    from beetle.infer import load_fold_predictor

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    config = load_config(run_dir / "config.yaml")
    dataset_csv = Path(dataset_csv or config.dataset_csv)
    splits_csv = Path(splits_csv or config.splits_csv)
    open_rasters = _raster_opener(config)

    written: list[Path] = []
    for fold in tuple(range(NUM_FOLDS)) if folds is None else tuple(folds):
        records = load_fold_slides(dataset_csv, splits_csv, fold)
        predictor = load_fold_predictor(run_dir, folds=(fold,))
        written += export_fold(
            predictor=predictor,
            records=records,
            fold=fold,
            output_dir=output_dir,
            open_rasters=open_rasters,
            chunk_px=chunk_px,
            overwrite=overwrite,
            scratch_dir=Path(scratch_dir) if scratch_dir else None,
        )
        del predictor
    print(f"Wrote {len(written)} out-of-fold slide masks under {output_dir}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beetle cv-predict", description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, default=None, help="defaults to the run config")
    parser.add_argument("--splits-csv", type=Path, default=None, help="defaults to the run config")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="subset of 0..4")
    parser.add_argument("--chunk-px", type=int, default=4096, help="chunk edge in mask pixels")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="local disk for the per-slide canvas memmap (up to ~20 GB); defaults to the system temp dir",
    )
    args = parser.parse_args(argv)
    run_cv_predict(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv,
        splits_csv=args.splits_csv,
        folds=args.folds,
        chunk_px=args.chunk_px,
        overwrite=args.overwrite,
        scratch_dir=args.scratch_dir,
    )
    return 0


def slide_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m beetle slide-predict",
        description=(
            "Predict whole slides inside ROI masks with the fold ensemble (all five folds "
            "by default). Input CSV columns: wsi_path, roi_mask_path, optional sample_id."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slides-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="subset of 0..4")
    parser.add_argument("--chunk-px", type=int, default=4096, help="chunk edge in mask pixels")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="local disk for the per-slide canvas memmap (up to ~20 GB); defaults to the system temp dir",
    )
    args = parser.parse_args(argv)
    run_slide_predict(
        run_dir=args.run_dir,
        slides_csv=args.slides_csv,
        output_dir=args.output_dir,
        folds=args.folds,
        chunk_px=args.chunk_px,
        overwrite=args.overwrite,
        scratch_dir=args.scratch_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
