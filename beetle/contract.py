"""BEETLE label vocabulary and External ROI/PNG/archive contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence
import zipfile

import numpy as np
from PIL import Image

NUM_FOLDS = 5
EXTERNAL_ROI_COUNT = 170
PIXEL_MAPPING = {
    "background": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}
SUBMISSION_LABELS = frozenset(
    value for value in PIXEL_MAPPING.values() if value != 0
)
NUM_CLASSES = len(SUBMISSION_LABELS)
MODEL_INDEX_TO_SUBMISSION_LABEL = np.asarray(sorted(SUBMISSION_LABELS), dtype=np.uint8)


@dataclass(frozen=True)
class ExternalRoi:
    roi_filename: str
    patient_id: str
    source_wsi: str
    native_spacing_um: float
    width: int
    height: int


def load_roi_sidecar(
    path: str | Path, *, expected_rois: int = EXTERNAL_ROI_COUNT
) -> tuple[ExternalRoi, ...]:
    """Load the ROI-to-WSI sidecar: filename, source, spacing, and dimensions per ROI."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rois")
    if payload.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError("ROI sidecar must use schema_version 1 with a rois list")
    if len(rows) != expected_rois:
        raise ValueError(
            f"ROI sidecar requires exactly {expected_rois} ROIs, got {len(rows)}"
        )
    records: list[ExternalRoi] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"ROI sidecar row {index} must be an object")
        filename = str(row.get("roi_filename", "")).strip()
        if (
            not filename
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".png"
        ):
            raise ValueError(f"ROI sidecar row {index} requires a flat .png roi_filename")
        patient_id = str(row.get("patient_id", "")).strip()
        source_wsi = str(row.get("source_wsi", "")).strip()
        if not patient_id or not source_wsi:
            raise ValueError(f"ROI {filename!r} requires patient_id and source_wsi")
        spacing = row.get("native_spacing_um")
        if (
            isinstance(spacing, bool)
            or not isinstance(spacing, (int, float))
            or not math.isfinite(float(spacing))
            or float(spacing) <= 0
        ):
            raise ValueError(f"ROI {filename!r} requires finite positive native_spacing_um")
        width, height = row.get("width"), row.get("height")
        if not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0
            for v in (width, height)
        ):
            raise ValueError(f"ROI {filename!r} requires positive integer width and height")
        records.append(
            ExternalRoi(
                roi_filename=filename,
                patient_id=patient_id,
                source_wsi=source_wsi,
                native_spacing_um=float(spacing),
                width=width,
                height=height,
            )
        )
    filenames = [record.roi_filename for record in records]
    if len(set(filenames)) != len(filenames):
        raise ValueError("ROI sidecar repeats roi_filename")
    return tuple(sorted(records, key=lambda record: record.roi_filename))


def exact_directory_paths(
    directory: str | Path, expected_names: Sequence[str], *, artifact_label: str
) -> tuple[Path, ...]:
    """Require a flat directory to contain exactly the declared basenames."""
    directory = Path(directory)
    expected = set(expected_names)
    observed = {path.name for path in directory.iterdir()}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{artifact_label} failed exact coverage; missing={missing}, extra={extra}"
        )
    return tuple(directory / name for name in sorted(expected))


def validate_roi_inputs(roi_dir: str | Path, records: Sequence[ExternalRoi]) -> None:
    """Validate External ROI basenames against sidecar-declared dimensions."""
    paths = exact_directory_paths(
        roi_dir,
        [record.roi_filename for record in records],
        artifact_label="External ROI filenames",
    )
    by_name = {record.roi_filename: record for record in records}
    for path in paths:
        record = by_name[path.name]
        with Image.open(path) as image:
            if image.size != (record.width, record.height):
                raise ValueError(
                    f"ROI {record.roi_filename!r} dimensions {image.size} disagree "
                    f"with sidecar {(record.width, record.height)}"
                )


def validate_submission_pngs(
    output_dir: str | Path, records: Sequence[ExternalRoi]
) -> tuple[Path, ...]:
    """Require exact coverage and BEETLE's PNG mode/dimension/label vocabulary."""
    paths = exact_directory_paths(
        output_dir,
        [record.roi_filename for record in records],
        artifact_label="Submission filenames",
    )
    by_name = {record.roi_filename: record for record in records}
    for path in paths:
        record = by_name[path.name]
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L":
                raise ValueError(
                    f"Submission {path.name!r} must be a single-channel grayscale PNG"
                )
            if image.size != (record.width, record.height):
                raise ValueError(
                    f"Submission {path.name!r} dimensions {image.size} != "
                    f"{(record.width, record.height)}"
                )
            labels = set(int(value) for value in np.unique(np.asarray(image)))
            if not labels <= SUBMISSION_LABELS:
                raise ValueError(
                    f"Submission {path.name!r} contains invalid labels {sorted(labels)}"
                )
    return paths


def write_flat_submission_zip(paths: Sequence[Path], zip_path: str | Path) -> Path:
    """Write a deterministic flat archive with no directory prefix."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(paths, key=lambda value: value.name):
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return zip_path
