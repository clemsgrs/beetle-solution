"""Curate the BEETLE development slides into soma's unified segmentation Manifest.

One row per development WSI pairs the slide with its annotation raster; soma samples
ROIs from these slides at train time. Splits preserve BEETLE's predefined patient
folds: for fold ``k``, a slide whose ``validation_fold == k`` is ``test``,
``== (k+1) % n_folds`` is ``tune``, else ``train``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
import re
from pathlib import Path

from PIL import Image

from beetle.contract import NUM_FOLDS, PIXEL_MAPPING
from soma.curation.manifest import CuratedManifest, write_manifest

FULL_COHORT_SLIDES = 587
FULL_COHORT_PATIENTS = 527

_PATIENT_ID_PATTERNS = {
    "tcga": re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-"),
    "rumc": re.compile(r"^(TC_S\d+_P\d+)_C\d+_B\d+$"),
    "jb": re.compile(r"^(\d+)[BS]$"),
}
# Three TCGA slides ship at ~0.657 µm/px instead of the ~0.525 µm/px cohort spacing.
# Their measured level-0 spacing is declared in the manifest (`spacing_at_level_0`)
# so soma reads them at native resolution instead of upsampling.
_NATIVE_LEVEL_0_EXCEPTIONS = (
    "TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25",
    "TCGA-OL-A66P-01Z-00-DX1.5ADD0D6D-37C6-4BC9-8C2B-64DB18BE99B3",
    "TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014",
)


def read_dev_rows(overview_csv: Path) -> list[dict]:
    """Read all released development rows; reconstruct missing public WSI paths."""
    rows = [
        dict(r)
        for r in csv.DictReader(overview_csv.open())
        if r["split"] == "development"
    ]
    if not rows:
        raise RuntimeError(f"No development rows found in {overview_csv}")
    for row in rows:
        if not row["wsi_path"].strip() and row["source"].strip().lower() in _PATIENT_ID_PATTERNS:
            row["wsi_path"] = f"images/development/wsis/{row['name']}.tif"
    return rows


def resolve_patient_id(row: dict) -> str:
    """Use the released patient_id, or derive it from the slide name by source."""
    released = row["patient_id"].strip()
    if released:
        return released
    pattern = _PATIENT_ID_PATTERNS.get(row["source"].strip().lower())
    match = pattern.match(row["name"].strip()) if pattern is not None else None
    if match is None:
        raise ValueError(
            f"Cannot recover missing patient_id for slide {row['name']!r} "
            f"from source {row['source']!r}."
        )
    return match.group(1)


def _read_level_0_tiff_spacing(path: Path) -> float:
    """Read isotropic level-0 spacing from TIFF resolution tags, in µm/px."""
    previous_pixel_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            x_resolution = float(image.tag_v2[282])
            y_resolution = float(image.tag_v2[283])
            resolution_unit = int(image.tag_v2[296])
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"Cannot read level-0 TIFF resolution metadata for {path.name!r}."
        ) from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous_pixel_limit

    micrometres_per_unit = {2: 25_400.0, 3: 10_000.0}.get(resolution_unit)
    if micrometres_per_unit is None or x_resolution <= 0 or y_resolution <= 0:
        raise ValueError(f"Invalid level-0 TIFF resolution metadata for {path.name!r}.")
    x_spacing = micrometres_per_unit / x_resolution
    y_spacing = micrometres_per_unit / y_resolution
    if not math.isclose(x_spacing, y_spacing, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Slide {path.name!r} is anisotropic at level 0.")
    return round((x_spacing + y_spacing) / 2, 9)


def build_dataset_rows(rows: list[dict], beetle_root: Path) -> list[dict]:
    """One unified-schema dataset row per slide (supervision column = label_mask_path)."""
    dataset_rows: list[dict] = []
    for r in rows:
        native_spacing = (
            _read_level_0_tiff_spacing(beetle_root / r["wsi_path"])
            if r["name"] in _NATIVE_LEVEL_0_EXCEPTIONS
            else None
        )
        dataset_rows.append(
            {
                "sample_id": r["name"],
                "image_path": str((beetle_root / r["wsi_path"]).resolve()),
                "label_mask_path": str(
                    (beetle_root / r["annotation_mask_path"]).resolve()
                ),
                "patient_id": resolve_patient_id(r),
                "source": r["source"],
                "specimen_type": r["specimen_type"],
                "validation_fold": r["validation_fold"],
                "spacing_at_level_0": native_spacing if native_spacing is not None else "",
            }
        )
    return dataset_rows


def build_split_rows(dataset_rows: list[dict]) -> list[dict]:
    """Slide-level CV splits from BEETLE's fold rotation (test/tune/train)."""
    fold_nums = sorted(
        {int(r["validation_fold"].replace("fold", "")) for r in dataset_rows}
    )
    n_folds = len(fold_nums)
    split_rows: list[dict] = []
    for k in fold_nums:
        tune_fold = (k + 1) % n_folds
        for r in dataset_rows:
            wf = int(r["validation_fold"].replace("fold", ""))
            split = "test" if wf == k else ("tune" if wf == tune_fold else "train")
            split_rows.append({"sample_id": r["sample_id"], "split": split, "fold": k})
    return split_rows


def validate_cohort(dataset_rows: list[dict]) -> None:
    """Require the full cohort, the five organizer folds, and no patient-fold leak."""
    num_slides = len(dataset_rows)
    num_patients = len({row["patient_id"] for row in dataset_rows})
    if (num_slides, num_patients) != (FULL_COHORT_SLIDES, FULL_COHORT_PATIENTS):
        raise ValueError(
            f"The BEETLE cohort must contain exactly {FULL_COHORT_SLIDES} slides / "
            f"{FULL_COHORT_PATIENTS} patients; resolved {num_slides} / {num_patients}."
        )
    observed_folds = {row["validation_fold"] for row in dataset_rows}
    expected_folds = {f"fold{fold}" for fold in range(NUM_FOLDS)}
    if observed_folds != expected_folds:
        raise ValueError(
            f"Expected organizer folds fold0..fold{NUM_FOLDS - 1}; "
            f"resolved {sorted(observed_folds)}."
        )
    folds_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in dataset_rows:
        folds_by_patient[row["patient_id"]].add(row["validation_fold"])
    leaking = {
        patient_id: sorted(folds)
        for patient_id, folds in folds_by_patient.items()
        if len(folds) != 1
    }
    if leaking:
        details = "; ".join(
            f"{patient_id}: {', '.join(folds)}"
            for patient_id, folds in sorted(leaking.items())
        )
        raise ValueError(f"Patient(s) cross organizer folds: {details}.")


def curate(
    overview_csv: str | Path, beetle_root: str | Path, output_dir: str | Path
) -> CuratedManifest:
    """Write dataset.csv + splits.csv + summary.json for the full development cohort."""
    overview_csv = Path(overview_csv)
    beetle_root = Path(beetle_root)
    rows = read_dev_rows(overview_csv)

    missing = [
        row["name"]
        for row in rows
        if not (beetle_root / row["wsi_path"]).is_file()
        or not (beetle_root / row["annotation_mask_path"]).is_file()
    ]
    if missing:
        raise ValueError(f"Missing local WSI or annotation file(s): {missing}.")

    dataset_rows = build_dataset_rows(rows, beetle_root)
    validate_cohort(dataset_rows)
    split_rows = build_split_rows(dataset_rows)
    summary = {
        "dataset": "BEETLE (breast-cancer segmentation, slide manifest)",
        "dataset_type": "segmentation",
        "num_slides": len(dataset_rows),
        "num_patients": len({row["patient_id"] for row in dataset_rows}),
        "num_classes": len(PIXEL_MAPPING) - 1,
        "pixel_mapping": PIXEL_MAPPING,
        "cv_folds": list(range(NUM_FOLDS)),
        "slides_per_fold": dict(Counter(r["validation_fold"] for r in dataset_rows)),
        "slides_per_source": dict(Counter(r["source"] for r in dataset_rows)),
    }
    manifest = write_manifest(
        Path(output_dir),
        dataset_type="segmentation",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )
    print(f"Wrote slide manifest ({len(dataset_rows)} slides) to {output_dir}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m beetle curate", description=__doc__
    )
    parser.add_argument(
        "--beetle-root",
        type=Path,
        required=True,
        help="root that the overview CSV's relative paths resolve against",
    )
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=None,
        help="data_overview.csv (default: <beetle-root>/data_overview.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <beetle-root>/curated_slide_manifest)",
    )
    args = parser.parse_args(argv)
    curate(
        args.overview_csv or (args.beetle_root / "data_overview.csv"),
        args.beetle_root,
        args.out or (args.beetle_root / "curated_slide_manifest"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
