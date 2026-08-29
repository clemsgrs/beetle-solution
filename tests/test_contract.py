import json

import numpy as np
import pytest
from PIL import Image

from beetle.contract import (
    load_roi_sidecar,
    validate_submission_pngs,
    write_flat_submission_zip,
)


def _sidecar(tmp_path, rois):
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps({"schema_version": 1, "rois": rois}))
    return path


def _roi_row(name, width=8, height=6):
    return {
        "roi_filename": name,
        "patient_id": "P1",
        "source_wsi": "wsi_1",
        "native_spacing_um": 0.5,
        "width": width,
        "height": height,
    }


def test_load_roi_sidecar_sorts_and_parses(tmp_path):
    path = _sidecar(tmp_path, [_roi_row("b.png"), _roi_row("a.png")])
    records = load_roi_sidecar(path, expected_rois=2)
    assert [r.roi_filename for r in records] == ["a.png", "b.png"]
    assert records[0].width == 8 and records[0].height == 6


def test_load_roi_sidecar_rejects_wrong_count(tmp_path):
    path = _sidecar(tmp_path, [_roi_row("a.png")])
    with pytest.raises(ValueError, match="exactly 2"):
        load_roi_sidecar(path, expected_rois=2)


def test_load_roi_sidecar_rejects_duplicate_and_nested_names(tmp_path):
    duplicated = _sidecar(tmp_path, [_roi_row("a.png"), _roi_row("a.png")])
    with pytest.raises(ValueError, match="repeats"):
        load_roi_sidecar(duplicated, expected_rois=2)
    nested = _sidecar(tmp_path, [_roi_row("sub/a.png")])
    with pytest.raises(ValueError, match="flat .png"):
        load_roi_sidecar(nested, expected_rois=1)


def _write_prediction(directory, name, width=8, height=6, label=1):
    array = np.full((height, width), label, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(directory / name)


def test_validate_submission_pngs_accepts_valid_predictions(tmp_path):
    records = load_roi_sidecar(
        _sidecar(tmp_path, [_roi_row("a.png"), _roi_row("b.png")]), expected_rois=2
    )
    out = tmp_path / "predictions"
    out.mkdir()
    _write_prediction(out, "a.png", label=1)
    _write_prediction(out, "b.png", label=4)
    paths = validate_submission_pngs(out, records)
    assert [p.name for p in paths] == ["a.png", "b.png"]


def test_validate_submission_pngs_rejects_bad_label_dims_and_coverage(tmp_path):
    records = load_roi_sidecar(_sidecar(tmp_path, [_roi_row("a.png")]), expected_rois=1)
    out = tmp_path / "predictions"
    out.mkdir()

    with pytest.raises(ValueError, match="missing"):
        validate_submission_pngs(out, records)

    _write_prediction(out, "a.png", label=0)
    with pytest.raises(ValueError, match="invalid labels"):
        validate_submission_pngs(out, records)

    _write_prediction(out, "a.png", width=9, label=2)
    with pytest.raises(ValueError, match="dimensions"):
        validate_submission_pngs(out, records)


def test_write_flat_submission_zip_is_deterministic(tmp_path):
    out = tmp_path / "predictions"
    out.mkdir()
    _write_prediction(out, "a.png")
    _write_prediction(out, "b.png")
    paths = [out / "b.png", out / "a.png"]
    zip_one = write_flat_submission_zip(paths, tmp_path / "one.zip")
    zip_two = write_flat_submission_zip(paths, tmp_path / "two.zip")
    assert zip_one.read_bytes() == zip_two.read_bytes()
