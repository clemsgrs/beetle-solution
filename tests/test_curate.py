import pytest

import csv

from beetle.curate import (
    CV_SPLITS,
    build_split_rows,
    resolve_patient_id,
    cv_split_rows,
    validate_cohort,
    validate_cv_splits,
    write_cv_splits,
)


def _dataset_row(sample_id, patient_id, fold):
    return {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "validation_fold": f"fold{fold}",
    }


def test_build_split_rows_rotates_test_tune_train():
    rows = [_dataset_row(f"s{fold}", f"p{fold}", fold) for fold in range(5)]
    split_rows = build_split_rows(rows)
    by_fold = {}
    for row in split_rows:
        by_fold.setdefault(row["fold"], {})[row["sample_id"]] = row["split"]
    for k in range(5):
        assert by_fold[k][f"s{k}"] == "test"
        assert by_fold[k][f"s{(k + 1) % 5}"] == "tune"
        trains = [s for s, split in by_fold[k].items() if split == "train"]
        assert len(trains) == 3


def test_validate_cohort_rejects_patient_fold_leak():
    # 587 slides / 527 patients: p526 owns the 61 surplus slides, spread across folds.
    rows = [_dataset_row(f"s{i}", f"p{min(i, 526)}", i % 5) for i in range(587)]
    with pytest.raises(ValueError, match="cross organizer folds"):
        validate_cohort(rows)


def test_validate_cohort_rejects_wrong_counts():
    rows = [_dataset_row(f"s{i}", f"p{i}", i % 5) for i in range(10)]
    with pytest.raises(ValueError, match="exactly 587"):
        validate_cohort(rows)


def test_resolve_patient_id_prefers_released_then_derives():
    assert resolve_patient_id({"patient_id": " P9 ", "source": "x", "name": "y"}) == "P9"
    assert (
        resolve_patient_id(
            {"patient_id": "", "source": "tcga", "name": "TCGA-AB-1234-01Z-x"}
        )
        == "TCGA-AB-1234"
    )
    assert (
        resolve_patient_id(
            {"patient_id": "", "source": "rumc", "name": "TC_S1_P002_C1_B4"}
        )
        == "TC_S1_P002"
    )
    with pytest.raises(ValueError, match="Cannot recover"):
        resolve_patient_id({"patient_id": "", "source": "unknown", "name": "slide"})


def test_cv_layout_keeps_tune_and_trains_on_the_rest():
    rows = [_dataset_row(f"s{fold}", f"p{fold}", fold) for fold in range(5)]
    nested = build_split_rows(rows)
    cv = cv_split_rows(nested)
    validate_cv_splits(cv)
    for before, after in zip(nested, cv):
        assert after["sample_id"] == before["sample_id"]
        assert after["fold"] == before["fold"]
        assert after["split"] == ("tune" if before["split"] == "tune" else "train")


def test_validate_cv_splits_rejects_leftover_test_and_overlapping_tunes():
    rows = [_dataset_row(f"s{fold}", f"p{fold}", fold) for fold in range(5)]
    nested = build_split_rows(rows)
    with pytest.raises(ValueError, match="train/tune only"):
        validate_cv_splits(nested)
    overlapping = [
        {**row, "split": "tune"} if row["sample_id"] == "s0" else row
        for row in cv_split_rows(nested)
    ]
    with pytest.raises(ValueError, match="exactly one fold"):
        validate_cv_splits(overlapping)


def test_write_cv_splits_writes_beside_the_curated_splits(tmp_path):
    rows = [_dataset_row(f"s{fold}", f"p{fold}", fold) for fold in range(5)]
    splits_csv = tmp_path / "splits.csv"
    with splits_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "fold"])
        writer.writeheader()
        writer.writerows(build_split_rows(rows))

    out = write_cv_splits(splits_csv)

    assert out == tmp_path / CV_SPLITS
    with out.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [row["split"] for row in written].count("tune") == 5
    assert [row["split"] for row in written].count("train") == 20
