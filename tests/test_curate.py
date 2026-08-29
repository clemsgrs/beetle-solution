import pytest

from beetle.curate import build_split_rows, resolve_patient_id, validate_cohort


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
