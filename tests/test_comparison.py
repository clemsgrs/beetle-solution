import json
from pathlib import Path

import numpy as np
import pytest

from beetle import comparison

VOCABULARY = ["other", "non_invasive_epithelium", "invasive_epithelium", "necrosis"]
IMPERFECT = [[4, 1, 0, 0], [0, 4, 1, 0], [0, 0, 4, 1], [1, 0, 0, 4]]
PERFECT = [[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]]


def _write_run(run_dir, matrix, split="tune"):
    for fold in range(5):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / f"confusion_evidence_{split}.json").write_text(json.dumps({
            "schema_version": 1,
            "records": [{
                "sample_id": f"roi-{fold}",
                "fold": fold,
                "class_vocabulary": VOCABULARY,
                "confusion_matrix": matrix,
            }],
        }))
    return run_dir


@pytest.fixture
def sources(tmp_path):
    cv_results = tmp_path / "cv_results.json"
    cv_results.write_text(json.dumps({
        "classes": [{"model_index": i, "name": name} for i, name in enumerate(VOCABULARY)],
        "folds": {str(fold): {"mean_dice": 0.8} for fold in range(5)},
        "patients": [
            {"patient_id": f"patient-{fold}", "fold": fold, "confusion_matrix": IMPERFECT}
            for fold in range(5)
        ],
    }))
    mapping = tmp_path / "roi_manifest.csv"
    mapping.write_text(
        "sample_id,patient_id\n" + "".join(f"roi-{f},patient-{f}\n" for f in range(5))
    )
    return tmp_path, cv_results, mapping


def test_report_pairs_every_later_attempt_with_every_earlier_one(sources):
    tmp_path, cv_results, mapping = sources

    report = comparison.build_report(
        attempt_sources=[
            ("attempt-01", cv_results),
            ("attempt-02", _write_run(tmp_path / "a02", IMPERFECT)),
            ("attempt-03", _write_run(tmp_path / "a03", PERFECT)),
        ],
        sample_patient_csv=mapping,
        spacing_exception_patient_ids=("patient-4",),
        primary_patient_count=5,
        sensitivity_patient_count=4,
        bootstrap_draws=20,
    )

    assert report["formal_comparator"] == "attempt-01"
    assert report["candidate"] == "attempt-03"
    assert report["attempts"]["attempt-03"] == {
        "fold_scores": [1.0] * 5,
        "mean": 1.0,
        "sample_standard_deviation": 0.0,
    }
    assert list(report["comparisons"]) == [
        "attempt-02_minus_attempt-01",
        "attempt-03_minus_attempt-01",
        "attempt-03_minus_attempt-02",
    ]
    gain = report["comparisons"]["attempt-03_minus_attempt-02"]
    assert gain["paired_fold_deltas"] == [0.2] * 5
    assert gain["mean_delta"] == 0.2
    assert gain["folds_improved"] == 5
    assert gain["patient_bootstrap"]["primary"] == {
        "seed": 0,
        "draws": 20,
        "pooled_macro_dice_delta": 0.2,
        "macro_dice_delta_percentile_95_ci": [0.2, 0.2],
    }
    assert report["comparisons"]["attempt-02_minus_attempt-01"]["folds_improved"] == 0
    assert report["pooled_metrics"]["attempt-02"] == {
        "dice_per_class": dict.fromkeys(VOCABULARY, 0.8),
        "macro_dice": 0.8,
        "pixel_micro_dice": 0.8,
    }
    sensitivity = report["patient_bootstrap"]["sensitivity"]
    assert sensitivity["patient_count"] == 4
    assert sensitivity["excluded_patient_ids"] == ["patient-4"]
    assert sensitivity["attempts"]["attempt-01"]["macro_dice_percentile_95_ci"] == [0.8, 0.8]


def test_paired_bootstrap_resamples_the_same_patients_for_both_attempts():
    heterogeneous = [np.asarray(IMPERFECT), np.asarray(PERFECT)] * 3

    result = comparison.paired_bootstrap_delta(heterogeneous, heterogeneous, draws=50)

    assert result["macro_dice_delta_percentile_95_ci"] == [0.0, 0.0]


def test_report_refuses_attempts_on_different_patients(sources):
    tmp_path, cv_results, mapping = sources
    mapping.write_text(
        "sample_id,patient_id\n" + "".join(f"roi-{f},other-{f}\n" for f in range(5))
    )

    with pytest.raises(ValueError, match="patient cohorts differ"):
        comparison.build_report(
            attempt_sources=[
                ("attempt-01", cv_results),
                ("attempt-03", _write_run(tmp_path / "a03", PERFECT)),
            ],
            sample_patient_csv=mapping,
            spacing_exception_patient_ids=(),
            primary_patient_count=5,
            sensitivity_patient_count=5,
            bootstrap_draws=5,
        )


def test_evidence_refuses_a_patient_split_across_folds(tmp_path):
    run = _write_run(tmp_path / "run", PERFECT)

    with pytest.raises(ValueError, match="multiple folds"):
        comparison.load_confusion_evidence(
            [run / f"fold_{fold}/confusion_evidence_tune.json" for fold in range(5)],
            {f"roi-{fold}": "patient-0" for fold in range(5)},
            label="attempt-03",
        )


def test_cli_passes_attempts_in_the_given_order(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        comparison, "build_report", lambda **kwargs: captured.update(kwargs) or {"ok": True}
    )
    output = tmp_path / "report.json"

    assert comparison.main([
        "--attempt", "attempt-01=cv_results.json",
        "--attempt", "attempt-03=run",
        "--sample-patient-csv", "roi_manifest.csv",
        "--output", str(output),
    ]) == 0

    assert captured["attempt_sources"] == [
        ("attempt-01", Path("cv_results.json")),
        ("attempt-03", Path("run")),
    ]
    assert captured["split"] == "tune"
    assert json.loads(output.read_text()) == {"ok": True}


def test_test_split_reads_scored_test_evidence_and_refuses_tune_records(sources):
    tmp_path, cv_results, mapping = sources
    baseline = _write_run(tmp_path / "a01", IMPERFECT, split="test")
    candidate = _write_run(tmp_path / "a03", PERFECT, split="test")
    _write_run(tmp_path / "a03", IMPERFECT, split="tune")
    cohort = dict(
        sample_patient_csv=mapping,
        spacing_exception_patient_ids=(),
        primary_patient_count=5,
        sensitivity_patient_count=5,
        bootstrap_draws=5,
    )

    report = comparison.build_report(
        attempt_sources=[("attempt-01", baseline), ("attempt-03", candidate)],
        split="test",
        **cohort,
    )

    assert report["attempts"]["attempt-03"]["fold_scores"] == [1.0] * 5
    with pytest.raises(ValueError, match="tune-fold scores"):
        comparison.build_report(
            attempt_sources=[("attempt-01", cv_results), ("attempt-03", candidate)],
            split="test",
            **cohort,
        )


def test_report_pools_patients_within_each_group(sources):
    tmp_path, cv_results, mapping = sources

    report = comparison.build_report(
        attempt_sources=[
            ("attempt-01", cv_results),
            ("attempt-03", _write_run(tmp_path / "a03", PERFECT)),
        ],
        sample_patient_csv=mapping,
        spacing_exception_patient_ids=(),
        primary_patient_count=5,
        sensitivity_patient_count=5,
        bootstrap_draws=5,
        patient_groups={"patient-0": "jb", "patient-1": "jb", "patient-2": "rumc",
                        "patient-3": "rumc", "patient-4": "rumc"},
    )

    assert list(report["per_group"]) == ["jb", "rumc"]
    assert report["per_group"]["jb"]["patient_count"] == 2
    assert report["per_group"]["jb"]["attempts"]["attempt-01"]["macro_dice"] == 0.8
    assert report["per_group"]["rumc"]["attempts"]["attempt-03"]["macro_dice"] == 1.0


def test_patient_groups_refuse_a_patient_spanning_two_groups(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("sample_id,patient_id,source\nwsi1,p1,rumc\nwsi2,p1,nki\n")

    with pytest.raises(ValueError, match="spans several source values"):
        comparison.read_patient_groups(dataset, "source")
