from pathlib import Path
import zipfile

import pytest

from beetle import release


def _completed_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_bytes(b"resolved config")
    for fold in range(5):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir()
        for name in release.FOLD_ARTIFACTS:
            (fold_dir / name).write_bytes(f"fold {fold} {name}".encode())
    environment = tmp_path / "environment.json"
    environment.write_text("{}")
    return run_dir, environment


def test_release_names_archives_after_the_attempt_and_keeps_evidence_order(tmp_path):
    run_dir, environment = _completed_run(tmp_path)
    report = tmp_path / "comparison_report.json"
    report.write_text("{}")
    recovery = tmp_path / "cache-access-recovery.json"
    recovery.write_text("{}")

    result = release.assemble_release(
        attempt_id="attempt-03",
        run_dir=run_dir,
        environment_path=environment,
        evidence={
            "comparison_report.json": report,
            "environment.json": environment,
            "cache-access-recovery.json": recovery,
        },
        output_dir=tmp_path / "release",
    )

    assert Path(result["weights_archive"]["path"]).name == "beetle-attempt-03-weights.zip"
    assert Path(result["evidence_archive"]["path"]).name == "beetle-attempt-03-evidence.zip"
    with zipfile.ZipFile(result["evidence_archive"]["path"]) as archive:
        names = archive.namelist()
    assert names[:3] == [
        "comparison_report.json",
        "environment.json",
        "cache-access-recovery.json",
    ]
    assert names[-1] == "artifact_checksums.json"
    assert "fold_4/confusion_evidence_tune.json" in names
    assert "fold_4/best_model.pt" not in names
    assert "fold_4/roi_batch_sampling.json" not in names


def test_release_refuses_missing_evidence_before_writing_archives(tmp_path):
    run_dir, environment = _completed_run(tmp_path)

    with pytest.raises(ValueError, match="release evidence is missing"):
        release.assemble_release(
            attempt_id="attempt-03",
            run_dir=run_dir,
            environment_path=environment,
            evidence={"comparison_report.json": tmp_path / "absent.json"},
            output_dir=tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()
