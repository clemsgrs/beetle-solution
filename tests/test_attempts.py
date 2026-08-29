import json
from pathlib import Path
import tomllib

from beetle.attempts import load_attempt_config


def test_attempt_01_resolves_to_the_submitted_scientific_protocol():
    config = load_attempt_config("configs/attempts/attempt-01.yaml")

    assert config.preprocessing.spacing_policy == "native_if_coarser"
    assert config.training.roi_batch_sampling == "uniform"
    assert config.training.class_request_ratios is None
    assert config.training.monitor == "dataset_global_mean_dice"
    assert config.evaluation.metrics == [
        "mean_dice",
        "dataset_global_mean_dice",
        "mean_iou",
        "dice_per_class",
    ]
    assert config.evaluation.save_segmentation_confusion_evidence is True


def test_attempt_01_uses_public_identity_for_run_metadata():
    config = load_attempt_config("configs/attempts/attempt-01.yaml")

    assert config.output_root == "data/beetle/runs/attempt-01"
    assert config.tags == ["beetle", "virchow2", "attempt-01"]


def test_attempt_01_provenance_uses_public_identity():
    provenance = Path("provenance/attempts/attempt-01")
    checkpoints = json.loads((provenance / "checkpoints.json").read_text())
    submission = json.loads((provenance / "submission.json").read_text())
    executions = json.loads((provenance / "executions.json").read_text())

    assert checkpoints["attempt_id"] == "attempt-01"
    assert "arm" not in checkpoints
    assert [entry["path"] for entry in checkpoints["checkpoints"]] == [
        f"fold_{fold}/best_model.pt" for fold in range(5)
    ]
    assert submission["attempt_id"] == "attempt-01"
    assert "selected_arm" not in submission
    assert "run_id" not in executions["historical_training"]


def test_project_pins_the_merged_soma_runtime():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    soma_dependency = next(
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("soma-pathology")
    )

    assert soma_dependency.endswith("@0ce1c688a7fbb8f0659a18d157aa2b5e2edfc05e")
