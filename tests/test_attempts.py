from pathlib import Path
import tomllib

from beetle.attempts import load_attempt_config


def test_uniform_attempt_resolves_to_the_submitted_scientific_protocol():
    config = load_attempt_config("configs/attempts/uniform.yaml")

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


def test_project_pins_the_merged_soma_runtime():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    soma_dependency = next(
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("soma-pathology")
    )

    assert soma_dependency.endswith("@0ce1c688a7fbb8f0659a18d157aa2b5e2edfc05e")
