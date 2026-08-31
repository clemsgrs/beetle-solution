import json
from dataclasses import asdict
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


def test_attempt_02_changes_only_decoder_depth_in_the_scientific_protocol():
    attempt_01 = asdict(load_attempt_config("configs/attempts/attempt-01.yaml"))
    attempt_02 = asdict(load_attempt_config("configs/attempts/attempt-02.yaml"))

    assert attempt_02["decoder"]["params"]["num_upsample_blocks"] == 4
    assert attempt_02["training"]["roi_batch_sampling"] == "uniform"
    assert attempt_02["output_root"] == "data/beetle/runs/attempt-02"
    assert attempt_02["tags"] == ["beetle", "virchow2", "attempt-02"]

    attempt_01["decoder"]["params"]["num_upsample_blocks"] = 4
    attempt_01["output_root"] = "data/beetle/runs/attempt-02"
    attempt_01["tags"] = ["beetle", "virchow2", "attempt-02"]
    assert attempt_02 == attempt_01


def test_attempt_02_locks_the_verified_attempt_01_cache_identity():
    lock = json.loads(
        Path("configs/attempts/attempt-02-cache-lock.json").read_text(encoding="utf-8")
    )

    assert lock == {
        "schema_version": 1,
        "attempt_id": "attempt-01",
        "dataset_sha256": (
            "7ac6294ca5d6e41a45eda0eb8669daab37673b35b59fe9de4033aa629f6d6cf5"
        ),
        "splits_sha256": (
            "2fa211abdf0d5af41a95a9e403e76db0c8d81083a5607267da9da941624a2ae6"
        ),
        "cache_root_name": (
            "virchow2_3158645804b69e3f3bc4439d4116edddf0840a72_"
            "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b_"
            "dense_fp16"
        ),
        "feature_namespace": "dense/95ab55f548038c00/dense_embeddings",
        "identity_file": "cache_payload_sha256_openslide.json",
        "manifest_file": "cache_payload_sha256_openslide.txt",
        "manifest_sha256": (
            "3efe739ae37f82f41dce62451101c60cfe3dca3ee87ec0e70ca13564f3e5de18"
        ),
        "manifest_entries": 249394,
        "payload_bytes": 437290638590,
        "tensor_files": 124697,
        "sidecar_files": 124697,
        "roi_draws_per_epoch_by_fold": {
            "0": 76480,
            "1": 73792,
            "2": 73984,
            "3": 76864,
            "4": 72832,
        },
    }


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

    assert soma_dependency.endswith("@4a3d6c84f9a3dc2832e585c6884a95d14a0f79bc")
