import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from beetle.attempt_02 import (
    assemble_release_archives,
    build_decoder_depth_report,
    capture_environment_provenance,
    _probe_candidate_worker,
    probe_candidate,
    run_preflight,
    run_training,
    validate_cache_payloads,
    validate_completed_run,
)


def test_probe_candidate_worker_returns_worker_json(monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout='{"passed": true, "peak_allocated_bytes": 123}',
        stderr="",
    )
    monkeypatch.setattr("beetle.attempt_02.subprocess.run", lambda *args, **kwargs: completed)

    assert _probe_candidate_worker(32, 2, device="cuda:0", timeout_seconds=7) == {
        "passed": True,
        "peak_allocated_bytes": 123,
    }


def test_probe_candidate_worker_records_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=7)

    monkeypatch.setattr("beetle.attempt_02.subprocess.run", timeout)

    assert _probe_candidate_worker(32, 2, device="cuda:0", timeout_seconds=7) == {
        "passed": False,
        "error_type": "TimeoutExpired",
        "error": "decoder probe exceeded 7 seconds",
    }


def test_environment_provenance_records_exact_runtime_identity(tmp_path, monkeypatch):
    import datetime
    import torch

    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 31, 12, 0, tzinfo=tz)

    distribution = SimpleNamespace(
        version="1.11.2",
        read_text=lambda name: json.dumps(
            {"vcs_info": {"commit_id": "soma-commit", "requested_revision": "soma-ref"}}
        ),
    )
    monkeypatch.setattr("beetle.attempt_02.distribution", lambda name: distribution)
    monkeypatch.setattr("beetle.attempt_02.datetime.datetime", FixedDateTime)
    monkeypatch.setattr("beetle.attempt_02.platform.python_version", lambda: "3.11.15")
    monkeypatch.setattr("beetle.attempt_02.platform.platform", lambda: "test-platform")
    monkeypatch.setattr("beetle.attempt_02.sys.executable", "/python")
    monkeypatch.setattr(torch, "__version__", "2.7.1+cu128")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90701)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "H200")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=150)
    )
    output = tmp_path / "environment.json"

    result = capture_environment_provenance(
        output_path=output, repository_commit="repo-commit"
    )

    assert result == {
        "schema_version": 1,
        "attempt_id": "attempt-02",
        "captured_at": "2026-08-31T12:00:00+00:00",
        "python": {"version": "3.11.15", "executable": "/python", "platform": "test-platform"},
        "soma": {"version": "1.11.2", "commit": "soma-commit", "requested_revision": "soma-ref"},
        "torch": {"version": "2.7.1+cu128", "cuda": "12.8", "cudnn": 90701},
        "gpus": [{"index": 0, "name": "H200", "total_memory_bytes": 150}],
        "repository_commit": "repo-commit",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == result


def _write_attempt_01_identity(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "beetle"
    manifest_dir = data_dir / "curated_slide_manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "dataset.csv").write_text("dataset\n", encoding="utf-8")
    (manifest_dir / "splits.csv").write_text("splits\n", encoding="utf-8")

    cache_root_name = (
        "virchow2_3158645804b69e3f3bc4439d4116edddf0840a72_"
        "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b_"
        "dense_fp16"
    )
    cache_root = data_dir / "cache" / cache_root_name
    feature_dir = cache_root / "dense/verified-namespace/dense_embeddings"
    feature_dir.mkdir(parents=True)
    payload_manifest = data_dir / "cache_payload_sha256_openslide.txt"
    payload_manifest.write_text("payload manifest\n", encoding="utf-8")
    identity = {
        "schema_version": 1,
        "status": "completed",
        "algorithm": "sha256",
        "feature_dir": str(feature_dir),
        "manifest_path": str(payload_manifest),
        "manifest_sha256": (
            "0424bf43bd3c56e8fa998718b44a24c4bcdc3506ac5b3c5d6121a5085b91bccf"
        ),
        "manifest_entries": 2,
        "payload_bytes": 20,
        "tensor_files": 1,
        "sidecar_files": 1,
    }
    identity_path = data_dir / "cache_payload_sha256_openslide.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    lock = {
        "schema_version": 1,
        "attempt_id": "attempt-01",
        "dataset_sha256": (
            "6b30f95b2f4c06ff5b7cc6d3b1c617745743c5c214966d9b978eaa4f48b5adae"
        ),
        "splits_sha256": (
            "83b74af247b2d7f85fd5b7688af2dce7a49f2f83ac9f953d28be8fc99cf6c131"
        ),
        "cache_root_name": cache_root_name,
        "feature_namespace": "dense/verified-namespace/dense_embeddings",
        "identity_file": "cache_payload_sha256_openslide.json",
        "manifest_file": "cache_payload_sha256_openslide.txt",
        "manifest_sha256": (
            "0424bf43bd3c56e8fa998718b44a24c4bcdc3506ac5b3c5d6121a5085b91bccf"
        ),
        "manifest_entries": 2,
        "payload_bytes": 20,
        "tensor_files": 1,
        "sidecar_files": 1,
        "roi_draws_per_epoch_by_fold": {
            "0": 640,
            "1": 576,
            "2": 512,
            "3": 448,
            "4": 384,
        },
    }
    lock_path = tmp_path / "attempt-01-cache-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return data_dir, lock_path


def test_preflight_proves_attempt_01_cache_and_protocol_reuse(tmp_path):
    data_dir, lock_path = _write_attempt_01_identity(tmp_path)

    result = run_preflight(
        data_dir=data_dir,
        cache_lock_path=lock_path,
        output_dir=tmp_path / "preflight",
        probe=lambda physical_batch_size, accumulation_steps: {"passed": True},
    )

    assert result["scientific_protocol"] == {
        "baseline_attempt_id": "attempt-01",
        "candidate_attempt_id": "attempt-02",
        "differences": [
            {
                "path": "decoder.params.num_upsample_blocks",
                "attempt-01": 2,
                "attempt-02": 4,
            }
        ],
        "sampling_policy": "uniform",
        "roi_draws_per_epoch_by_fold": {
            "0": 640,
            "1": 576,
            "2": 512,
            "3": 448,
            "4": 384,
        },
    }
    assert result["cache_reuse"] == {
        "source_attempt_id": "attempt-01",
        "dataset_sha256": (
            "6b30f95b2f4c06ff5b7cc6d3b1c617745743c5c214966d9b978eaa4f48b5adae"
        ),
        "splits_sha256": (
            "83b74af247b2d7f85fd5b7688af2dce7a49f2f83ac9f953d28be8fc99cf6c131"
        ),
        "cache_root_name": (
            "virchow2_3158645804b69e3f3bc4439d4116edddf0840a72_"
            "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b_"
            "dense_fp16"
        ),
        "feature_namespace": "dense/verified-namespace/dense_embeddings",
        "manifest_sha256": (
            "0424bf43bd3c56e8fa998718b44a24c4bcdc3506ac5b3c5d6121a5085b91bccf"
        ),
        "manifest_entries": 2,
        "payload_bytes": 20,
        "tensor_files": 1,
        "sidecar_files": 1,
        "verified": True,
    }
    saved = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert saved == result


def test_preflight_refuses_attempt_01_cache_drift_before_probing(tmp_path):
    data_dir, lock_path = _write_attempt_01_identity(tmp_path)
    (data_dir / "cache_payload_sha256_openslide.txt").write_text(
        "different payload manifest\n", encoding="utf-8"
    )
    probed = []

    with pytest.raises(ValueError, match="cache drift"):
        run_preflight(
            data_dir=data_dir,
            cache_lock_path=lock_path,
            output_dir=tmp_path / "preflight",
            probe=lambda physical, accumulation: probed.append(
                (physical, accumulation)
            ),
        )

    assert probed == []


def test_preflight_probes_in_order_and_freezes_largest_passing_candidate(tmp_path):
    data_dir, lock_path = _write_attempt_01_identity(tmp_path)
    calls = []

    def probe(physical_batch_size, accumulation_steps):
        calls.append((physical_batch_size, accumulation_steps))
        return {"passed": physical_batch_size <= 32}

    result = run_preflight(
        data_dir=data_dir,
        cache_lock_path=lock_path,
        output_dir=tmp_path / "preflight",
        probe=probe,
    )

    assert calls == [(64, 1), (32, 2), (16, 4), (8, 8)]
    assert result["execution"]["selected"] == {
        "physical_batch_size": 32,
        "accumulation_steps": 2,
        "effective_batch_size": 64,
    }
    assert result["execution"]["frozen_fold_ids"] == [0, 1, 2, 3, 4]
    assert result["scientific_protocol"]["roi_draws_per_epoch_by_fold"] == {
        "0": 640,
        "1": 576,
        "2": 512,
        "3": 448,
        "4": 384,
    }
    resolved = yaml.safe_load(
        (tmp_path / "preflight/resolved/attempt-02.yaml").read_text(encoding="utf-8")
    )
    assert resolved["training"]["batch_size"] == 64
    assert resolved["training"]["gradient_accumulation"] == 1


def test_decoder_probe_performs_one_effective_batch_optimizer_step():
    result = probe_candidate(
        physical_batch_size=1,
        accumulation_steps=2,
        device="cpu",
        optimizer_steps=1,
    )

    assert result["passed"] is True
    assert result["physical_batch_size"] == 1
    assert result["accumulation_steps"] == 2
    assert result["effective_batch_size"] == 2
    assert result["optimizer_steps"] == 1
    assert result["microbatches"] == 2
    assert result["num_upsample_blocks"] == 4
    assert result["parameters_changed"] is True
    assert result["feature_shape"] == [1, 1280, 37, 37]
    assert result["logits_shape"] == [1, 4, 512, 512]


def test_cache_payload_gate_requires_complete_strict_reuse(tmp_path):
    feature_dir = tmp_path / "cache/dense/key/dense_embeddings"
    feature_dir.mkdir(parents=True)
    for sample_id in ("roi-0", "roi-1"):
        (feature_dir / f"{sample_id}.pt").write_bytes(b"tensor")
        (feature_dir / f"{sample_id}.meta.json").write_text("{}", encoding="utf-8")
    store = SimpleNamespace(feature_dir=feature_dir)
    dataset = SimpleNamespace(sample_ids=("roi-0", "roi-1"))
    calls = []

    def strict_cache_context(config_path, work_dir):
        calls.append((Path(config_path), Path(work_dir)))
        return SimpleNamespace(feature_store=store, dataset=dataset)

    output = tmp_path / "strict-cache-validation.json"
    result = validate_cache_payloads(
        config_path=tmp_path / "attempt-02.yaml",
        work_dir=tmp_path / "validation-work",
        output_path=output,
        strict_cache_context=strict_cache_context,
    )

    assert calls == [
        (tmp_path / "attempt-02.yaml", tmp_path / "validation-work")
    ]
    assert result == {
        "schema_version": 1,
        "status": "completed",
        "reuse_policy": "strict",
        "payload_validation": True,
        "feature_dir": str(feature_dir.resolve()),
        "roi_grids": 2,
        "tensor_files": 2,
        "sidecar_files": 2,
        "payload_bytes": 12,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_training_launch_requires_both_completed_pretraining_gates(tmp_path, monkeypatch):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "completed",
                "attempt_id": "attempt-02",
                "cache_reuse": {
                    "verified": True,
                    "cache_root_name": "locked-cache",
                    "feature_namespace": "dense/key/dense_embeddings",
                    "manifest_entries": 10,
                    "payload_bytes": 100,
                    "tensor_files": 5,
                    "sidecar_files": 5,
                },
                "execution": {
                    "selected": {
                        "physical_batch_size": 32,
                        "accumulation_steps": 2,
                        "effective_batch_size": 64,
                    },
                    "frozen_fold_ids": [0, 1, 2, 3, 4],
                },
            }
        ),
        encoding="utf-8",
    )
    validation = tmp_path / "strict-cache-validation.json"
    validation.write_text(
        json.dumps(
            {
                "status": "completed",
                "reuse_policy": "strict",
                "payload_validation": True,
                "feature_dir": "/data/locked-cache/dense/key/dense_embeddings",
                "roi_grids": 5,
                "tensor_files": 5,
                "sidecar_files": 5,
                "payload_bytes": 100,
            }
        ),
        encoding="utf-8",
    )
    from soma.config import save_config
    from beetle.attempts import load_attempt_config

    resolved_dir = tmp_path / "resolved"
    resolved_dir.mkdir()
    save_config(
        load_attempt_config("configs/attempts/attempt-02.yaml"),
        resolved_dir / "attempt-02.yaml",
    )
    launched = []
    monkeypatch.chdir(tmp_path)

    result = run_training(
        preflight_path=preflight,
        strict_validation_path=validation,
        run_id="attempt-02-test",
        trainer=lambda config: launched.append(config) or "finished",
    )

    assert result == "finished"
    assert len(launched) == 1
    config = launched[0]
    assert config.run_id == "attempt-02-test"
    assert config.decoder.params["num_upsample_blocks"] == 4
    assert config.training.batch_size == 32
    assert config.training.gradient_accumulation == 2

    invalid = json.loads(preflight.read_text(encoding="utf-8"))
    invalid["execution"]["frozen_fold_ids"] = [0, 1, 2, 3]
    preflight.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="pretraining gate"):
        run_training(
            preflight_path=preflight,
            strict_validation_path=validation,
            run_id="must-not-launch",
            trainer=lambda config: pytest.fail("invalid evidence launched training"),
        )


def test_training_launch_refuses_protocol_drift_after_preflight(tmp_path, monkeypatch):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "status": "completed", "attempt_id": "attempt-02",
        "cache_reuse": {"verified": True, "cache_root_name": "cache", "feature_namespace": "features", "tensor_files": 1, "sidecar_files": 1, "payload_bytes": 10},
        "execution": {"selected": {"physical_batch_size": 64, "accumulation_steps": 1, "effective_batch_size": 64}, "frozen_fold_ids": [0, 1, 2, 3, 4]},
    }), encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({
        "status": "completed", "reuse_policy": "strict", "payload_validation": True,
        "feature_dir": "/cache/features", "roi_grids": 1, "tensor_files": 1, "sidecar_files": 1, "payload_bytes": 10,
    }), encoding="utf-8")
    resolved = tmp_path / "resolved/attempt-02.yaml"
    resolved.parent.mkdir()
    resolved.write_text("resolved", encoding="utf-8")
    monkeypatch.setattr("soma.config.load_config", lambda path: "preflight-protocol")
    monkeypatch.setattr("beetle.attempt_02.load_attempt_config", lambda path, overrides=None: "current-protocol")
    monkeypatch.setattr("beetle.attempt_02._scientific_protocol", lambda value: {"identity": value})

    with pytest.raises(ValueError, match="protocol drift"):
        run_training(
            preflight_path=preflight,
            strict_validation_path=validation,
            run_id="must-not-launch",
            trainer=lambda config: pytest.fail("protocol drift launched training"),
        )


def test_completed_run_gate_requires_every_fold_artifact_and_records_checksums(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_bytes(b"resolved config")
    environment = tmp_path / "environment.json"
    environment.write_bytes(b"environment provenance")
    required = (
        "best_model.pt",
        "training_history.json",
        "roi_batch_sampling.json",
        "confusion_evidence_tune.json",
        "metrics.json",
        "segmentation_roi_population.json",
    )
    for fold in range(5):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir()
        for name in required:
            (fold_dir / name).write_bytes(f"fold {fold} {name}".encode())

    result = validate_completed_run(run_dir=run_dir, environment_path=environment)

    assert result["status"] == "completed"
    assert list(result["folds"]) == ["0", "1", "2", "3", "4"]
    assert result["folds"]["0"]["best_model.pt"] == {
        "bytes": 20,
        "sha256": "69c7d7bc78ea42e758a25773d3b421b248e3bf22d76088b6a5fc647caa6ca851",
    }
    assert result["resolved_config"]["bytes"] == 15
    assert result["environment_provenance"]["bytes"] == 22

    (run_dir / "fold_4/confusion_evidence_tune.json").unlink()
    with pytest.raises(ValueError, match="fold 4.*confusion_evidence_tune.json"):
        validate_completed_run(run_dir=run_dir, environment_path=environment)


def test_release_archives_contain_five_checkpoints_and_compact_evidence(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_bytes(b"resolved config")
    for fold in range(5):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir()
        for name in (
            "best_model.pt",
            "training_history.json",
            "roi_batch_sampling.json",
            "confusion_evidence_tune.json",
            "metrics.json",
            "segmentation_roi_population.json",
        ):
            (fold_dir / name).write_bytes(f"fold {fold} {name}".encode())
    preflight_dir = tmp_path / "preflight"
    (preflight_dir / "resolved").mkdir(parents=True)
    preflight = preflight_dir / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    (preflight_dir / "resolved/attempt-01.yaml").write_text("baseline", encoding="utf-8")
    (preflight_dir / "resolved/attempt-02.yaml").write_text("candidate", encoding="utf-8")
    strict = tmp_path / "strict-cache-validation.json"
    strict.write_text("{}", encoding="utf-8")
    report = tmp_path / "decoder_depth_report.json"
    report.write_text("{}", encoding="utf-8")
    environment = tmp_path / "environment.json"
    environment.write_text("{}", encoding="utf-8")

    result = assemble_release_archives(
        run_dir=run_dir,
        preflight_path=preflight,
        strict_validation_path=strict,
        report_path=report,
        environment_path=environment,
        output_dir=tmp_path / "release",
    )

    assert result["status"] == "completed"
    with zipfile.ZipFile(result["weights_archive"]["path"]) as archive:
        assert archive.namelist() == [
            "config.yaml",
            "fold_0/best_model.pt",
            "fold_1/best_model.pt",
            "fold_2/best_model.pt",
            "fold_3/best_model.pt",
            "fold_4/best_model.pt",
            "artifact_checksums.json",
        ]
    with zipfile.ZipFile(result["evidence_archive"]["path"]) as archive:
        names = archive.namelist()
    assert "decoder_depth_report.json" in names
    assert "resolved/attempt-01.yaml" in names
    assert "resolved/attempt-02.yaml" in names
    assert "fold_4/training_history.json" in names
    assert "fold_4/confusion_evidence_tune.json" in names
    assert "fold_4/metrics.json" in names
    assert "fold_4/segmentation_roi_population.json" in names
    assert "fold_4/sampler_audit.json" not in names


def test_decoder_depth_report_records_paired_and_patient_level_endpoints(tmp_path):
    vocabulary = [
        "other",
        "non_invasive_epithelium",
        "invasive_epithelium",
        "necrosis",
    ]
    baseline_matrix = [
        [4, 1, 0, 0],
        [0, 4, 1, 0],
        [0, 0, 4, 1],
        [1, 0, 0, 4],
    ]
    candidate_matrix = [
        [5, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 5],
    ]
    patients = []
    evidence_paths = []
    mapping_lines = ["sample_id,patient_id"]
    for fold in range(5):
        patient_id = f"patient-{fold}"
        sample_id = f"roi-{fold}"
        patients.append(
            {
                "patient_id": patient_id,
                "fold": fold,
                "confusion_matrix": baseline_matrix,
            }
        )
        evidence = tmp_path / f"fold-{fold}.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "records": [
                        {
                            "sample_id": sample_id,
                            "fold": fold,
                            "class_vocabulary": vocabulary,
                            "confusion_matrix": candidate_matrix,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        evidence_paths.append(evidence)
        mapping_lines.append(f"{sample_id},{patient_id}")
    baseline = tmp_path / "attempt-01.json"
    baseline.write_text(
        json.dumps(
            {
                "attempt_id": "attempt-01",
                "folds": {
                    str(fold): {"mean_dice": 0.8} for fold in range(5)
                },
                "patients": patients,
            }
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "roi_manifest.csv"
    mapping.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")

    report = build_decoder_depth_report(
        attempt_01_report=baseline,
        attempt_02_evidence=evidence_paths,
        sample_patient_csv=mapping,
        spacing_exception_patient_ids=("patient-4",),
        primary_patient_count=5,
        sensitivity_patient_count=4,
        bootstrap_draws=20,
    )

    assert report["attempts"] == {
        "attempt-01": {
            "decoder_upsample_blocks": 2,
            "fold_scores": [0.8, 0.8, 0.8, 0.8, 0.8],
            "mean": 0.8,
            "sample_standard_deviation": 0.0,
        },
        "attempt-02": {
            "decoder_upsample_blocks": 4,
            "fold_scores": [1.0, 1.0, 1.0, 1.0, 1.0],
            "mean": 1.0,
            "sample_standard_deviation": 0.0,
        },
    }
    assert report["paired_fold_deltas"] == [0.2, 0.2, 0.2, 0.2, 0.2]
    assert report["pooled_metrics"]["attempt-01"] == {
        "dice_per_class": dict.fromkeys(vocabulary, 0.8),
        "macro_dice": 0.8,
        "pixel_micro_dice": 0.8,
    }
    assert report["pooled_metrics"]["attempt-02"] == {
        "dice_per_class": dict.fromkeys(vocabulary, 1.0),
        "macro_dice": 1.0,
        "pixel_micro_dice": 1.0,
    }
    assert report["patient_bootstrap"]["primary_527_patient"]["patient_count"] == 5
    assert report["patient_bootstrap"]["derived_524_patient"] == {
        "patient_count": 4,
        "evaluation_only": True,
        "excluded_patient_ids": ["patient-4"],
        "attempts": {
            "attempt-01": {
                "seed": 0,
                "draws": 20,
                "macro_dice_percentile_95_ci": [0.8, 0.8],
            },
            "attempt-02": {
                "seed": 0,
                "draws": 20,
                "macro_dice_percentile_95_ci": [1.0, 1.0],
            },
        },
    }
    assert report["formal_comparator"] == "attempt-01"
    assert report["historical_motivation"] == {
        "source": "clemsgrs/soma#127",
        "role": "historical motivation only; not a formal comparator",
    }
    assert report["external_model_superseded"] is False
    assert report["interpretation"] == (
        "Attempt 02's five-fold mean was 0.200000 higher than Attempt 01. "
        "This decoder-depth ablation is complete regardless of direction and does not "
        "select or supersede an External model."
    )
