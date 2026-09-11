import json

import pytest

from beetle import attempt_03


@pytest.fixture
def locked_cache(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "curated_slide_manifest"
    manifest_dir.mkdir()
    (manifest_dir / "dataset.csv").write_text("dataset\n")
    (manifest_dir / "splits.csv").write_text("splits\n")
    features = tmp_path / "cache/encoder/dense"
    features.mkdir(parents=True)
    tensor = features / "roi.pt"
    tensor.write_bytes(b"original")
    sidecar = features / "roi.meta.json"
    sidecar.write_text("{}")
    manifest = tmp_path / "hashes.txt"
    manifest.write_text("".join(
        f"{attempt_03.sha256(p)}  ./{p.name}\n" for p in (tensor, sidecar)
    ))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "dataset_sha256": attempt_03.sha256(manifest_dir / "dataset.csv"),
        "splits_sha256": attempt_03.sha256(manifest_dir / "splits.csv"),
        "manifest_file": "hashes.txt", "manifest_sha256": attempt_03.sha256(manifest),
        "cache_root_name": "encoder", "feature_namespace": "dense",
        "manifest_entries": 2, "tensor_files": 1, "sidecar_files": 1,
        "payload_bytes": 8,
    }))
    monkeypatch.setattr(attempt_03, "LOCK", lock)
    return tmp_path, features


def test_full_cache_audit_records_verified_counts(locked_cache):
    data, _ = locked_cache
    report = attempt_03.validate_cache(data, data / "audit.json")
    assert report["status"] == "completed"
    assert report["tensor_files"] == report["sidecar_files"] == 1
    assert report["payload_bytes"] == 8


@pytest.mark.parametrize("drift", ["same_size_corruption", "missing", "extra", "splits"])
def test_cache_audit_refuses_drift(locked_cache, drift):
    data, features = locked_cache
    if drift == "same_size_corruption":
        (features / "roi.pt").write_bytes(b"modified")
    elif drift == "missing":
        (features / "roi.pt").unlink()
    elif drift == "extra":
        (features / "extra.pt").write_bytes(b"extra")
    else:
        (data / "curated_slide_manifest/splits.csv").write_text("changed\n")
    with pytest.raises(ValueError):
        attempt_03.validate_cache(data, data / "audit.json")
    assert not (data / "audit.json").exists()


def test_cache_failure_invalidates_previous_success(locked_cache):
    data, features = locked_cache
    output = data / "audit.json"
    attempt_03.validate_cache(data, output)
    (features / "roi.pt").write_bytes(b"modified")
    with pytest.raises(ValueError):
        attempt_03.validate_cache(data, output)
    assert not output.exists()


def test_prepare_preserves_protocol_and_uses_verified_runtime_paths(locked_cache):
    from soma.config import load_config, save_config

    data, _ = locked_cache
    evidence = data / "evidence"
    attempt_03.validate_cache(data, evidence / "cache-validation.json")
    config = attempt_03.validate_protocol()
    save_config(config, evidence / "resolved-attempt-03.yaml")
    selected = {"physical_batch_size": 32, "accumulation_steps": 2, "effective_batch_size": 64}
    attempt_03.write_json(evidence / "preflight.json", {
        "status": "completed", "attempt_id": "attempt-03",
        "execution": {"selected": selected, "candidates": [dict(selected, passed=True)]},
    })
    launch = attempt_03.prepare(data, evidence, data / "runs")
    resolved = load_config(launch)
    assert resolved.decoder == config.decoder
    assert resolved.dataset_csv == str(data / "curated_slide_manifest/dataset.csv")
    assert resolved.cache.root_dir == str(data / "cache/encoder")
    assert resolved.training.batch_size == 32
    assert resolved.training.gradient_accumulation == 2
    assert resolved.evaluation == config.evaluation
    assert resolved.training.learning_rate == config.training.learning_rate

    (data / "curated_slide_manifest/splits.csv").write_text("drift\n")
    with pytest.raises(ValueError, match="splits.csv changed"):
        attempt_03.prepare(data, evidence, data / "runs")
