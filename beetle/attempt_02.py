"""Attempt 02 preflight, execution, and evidence assembly."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Callable, Sequence

from beetle.attempts import REPO_ROOT, load_attempt_config
from beetle.comparison import (
    SPACING_EXCEPTION_PATIENT_IDS,
    compare_attempts,
    load_confusion_evidence,
    load_cv_results,
    read_sample_patients,
)
from beetle.release import assemble_release, validate_completed_run

ATTEMPT_01_CONFIG = REPO_ROOT / "configs/attempts/attempt-01.yaml"
ATTEMPT_02_CONFIG = REPO_ROOT / "configs/attempts/attempt-02.yaml"
DEFAULT_CACHE_LOCK = REPO_ROOT / "configs/attempts/attempt-02-cache-lock.json"
BATCH_CANDIDATES = ((64, 1), (32, 2), (16, 4), (8, 8))
def _strict_cache_context(config_path: Path, work_dir: Path):
    from soma.config import load_config
    from soma.pipeline import Pipeline

    config = load_config(config_path)
    if config.cache.reuse_policy != "strict" or config.cache.validate_payloads is not True:
        raise ValueError(
            "Attempt 02 cache validation requires reuse_policy=strict and "
            "validate_payloads=true"
        )
    return Pipeline(config)._build_slide_manifest_dense_context(run_dir=work_dir)


def validate_cache_payloads(
    *,
    config_path: str | Path,
    work_dir: str | Path,
    output_path: str | Path,
    strict_cache_context: Callable[[Path, Path], object] | None = None,
) -> dict:
    """Resolve the exact cache with Soma's strict full-payload validation enabled."""
    config_path = Path(config_path)
    work_dir = Path(work_dir)
    output_path = Path(output_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    context_builder = strict_cache_context or _strict_cache_context
    context = context_builder(config_path, work_dir)
    sample_ids = list(context.dataset.sample_ids)
    store = context.feature_store
    store.validate_coverage(sample_ids) if hasattr(store, "validate_coverage") else None
    feature_dir = Path(store.feature_dir).resolve()
    tensors = sorted(feature_dir.rglob("*.pt"))
    sidecars = sorted(feature_dir.rglob("*.meta.json"))
    if len(tensors) != len(sample_ids) or len(sidecars) != len(sample_ids):
        raise ValueError(
            "Attempt 02 strict cache validation found incomplete payload coverage: "
            f"ROIs={len(sample_ids)}, tensors={len(tensors)}, sidecars={len(sidecars)}"
        )
    result = {
        "schema_version": 1,
        "status": "completed",
        "reuse_policy": "strict",
        "payload_validation": True,
        "feature_dir": str(feature_dir),
        "roi_grids": len(sample_ids),
        "tensor_files": len(tensors),
        "sidecar_files": len(sidecars),
        "payload_bytes": sum(path.stat().st_size for path in tensors),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return result


def run_training(
    *,
    preflight_path: str | Path,
    strict_validation_path: str | Path,
    run_id: str,
    trainer: Callable[[object], object] | None = None,
):
    """Launch all folds only after the immutable pretraining evidence is complete."""
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    validation = json.loads(
        Path(strict_validation_path).read_text(encoding="utf-8")
    )
    cache = preflight.get("cache_reuse", {})
    execution = preflight.get("execution", {})
    selected = execution.get("selected", {})
    selected_pair = (
        selected.get("physical_batch_size"),
        selected.get("accumulation_steps"),
    )
    valid_selection = (
        selected_pair in BATCH_CANDIDATES
        and selected.get("effective_batch_size") == 64
    )
    expected_feature_suffix = str(
        Path(str(cache.get("cache_root_name", "")))
        / str(cache.get("feature_namespace", ""))
    )
    valid = (
        preflight.get("status") == "completed"
        and preflight.get("attempt_id") == "attempt-02"
        and cache.get("verified") is True
        and valid_selection
        and execution.get("frozen_fold_ids") == [0, 1, 2, 3, 4]
        and validation.get("status") == "completed"
        and validation.get("reuse_policy") == "strict"
        and validation.get("payload_validation") is True
        and str(validation.get("feature_dir", "")).endswith(expected_feature_suffix)
        and validation.get("tensor_files") == cache.get("tensor_files")
        and validation.get("sidecar_files") == cache.get("sidecar_files")
        and validation.get("payload_bytes") == cache.get("payload_bytes")
        and validation.get("roi_grids") == cache.get("tensor_files")
    )
    if not valid:
        raise ValueError(
            "Attempt 02 pretraining gate is incomplete or disagrees with the locked cache"
        )
    from soma.config import load_config

    resolved_path = Path(preflight_path).parent / "resolved/attempt-02.yaml"
    if not resolved_path.is_file():
        raise ValueError("Attempt 02 pretraining gate is missing resolved protocol evidence")
    resolved = load_config(resolved_path)
    current = load_attempt_config(ATTEMPT_02_CONFIG)
    if _scientific_protocol(resolved) != _scientific_protocol(current):
        raise ValueError("Attempt 02 protocol drift after preflight")
    config = load_attempt_config(
        ATTEMPT_02_CONFIG,
        overrides={
            "run": {"run_id": run_id},
            "training": {
                "batch_size": selected["physical_batch_size"],
                "gradient_accumulation": selected["accumulation_steps"],
            },
        },
    )
    if (
        config.decoder.params.get("num_upsample_blocks") != 4
        or config.training.batch_size != selected["physical_batch_size"]
        or config.training.gradient_accumulation != selected["accumulation_steps"]
    ):
        raise ValueError("Attempt 02 pretraining gate disagrees with the launch config")
    if trainer is not None:
        return trainer(config)

    from soma.pipeline import Pipeline

    result = Pipeline(config).run()
    from beetle.record import record_training

    record_training("attempt-02", result.run_dir, config)
    return result


def build_decoder_depth_report(
    *,
    attempt_01_report: str | Path,
    attempt_02_evidence: Sequence[str | Path],
    sample_patient_csv: str | Path,
    spacing_exception_patient_ids: Sequence[str],
    primary_patient_count: int = 527,
    sensitivity_patient_count: int = 524,
    bootstrap_draws: int = 10_000,
) -> dict:
    """Build the paired Attempt 02 endpoint from held-out confusion evidence."""
    _, baseline_patients, baseline_fold_scores = load_cv_results(attempt_01_report)
    vocabulary, candidate_patients, candidate_fold_scores = load_confusion_evidence(
        attempt_02_evidence,
        read_sample_patients(sample_patient_csv),
        label="Attempt 02",
    )
    report = compare_attempts(
        {
            "attempt-01": (baseline_fold_scores, baseline_patients),
            "attempt-02": (candidate_fold_scores, candidate_patients),
        },
        vocabulary=vocabulary,
        spacing_exception_patient_ids=spacing_exception_patient_ids,
        primary_patient_count=primary_patient_count,
        sensitivity_patient_count=sensitivity_patient_count,
        bootstrap_draws=bootstrap_draws,
    )
    attempts = report["attempts"]
    attempts["attempt-01"]["decoder_upsample_blocks"] = 2
    attempts["attempt-02"]["decoder_upsample_blocks"] = 4
    mean_delta = statistics.mean(candidate_fold_scores) - statistics.mean(
        baseline_fold_scores
    )
    direction = "higher" if mean_delta > 0 else "lower" if mean_delta < 0 else "unchanged"
    if direction == "unchanged":
        comparison = "Attempt 02's five-fold mean was unchanged from Attempt 01."
    else:
        comparison = (
            f"Attempt 02's five-fold mean was {abs(mean_delta):.6f} {direction} "
            "than Attempt 01."
        )
    # Published schema: the Attempt 02 release report hash depends on this shape.
    return {
        "schema_version": 1,
        "attempts": attempts,
        "paired_fold_deltas": report["comparisons"]["attempt-02_minus_attempt-01"][
            "paired_fold_deltas"
        ],
        "pooled_metrics": report["pooled_metrics"],
        "patient_bootstrap": {
            "primary_527_patient": report["patient_bootstrap"]["primary"],
            "derived_524_patient": report["patient_bootstrap"]["sensitivity"],
        },
        "formal_comparator": "attempt-01",
        "historical_motivation": {
            "source": "clemsgrs/soma#127",
            "role": "historical motivation only; not a formal comparator",
        },
        "external_model_superseded": False,
        "interpretation": (
            f"{comparison} This decoder-depth ablation is complete regardless of "
            "direction and does not select or supersede an External model."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_release_archives(
    *,
    run_dir: str | Path,
    preflight_path: str | Path,
    strict_validation_path: str | Path,
    report_path: str | Path,
    environment_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Package five weights and compact publication evidence after all-fold gating."""
    resolved_dir = Path(preflight_path).parent / "resolved"
    return assemble_release(
        attempt_id="attempt-02",
        run_dir=run_dir,
        environment_path=environment_path,
        evidence={
            "preflight.json": preflight_path,
            "strict-cache-validation.json": strict_validation_path,
            "decoder_depth_report.json": report_path,
            "environment.json": environment_path,
            "resolved/attempt-01.yaml": resolved_dir / "attempt-01.yaml",
            "resolved/attempt-02.yaml": resolved_dir / "attempt-02.yaml",
        },
        output_dir=output_dir,
    )


def capture_environment_provenance(
    *, output_path: str | Path, repository_commit: str | None = None
) -> dict:
    """Record the runtime and hardware identity used for the real five-fold run."""
    import torch

    soma_distribution = distribution("soma-pathology")
    direct_url = json.loads(soma_distribution.read_text("direct_url.json") or "{}")
    git_commit = repository_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "attempt_id": "attempt-02",
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "soma": {
            "version": soma_distribution.version,
            "commit": direct_url.get("vcs_info", {}).get("commit_id"),
            "requested_revision": direct_url.get("vcs_info", {}).get(
                "requested_revision"
            ),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    index
                ).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
        "repository_commit": git_commit,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _scientific_protocol(config) -> dict:
    payload = asdict(config)
    payload.pop("output_root")
    payload.pop("tags")
    return payload


def _differences(left, right, path: str = "") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[dict] = []
        for key in sorted(left.keys() | right.keys()):
            child_path = f"{path}.{key}" if path else key
            differences.extend(
                _differences(left.get(key), right.get(key), child_path)
            )
        return differences
    if left == right:
        return []
    return [{"path": path, "attempt-01": left, "attempt-02": right}]


def _validate_protocol() -> dict:
    baseline = load_attempt_config(ATTEMPT_01_CONFIG)
    candidate = load_attempt_config(ATTEMPT_02_CONFIG)
    differences = _differences(
        _scientific_protocol(baseline), _scientific_protocol(candidate)
    )
    expected = [
        {
            "path": "decoder.params.num_upsample_blocks",
            "attempt-01": 2,
            "attempt-02": 4,
        }
    ]
    if differences != expected:
        raise ValueError(
            "Attempt 02 protocol drift: expected only the two-to-four decoder-depth "
            f"change, observed {differences!r}"
        )
    if candidate.training.roi_batch_sampling != "uniform":
        raise ValueError("Attempt 02 protocol drift: sampling policy must remain uniform")
    return {
        "baseline_attempt_id": "attempt-01",
        "candidate_attempt_id": "attempt-02",
        "differences": differences,
        "sampling_policy": "uniform",
    }


def _validate_cache_reuse(data_dir: Path, cache_lock_path: Path) -> dict:
    lock = json.loads(cache_lock_path.read_text(encoding="utf-8"))
    identity_file = lock.get("identity_file", "")
    manifest_file = lock.get("manifest_file", "")
    if (
        not isinstance(identity_file, str)
        or Path(identity_file).name != identity_file
        or not isinstance(manifest_file, str)
        or Path(manifest_file).name != manifest_file
    ):
        raise ValueError("Attempt 02 cache drift: invalid locked evidence filename")
    identity_path = data_dir / identity_file
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    manifest_path = data_dir / manifest_file
    config = load_attempt_config(ATTEMPT_01_CONFIG)
    cache_root_name = Path(config.cache.root_dir).name
    feature_dir = data_dir / "cache" / cache_root_name / lock["feature_namespace"]

    observed = {
        "source_attempt_id": lock.get("attempt_id"),
        "dataset_sha256": _sha256(data_dir / "curated_slide_manifest/dataset.csv"),
        "splits_sha256": _sha256(data_dir / "curated_slide_manifest/splits.csv"),
        "cache_root_name": cache_root_name,
        "feature_namespace": lock.get("feature_namespace"),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_entries": identity.get("manifest_entries"),
        "payload_bytes": identity.get("payload_bytes"),
        "tensor_files": identity.get("tensor_files"),
        "sidecar_files": identity.get("sidecar_files"),
    }
    expected = {
        "source_attempt_id": "attempt-01",
        "dataset_sha256": lock.get("dataset_sha256"),
        "splits_sha256": lock.get("splits_sha256"),
        "cache_root_name": lock.get("cache_root_name"),
        "feature_namespace": lock.get("feature_namespace"),
        "manifest_sha256": lock.get("manifest_sha256"),
        "manifest_entries": lock.get("manifest_entries"),
        "payload_bytes": lock.get("payload_bytes"),
        "tensor_files": lock.get("tensor_files"),
        "sidecar_files": lock.get("sidecar_files"),
    }
    identity_feature_dir = Path(identity.get("feature_dir", "")).resolve()
    identity_manifest = Path(identity.get("manifest_path", "")).resolve()
    if (
        observed != expected
        or identity.get("status") != "completed"
        or identity.get("manifest_sha256") != expected["manifest_sha256"]
        or identity_feature_dir != feature_dir.resolve()
        or identity_manifest != manifest_path.resolve()
        or not feature_dir.is_dir()
    ):
        raise ValueError(
            "Attempt 02 cache drift: the supplied cache is not the verified "
            "Attempt 01 feature-cache identity"
        )
    return {**observed, "verified": True}


def probe_candidate(
    *,
    physical_batch_size: int,
    accumulation_steps: int,
    device: str,
    optimizer_steps: int = 2,
) -> dict:
    """Exercise the four-block decoder with complete effective-batch updates."""
    import torch

    from soma.decoders.registry import build_decoder_for_grid
    from soma.dense.geometry import compute_dense_geometry
    from soma.tasks.segmentation import SegmentationHead
    from soma.training.model import SegmentationModel

    if physical_batch_size < 1 or accumulation_steps < 1 or optimizer_steps < 1:
        raise ValueError("batch, accumulation, and optimizer-step counts must be positive")
    resolved_device = torch.device(device)
    torch.manual_seed(0)
    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()

    geometry = compute_dense_geometry(target_size=512, patch_size=14)
    decoder = build_decoder_for_grid(
        "lightweight_conv",
        {"hidden_dim": 256, "num_upsample_blocks": 4, "num_groups": 32},
        geometry=geometry,
        input_dim=1280,
        num_classes=4,
    )
    head = SegmentationHead(num_classes=4, geometry=geometry)
    model = SegmentationModel(decoder=decoder, task_head=head).to(resolved_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()
    features = torch.randn(
        physical_batch_size,
        1280,
        *geometry.grid_shape,
        device=resolved_device,
        dtype=torch.float32,
    )
    base_mask = (
        torch.arange(512 * 512, device=resolved_device, dtype=torch.long)
        .remainder(4)
        .reshape(512, 512)
    )
    masks = base_mask.unsqueeze(0).repeat(physical_batch_size, 1, 1)
    masks[:, -1, :] = 255

    started = perf_counter()
    final_loss = None
    logits_shape = None
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            output = model(features)
            loss = head.compute_loss(output.logits, {"mask": masks})
            (loss / accumulation_steps).backward()
            final_loss = float(loss.detach().cpu())
            logits_shape = list(output.logits.shape)
        optimizer.step()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    result = {
        "passed": True,
        "physical_batch_size": physical_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": physical_batch_size * accumulation_steps,
        "optimizer_steps": optimizer_steps,
        "microbatches": optimizer_steps * accumulation_steps,
        "num_upsample_blocks": 4,
        "parameters_changed": not torch.equal(before, first_parameter.detach()),
        "feature_shape": list(features.shape),
        "logits_shape": logits_shape,
        "final_loss": final_loss,
        "elapsed_seconds": perf_counter() - started,
    }
    if resolved_device.type == "cuda":
        props = torch.cuda.get_device_properties(None)
        result.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "total_memory_bytes": props.total_memory,
            }
        )
    return result


def _probe_candidate_worker(
    physical_batch_size: int,
    accumulation_steps: int,
    *,
    device: str,
    timeout_seconds: int,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "beetle.attempt_02",
        "probe-worker",
        "--physical-batch-size",
        str(physical_batch_size),
        "--accumulation-steps",
        str(accumulation_steps),
        "--device",
        device,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error_type": "TimeoutExpired",
            "error": f"decoder probe exceeded {timeout_seconds} seconds",
        }
    if completed.returncode:
        return {
            "passed": False,
            "error_type": "SubprocessError",
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "error_type": "SubprocessProtocolError",
            "error": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "passed": False,
            "error_type": "SubprocessProtocolError",
            "error": "probe worker did not return a JSON object",
        }
    return payload


def run_preflight(
    *,
    data_dir: str | Path,
    cache_lock_path: str | Path,
    output_dir: str | Path,
    probe: Callable[[int, int], dict] | None = None,
    device: str = "cuda:0",
    probe_timeout_seconds: int = 300,
) -> dict:
    """Refuse scientific/cache drift and freeze one execution batch for five folds."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _validate_protocol()
    cache_lock_path = Path(cache_lock_path)
    cache_reuse = _validate_cache_reuse(data_dir, cache_lock_path)
    cache_lock = json.loads(cache_lock_path.read_text(encoding="utf-8"))
    draw_budgets = cache_lock.get("roi_draws_per_epoch_by_fold")
    if (
        not isinstance(draw_budgets, dict)
        or set(draw_budgets) != {"0", "1", "2", "3", "4"}
        or any(
            isinstance(draws, bool)
            or not isinstance(draws, int)
            or draws < 1
            or draws % 64
            for draws in draw_budgets.values()
        )
    ):
        raise ValueError(
            "Attempt 02 protocol drift: the five Attempt 01 ROI draw budgets "
            "must be positive whole effective batches"
        )
    protocol["roi_draws_per_epoch_by_fold"] = draw_budgets
    from soma.config import save_config

    resolved_dir = output_dir / "resolved"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    save_config(load_attempt_config(ATTEMPT_01_CONFIG), resolved_dir / "attempt-01.yaml")
    save_config(load_attempt_config(ATTEMPT_02_CONFIG), resolved_dir / "attempt-02.yaml")
    report_probe_progress = probe is None
    if report_probe_progress:
        probe = lambda physical, accumulation: _probe_candidate_worker(
            physical,
            accumulation,
            device=device,
            timeout_seconds=probe_timeout_seconds,
        )
    attempts = []
    for physical_batch_size, accumulation_steps in BATCH_CANDIDATES:
        if report_probe_progress:
            print(
                f"probing {physical_batch_size}x{accumulation_steps} on {device}",
                flush=True,
            )
        attempt = dict(probe(physical_batch_size, accumulation_steps))
        attempts.append(
            {
                "physical_batch_size": physical_batch_size,
                "accumulation_steps": accumulation_steps,
                "effective_batch_size": 64,
                **attempt,
            }
        )
        if report_probe_progress:
            outcome = "passed" if attempt.get("passed") is True else "failed"
            print(
                f"probe {physical_batch_size}x{accumulation_steps}: {outcome}",
                flush=True,
            )
    passing = [attempt for attempt in attempts if attempt.get("passed") is True]
    if not passing:
        raise RuntimeError("no Attempt 02 batch candidate passed preflight")
    selected = passing[0]
    result = {
        "schema_version": 1,
        "status": "completed",
        "attempt_id": "attempt-02",
        "scientific_protocol": protocol,
        "cache_reuse": cache_reuse,
        "execution": {
            "candidates": attempts,
            "selected": {
                key: selected[key]
                for key in (
                    "physical_batch_size",
                    "accumulation_steps",
                    "effective_batch_size",
                )
            },
            "frozen_fold_ids": [0, 1, 2, 3, 4],
        },
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--cache-lock", type=Path, default=DEFAULT_CACHE_LOCK)
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--probe-timeout-seconds", type=int, default=300)
    validate_cache = subparsers.add_parser("validate-cache")
    validate_cache.add_argument("--config", type=Path, required=True)
    validate_cache.add_argument("--work-dir", type=Path, required=True)
    validate_cache.add_argument("--output", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--preflight", type=Path, required=True)
    train.add_argument("--strict-cache-validation", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--attempt-01-report", type=Path, required=True)
    report.add_argument(
        "--attempt-02-evidence", type=Path, action="append", required=True
    )
    report.add_argument("--sample-patient-csv", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--bootstrap-draws", type=int, default=10_000)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--output", type=Path, required=True)
    environment.add_argument("--repository-commit")
    package = subparsers.add_parser("package")
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--preflight", type=Path, required=True)
    package.add_argument("--strict-cache-validation", type=Path, required=True)
    package.add_argument("--report", type=Path, required=True)
    package.add_argument("--environment", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    worker = subparsers.add_parser("probe-worker", help=argparse.SUPPRESS)
    worker.add_argument("--physical-batch-size", type=int, required=True)
    worker.add_argument("--accumulation-steps", type=int, required=True)
    worker.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        run_preflight(
            data_dir=args.data_dir,
            cache_lock_path=args.cache_lock,
            output_dir=args.output_dir,
            device=args.device,
            probe_timeout_seconds=args.probe_timeout_seconds,
        )
        return 0
    if args.command == "validate-cache":
        validate_cache_payloads(
            config_path=args.config,
            work_dir=args.work_dir,
            output_path=args.output,
        )
        return 0
    if args.command == "train":
        run_training(
            preflight_path=args.preflight,
            strict_validation_path=args.strict_cache_validation,
            run_id=args.run_id,
        )
        return 0
    if args.command == "report":
        if len(args.attempt_02_evidence) != 5:
            raise ValueError("Attempt 02 report requires exactly five fold evidence files")
        result = build_decoder_depth_report(
            attempt_01_report=args.attempt_01_report,
            attempt_02_evidence=args.attempt_02_evidence,
            sample_patient_csv=args.sample_patient_csv,
            spacing_exception_patient_ids=SPACING_EXCEPTION_PATIENT_IDS,
            bootstrap_draws=args.bootstrap_draws,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    if args.command == "environment":
        capture_environment_provenance(
            output_path=args.output, repository_commit=args.repository_commit
        )
        return 0
    if args.command == "package":
        result = assemble_release_archives(
            run_dir=args.run_dir,
            preflight_path=args.preflight,
            strict_validation_path=args.strict_cache_validation,
            report_path=args.report,
            environment_path=args.environment,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    try:
        result = probe_candidate(
            physical_batch_size=args.physical_batch_size,
            accumulation_steps=args.accumulation_steps,
            device=args.device,
        )
    except Exception as exc:
        result = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
