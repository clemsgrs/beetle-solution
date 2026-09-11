"""Gate a completed five-fold run and package its GitHub release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import zipfile

FOLD_ARTIFACTS = (
    "best_model.pt",
    "training_history.json",
    "roi_batch_sampling.json",
    "confusion_evidence_tune.json",
    "metrics.json",
    "segmentation_roi_population.json",
)
EVIDENCE_FOLD_ARTIFACTS = (
    "training_history.json",
    "confusion_evidence_tune.json",
    "metrics.json",
    "segmentation_roi_population.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_completed_run(
    *, run_dir: str | Path, environment_path: str | Path
) -> dict:
    """Gate publication on five real folds and identify every retained artifact."""
    run_dir = Path(run_dir)
    environment_path = Path(environment_path)
    resolved_config = run_dir / "config.yaml"
    if not resolved_config.is_file():
        raise ValueError("Completed-run gate is missing resolved config.yaml")
    if not environment_path.is_file():
        raise ValueError("Completed-run gate is missing environment provenance")
    folds = {}
    for fold in range(5):
        fold_dir = run_dir / f"fold_{fold}"
        identities = {}
        for name in FOLD_ARTIFACTS:
            path = fold_dir / name
            if not path.is_file():
                raise ValueError(f"Completed-run gate: fold {fold} is missing {name}")
            identities[name] = artifact_identity(path)
        folds[str(fold)] = identities
    return {
        "schema_version": 1,
        "status": "completed",
        "resolved_config": artifact_identity(resolved_config),
        "environment_provenance": artifact_identity(environment_path),
        "folds": folds,
    }


def assemble_release(
    *,
    attempt_id: str,
    run_dir: str | Path,
    environment_path: str | Path,
    evidence: Mapping[str, str | Path],
    output_dir: str | Path,
) -> dict:
    """Package five checkpoints and compact evidence after all-fold gating.

    ``evidence`` maps archive names to files and keeps its order; each fold's
    history, confusion evidence, metrics, and ROI population follow it.
    """
    run_dir = Path(run_dir)
    environment_path = Path(environment_path)
    output_dir = Path(output_dir)
    evidence = {name: Path(path) for name, path in evidence.items()}
    missing = [
        str(path) for path in (environment_path, *evidence.values()) if not path.is_file()
    ]
    if missing:
        raise ValueError(f"{attempt_id} release evidence is missing: {missing}")
    manifest = validate_completed_run(run_dir=run_dir, environment_path=environment_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "artifact_checksums.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    weights_path = output_dir / f"beetle-{attempt_id}-weights.zip"
    with zipfile.ZipFile(weights_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(run_dir / "config.yaml", "config.yaml")
        for fold in range(5):
            archive.write(run_dir / f"fold_{fold}/best_model.pt", f"fold_{fold}/best_model.pt")
        archive.write(manifest_path, "artifact_checksums.json")

    evidence_path = output_dir / f"beetle-{attempt_id}-evidence.zip"
    with zipfile.ZipFile(evidence_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in evidence.items():
            archive.write(path, name)
        for fold in range(5):
            for name in EVIDENCE_FOLD_ARTIFACTS:
                archive.write(run_dir / f"fold_{fold}/{name}", f"fold_{fold}/{name}")
        archive.write(manifest_path, "artifact_checksums.json")
    return {
        "schema_version": 1,
        "status": "completed",
        "artifact_manifest": {"path": str(manifest_path), **artifact_identity(manifest_path)},
        "weights_archive": {"path": str(weights_path), **artifact_identity(weights_path)},
        "evidence_archive": {"path": str(evidence_path), **artifact_identity(evidence_path)},
    }


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"--evidence expects NAME=PATH, got {value!r}")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=_named_path,
        action="append",
        default=[],
        help="ARCHIVE_NAME=PATH, repeated in archive order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    names = [name for name, _ in args.evidence]
    if len(names) != len(set(names)):
        parser.error("--evidence archive names must be unique")
    result = assemble_release(
        attempt_id=args.attempt,
        run_dir=args.run_dir,
        environment_path=args.environment,
        evidence=dict(args.evidence),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
