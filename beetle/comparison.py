"""Compare attempts on paired held-out development evidence.

Each attempt contributes five fold scores and one confusion matrix per patient.
The first attempt given is the formal comparator. Every later attempt is paired
with every earlier one on identical patients, so a bootstrap difference
resamples the same patients for both sides.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import numpy as np

FOLD_IDS = tuple(range(5))
EVIDENCE_NAME = "confusion_evidence_tune.json"
SPACING_EXCEPTION_PATIENT_IDS = (
    "TCGA-OL-A66I",
    "TCGA-OL-A66P",
    "TCGA-OL-A6VO",
)


def rounded(value: float) -> float:
    return round(float(value), 12)


def confusion_metrics(matrix: np.ndarray, vocabulary: Sequence[str]) -> dict:
    true_positive = np.diag(matrix).astype(np.float64)
    denominators = matrix.sum(axis=0) + matrix.sum(axis=1)
    dice = np.divide(
        2.0 * true_positive,
        denominators,
        out=np.zeros_like(true_positive),
        where=denominators != 0,
    )
    total = int(matrix.sum())
    return {
        "dice_per_class": {
            name: rounded(dice[index]) for index, name in enumerate(vocabulary)
        },
        "macro_dice": rounded(dice.mean()),
        "pixel_micro_dice": rounded(true_positive.sum() / total),
    }


def _macro_dice(matrix: np.ndarray) -> float:
    return confusion_metrics(matrix, range(matrix.shape[0]))["macro_dice"]


def bootstrap_macro_dice(matrices: Sequence[np.ndarray], *, draws: int) -> dict:
    stacked = np.stack(matrices)
    rng = np.random.default_rng(0)
    replicates = []
    for _ in range(draws):
        indices = rng.integers(0, len(stacked), size=len(stacked))
        replicates.append(_macro_dice(stacked[indices].sum(axis=0)))
    low, high = np.percentile(replicates, [2.5, 97.5])
    return {
        "seed": 0,
        "draws": draws,
        "macro_dice_percentile_95_ci": [rounded(low), rounded(high)],
    }


def paired_bootstrap_delta(
    candidate: Sequence[np.ndarray],
    comparator: Sequence[np.ndarray],
    *,
    draws: int,
) -> dict:
    """Bound the pooled macro Dice gap, resampling the same patients for both."""
    candidate = np.stack(candidate)
    comparator = np.stack(comparator)
    if candidate.shape != comparator.shape:
        raise ValueError("Paired bootstrap requires one matrix per patient on both sides")
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(draws):
        indices = rng.integers(0, len(candidate), size=len(candidate))
        deltas.append(
            _macro_dice(candidate[indices].sum(axis=0))
            - _macro_dice(comparator[indices].sum(axis=0))
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    observed = _macro_dice(candidate.sum(axis=0)) - _macro_dice(comparator.sum(axis=0))
    return {
        "seed": 0,
        "draws": draws,
        "pooled_macro_dice_delta": rounded(observed),
        "macro_dice_delta_percentile_95_ci": [rounded(low), rounded(high)],
    }


def read_sample_patients(path: str | Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["sample_id"]): str(row["patient_id"])
        for row in rows
        if row.get("sample_id") and row.get("patient_id")
    }


def load_confusion_evidence(
    evidence_paths: Sequence[str | Path],
    sample_to_patient: Mapping[str, str],
    *,
    label: str,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], list[float]]:
    """Sum held-out ROI confusion matrices per patient and score each fold."""
    records = []
    for path in evidence_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(payload.get("records", []))
    if not records:
        raise ValueError(f"{label} confusion evidence is empty")
    vocabulary = tuple(records[0]["class_vocabulary"])
    by_patient: dict[str, list[np.ndarray]] = {}
    patient_folds: dict[str, int] = {}
    fold_matrices: dict[int, list[np.ndarray]] = {fold: [] for fold in FOLD_IDS}
    seen_samples: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen_samples:
            raise ValueError(f"{label} sample appears more than once: {sample_id}")
        seen_samples.add(sample_id)
        if tuple(record["class_vocabulary"]) != vocabulary:
            raise ValueError(f"{label} evidence disagrees on class vocabulary")
        patient_id = sample_to_patient.get(sample_id)
        if patient_id is None:
            raise ValueError(f"{label} sample has no patient mapping: {sample_id}")
        fold = int(record["fold"])
        matrix = np.asarray(record["confusion_matrix"], dtype=np.int64)
        if patient_folds.setdefault(patient_id, fold) != fold:
            raise ValueError(f"{label} patient appears in multiple folds: {patient_id}")
        by_patient.setdefault(patient_id, []).append(matrix)
        fold_matrices.setdefault(fold, []).append(matrix)
    if set(fold_matrices) != set(FOLD_IDS) or any(not fold_matrices[x] for x in FOLD_IDS):
        raise ValueError(f"{label} requires held-out confusion evidence for folds 0-4")
    patients = {
        patient_id: np.stack(matrices).sum(axis=0)
        for patient_id, matrices in by_patient.items()
    }
    fold_scores = [
        confusion_metrics(np.stack(fold_matrices[fold]).sum(axis=0), vocabulary)[
            "macro_dice"
        ]
        for fold in FOLD_IDS
    ]
    return vocabulary, patients, fold_scores


def load_cv_results(
    path: str | Path,
) -> tuple[tuple[str, ...] | None, dict[str, np.ndarray], list[float]]:
    """Read a released per-patient cross-validation record (the Attempt 01 format)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = payload.get("classes")
    vocabulary = (
        tuple(entry["name"] for entry in sorted(classes, key=lambda x: x["model_index"]))
        if classes
        else None
    )
    patients = {
        str(record["patient_id"]): np.asarray(record["confusion_matrix"], dtype=np.int64)
        for record in payload.get("patients", [])
    }
    fold_scores = [float(payload["folds"][str(fold)]["mean_dice"]) for fold in FOLD_IDS]
    return vocabulary, patients, fold_scores


def compare_attempts(
    attempts: Mapping[str, tuple[Sequence[float], Mapping[str, np.ndarray]]],
    *,
    vocabulary: Sequence[str],
    spacing_exception_patient_ids: Sequence[str] = SPACING_EXCEPTION_PATIENT_IDS,
    primary_patient_count: int = 527,
    sensitivity_patient_count: int = 524,
    bootstrap_draws: int = 10_000,
) -> dict:
    """Summarize each attempt and pair every later attempt with every earlier one."""
    names = list(attempts)
    if len(names) < 2:
        raise ValueError("A comparison requires at least two attempts")
    formal = names[0]
    cohort = set(attempts[formal][1])
    for name in names[1:]:
        if set(attempts[name][1]) != cohort:
            raise ValueError(f"{name} and {formal} patient cohorts differ")
    if len(cohort) != primary_patient_count:
        raise ValueError(
            f"Primary cohort has {len(cohort)} patients; expected {primary_patient_count}"
        )
    excluded = sorted(str(value) for value in spacing_exception_patient_ids)
    if not set(excluded).issubset(cohort):
        raise ValueError("Spacing-exception patients are absent from the cohort")
    primary_ids = sorted(cohort)
    sensitivity_ids = sorted(cohort - set(excluded))
    if len(sensitivity_ids) != sensitivity_patient_count:
        raise ValueError(
            f"Sensitivity cohort has {len(sensitivity_ids)} patients; "
            f"expected {sensitivity_patient_count}"
        )

    def matrices(name: str, patient_ids: Sequence[str]) -> list[np.ndarray]:
        return [attempts[name][1][patient_id] for patient_id in patient_ids]

    def cohort_summary(patient_ids: Sequence[str]) -> dict:
        return {
            "patient_count": len(patient_ids),
            "attempts": {
                name: bootstrap_macro_dice(matrices(name, patient_ids), draws=bootstrap_draws)
                for name in names
            },
        }

    sensitivity = cohort_summary(sensitivity_ids)
    sensitivity.update({"evaluation_only": True, "excluded_patient_ids": excluded})

    comparisons = {}
    for later, candidate in enumerate(names[1:], start=1):
        for comparator in names[:later]:
            deltas = [
                candidate_score - comparator_score
                for candidate_score, comparator_score in zip(
                    attempts[candidate][0], attempts[comparator][0], strict=True
                )
            ]
            comparisons[f"{candidate}_minus_{comparator}"] = {
                "paired_fold_deltas": [rounded(value) for value in deltas],
                "mean_delta": rounded(statistics.mean(deltas)),
                "folds_improved": sum(value > 0 for value in deltas),
                "patient_bootstrap": {
                    cohort_name: paired_bootstrap_delta(
                        matrices(candidate, patient_ids),
                        matrices(comparator, patient_ids),
                        draws=bootstrap_draws,
                    )
                    for cohort_name, patient_ids in (
                        ("primary", primary_ids),
                        ("sensitivity", sensitivity_ids),
                    )
                },
            }

    return {
        "schema_version": 1,
        "formal_comparator": formal,
        "candidate": names[-1],
        "attempts": {
            name: {
                "fold_scores": [rounded(value) for value in scores],
                "mean": rounded(statistics.mean(scores)),
                "sample_standard_deviation": rounded(statistics.stdev(scores)),
            }
            for name, (scores, _) in attempts.items()
        },
        "pooled_metrics": {
            name: confusion_metrics(np.stack(matrices(name, primary_ids)).sum(axis=0), vocabulary)
            for name in names
        },
        "comparisons": comparisons,
        "patient_bootstrap": {
            "primary": cohort_summary(primary_ids),
            "sensitivity": sensitivity,
        },
    }


def build_report(
    *,
    attempt_sources: Sequence[tuple[str, Path]],
    sample_patient_csv: str | Path,
    bootstrap_draws: int = 10_000,
    **cohorts,
) -> dict:
    """Load each attempt from a released CV record (a file) or a run directory."""
    sample_to_patient = read_sample_patients(sample_patient_csv)
    attempts: dict[str, tuple[list[float], dict[str, np.ndarray]]] = {}
    vocabularies = set()
    for name, path in attempt_sources:
        if name in attempts:
            raise ValueError(f"Attempt given more than once: {name}")
        path = Path(path)
        if path.is_dir():
            vocabulary, patients, scores = load_confusion_evidence(
                [path / f"fold_{fold}" / EVIDENCE_NAME for fold in FOLD_IDS],
                sample_to_patient,
                label=name,
            )
        else:
            vocabulary, patients, scores = load_cv_results(path)
        if vocabulary is not None:
            vocabularies.add(vocabulary)
        attempts[name] = (scores, patients)
    if len(vocabularies) != 1:
        raise ValueError("Attempts disagree on, or do not declare, the class vocabulary")
    return compare_attempts(
        attempts,
        vocabulary=vocabularies.pop(),
        bootstrap_draws=bootstrap_draws,
        **cohorts,
    )


def _attempt_source(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"--attempt expects NAME=PATH, got {value!r}")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt",
        dest="attempts",
        type=_attempt_source,
        action="append",
        required=True,
        help=(
            "NAME=PATH, repeated in order; PATH is a released cv_results.json or a "
            "run directory. The first attempt is the formal comparator."
        ),
    )
    parser.add_argument("--sample-patient-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args(argv)
    result = build_report(
        attempt_sources=args.attempts,
        sample_patient_csv=args.sample_patient_csv,
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
