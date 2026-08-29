"""Write-only attempt recorder.

Called at the end of ``train`` and ``infer``. It only writes evidence under
``provenance/attempts/<attempt>/`` — it never verifies anything, and a recording
failure never fails the run.
"""

from __future__ import annotations

import datetime
import json
import shutil
import traceback
from pathlib import Path

from beetle.attempts import REPO_ROOT

ATTEMPTS_DIR = REPO_ROOT / "provenance" / "attempts"


def record_training(attempt: str, run_dir: Path, config) -> None:
    _best_effort(_record_training, attempt, run_dir, config)


def record_inference(attempt: str, run_dir: Path, zip_path: Path, roi_count: int) -> None:
    _best_effort(_record_inference, attempt, run_dir, zip_path, roi_count)


def _record_training(attempt: str, run_dir: Path, config) -> None:
    from soma.config import save_config

    out = ATTEMPTS_DIR / attempt
    out.mkdir(parents=True, exist_ok=True)
    save_config(config, out / "config.yaml")
    summary = run_dir / "summary.json"
    if summary.is_file():
        shutil.copy2(summary, out / "metrics.json")
    _write_json(
        out / "training.json",
        {
            "run_dir": str(Path(run_dir).resolve()),
            "soma": _soma_identity(),
            "recorded_at": _now(),
        },
    )


def _record_inference(attempt: str, run_dir: Path, zip_path: Path, roi_count: int) -> None:
    out = ATTEMPTS_DIR / attempt
    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / "inference.json",
        {
            "run_dir": str(Path(run_dir).resolve()),
            "submission_zip": str(Path(zip_path).resolve()),
            "roi_count": roi_count,
            "soma": _soma_identity(),
            "recorded_at": _now(),
        },
    )


def _soma_identity() -> dict:
    identity: dict = {}
    try:
        from importlib.metadata import distribution

        dist = distribution("soma-pathology")
        identity["version"] = dist.version
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            commit = payload.get("vcs_info", {}).get("commit_id")
            if commit:
                identity["commit"] = commit
    except Exception:
        pass
    return identity


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _best_effort(function, *args) -> None:
    try:
        function(*args)
    except Exception:
        print("warning: attempt recording failed (run unaffected)")
        traceback.print_exc()
