"""Resolve one attempt: base config + one attempt overlay, merged by soma."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from soma.config import PipelineConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def resolve_base_config(path: str | Path = BASE_CONFIG) -> Path:
    """Return the base config path, or fail clearly outside a source checkout."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Base config not found: {path}. The curate/extract/train commands "
            "run from a repository checkout; run from one or pass an explicit "
            "config path."
        )
    return path


def load_attempt_config(
    attempt_path: str | Path,
    *,
    base_path: str | Path = BASE_CONFIG,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Merge the attempt overlay (and optional overrides) over the base config."""
    base_path = resolve_base_config(base_path)
    overlay = yaml.safe_load(Path(attempt_path).read_text(encoding="utf-8")) or {}
    if not isinstance(overlay, dict):
        raise TypeError(f"Attempt overlay must be a mapping: {attempt_path}")
    if overrides:
        overlay = _deep_merge(overlay, overrides)
    return load_config(base_path, overrides=overlay)


def attempt_name(attempt_path: str | Path) -> str:
    return Path(attempt_path).stem


def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``--set a.b.c=value`` strings into a nested override dict."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw_value = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--set expects key=value, got {pair!r}")
        cursor = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return overrides


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
