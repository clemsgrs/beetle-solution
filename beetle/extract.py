"""Populate and validate the shared dense feature cache without training decoders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from beetle.attempts import BASE_CONFIG, parse_set_overrides
from soma.pipeline import Pipeline


def extract_cache(config, work_dir: str | Path) -> dict:
    """Run the pipeline's cache preparation seam and stop before training."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(config)
    context = pipeline._build_slide_manifest_dense_context(run_dir=work_dir)
    store = context.feature_store
    roi_ids = list(context.dataset.sample_ids)
    store.validate_coverage(roi_ids)
    summary = {
        "feature_dir": str(Path(store.feature_dir).resolve()),
        "parent_slides": len(pipeline.dataset.sample_ids),
        "roi_grids": len(roi_ids),
        "feature_dim": int(store.feature_dim),
        "grid_shape": list(store.grid_shape),
    }
    print(
        f"Cache complete: {summary['roi_grids']} ROI grids from "
        f"{summary['parent_slides']} slides in {summary['feature_dir']}"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m beetle extract", description=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_CONFIG,
        help="config to extract with (default: configs/base.yaml)",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    from soma.config import load_config

    config = load_config(args.config, overrides=parse_set_overrides(args.overrides))
    extract_cache(config, args.work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
