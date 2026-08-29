"""Train the five fold decoders for one attempt (base config + attempt overlay)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from beetle.attempts import attempt_name, load_attempt_config, parse_set_overrides
from beetle.record import record_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m beetle train", description=__doc__)
    parser.add_argument(
        "--attempt",
        type=Path,
        required=True,
        help="attempt overlay YAML (configs/attempts/<name>.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="extra dotted overrides, e.g. --set run.resume=true",
    )
    args = parser.parse_args(argv)

    config = load_attempt_config(
        args.attempt, overrides=parse_set_overrides(args.overrides)
    )
    from soma.pipeline import Pipeline

    result = Pipeline(config).run()
    record_training(attempt_name(args.attempt), result.run_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
