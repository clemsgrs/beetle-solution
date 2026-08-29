"""Entry point: ``python -m beetle {curate, extract, train, infer}``."""

from __future__ import annotations

import sys

_COMMANDS = ("curate", "extract", "train", "infer")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in _COMMANDS:
        print(f"usage: python -m beetle {{{','.join(_COMMANDS)}}} ...", file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    if command == "curate":
        from beetle.curate import main as run
    elif command == "extract":
        from beetle.extract import main as run
    elif command == "train":
        from beetle.train import main as run
    else:
        from beetle.infer import main as run
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
