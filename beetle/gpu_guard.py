"""Independent, device-scoped, fail-closed monitor for one worker group."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time

from beetle.gpu_job import gpu_pids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pgid', type=int)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--expected-pid', required=True)
    args = parser.parse_args()

    def log(message):
        with args.log.open('a') as handle:
            handle.write(f'{datetime.now(timezone.utc).isoformat()} {message}\n')

    def alive():
        try:
            os.killpg(args.pgid, 0)
            return True
        except ProcessLookupError:
            return False

    ours = {args.expected_pid}
    try:
        if gpu_pids() != ours:
            raise RuntimeError('GPU ownership changed before guard startup')
        log(f'guard started; pgid={args.pgid}; baseline GPU pids={sorted(ours)}')
        while alive():
            foreign = gpu_pids() - ours
            if foreign:
                raise RuntimeError(f'foreign GPU process detected: {sorted(foreign)}')
            time.sleep(args.interval)
        log('job has exited; guard stopping')
        return 0
    except BaseException as exc:
        log(f'{exc}; stopping job')
        try:
            os.killpg(args.pgid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            if alive():
                os.killpg(args.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
