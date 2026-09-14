"""Relaunch a guarded GPU job each time the shared GPU has stayed idle long enough.

Every launch runs under ``beetle.gpu_job`` in a fresh ``run-NN`` job directory, so the
guard still stops our process group the moment another GPU process appears. This loop
only decides when to launch again: it waits for an unbroken idle window after a yield,
and stops for a person on any stop that was not a yield.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from beetle.gpu_job import gpu_pids, write_json

# Worker errors raised when another process takes the GPU during the CUDA handshake.
WORKER_YIELD_ERRORS = (
    'GPU became busy before CUDA initialization',
    'Expected exactly our one GPU client',
    'GPU ownership changed during guard handshake',
)


def log(path: Path, message: str) -> None:
    with path.open('a') as handle:
        handle.write(f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}\n')


def next_job_dir(jobs: Path) -> Path:
    numbers = [int(match.group(1)) for path in jobs.glob('run-*')
               if (match := re.fullmatch(r'run-(\d+)', path.name))]
    return jobs / f'run-{max(numbers, default=0) + 1:02d}'


def wait_for_idle(idle_seconds: float, poll_seconds: float, stop_file: Path, say,
                  clock=time.monotonic, sleep=time.sleep) -> bool:
    """Return True once no GPU process was seen for ``idle_seconds``; False on a stop request."""
    idle_since = None
    reported = None
    while not stop_file.exists():
        try:
            busy = gpu_pids()
        except Exception as exc:
            # An unanswered query is never evidence of an idle GPU.
            busy = {f'query failed: {exc!r}'}
        if busy:
            if busy != reported:
                say(f'GPU busy: {sorted(busy)}')
                reported = busy
            idle_since = None
        else:
            if idle_since is None:
                idle_since = clock()
                reported = None
            if clock() - idle_since >= idle_seconds:
                return True
        sleep(poll_seconds)
    return False


def yielded(job_dir: Path, status: dict) -> bool:
    """Whether a stopped job made way for another GPU user rather than failing."""
    reason = status.get('reason', '')
    if 'GPU already occupied' in reason or 'Another GPU process appeared' in reason:
        return True
    for name, markers in (('gpu_guard.log', ('foreign GPU process detected',)),
                          ('run.log', WORKER_YIELD_ERRORS)):
        path = job_dir / name
        if path.exists() and any(marker in path.read_text(errors='replace') for marker in markers):
            return True
    # Handshake races surface as ambiguous-ownership errors; a busy GPU right after says why.
    try:
        return bool(gpu_pids())
    except Exception:
        return False


def resume(command: list[str], jobs: Path, idle_seconds: float, poll_seconds: float = 5,
           max_quick_yields: int = 6, quick_seconds: float = 1200) -> int:
    jobs.mkdir(parents=True, exist_ok=True)
    lock = (jobs / 'autoresume.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # One loop per run.
    stop_file = jobs / 'autoresume.stop'
    status_path = jobs / 'autoresume-status.json'
    state = {'pid': os.getpid(), 'state': 'starting', 'idle_seconds': idle_seconds,
             'stop_file': str(stop_file), 'command': command, 'runs': []}

    def say(message: str) -> None:
        log(jobs / 'autoresume.log', message)

    wait = 0  # The first launch only needs the GPU idle now.
    quick = 0
    child = None
    try:
        while True:
            state.update(state='waiting', job_dir=None)
            write_json(status_path, state)
            if not wait_for_idle(wait, poll_seconds, stop_file, say):
                state.update(state='stopped', reason=f'{stop_file.name} present')
                say(f'stopping: {stop_file} present')
                return 0
            job_dir = next_job_dir(jobs)
            started = time.monotonic()
            state.update(state='running', job_dir=str(job_dir))
            write_json(status_path, state)
            say(f'launching {job_dir.name}')
            worker = [part.replace('{job_dir}', str(job_dir)) for part in command]
            child = subprocess.Popen([sys.executable, '-u', '-m', 'beetle.gpu_job',
                                      '--job-dir', str(job_dir), '--', *worker],
                                     stdin=subprocess.DEVNULL)
            code = child.wait()
            child = None
            status_file = job_dir / 'guard-status.json'
            job = json.loads(status_file.read_text()) if status_file.exists() else {}
            minutes = round((time.monotonic() - started) / 60, 1)
            entry = {'job_dir': job_dir.name, 'exit_code': code, 'status': job.get('status', 'unknown'),
                     'reason': job.get('reason', ''), 'minutes': minutes}
            state['runs'].append(entry)
            if entry['status'] == 'completed':
                state.update(state='completed')
                say(f'{job_dir.name} completed after {minutes} min')
                return 0
            if not yielded(job_dir, job):
                state.update(state='needs_attention', reason=f'{job_dir.name} stopped without yielding')
                say(f'{job_dir.name} stopped without yielding ({entry["status"]}: {entry["reason"]}); '
                    'not relaunching')
                return 1
            entry['yielded'] = True
            quick = quick + 1 if minutes * 60 < quick_seconds else 0
            if quick >= max_quick_yields:
                state.update(state='needs_attention',
                             reason=f'{quick} consecutive yields within {quick_seconds / 60:g} min of launch')
                say(f'not relaunching: {state["reason"]}')
                return 1
            say(f'{job_dir.name} yielded after {minutes} min ({entry["reason"]}); '
                f'relaunching after {idle_seconds / 60:g} min of idle GPU')
            wait = idle_seconds
    finally:
        if child is not None and child.poll() is None:
            child.terminate()  # gpu_job turns SIGTERM into stopping the worker group.
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if state['state'] in ('starting', 'waiting', 'running'):
            state.update(state='stopped', reason='resume loop exited')
            say('resume loop exited')
        write_json(status_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs-dir', type=Path, required=True)
    parser.add_argument('--idle-minutes', type=float, default=10)
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='worker command; {job_dir} is replaced by each launch\'s job directory')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a worker command is required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    return resume(command, args.jobs_dir.resolve(), idle_seconds=args.idle_minutes * 60)


if __name__ == '__main__':
    raise SystemExit(main())
