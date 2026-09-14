"""Relaunch a guarded GPU job on whichever shared GPU has stayed idle long enough.

Every launch runs under ``beetle.gpu_job`` in a fresh ``run-NN`` job directory, pinned to
one physical GPU, so the guard still stops our process group the moment another process
appears on that GPU. This loop only decides when and where to launch again. It watches
every candidate GPU, including while a job runs, so after a yield it can move straight to
a GPU that has already shown no process for an unbroken idle window. If none has, it
waits for one. It stops for a person on any stop that was not a yield.
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


def short(uuid: str) -> str:
    return uuid[:12]


def next_job_dir(jobs: Path) -> Path:
    numbers = [int(match.group(1)) for path in jobs.glob('run-*')
               if (match := re.fullmatch(r'run-(\d+)', path.name))]
    return jobs / f'run-{max(numbers, default=0) + 1:02d}'


def gpu_processes(uuids: list[str]) -> dict[str, set[str]]:
    """PIDs on each candidate GPU; raises unless every candidate is reported."""
    def query(*args: str) -> str:
        return subprocess.run(['nvidia-smi', *args, '--format=csv,noheader'],
                              capture_output=True, text=True, check=True, timeout=10).stdout

    missing = set(uuids) - set(query('--query-gpu=uuid').split())
    if missing:
        raise RuntimeError(f'GPUs not reported by nvidia-smi: {sorted(missing)}')
    apps = query('--query-compute-apps=gpu_uuid,pid')
    busy = {uuid: set() for uuid in uuids}
    for line in filter(str.strip, apps.splitlines()):
        uuid, _, pid = (part.strip() for part in line.partition(','))
        if not uuid.startswith('GPU-') or not pid.isdigit():
            raise RuntimeError(f'Unexpected GPU process query: {apps!r}')
        if uuid in busy:
            busy[uuid].add(pid)
    return busy


class IdleGpus:
    """Remember, for each candidate GPU, since when it has shown no process."""

    def __init__(self, uuids: list[str], query=gpu_processes, clock=time.monotonic):
        self.uuids = list(uuids)
        self.query = query
        self.clock = clock
        self.idle_since: dict[str, float] = {}

    def observe(self) -> dict[str, list[str]]:
        """Poll once and return the busy GPUs, keyed by short UUID."""
        now = self.clock()
        try:
            busy = self.query(self.uuids)
        except Exception as exc:
            # An unanswered query is never evidence of an idle GPU: every window restarts.
            self.idle_since.clear()
            return {'query failed': [repr(exc)]}
        for uuid in self.uuids:
            if busy[uuid]:
                self.idle_since.pop(uuid, None)
            else:
                self.idle_since.setdefault(uuid, now)
        return {short(uuid): sorted(pids) for uuid, pids in busy.items() if pids}

    def idlest(self, seconds: float) -> str | None:
        """The GPU idle the longest, if that is at least ``seconds``."""
        now = self.clock()
        ready = [(since, uuid) for uuid, since in self.idle_since.items() if now - since >= seconds]
        return min(ready)[1] if ready else None


def wait_for_gpu(gpus: IdleGpus, idle_seconds: float, poll_seconds: float, stop_file: Path, say,
                 sleep=time.sleep) -> str | None:
    """Return a GPU once it has shown no process for ``idle_seconds``; None on a stop request."""
    reported = None
    while not stop_file.exists():
        busy = gpus.observe()
        chosen = gpus.idlest(idle_seconds)
        if chosen is not None:
            return chosen
        if busy != reported:
            say(f'no GPU idle for {idle_seconds / 60:g} min yet; busy: {busy}')
            reported = busy
        sleep(poll_seconds)
    return None


def yielded(job_dir: Path, status: dict, gpu: str) -> bool:
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
        return bool(gpu_pids(gpu))
    except Exception:
        return False


def resume(command: list[str], jobs: Path, gpus: list[str], idle_seconds: float,
           poll_seconds: float = 5, max_quick_yields: int = 6, quick_seconds: float = 1200,
           tracker: IdleGpus | None = None) -> int:
    jobs.mkdir(parents=True, exist_ok=True)
    lock = (jobs / 'autoresume.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # One loop per run.
    stop_file = jobs / 'autoresume.stop'
    status_path = jobs / 'autoresume-status.json'
    tracker = tracker or IdleGpus(gpus)
    state = {'pid': os.getpid(), 'state': 'starting', 'idle_seconds': idle_seconds, 'gpus': gpus,
             'stop_file': str(stop_file), 'command': command, 'runs': []}

    def say(message: str) -> None:
        log(jobs / 'autoresume.log', message)

    wait = 0  # The first launch only needs a GPU idle now.
    quick = 0
    child = None
    try:
        while True:
            state.update(state='waiting', job_dir=None, gpu=None)
            write_json(status_path, state)
            gpu = wait_for_gpu(tracker, wait, poll_seconds, stop_file, say)
            if gpu is None:
                state.update(state='stopped', reason=f'{stop_file.name} present')
                say(f'stopping: {stop_file} present')
                return 0
            job_dir = next_job_dir(jobs)
            started = time.monotonic()
            state.update(state='running', job_dir=str(job_dir), gpu=gpu)
            write_json(status_path, state)
            say(f'launching {job_dir.name} on {gpu}')
            worker = [part.replace('{job_dir}', str(job_dir)) for part in command]
            child = subprocess.Popen([sys.executable, '-u', '-m', 'beetle.gpu_job',
                                      '--job-dir', str(job_dir), '--', *worker],
                                     stdin=subprocess.DEVNULL,
                                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
            while child.poll() is None:
                tracker.observe()  # Idle history on the other GPUs lets a yield move at once.
                time.sleep(poll_seconds)
            code = child.wait()
            child = None
            status_file = job_dir / 'guard-status.json'
            job = json.loads(status_file.read_text()) if status_file.exists() else {}
            minutes = round((time.monotonic() - started) / 60, 1)
            entry = {'job_dir': job_dir.name, 'gpu': gpu, 'exit_code': code,
                     'status': job.get('status', 'unknown'), 'reason': job.get('reason', ''),
                     'minutes': minutes}
            state['runs'].append(entry)
            if entry['status'] == 'completed':
                state.update(state='completed')
                say(f'{job_dir.name} completed after {minutes} min')
                return 0
            if not yielded(job_dir, job, gpu):
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
            say(f'{job_dir.name} yielded {short(gpu)} after {minutes} min ({entry["reason"]}); '
                f'relaunching on a GPU idle for {idle_seconds / 60:g} min')
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


def gpu_list(value: str) -> list[str]:
    uuids = [part.strip() for part in value.split(',') if part.strip()]
    if not uuids or len(set(uuids)) != len(uuids) or not all(uuid.startswith('GPU-') for uuid in uuids):
        raise argparse.ArgumentTypeError('expected distinct physical GPU UUIDs, comma-separated')
    return uuids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=gpu_list, required=True,
                        help='candidate GPU UUIDs, comma-separated; each launch uses one')
    parser.add_argument('--idle-minutes', type=float, default=10)
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='worker command; {job_dir} is replaced by each launch\'s job directory')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a worker command is required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    return resume(command, args.jobs_dir.resolve(), args.gpus, idle_seconds=args.idle_minutes * 60)


if __name__ == '__main__':
    raise SystemExit(main())
