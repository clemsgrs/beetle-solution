"""Run one CUDA client under the shared-GPU guard, with fail-closed supervision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GUARD = Path(__file__).with_name('gpu_guard.py')


def gpu_uuid() -> str:
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must select one physical GPU UUID')
    return uuid


def gpu_pids(uuid: str | None = None) -> set[str]:
    result = subprocess.run(
        ['nvidia-smi', '--id', uuid or gpu_uuid(), '--query-compute-apps=pid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=5,
    )
    lines = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if any(not pid.isdigit() for pid in lines):
        raise RuntimeError(f'Unexpected GPU process query: {result.stdout!r}')
    return lines


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def claim_gpu(job_dir: Path):
    """Hold a CUDA allocation while the supervisor validates and arms the guard."""
    import torch

    if gpu_pids():
        raise RuntimeError('GPU became busy before CUDA initialization')
    token = torch.ones(1, device='cuda:0')
    torch.cuda.synchronize()
    ours = gpu_pids()
    if len(ours) != 1:
        raise RuntimeError(f'Expected exactly our one GPU client, got {sorted(ours)}')
    write_json(job_dir / 'gpu-ready.json', {'gpu_pids': sorted(ours), 'pid': os.getpid()})
    deadline = time.monotonic() + 30
    while not (job_dir / 'gpu-go').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Guard did not arm within 30 seconds')
        time.sleep(0.1)
    if gpu_pids() != ours:
        raise RuntimeError('GPU ownership changed during guard handshake')
    return token


def stop_group(process: subprocess.Popen) -> None:
    """Terminate only the process group created for this job, then escalate."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def supervise(command: list[str], job_dir: Path) -> int:
    job_dir.mkdir(parents=True, exist_ok=True)
    # A fresh directory prevents stale handshake files from authorizing a new run.
    status = job_dir / 'guard-status.json'
    if status.exists() or (job_dir / 'gpu-ready.json').exists():
        raise RuntimeError('Use a fresh job directory; automatic restart is disabled')
    state = {'automatic_restart': False, 'poll_seconds': 1, 'command': command}
    worker = guard = None
    try:
        busy = gpu_pids()
        if busy:
            state.update(status='paused', reason=f'GPU already occupied: {sorted(busy)}')
            write_json(status, state)
            return 75
        with (job_dir / 'run.log').open('ab', buffering=0) as log:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='arming', pid=worker.pid, pgid=worker.pid)
        write_json(status, state)
        deadline = time.monotonic() + 120
        ready = job_dir / 'gpu-ready.json'
        while not ready.exists():
            if worker.poll() is not None:
                raise RuntimeError(f'Worker exited before GPU handshake: {worker.returncode}')
            if time.monotonic() > deadline:
                raise RuntimeError('GPU initialization timed out')
            time.sleep(0.1)
        own = json.loads(ready.read_text())
        ours = set(own['gpu_pids'])
        if own['pid'] != worker.pid or len(ours) != 1 or gpu_pids() != ours:
            raise RuntimeError('GPU ownership is ambiguous; refusing to arm')
        guard_log = job_dir / 'gpu_guard.log'
        with (job_dir / 'gpu_guard.out').open('ab', buffering=0) as log:
            guard = subprocess.Popen(
                [sys.executable, str(GUARD), str(worker.pid), '--log', str(guard_log),
                 '--interval', '1', '--expected-pid', next(iter(ours))], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        deadline = time.monotonic() + 10
        expected = f'baseline GPU pids={sorted(ours)}'
        while not guard_log.exists() or expected not in guard_log.read_text():
            if guard.poll() is not None or time.monotonic() > deadline or gpu_pids() != ours:
                raise RuntimeError('Guard failed to establish the expected GPU baseline')
            time.sleep(0.1)
        if gpu_pids() != ours:
            raise RuntimeError('Another GPU process appeared while arming')
        state.update(status='running', gpu_pids=sorted(ours), guard_pid=guard.pid)
        write_json(status, state)
        (job_dir / 'gpu-go').touch()
        while worker.poll() is None:
            # Independent check also closes query failures and unexpected guard exits.
            foreign = gpu_pids() - ours
            if foreign:
                raise RuntimeError(f'Another GPU process appeared: {sorted(foreign)}')
            if guard.poll() is not None:
                if worker.poll() is not None:
                    break
                raise RuntimeError('GPU guard exited while the worker was active')
            time.sleep(1)
        yielded = guard_log.exists() and 'foreign GPU process detected' in guard_log.read_text()
        state.update(status='paused' if yielded else ('completed' if worker.returncode == 0 else 'failed'),
                     exit_code=worker.returncode)
        if yielded:
            state['reason'] = 'Shared GPU guard stopped our job for another GPU process'
        return worker.returncode
    except BaseException as exc:
        state.update(status='paused', reason=str(exc))
        if worker is not None:
            stop_group(worker)
        return 75
    finally:
        if worker is not None and worker.poll() is None:
            stop_group(worker)
        if guard is not None and guard.poll() is None:
            guard.terminate()
            try:
                guard.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guard.kill()
                guard.wait()
        write_json(status, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a worker command is required')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    return supervise(command, args.job_dir.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
