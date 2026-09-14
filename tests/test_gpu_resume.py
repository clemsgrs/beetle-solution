import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from beetle import gpu_resume

A, B = 'GPU-aaaa', 'GPU-bbbb'


def tracker(queries, now):
    """An idle tracker over GPUs A and B fed by a scripted sequence of queries."""
    queries = iter(queries)

    def query(uuids):
        value = next(queries)
        if isinstance(value, Exception):
            raise value
        return {uuid: set(value.get(uuid, ())) for uuid in uuids}

    return gpu_resume.IdleGpus([A, B], query=query, clock=lambda: now[0])


def ticking(now):
    return lambda seconds: now.__setitem__(0, now[0] + seconds)


def test_waits_for_an_unbroken_idle_window_on_one_gpu(tmp_path):
    now = [0.0]
    busy = {B: {'1'}}
    gpus = tracker([busy, busy, {A: {'999'}, **busy}, busy, busy, busy, busy], now)
    chosen = gpu_resume.wait_for_gpu(gpus, 20, 10, tmp_path / 'stop', Mock(), sleep=ticking(now))
    # A is idle at 0-10 s, busy at 20 s, then idle from 30 s: its window restarts and ends at 50 s.
    assert (chosen, now[0]) == (A, 50)


def test_picks_the_gpu_idle_longest():
    now = [0.0]
    gpus = tracker([{A: {'1'}}, {}], now)
    gpus.observe()  # B idle since 0.
    now[0] = 30
    gpus.observe()  # A idle since 30.
    now[0] = 100
    assert gpus.idlest(60) == B
    assert gpus.idlest(0) == B


def test_query_failure_restarts_every_window_and_stop_file_ends_wait(tmp_path):
    now = [0.0]
    gpus = tracker([{}, RuntimeError('nvidia-smi failed'), {}], now)
    gpus.observe()
    now[0] = 100
    assert gpus.observe() == {'query failed': ["RuntimeError('nvidia-smi failed')"]}
    assert gpus.idlest(0) is None
    gpus.observe()
    assert gpus.idlest(1) is None
    stop = tmp_path / 'stop'
    stop.touch()
    assert gpu_resume.wait_for_gpu(gpus, 0, 1, stop, Mock(), sleep=lambda _: None) is None


def test_process_query_requires_every_candidate(monkeypatch):
    outputs = {'--query-gpu=uuid': f'{A}\n{B}\n', '--query-compute-apps=gpu_uuid,pid': f'{A}, 12\nGPU-other, 13\n'}
    run = lambda command, **kwargs: Mock(stdout=outputs[command[1]])
    monkeypatch.setattr(gpu_resume.subprocess, 'run', run)
    assert gpu_resume.gpu_processes([A, B]) == {A: {'12'}, B: set()}
    with pytest.raises(RuntimeError, match='not reported'):
        gpu_resume.gpu_processes([A, 'GPU-gone'])
    outputs['--query-compute-apps=gpu_uuid,pid'] = f'{A}, [N/A]\n'
    with pytest.raises(RuntimeError, match='Unexpected'):
        gpu_resume.gpu_processes([A, B])


@pytest.mark.parametrize('case, expected', [
    ('supervisor_saw_foreign', True), ('guard_saw_foreign', True),
    ('busy_during_handshake', True), ('crash_on_idle_gpu', False)])
def test_yield_classification(tmp_path, monkeypatch, case, expected):
    monkeypatch.setattr(gpu_resume, 'gpu_pids', lambda uuid: set())
    status = {'status': 'failed', 'reason': ''}
    if case == 'supervisor_saw_foreign':
        status = {'status': 'paused', 'reason': "Another GPU process appeared: ['999']"}
    elif case == 'guard_saw_foreign':
        (tmp_path / 'gpu_guard.log').write_text("foreign GPU process detected: ['999']; stopping job\n")
    elif case == 'busy_during_handshake':
        (tmp_path / 'run.log').write_text('RuntimeError: GPU became busy before CUDA initialization\n')
    else:
        (tmp_path / 'run.log').write_text('ValueError: Prepared config changed\n')
    assert gpu_resume.yielded(tmp_path, status, A) is expected


def fake_jobs(monkeypatch, outcomes, launched):
    """Each launch writes the next guard status into its job directory."""
    outcomes = iter(outcomes)

    def spawn(command, env, **kwargs):
        job_dir = Path(command[command.index('--job-dir') + 1])
        job_dir.mkdir()
        launched.append((job_dir.name, env['CUDA_VISIBLE_DEVICES'], command[command.index('--') + 1:]))
        status, reason, guard_log = next(outcomes)
        (job_dir / 'guard-status.json').write_text(json.dumps({'status': status, 'reason': reason}))
        if guard_log:
            (job_dir / 'gpu_guard.log').write_text(guard_log)
        return Mock(wait=Mock(return_value=0 if status == 'completed' else 75), poll=Mock(return_value=0))

    monkeypatch.setattr(gpu_resume.subprocess, 'Popen', spawn)
    monkeypatch.setattr(gpu_resume, 'gpu_pids', lambda uuid: set())


YIELD = ('paused', 'Shared GPU guard stopped our job for another GPU process',
         "foreign GPU process detected: ['999']; stopping job\n")


def test_relaunches_after_a_yield_until_completed(tmp_path, monkeypatch):
    (tmp_path / 'run-04').mkdir()
    waits, gpus = [], iter([A, B])
    monkeypatch.setattr(gpu_resume, 'wait_for_gpu', lambda tracker, seconds, *args: waits.append(seconds) or next(gpus))
    launched = []
    fake_jobs(monkeypatch, [YIELD, ('completed', '', None)], launched)
    assert gpu_resume.resume(['worker', '--job-dir', '{job_dir}'], tmp_path, [A, B], idle_seconds=600) == 0
    assert [(name, gpu) for name, gpu, _ in launched] == [('run-05', A), ('run-06', B)]
    assert launched[1][2] == ['worker', '--job-dir', str(tmp_path / 'run-06')]
    assert waits == [0, 600]
    state = json.loads((tmp_path / 'autoresume-status.json').read_text())
    assert state['state'] == 'completed' and state['runs'][0]['yielded'] and state['runs'][0]['gpu'] == A


def test_yield_moves_at_once_to_a_gpu_that_stayed_idle_during_the_run(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(gpu_resume.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(gpu_resume.time, 'sleep', ticking(now))
    # B is taken before the first launch and frees up during our run on A; then A is taken.
    queries = [{B: {'7'}}] + [{A: {'1'}}] * 200 + [{A: {'999'}}] * 5
    gpus = tracker(queries, now)
    launched = []
    fake_jobs(monkeypatch, [YIELD, ('completed', '', None)], launched)

    real_spawn = gpu_resume.subprocess.Popen
    def spawn(command, **kwargs):
        process = real_spawn(command, **kwargs)
        if len(launched) == 1:  # The first job runs 1,000 s of polls, then yields.
            polls = iter([None] * 200 + [75])
            process.poll = Mock(side_effect=lambda: next(polls))
        return process
    monkeypatch.setattr(gpu_resume.subprocess, 'Popen', spawn)

    assert gpu_resume.resume(['worker'], tmp_path, [A, B], idle_seconds=600, tracker=gpus) == 0
    assert [gpu for _, gpu, _ in launched] == [A, B]
    assert now[0] == 1000  # No extra wait: B had been idle for 995 s when A yielded.


def test_failure_is_left_for_a_person(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_resume, 'wait_for_gpu', lambda *args: A)
    launched = []
    fake_jobs(monkeypatch, [('failed', '', None), ('completed', '', None)], launched)
    assert gpu_resume.resume(['worker'], tmp_path, [A, B], idle_seconds=600) == 1
    assert len(launched) == 1
    assert json.loads((tmp_path / 'autoresume-status.json').read_text())['state'] == 'needs_attention'


def test_repeated_quick_yields_stop_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_resume, 'wait_for_gpu', lambda *args: A)
    launched = []
    fake_jobs(monkeypatch, [('paused', 'GPU already occupied: [999]', None)] * 3, launched)
    assert gpu_resume.resume(['worker'], tmp_path, [A, B], idle_seconds=600, max_quick_yields=3) == 1
    assert len(launched) == 3


def test_stop_file_prevents_launch(tmp_path, monkeypatch):
    (tmp_path / 'autoresume.stop').touch()
    launch = Mock()
    monkeypatch.setattr(gpu_resume.subprocess, 'Popen', launch)
    gpus = tracker([{}], [0.0])
    assert gpu_resume.resume(['worker'], tmp_path, [A, B], idle_seconds=600, tracker=gpus) == 0
    launch.assert_not_called()


def test_gpu_list_refuses_indices_and_duplicates():
    assert gpu_resume.gpu_list(f'{A},{B}') == [A, B]
    for value in ('0,1', f'{A},{A}', ''):
        with pytest.raises(Exception):
            gpu_resume.gpu_list(value)
