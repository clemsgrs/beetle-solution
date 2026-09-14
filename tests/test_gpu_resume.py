import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from beetle import gpu_resume


def test_waits_for_an_unbroken_idle_window(tmp_path, monkeypatch):
    now = [0.0]
    queries = iter([set(), set(), {'999'}, set(), set(), set(), set()])
    monkeypatch.setattr(gpu_resume, 'gpu_pids', lambda: next(queries))
    sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    assert gpu_resume.wait_for_idle(20, 10, tmp_path / 'stop', Mock(), clock=lambda: now[0], sleep=sleep)
    # Idle at 0-10 s, busy at 20 s, then idle from 30 s: the window restarts and ends at 50 s.
    assert now[0] == 50


def test_query_failure_is_not_idle_and_stop_file_ends_wait(tmp_path, monkeypatch):
    stop = tmp_path / 'stop'
    def query():
        stop.touch()
        raise RuntimeError('nvidia-smi failed')
    monkeypatch.setattr(gpu_resume, 'gpu_pids', query)
    assert not gpu_resume.wait_for_idle(0, 1, stop, Mock(), sleep=lambda _: None)


@pytest.mark.parametrize('case, expected', [
    ('supervisor_saw_foreign', True), ('guard_saw_foreign', True),
    ('busy_during_handshake', True), ('crash_on_idle_gpu', False)])
def test_yield_classification(tmp_path, monkeypatch, case, expected):
    monkeypatch.setattr(gpu_resume, 'gpu_pids', lambda: set())
    status = {'status': 'failed', 'reason': ''}
    if case == 'supervisor_saw_foreign':
        status = {'status': 'paused', 'reason': "Another GPU process appeared: ['999']"}
    elif case == 'guard_saw_foreign':
        (tmp_path / 'gpu_guard.log').write_text("foreign GPU process detected: ['999']; stopping job\n")
    elif case == 'busy_during_handshake':
        (tmp_path / 'run.log').write_text('RuntimeError: GPU became busy before CUDA initialization\n')
    else:
        (tmp_path / 'run.log').write_text('ValueError: Prepared config changed\n')
    assert gpu_resume.yielded(tmp_path, status) is expected


def fake_jobs(monkeypatch, outcomes, launched):
    """Each launch writes the next guard status into its job directory."""
    outcomes = iter(outcomes)

    def spawn(command, **kwargs):
        job_dir = Path(command[command.index('--job-dir') + 1])
        job_dir.mkdir()
        launched.append((job_dir.name, command[command.index('--') + 1:]))
        status, reason, guard_log = next(outcomes)
        (job_dir / 'guard-status.json').write_text(json.dumps({'status': status, 'reason': reason}))
        if guard_log:
            (job_dir / 'gpu_guard.log').write_text(guard_log)
        return Mock(wait=Mock(return_value=0 if status == 'completed' else 75), poll=Mock(return_value=0))

    monkeypatch.setattr(gpu_resume.subprocess, 'Popen', spawn)
    monkeypatch.setattr(gpu_resume, 'gpu_pids', lambda: set())


def test_relaunches_after_a_yield_until_completed(tmp_path, monkeypatch):
    (tmp_path / 'run-04').mkdir()
    waits = []
    monkeypatch.setattr(gpu_resume, 'wait_for_idle', lambda seconds, *args: waits.append(seconds) or True)
    launched = []
    fake_jobs(monkeypatch, [
        ('paused', 'Shared GPU guard stopped our job for another GPU process',
         "foreign GPU process detected: ['999']; stopping job\n"),
        ('completed', '', None)], launched)
    assert gpu_resume.resume(['worker', '--job-dir', '{job_dir}'], tmp_path, idle_seconds=600) == 0
    assert [name for name, _ in launched] == ['run-05', 'run-06']
    assert launched[1][1] == ['worker', '--job-dir', str(tmp_path / 'run-06')]
    assert waits == [0, 600]
    state = json.loads((tmp_path / 'autoresume-status.json').read_text())
    assert state['state'] == 'completed' and state['runs'][0]['yielded']


def test_failure_is_left_for_a_person(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_resume, 'wait_for_idle', lambda *args: True)
    launched = []
    fake_jobs(monkeypatch, [('failed', '', None), ('completed', '', None)], launched)
    assert gpu_resume.resume(['worker'], tmp_path, idle_seconds=600) == 1
    assert len(launched) == 1
    assert json.loads((tmp_path / 'autoresume-status.json').read_text())['state'] == 'needs_attention'


def test_repeated_quick_yields_stop_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_resume, 'wait_for_idle', lambda *args: True)
    launched = []
    fake_jobs(monkeypatch, [('paused', 'GPU already occupied: [999]', None)] * 3, launched)
    assert gpu_resume.resume(['worker'], tmp_path, idle_seconds=600, max_quick_yields=3) == 1
    assert len(launched) == 3


def test_stop_file_prevents_launch(tmp_path, monkeypatch):
    (tmp_path / 'autoresume.stop').touch()
    monkeypatch.setattr(gpu_resume, 'gpu_pids', lambda: set())
    launch = Mock()
    monkeypatch.setattr(gpu_resume.subprocess, 'Popen', launch)
    assert gpu_resume.resume(['worker'], tmp_path, idle_seconds=600) == 0
    launch.assert_not_called()
