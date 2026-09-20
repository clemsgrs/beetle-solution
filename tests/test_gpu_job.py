import json
import subprocess
from unittest.mock import Mock

import pytest

from beetle import gpu_job


def test_busy_gpu_does_not_start_a_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_job, 'gpu_pids', lambda: {'999'})
    launch = Mock()
    monkeypatch.setattr(gpu_job.subprocess, 'Popen', launch)
    assert gpu_job.supervise(['worker'], tmp_path) == 75
    launch.assert_not_called()
    assert json.loads((tmp_path / 'guard-status.json').read_text())['status'] == 'paused'


@pytest.mark.parametrize('failure', ['foreign', 'query_error', 'guard_exit'])
def test_active_job_stops_on_foreign_process_or_lost_monitoring(tmp_path, monkeypatch, failure):
    worker = Mock(pid=123, returncode=None)
    worker.poll.return_value = None
    guard = Mock(pid=456)
    guard.poll.return_value = 1 if failure == 'guard_exit' else None

    def spawn(command, **kwargs):
        if command == ['worker']:
            gpu_job.write_json(tmp_path / 'gpu-ready.json', {'pid': 123, 'gpu_pids': ['12345']})
            return worker
        (tmp_path / 'gpu_guard.log').write_text("guard started; baseline GPU pids=['12345']\n")
        return guard

    queries = iter([set(), {'12345'}, {'12345'}, {'12345', '999'}])

    def query():
        value = next(queries)
        if '999' in value and failure == 'query_error':
            raise subprocess.TimeoutExpired('nvidia-smi', 5)
        return {'12345'} if '999' in value and failure == 'guard_exit' else value

    stopped = []

    def stop(process):
        stopped.append(process.pid)
        process.poll.return_value = -15

    monkeypatch.setattr(gpu_job, 'gpu_pids', query)
    monkeypatch.setattr(gpu_job.subprocess, 'Popen', spawn)
    monkeypatch.setattr(gpu_job, 'stop_group', stop)
    assert gpu_job.supervise(['worker'], tmp_path) == 75
    assert stopped == [123]
    status = json.loads((tmp_path / 'guard-status.json').read_text())
    assert status['status'] == 'paused'
    assert status['automatic_restart'] is False


def test_query_errors_are_not_treated_as_an_idle_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    monkeypatch.setattr(gpu_job.subprocess, 'run',
                        Mock(side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')))
    with pytest.raises(subprocess.CalledProcessError):
        gpu_job.gpu_pids()


def test_query_scoped_to_selected_physical_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-selected')
    query = Mock(return_value=Mock(stdout='123\n'))
    monkeypatch.setattr(gpu_job.subprocess, 'run', query)
    assert gpu_job.gpu_pids() == {'123'}
    assert query.call_args.args[0] == [
        'nvidia-smi', '--id', 'GPU-selected',
        '--query-compute-apps=pid', '--format=csv,noheader']


@pytest.mark.parametrize('failure', ['foreign', 'query_error', 'baseline'])
def test_independent_guard_stops_only_our_group(tmp_path, monkeypatch, failure):
    from beetle import gpu_guard
    monkeypatch.setattr('sys.argv', ['guard', '123', '--log', str(tmp_path / 'guard.log'),
                                   '--expected-pid', '100'])
    queries = [set()] if failure == 'baseline' else [
        {'100'}, RuntimeError('query failed') if failure == 'query_error' else {'100', '999'}]
    monkeypatch.setattr(gpu_guard, 'gpu_pids', Mock(side_effect=queries))
    signals = []
    def killpg(pgid, sig):
        assert pgid == 123
        if signals:
            raise ProcessLookupError
        if sig:
            signals.append(sig)
    monkeypatch.setattr(gpu_guard.os, 'killpg', killpg)
    assert gpu_guard.main() == 75
    assert signals == [gpu_guard.signal.SIGTERM]
