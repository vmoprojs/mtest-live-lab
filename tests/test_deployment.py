import subprocess
import pytest
from streamlit.testing.v1 import AppTest
from lab.data import BootstrapConfig, DataError, simulate
from lab.engines import EngineError, ROOT, run_python
from lab.runtime import analysis_slot, prepare_runtime


def test_public_defaults_and_fresh_session():
    a = AppTest.from_file(str(ROOT/'app.py'), default_timeout=40).run()
    assert not a.exception
    assert a.tabs[0].label == 'B · MTest en R'
    assert a.slider(key='rho').value == .95
    assert a.select_slider(key='n_boot').value == 200
    assert a.number_input(key='nsam').value == 300
    assert 'result_R' not in a.session_state
    a.session_state['private_marker'] = 'session-a'
    b = AppTest.from_file(str(ROOT/'app.py'), default_timeout=40).run()
    assert 'private_marker' not in b.session_state
    assert 'result_R' not in b.session_state


def test_capacity_is_released_even_after_error():
    with pytest.raises(ValueError):
        with analysis_slot():
            with pytest.raises(EngineError, match='otro análisis'):
                with analysis_slot(): pass
            raise ValueError()
    with analysis_slot(): pass


def test_public_workload_limits():
    for cfg in [BootstrapConfig(n_boot=501), BootstrapConfig(nsam=2001), BootstrapConfig(500,1000)]:
        with pytest.raises(DataError): cfg.validate(8)


def test_python_timeout_and_sanitized_failure(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 90)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(EngineError, match='Python superó 90'):
        run_python(simulate(), 'y', ['X1','X2'], BootstrapConfig())


def test_initializer_is_idempotent():
    assert prepare_runtime()['available']
    assert prepare_runtime()['available']


def test_rscript_discovery_without_activated_path(tmp_path, monkeypatch):
    import lab.engines as engines
    binary = tmp_path/'bin'/'Rscript'
    binary.parent.mkdir()
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    monkeypatch.delenv('MTEST_RSCRIPT', raising=False)
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(engines.sys, 'prefix', str(tmp_path))
    assert engines.rscript_path() == str(binary)


def test_public_server_configuration():
    import tomllib
    config = tomllib.loads((ROOT/'.streamlit/config.toml').read_text())
    assert config['server']['address'] == '0.0.0.0'
    assert config['server']['fileWatcherType'] == 'auto'
    assert config['server']['maxUploadSize'] == 2
    assert config['server'].get('enableXsrfProtection', True)
    assert config['server'].get('enableCORS', True)
