"""``python -m src.mcp_server``: picking the transport, and refusing to start badly."""

from __future__ import annotations

from typing import Any

import pytest
import uvicorn

from src.core.config import PROJECT_ROOT
from src.mcp_server import __main__ as launcher
from src.mcp_server import server as server_module
from tests.unit.mcp_server.support import TOKEN


class FakeServer:
    def __init__(self) -> None:
        self.transports: list[str] = []

    def run(self, transport: str) -> None:
        self.transports.append(transport)


@pytest.fixture
def fake_server(monkeypatch, tmp_path) -> FakeServer:
    monkeypatch.chdir(tmp_path)  # main() changes folder; this restores it afterwards
    fake = FakeServer()
    monkeypatch.setattr(server_module, "build_server", lambda: fake)
    return fake


def test_default_is_stdio_from_the_project_root(fake_server):
    assert launcher.main([]) == 0
    assert fake_server.transports == ["stdio"]
    assert PROJECT_ROOT.samefile(".")


def test_http_flag(monkeypatch, fake_server):
    seen: dict[str, Any] = {}

    def serve(build, settings, *, port):
        seen.update(build=build, port=port)
        return 0

    monkeypatch.setattr(launcher, "serve_http", serve)
    assert launcher.main(["--http", "--port", "9000"]) == 0
    assert seen["port"] == 9000 and seen["build"]() is fake_server


def test_bad_config_exits_with_a_message(monkeypatch, fake_server, tmp_path, capsys):
    monkeypatch.setenv("FINNIE_CONFIG", str(tmp_path / "missing.yaml"))
    assert launcher.main([]) == 2
    assert "Config file not found" in capsys.readouterr().err
    assert fake_server.transports == []


def test_http_refuses_to_start_without_a_token(monkeypatch, make_settings, services, capsys):
    started = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(a))
    code = launcher.serve_http(lambda: server_module.build_server(services), make_settings())
    assert code == 2 and not started
    assert "MCP_API_TOKEN isn't set" in capsys.readouterr().err


def test_http_serves_on_localhost(monkeypatch, make_settings, services, capsys):
    started: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.append(kwargs))
    settings = make_settings(mcp_api_token=TOKEN)
    code = launcher.serve_http(lambda: server_module.build_server(services), settings, port=9001)
    assert code == 0
    assert started == [{"host": "127.0.0.1", "port": 9001, "log_level": "warning"}]
    err = capsys.readouterr().err
    assert "http://127.0.0.1:9001/mcp" in err and TOKEN not in err
