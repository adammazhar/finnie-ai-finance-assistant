import os
import sys

from src.web_app import __main__ as launcher


def test_launcher_starts_streamlit_from_the_project_root(monkeypatch, tmp_path):
    calls = {}

    def fake_main(standalone_mode):
        calls["argv"] = list(sys.argv)
        calls["cwd"] = os.getcwd()
        return None

    monkeypatch.chdir(tmp_path)  # started from somewhere else
    monkeypatch.setattr(launcher.cli, "main", fake_main)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    assert launcher.main(["--server.port", "8600"]) == 0
    assert calls["cwd"] == str(launcher.ROOT)  # so .streamlit/config.toml (the theme) loads
    assert calls["argv"] == [
        "streamlit",
        "run",
        str(launcher.ROOT / "src" / "web_app" / "app.py"),
        "--server.port",
        "8600",
    ]
