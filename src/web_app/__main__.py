"""``python -m src.web_app``: start Finnie from the project root.

Streamlit reads ``.streamlit/config.toml`` (the theme, among others) from the folder it's
started in. Started from anywhere else, the app falls back to Streamlit's default theme,
possibly dark. This launcher always starts it from the project root. Extra arguments are
passed to ``streamlit run``, e.g. ``python -m src.web_app --server.port 8600``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT)
    sys.argv = ["streamlit", "run", str(ROOT / "src" / "web_app" / "app.py"), *(argv or [])]
    return int(cli.main(standalone_mode=False) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
