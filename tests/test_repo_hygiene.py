"""Last line of defense against committing secrets or course materials."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from src.core.config import PROJECT_ROOT
from src.utils.logging import SECRET_PATTERNS

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

MIN_SECRET_LEN = 12


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout


def committable_files() -> list[str]:
    """Tracked files plus untracked files that are not ignored (i.e. could be staged)."""
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return [p for p in out.split("\0") if p and (PROJECT_ROOT / p).is_file()]


def real_secret_values() -> list[str]:
    env = PROJECT_ROOT / ".env"
    if not env.is_file():
        return []
    values = []
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        name, sep, value = line.partition("=")
        value = value.strip().strip("'\"")
        if sep and name.strip().endswith(("_KEY", "_TOKEN")) and len(value) >= MIN_SECRET_LEN:
            values.append(value)
    return values


@pytest.mark.parametrize("path", [".env", "docs/ik/anything.docx"])
def test_sensitive_paths_are_gitignored(path):
    assert git("check-ignore", path).strip() == path


def test_nothing_sensitive_is_tracked():
    tracked = git("ls-files").splitlines()
    assert ".env" not in tracked
    assert not [p for p in tracked if p.startswith("docs/ik/")]


def test_no_secrets_in_committable_files():
    real = real_secret_values()
    offenders = []
    for rel in committable_files():
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        if any(value in text for value in real):
            offenders.append(f"{rel}: contains a value from .env")
        for pattern in SECRET_PATTERNS[:3]:  # key-shaped tokens (sk-, sk-ant-, tvly-)
            for match in pattern.finditer(text):
                token = match.group(0)
                if not re.search(r"test|fake|example|0{6}|abcdef|leaked|secret", token):
                    offenders.append(f"{rel}: {token[:8]}...")
    assert not offenders, "Possible secrets found:\n" + "\n".join(offenders)
