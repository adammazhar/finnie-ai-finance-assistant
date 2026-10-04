"""Link checker for every source URL in the knowledge base and reference data.

It fails (non-zero exit) when a URL is unreachable or looks invented:

- the domain isn't on the allow-list of reputable sources
- the URL contains placeholder text (``example.com``, ``...``, ``your-url-here``)
- DNS failure, timeout, or a 4xx/5xx response after retries
- a "soft 404": the site answers 200 for any path (investor.gov's glossary does this), and
  the cited page has the same <title> as a made-up sibling URL on that site

Requests identify themselves honestly (``FinnieLinkChecker/1.0``) and never impersonate a
browser. They try HEAD first, fall back to GET when HEAD is refused, follow redirects, and
retry transient failures with backoff. A site that refuses automated checks (HTTP
401/403/429) fails unless the URL has an entry in ``data/knowledge_base/link_allowlist.yaml``
that gives a reason; such URLs are reported as allow-listed, never silently skipped.
Run: ``python scripts/check_kb_links.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from src.core.config import PROJECT_ROOT
from src.rag.knowledge_base import GLOSSARY_FILE, KB_ROOT, ArticleError, parse_article

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ALLOWLIST_FILE = KB_ROOT / "link_allowlist.yaml"
REPORT_FILE = KB_ROOT / "link_report.json"

ALLOWED_DOMAINS = frozenset(
    {
        # US government and regulators
        "investor.gov",
        "sec.gov",
        "finra.org",
        "irs.gov",
        "treasury.gov",
        "treasurydirect.gov",
        "federalreserve.gov",
        "federalreservehistory.org",
        "newyorkfed.org",
        "stlouisfed.org",
        "consumerfinance.gov",
        "fdic.gov",
        "ssa.gov",
        "dol.gov",
        "bls.gov",
        "bea.gov",
        "census.gov",
        "ftc.gov",
        "usa.gov",
        "mymoney.gov",
        "studentaid.gov",
        "ed.gov",
        "sipc.org",
        "nber.org",
        "cbo.gov",
        "healthcare.gov",
        "annualcreditreport.com",
        "identitytheft.gov",
        "fiscaldata.treasury.gov",
        "cftc.gov",
        # Educational references
        "bogleheads.org",
        "myfico.com",
        "khanacademy.org",
        # Index providers and exchanges
        "spglobal.com",
        "nasdaq.com",
        "nyse.com",
        "msci.com",
        "ftserussell.com",
        # Fund providers (expense ratio sources)
        "vanguard.com",
        "ishares.com",
        "blackrock.com",
        "invesco.com",
        "ssga.com",
        "spdrgoldshares.com",
        "schwab.com",
        "schwabassetmanagement.com",
        "fidelity.com",
    }
)
PLACEHOLDER_URL = re.compile(
    r"example\.(com|org|net)|\.\.\.|localhost|lorem|placeholder|your-(?:url|link|site|page)|xxx",
    re.IGNORECASE,
)
BLOCKED_STATUSES = frozenset({401, 403, 429})
HEAD_REFUSED = frozenset({400, 403, 404, 405, 406, 429, 501})
CHECKER_UA = "FinnieLinkChecker/1.0 (+education project; link validation)"
SOFT_404_PROBE = "finnie-link-check-no-such-page-7f3a9"
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
CANONICAL = re.compile(r"<link\b[^>]*\brel=[\"']canonical[\"'][^>]*>", re.IGNORECASE)
HREF = re.compile(r"\bhref=[\"']([^\"']+)[\"']", re.IGNORECASE)
SLUG_STOPWORDS = frozenset(
    {"the", "and", "for", "www", "com", "html", "htm", "asp", "php", "index"}
)
_PROBE_CACHE: dict[str, tuple[int, str | None, str | None]] = {}
_PROBE_LOCK = threading.Lock()


@dataclass
class LinkResult:
    """Outcome of checking one URL.

    ``used_by`` lists the files that cite it; ``allowlisted`` marks a failure overridden by the
    allow-list.
    """

    url: str
    ok: bool
    status: int | None = None
    final_url: str | None = None
    method: str | None = None
    error: str | None = None
    allowlisted: bool = False
    allowlist_reason: str | None = None
    used_by: list[str] = field(default_factory=list)


def domain_allowed(url: str) -> bool:
    """Whether the URL's host is an allow-listed domain or a subdomain of one."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith(f".{d}") for d in ALLOWED_DOMAINS)


# ---- collection -----------------------------------------------------------------------


def _source_urls(node: Any) -> Iterable[str]:
    """Every ``source_url`` / ``url`` string nested anywhere in a YAML document."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("source_url", "url") and isinstance(value, str):
                yield value
            else:
                yield from _source_urls(value)
    elif isinstance(node, list):
        for item in node:
            yield from _source_urls(item)


def _display_path(path: Path) -> str:
    """Project-relative path for reports, so they don't embed local user directories."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_urls(
    kb_root: Path = KB_ROOT,
    reference_dir: Path = REFERENCE_DIR,
    paths: Sequence[Path] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Map each URL to the files that cite it. Returns (urls, files that couldn't be read)."""
    urls: dict[str, list[str]] = {}
    unreadable: list[str] = []

    def add(url: str, origin: Path) -> None:
        urls.setdefault(url, []).append(_display_path(origin))

    targets = (
        list(paths)
        if paths
        else [
            *sorted(kb_root.glob("*/*.md")),
            kb_root / GLOSSARY_FILE,
            *sorted(reference_dir.glob("*.yaml")),
        ]
    )
    for path in targets:
        if not path.is_file():
            continue
        if path.suffix == ".md":
            try:
                for source in parse_article(path).meta.sources:
                    add(source.url, path)
            except ArticleError as exc:
                unreadable.append(str(exc))
        else:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            for url in _source_urls(data):
                add(url, path)
    return urls, unreadable


# ---- checking -------------------------------------------------------------------------


def _request(
    session: requests.Session, method: str, url: str, agent: str, timeout: float
) -> requests.Response:
    response = session.request(
        method,
        url,
        headers={"User-Agent": agent, "Accept": "text/html,application/pdf,*/*"},
        allow_redirects=True,
        timeout=timeout,
        stream=method == "GET",
    )
    response.close()
    return response


def _normalize(url: str) -> str:
    parts = urlparse(url)
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}"


def _page_info(
    session: requests.Session, url: str, timeout: float
) -> tuple[int, str | None, str | None]:
    """Status, normalized <title>, and canonical URL of a page."""
    response = session.get(
        url, headers={"User-Agent": CHECKER_UA}, allow_redirects=True, timeout=timeout
    )
    head = response.text[:200_000]
    match = TITLE.search(head)
    title = " ".join(match.group(1).split()).lower() if match else None
    tag = CANONICAL.search(head)
    href = HREF.search(tag.group(0)) if tag else None
    return response.status_code, title, href.group(1) if href else None


def _sibling(url: str) -> str:
    parts = urlparse(url)
    parent = parts.path.rstrip("/").rsplit("/", 1)[0]
    return parts._replace(path=f"{parent}/{SOFT_404_PROBE}", query="", fragment="").geturl()


def _words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and not w.isdigit()} - SLUG_STOPWORDS


def title_matches_slug(url: str, title: str) -> bool | None:
    """Whether the page title shares a word with the URL's last path segment.

    ``None`` when the segment has no meaningful words (e.g. a numeric ID) to compare.
    """
    slug_words = _words(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    if not slug_words:
        return None
    return bool(slug_words & _words(title))


def is_soft_404(session: requests.Session, url: str, timeout: float = 15) -> bool:
    """True when a 200 page is indistinguishable, by title, from a made-up sibling URL.

    One probe per directory is cached. PDFs and pages without a <title> are never flagged,
    and a failure during this extra check never fails the link on its own.

    Some sites (iShares) route by an ID and ignore the rest of the path, so a made-up sibling
    serves the real page. Those aren't soft 404s: the cited page's title then reflects its own
    URL (it shares a word with the last path segment), or the probe's canonical link points
    back at the cited page. A generic not-found page's title matches neither.
    """
    if urlparse(url).path.lower().endswith(".pdf"):
        return False
    probe_url = _sibling(url)
    try:
        with _PROBE_LOCK:
            probe = _PROBE_CACHE.get(probe_url)
        if probe is None:
            probe = _page_info(session, probe_url, timeout)
            with _PROBE_LOCK:
                _PROBE_CACHE[probe_url] = probe
        probe_status, probe_title, probe_canonical = probe
        if probe_status != 200 or probe_title is None:
            return False
        if probe_canonical and _normalize(probe_canonical) == _normalize(url):
            return False
        status, title, _ = _page_info(session, url, timeout)
    except requests.RequestException:
        return False
    if status != 200 or title != probe_title:
        return False
    return title_matches_slug(url, title) is not True


def check_url(
    session: requests.Session,
    url: str,
    *,
    attempts: int = 3,
    timeout: float = 15,
    sleep: Callable[[float], None] = time.sleep,
) -> LinkResult:
    """Check one URL: placeholder, https, and domain rules, then HEAD/GET with retries.

    A page that loads is also probed for a soft 404. Never raises for network errors; a failure
    comes back as ``ok=False`` with an ``error``.
    """
    if PLACEHOLDER_URL.search(url):
        return LinkResult(url=url, ok=False, error="looks like a placeholder URL")
    if not url.startswith("https://"):
        return LinkResult(url=url, ok=False, error="must use https")
    if not domain_allowed(url):
        return LinkResult(url=url, ok=False, error="domain is not on the allow-list")

    last = LinkResult(url=url, ok=False, error="not checked")
    for attempt in range(attempts):
        if attempt:
            # Bot-protection and rate-limit responses need a longer pause than server errors.
            throttled = last.status in BLOCKED_STATUSES
            sleep(5 * 3 ** (attempt - 1) if throttled else 2 ** (attempt - 1))
        for method in ("HEAD", "GET"):
            try:
                response = _request(session, method, url, CHECKER_UA, timeout)
            except requests.RequestException as exc:
                last = LinkResult(url=url, ok=False, method=method, error=type(exc).__name__)
                break  # transport failure: back off and retry the whole sequence
            last = LinkResult(
                url=url,
                ok=200 <= response.status_code < 400,
                status=response.status_code,
                final_url=response.url,
                method=method,
            )
            if last.ok:
                if is_soft_404(session, url, timeout):
                    last.ok = False
                    last.error = (
                        "soft 404: the page looks the same as the site's page for a made-up URL"
                    )
                return last
            if method == "HEAD" and response.status_code not in HEAD_REFUSED:
                break
        if last.status is not None and last.status < 500 and last.status not in BLOCKED_STATUSES:
            break  # a definite client error (e.g. 404) won't fix itself
    if last.error is None:
        last.error = f"HTTP {last.status}"
    return last


def load_allowlist(path: Path = ALLOWLIST_FILE) -> dict[str, str]:
    """URL -> reason. Every entry must explain why the URL can't be checked automatically."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("urls") or {}
    missing = [url for url, reason in entries.items() if not str(reason or "").strip()]
    if missing:
        raise ValueError(f"Allow-list entries need a reason: {missing}")
    return {url: str(reason).strip() for url, reason in entries.items()}


def check_links(
    urls: dict[str, list[str]],
    *,
    allowlist: dict[str, str] | None = None,
    session: requests.Session | None = None,
    workers: int = 8,
    per_host: int = 2,
    sleep: Callable[[float], None] | None = None,
) -> list[LinkResult]:
    """Check URLs in parallel, but never more than ``per_host`` at once against one site."""
    pause = sleep or time.sleep  # resolved at call time so tests can patch time.sleep
    allowlist = allowlist or {}
    session = session or requests.Session()
    host_limits: dict[str, threading.Semaphore] = {}
    limits_lock = threading.Lock()

    def host_limit(url: str) -> threading.Semaphore:
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        with limits_lock:
            return host_limits.setdefault(host, threading.Semaphore(max(1, per_host)))

    def run(url: str) -> LinkResult:
        with host_limit(url):
            result = check_url(session, url, sleep=pause)
        result.used_by = urls[url]
        if not result.ok and url in allowlist and domain_allowed(url):
            result.ok, result.allowlisted, result.allowlist_reason = True, True, allowlist[url]
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return sorted(pool.map(run, urls), key=lambda r: r.url)


def write_report(results: Sequence[LinkResult], path: Path, stale: Sequence[str]) -> None:
    """Write the JSON report: totals, stale allow-list entries, and every result."""
    failed = [r for r in results if not r.ok]
    report = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "allowlisted": sum(r.allowlisted for r in results),
        "stale_allowlist_entries": list(stale),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None, *, session: requests.Session | None = None) -> int:
    """Command-line entry point for ``scripts/check_kb_links.py``.

    Checks the links, writes the JSON report unless ``--no-report``, and prints failures. Returns 1
    if a link failed, a file was unreadable, or an allow-list entry is no longer cited; otherwise 0.
    """
    parser = argparse.ArgumentParser(description="Check knowledge base and reference links.")
    parser.add_argument("paths", nargs="*", type=Path, help="limit to these files")
    parser.add_argument("--report", type=Path, default=REPORT_FILE)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    urls, unreadable = collect_urls(paths=args.paths or None)
    allowlist = load_allowlist()
    results = check_links(urls, allowlist=allowlist, session=session, workers=args.workers)
    stale = [] if args.paths else sorted(set(allowlist) - set(urls))
    if not args.no_report:
        write_report(results, args.report, stale)

    failed = [r for r in results if not r.ok]
    for r in results:
        if r.allowlisted:
            print(f"ALLOWLISTED {r.url} ({r.error}): {r.allowlist_reason}")
    for r in failed:
        print(f"FAILED {r.url}: {r.error} (cited by {', '.join(r.used_by)})", file=sys.stderr)
    for problem in unreadable:
        print(f"UNREADABLE {problem}", file=sys.stderr)
    for url in stale:
        print(f"STALE allow-list entry (no longer cited): {url}", file=sys.stderr)
    print(
        f"{len(results)} URLs: {len(results) - len(failed)} ok, {len(failed)} failed, "
        f"{sum(r.allowlisted for r in results)} allow-listed"
    )
    return 1 if failed or unreadable or stale else 0
