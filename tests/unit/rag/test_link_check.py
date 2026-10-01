import json
from pathlib import Path

import pytest
import requests
import responses
import yaml

from src.rag import link_check
from src.rag.link_check import (
    check_links,
    check_url,
    collect_urls,
    domain_allowed,
    load_allowlist,
    main,
)
from tests.unit.rag.test_knowledge_base import BODY, front, write

GOOD = "https://www.investor.gov/good"


@pytest.fixture
def session():
    link_check._PROBE_CACHE.clear()
    return requests.Session()


def no_sleep(_):
    return None


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://www.irs.gov/x", True),
        ("https://irs.gov/x", True),
        ("https://fund-docs.vanguard.com/F0970.pdf", True),
        ("https://notirs.gov/x", False),
        ("https://irs.gov.evil.com/x", False),
        ("https://randomblog.com/x", False),
    ],
)
def test_domain_allow_list(url, allowed):
    assert domain_allowed(url) is allowed


def test_real_urls_with_your_in_the_path_are_not_placeholders():
    from src.rag.link_check import PLACEHOLDER_URL

    assert not PLACEHOLDER_URL.search("https://www.consumerfinance.gov/blog/how-reduce-your-debt/")
    assert PLACEHOLDER_URL.search("https://www.irs.gov/your-url-here")


@pytest.mark.parametrize(
    ("url", "error"),
    [
        ("https://www.example.com/x", "placeholder"),
        ("https://www.irs.gov/.../x", "placeholder"),
        ("http://www.irs.gov/x", "https"),
        ("https://randomblog.com/x", "allow-list"),
    ],
)
def test_rejected_without_network(session, url, error):
    result = check_url(session, url, sleep=no_sleep)
    assert not result.ok and error in result.error


@responses.activate
def test_head_success(session):
    responses.add(responses.HEAD, GOOD, status=200)
    result = check_url(session, GOOD, sleep=no_sleep)
    assert result.ok and result.status == 200 and result.method.startswith("HEAD")


@responses.activate
def test_head_refused_then_get(session):
    responses.add(responses.HEAD, GOOD, status=405)
    responses.add(responses.GET, GOOD, status=200)
    result = check_url(session, GOOD, sleep=no_sleep)
    assert result.ok and result.method == "GET"


@responses.activate
def test_bot_blocked_site_fails_and_is_never_sent_a_browser_agent(session):
    responses.add(responses.HEAD, GOOD, status=403)
    responses.add(responses.GET, GOOD, status=403)
    result = check_url(session, GOOD, sleep=no_sleep)
    assert not result.ok and result.error == "HTTP 403"
    agents = {call.request.headers["User-Agent"] for call in responses.calls}
    assert agents == {link_check.CHECKER_UA}


@responses.activate
def test_not_found_fails_without_retrying(session):
    responses.add(responses.HEAD, GOOD, status=404)
    responses.add(responses.GET, GOOD, status=404)
    sleeps = []
    result = check_url(session, GOOD, sleep=sleeps.append)
    assert not result.ok and result.error == "HTTP 404" and sleeps == []


@responses.activate
def test_server_errors_retry_with_backoff(session):
    responses.add(responses.HEAD, GOOD, status=503)
    sleeps = []
    result = check_url(session, GOOD, sleep=sleeps.append)
    assert not result.ok and result.error == "HTTP 503" and sleeps == [1, 2]


@responses.activate
def test_transport_errors(session):
    responses.add(responses.HEAD, GOOD, body=requests.ConnectionError("dns"))
    sleeps = []
    result = check_url(session, GOOD, sleep=sleeps.append)
    assert not result.ok and result.error == "ConnectionError" and len(sleeps) == 2


@responses.activate
def test_recovers_on_retry(session):
    responses.add(responses.HEAD, GOOD, body=requests.Timeout())
    responses.add(responses.HEAD, GOOD, status=200)
    assert check_url(session, GOOD, sleep=no_sleep).ok


def test_collect_urls(tmp_path):
    kb, ref = tmp_path / "kb", tmp_path / "ref"
    write(kb, "stocks", "stocks-001", front() + BODY)
    write(kb, "stocks", "stocks-002", "broken, no front matter")
    (kb / "glossary.yaml").write_text(
        yaml.safe_dump({"sources": [{"name": "g", "url": "https://www.finra.org/g"}]}),
        encoding="utf-8",
    )
    ref.mkdir()
    (ref / "tax.yaml").write_text(
        yaml.safe_dump(
            {
                "figures": {"a": {"source_url": "https://www.irs.gov/a"}},
                "list": [{"source_url": "https://www.investor.gov/stocks"}],
            }
        ),
        encoding="utf-8",
    )
    urls, unreadable = collect_urls(kb, ref)
    assert set(urls) == {
        "https://www.investor.gov/stocks",
        "https://www.finra.org/g",
        "https://www.irs.gov/a",
    }
    assert len(urls["https://www.investor.gov/stocks"]) == 2  # cited by an article and ref data
    assert len(unreadable) == 1 and "stocks-002.md" in unreadable[0]
    only, _ = collect_urls(kb, ref, paths=[ref / "tax.yaml", ref / "missing.yaml"])
    assert set(only) == {"https://www.irs.gov/a", "https://www.investor.gov/stocks"}


def test_allowlist(tmp_path):
    assert load_allowlist(tmp_path / "none.yaml") == {}
    path = tmp_path / "allow.yaml"
    path.write_text(
        yaml.safe_dump({"urls": {GOOD: "Blocks bots; checked by hand"}}), encoding="utf-8"
    )
    assert load_allowlist(path) == {GOOD: "Blocks bots; checked by hand"}
    path.write_text(yaml.safe_dump({"urls": {GOOD: ""}}), encoding="utf-8")
    with pytest.raises(ValueError, match="need a reason"):
        load_allowlist(path)


@responses.activate
def test_check_links_applies_allowlist_to_allowed_domains_only(session):
    bad_domain = "https://randomblog.com/x"
    responses.add(responses.HEAD, GOOD, status=404)
    responses.add(responses.GET, GOOD, status=404)
    results = check_links(
        {GOOD: ["a.md"], bad_domain: ["b.md"]},
        allowlist={GOOD: "reason", bad_domain: "reason"},
        session=session,
        sleep=no_sleep,
        workers=2,
    )
    by_url = {r.url: r for r in results}
    assert by_url[GOOD].ok and by_url[GOOD].allowlisted and by_url[GOOD].used_by == ["a.md"]
    assert not by_url[bad_domain].ok  # the allow-list can't rescue an unapproved domain


@responses.activate
def test_main_writes_report_and_fails_on_bad_links(tmp_path, monkeypatch, capsys):
    kb, ref = tmp_path / "kb", tmp_path / "ref"
    ref.mkdir()
    good_url, dead_url = "https://www.investor.gov/stocks", "https://www.sec.gov/dead"
    write(kb, "stocks", "stocks-001", front() + BODY)
    write(
        kb,
        "stocks",
        "stocks-002",
        front(
            id="stocks-002", title="Dead link article", sources=[{"name": "SEC", "url": dead_url}]
        )
        + BODY,
    )
    allow = kb / "link_allowlist.yaml"
    allow.write_text(
        yaml.safe_dump({"urls": {"https://www.irs.gov/old": "no longer cited"}}), encoding="utf-8"
    )
    monkeypatch.setattr(link_check, "KB_ROOT", kb)
    monkeypatch.setattr(link_check, "REFERENCE_DIR", ref)
    monkeypatch.setattr(link_check, "ALLOWLIST_FILE", allow)
    monkeypatch.setattr(link_check, "collect_urls", lambda paths=None: collect_urls(kb, ref, paths))
    monkeypatch.setattr(link_check, "load_allowlist", lambda: load_allowlist(allow))
    monkeypatch.setattr(link_check.time, "sleep", no_sleep)
    responses.add(responses.HEAD, good_url, status=200)
    responses.add(responses.HEAD, dead_url, status=404)
    responses.add(responses.GET, dead_url, status=404)

    report = tmp_path / "report.json"
    assert main(["--report", str(report), "--workers", "1"]) == 1
    out = capsys.readouterr()
    assert "FAILED https://www.sec.gov/dead: HTTP 404" in out.err
    assert "STALE allow-list entry" in out.err
    data = json.loads(report.read_text(encoding="utf-8"))
    assert (data["total"], data["passed"], data["failed"]) == (2, 1, 1)
    assert data["stale_allowlist_entries"] == ["https://www.irs.gov/old"]

    # checking only the good file passes and skips the stale-entry check
    assert main([str(kb / "stocks" / "stocks-001.md"), "--no-report"]) == 0
    assert "1 URLs: 1 ok, 0 failed, 0 allow-listed" in capsys.readouterr().out


@responses.activate
def test_main_reports_allowlisted_and_unreadable(tmp_path, monkeypatch, capsys):
    kb = tmp_path / "kb"
    url = "https://www.investor.gov/stocks"
    write(kb, "stocks", "stocks-001", front() + BODY)
    write(kb, "stocks", "stocks-002", "no front matter")
    monkeypatch.setattr(link_check, "load_allowlist", lambda: {url: "blocks bots"})
    monkeypatch.setattr(link_check.time, "sleep", no_sleep)
    responses.add(responses.HEAD, url, status=403)
    responses.add(responses.GET, url, status=403)
    code = main(
        [str(kb / "stocks" / "stocks-001.md"), str(kb / "stocks" / "stocks-002.md"), "--no-report"]
    )
    out = capsys.readouterr()
    assert code == 1  # the unreadable article still fails the gate
    assert f"ALLOWLISTED {url} (HTTP 403): blocks bots" in out.out
    assert "UNREADABLE" in out.err


GLOSSARY = "https://www.investor.gov/introduction-investing/investing-basics/glossary"
PROBE = f"{GLOSSARY}/{link_check.SOFT_404_PROBE}"


def page(title):
    return f"<html><head><title>{title}</title></head><body>x</body></html>"


@responses.activate
def test_soft_404_detected_when_page_matches_made_up_sibling(session):
    fake = f"{GLOSSARY}/limit-order"
    responses.add(responses.HEAD, fake, status=200)
    responses.add(responses.GET, PROBE, body=page("Glossary | Investor.gov"))
    responses.add(responses.GET, fake, body=page("Glossary  |\n Investor.gov"))
    result = check_url(session, fake, sleep=no_sleep)
    assert not result.ok and result.error.startswith("soft 404")


@responses.activate
def test_real_page_on_soft_404_site_passes_and_probe_is_cached(session):
    real, other = f"{GLOSSARY}/market-order", f"{GLOSSARY}/mutual-funds"
    responses.add(responses.GET, PROBE, body=page("Glossary | Investor.gov"))
    for url, title in (
        (real, "Market Order | Investor.gov"),
        (other, "Mutual Funds | Investor.gov"),
    ):
        responses.add(responses.HEAD, url, status=200)
        responses.add(responses.GET, url, body=page(title))
    assert check_url(session, real, sleep=no_sleep).ok
    assert check_url(session, other, sleep=no_sleep).ok
    assert sum(call.request.url == PROBE for call in responses.calls) == 1


@responses.activate
def test_no_soft_404_when_site_returns_real_404s(session):
    url = "https://www.sec.gov/page"
    responses.add(responses.HEAD, url, status=200)
    responses.add(
        responses.GET,
        f"https://www.sec.gov/{link_check.SOFT_404_PROBE}",
        status=404,
        body=page("Not found"),
    )
    assert check_url(session, url, sleep=no_sleep).ok


@responses.activate
def test_soft_404_check_skips_pdfs_and_untitled_pages_and_tolerates_errors(session):
    pdf = "https://fund-docs.vanguard.com/F0970.pdf"
    responses.add(responses.HEAD, pdf, status=200)
    assert check_url(session, pdf, sleep=no_sleep).ok  # no probe request registered for PDFs

    untitled = "https://www.finra.org/a/page"
    responses.add(responses.HEAD, untitled, status=200)
    responses.add(
        responses.GET,
        f"https://www.finra.org/a/{link_check.SOFT_404_PROBE}",
        body="<html>no title</html>",
    )
    assert check_url(session, untitled, sleep=no_sleep).ok

    flaky = "https://www.irs.gov/b/page"
    responses.add(responses.HEAD, flaky, status=200)  # probe request raises (not registered)
    assert check_url(session, flaky, sleep=no_sleep).ok


@responses.activate
def test_site_that_ignores_the_slug_is_not_a_soft_404(session):
    real = "https://www.ishares.com/us/products/239733/ishares-conservative-allocation-etf"
    probe = f"https://www.ishares.com/us/products/239733/{link_check.SOFT_404_PROBE}"
    fund_page = page("iShares Core 30/70 Conservative Allocation ETF | AOK").replace(
        "<head>", f'<head><link rel="canonical" href="{real}/">'
    )
    responses.add(responses.HEAD, real, status=200)
    responses.add(responses.GET, probe, body=fund_page)
    responses.add(responses.GET, real, body=fund_page)
    assert check_url(session, real, sleep=no_sleep).ok


@responses.activate
def test_id_routed_page_whose_title_matches_its_slug_is_not_a_soft_404(session):
    real = "https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf"
    probe = f"https://www.ishares.com/us/products/314116/{link_check.SOFT_404_PROBE}"
    fund_page = page("iShares 0-3 Month Treasury Bond ETF | SGOV")  # no canonical link
    responses.add(responses.HEAD, real, status=200)
    responses.add(responses.GET, probe, body=fund_page)
    responses.add(responses.GET, real, body=fund_page)
    assert check_url(session, real, sleep=no_sleep).ok


@responses.activate
def test_numeric_slug_falls_back_to_title_match(session):
    url = "https://www.sec.gov/files/12345"
    responses.add(responses.HEAD, url, status=200)
    responses.add(
        responses.GET,
        f"https://www.sec.gov/files/{link_check.SOFT_404_PROBE}",
        body=page("SEC.gov | Home"),
    )
    responses.add(responses.GET, url, body=page("SEC.gov | Home"))
    assert not check_url(session, url, sleep=no_sleep).ok


@pytest.mark.parametrize(
    ("url", "title", "expected"),
    [
        ("https://www.investor.gov/glossary/limit-order", "Glossary | Investor.gov", False),
        ("https://www.investor.gov/glossary/market-order", "Market Order | Investor.gov", True),
        ("https://www.schwab.com/summary/swtsx", "Charles Schwab", False),
        ("https://www.sec.gov/files/12345", "Anything", None),
        ("https://www.sec.gov/index.html", "Home", None),
    ],
)
def test_title_matches_slug(url, title, expected):
    assert link_check.title_matches_slug(url, title) is expected


def test_report_paths_are_project_relative():
    from src.core.config import PROJECT_ROOT

    inside = PROJECT_ROOT / "data" / "reference" / "tax_2026.yaml"
    assert link_check._display_path(inside) == "data/reference/tax_2026.yaml"
    outside = Path("/tmp/elsewhere/file.md")
    assert link_check._display_path(outside) == outside.as_posix()


@responses.activate
def test_throttled_responses_back_off_longer(session):
    responses.add(responses.HEAD, GOOD, status=429)
    responses.add(responses.GET, GOOD, status=429)
    sleeps = []
    result = check_url(session, GOOD, sleep=sleeps.append)
    assert not result.ok and result.status == 429 and sleeps == [5, 15]


def test_per_host_concurrency_is_capped(monkeypatch):
    import threading
    import time as real_time

    active: dict[str, int] = {}
    peak: dict[str, int] = {}
    lock = threading.Lock()

    def fake_check(session, url, sleep):
        host = url.split("/")[2]
        with lock:
            active[host] = active.get(host, 0) + 1
            peak[host] = max(peak.get(host, 0), active[host])
        real_time.sleep(0.02)
        with lock:
            active[host] -= 1
        return link_check.LinkResult(url=url, ok=True)

    monkeypatch.setattr(link_check, "check_url", fake_check)
    urls = {f"https://www.investor.gov/p{i}": ["a"] for i in range(8)}
    urls |= {f"https://www.sec.gov/p{i}": ["b"] for i in range(8)}
    results = check_links(urls, workers=8, per_host=2)
    assert len(results) == 16 and all(r.ok for r in results)
    assert peak == {"www.investor.gov": 2, "www.sec.gov": 2}
