"""Refresh the local list of company tickers from the SEC: python scripts/update_sec_tickers.py

The SEC asks automated tools to identify themselves with contact details in the User-Agent
(https://www.sec.gov/os/accessing-edgar-data). Set SEC_CONTACT_EMAIL (in .env or the
environment) to your email address; the script refuses to run without it. The file is
downloaded once and stored in data/reference/, so the app never calls the SEC itself.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from dotenv import load_dotenv

from src.data.symbols import SEC_FILE, SEC_URL

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    contact = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if "@" not in contact:
        print(
            "Set SEC_CONTACT_EMAIL to your email address: the SEC requires a contact in the "
            "User-Agent of automated requests.",
            file=sys.stderr,
        )
        return 2
    response = requests.get(
        SEC_URL,
        headers={"User-Agent": f"Finnie financial-education app ({contact})"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().values()  # {"0": {"cik_str", "ticker", "title"}, ...}, largest first
    seen: set[str] = set()
    companies = []
    for row in rows:
        ticker = str(row["ticker"]).upper()
        if ticker not in seen:
            seen.add(ticker)
            companies.append([ticker, str(row["title"])])
    payload = {
        "source": SEC_URL,
        "retrieved": datetime.now(UTC).date().isoformat(),
        "note": "Ticker and company name only, in the SEC's order (largest companies first).",
        "companies": companies,
    }
    SEC_FILE.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"{len(companies):,} tickers written to {SEC_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
