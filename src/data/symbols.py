"""Find a ticker from a company, fund, or index name ("Apple" -> AAPL, "S&P 500" -> ^GSPC, SPY).

Three local sources, merged into one directory, nothing fetched at run time:

- **Indexes**: the major U.S. indexes, by their Yahoo symbols.
- **Finnie's catalog** (``data/reference/securities.yaml``): ETFs, mutual funds, and money
  market funds, with their type.
- **The SEC's company list** (``data/reference/sec_company_tickers.json``): every ticker the
  SEC maps to a registered company, refreshed with ``scripts/update_sec_tickers.py``.

When nothing local matches, :func:`yahoo_search` asks Yahoo Finance's search as a fallback.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from src.core.config import PROJECT_ROOT
from src.core.reference import SecurityCatalog, get_catalog

logger = logging.getLogger(__name__)

SEC_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FILE = PROJECT_ROOT / "data" / "reference" / "sec_company_tickers.json"
Kind = Literal["Index", "ETF", "Mutual fund", "Money market", "Fund", "Stock", "Crypto", "Other"]
Source = Literal["finnie", "sec", "yahoo"]

INDEXES: dict[str, str] = {
    "^GSPC": "S&P 500 Index",
    "^DJI": "Dow Jones Industrial Average",
    "^NDX": "Nasdaq-100 Index",
    "^IXIC": "Nasdaq Composite Index",
    "^RUT": "Russell 2000 Index",
    "^VIX": "CBOE Volatility Index (VIX)",
}
_CATALOG_KINDS: dict[str, Kind] = {
    "etf": "ETF",
    "mutual_fund": "Mutual fund",
    "money_market": "Money market",
    "stock": "Stock",
}
_YAHOO_KINDS: dict[str, Kind] = {
    "EQUITY": "Stock",
    "ETF": "ETF",
    "MUTUALFUND": "Mutual fund",
    "MONEYMARKET": "Money market",
    "INDEX": "Index",
    "CRYPTOCURRENCY": "Crypto",
}
WORD = re.compile(r"[a-z0-9&]+")


class SymbolMatch(BaseModel):
    """One suggestion: the ticker, its name and kind, and where the entry came from."""

    ticker: str
    name: str
    kind: Kind
    source: Source

    @property
    def label(self) -> str:
        """How the suggestion reads in a list: "AAPL · Apple Inc. · Stock"."""
        return f"{self.ticker} · {self.name} · {self.kind}"


def _normal(text: str) -> str:
    """Lower-case words only, without a leading "the" ("The Coca-Cola Company" -> "coca cola
    company"), so names match the way people type them."""
    words = WORD.findall(text.lower())
    return " ".join(words[1:] if words[:1] == ["the"] and len(words) > 1 else words)


def _same_ticker(ticker: str) -> str:
    """BRK.B and BRK-B are the same share class written two ways."""
    return ticker.replace("-", ".")


def _sec_kind(name: str) -> Kind:
    """The SEC's list also has ETFs and closed-end funds; tell them apart by name."""
    if re.search(r"\bETF\b|exchange[- ]traded", name, re.IGNORECASE):
        return "ETF"
    if re.search(r"\bfund\b", name, re.IGNORECASE):
        return "Fund"
    return "Stock"


class SymbolDirectory:
    """Indexes, Finnie's funds, and SEC companies, searchable by ticker or name."""

    def __init__(self, entries: Sequence[SymbolMatch]) -> None:
        unique: dict[str, SymbolMatch] = {}
        for entry in entries:  # the first source to list a ticker wins (indexes, then funds)
            unique.setdefault(_same_ticker(entry.ticker), entry)
        self.entries = list(unique.values())
        self._names = [_normal(e.name) for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, ticker: str) -> SymbolMatch | None:
        """The entry for an exact ticker, or ``None``."""
        wanted = _same_ticker(ticker.strip().upper().lstrip("$"))
        return next((e for e in self.entries if _same_ticker(e.ticker) == wanted), None)

    def search(self, query: str, limit: int = 8) -> list[SymbolMatch]:
        """Best matches for a ticker or name, best first.

        Order: exact ticker, ticker prefix, name starting with the query, a word in the name
        starting with it, then the query anywhere in the name. Ties keep the directory's
        order (indexes, then Finnie's funds, then SEC companies, largest first).
        """
        text = _normal(query)
        symbol = _same_ticker(query.strip().upper().lstrip("$"))
        if not text and not symbol:
            return []
        scored: list[tuple[int, int, SymbolMatch]] = []
        for position, (entry, name) in enumerate(zip(self.entries, self._names, strict=True)):
            ticker = _same_ticker(entry.ticker)
            if ticker == symbol:
                score = 0
            elif symbol and ticker.lstrip("^").startswith(symbol.lstrip("^")):
                score = 1
            elif text and name.startswith(text):
                score = 2
            elif text and f" {text}" in f" {name}":
                score = 3
            elif text and text in name:
                score = 4
            else:
                continue
            scored.append((score, position, entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in scored[:limit]]


def _catalog_entries(catalog: SecurityCatalog) -> list[SymbolMatch]:
    entries = []
    for ticker in catalog.tickers():
        info = catalog.get(ticker)
        if info is None or info.type == "cash":
            continue
        entries.append(
            SymbolMatch(
                ticker=ticker,
                name=info.name,
                kind=_CATALOG_KINDS.get(info.type, "Other"),
                source="finnie",
            )
        )
    return entries


def load_sec_companies(path: Path = SEC_FILE) -> list[SymbolMatch]:
    """The stored SEC company list, or ``[]`` (with a warning) if the file is missing."""
    if not path.is_file():
        logger.warning("SEC ticker file missing; run scripts/update_sec_tickers.py")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        SymbolMatch(ticker=ticker, name=name, kind=_sec_kind(name), source="sec")
        for ticker, name in data["companies"]
    ]


def build_directory(
    catalog: SecurityCatalog | None = None, sec_path: Path = SEC_FILE
) -> SymbolDirectory:
    """The full directory: indexes first, then Finnie's catalog, then SEC companies."""
    indexes = [
        SymbolMatch(ticker=t, name=n, kind="Index", source="finnie") for t, n in INDEXES.items()
    ]
    return SymbolDirectory(
        indexes + _catalog_entries(catalog or get_catalog()) + load_sec_companies(sec_path)
    )


@cache
def get_symbol_directory() -> SymbolDirectory:
    """Process-wide directory (loaded once)."""
    return build_directory()


def yahoo_search(
    query: str, limit: int = 8, search: Callable[..., Any] | None = None
) -> list[SymbolMatch]:
    """Yahoo Finance's search, for names the local directory doesn't know. ``[]`` on any
    failure: it's a convenience, so an outage only means fewer suggestions."""
    if not query.strip():
        return []
    try:
        if search is None:
            import yfinance as yf

            search = yf.Search
        quotes = search(
            query,
            max_results=limit,
            news_count=0,
            lists_count=0,
            include_cb=False,
            recommended=0,
            raise_errors=False,
        ).quotes
    except Exception:
        logger.warning("Yahoo search failed", exc_info=True)
        return []
    matches = []
    for quote in quotes or []:
        symbol = str(quote.get("symbol") or "").upper()
        name = quote.get("longname") or quote.get("shortname") or symbol
        if symbol:
            kind = _YAHOO_KINDS.get(str(quote.get("quoteType") or "").upper(), "Other")
            matches.append(SymbolMatch(ticker=symbol, name=str(name), kind=kind, source="yahoo"))
    return matches[:limit]
