"""Finding a ticker by company, fund, or index name (Markets search)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from src.core.reference import get_catalog
from src.data.symbols import (
    INDEXES,
    SymbolDirectory,
    SymbolMatch,
    build_directory,
    get_symbol_directory,
    load_sec_companies,
    yahoo_search,
)


@pytest.fixture
def sec_file(tmp_path):
    path = tmp_path / "sec.json"
    companies = [
        ["NVDA", "NVIDIA CORP"],
        ["AAPL", "Apple Inc."],
        ["KO", "The Coca-Cola Company"],
        ["CCEP", "COCA-COLA EUROPACIFIC PARTNERS plc"],
        ["BRK-B", "BERKSHIRE HATHAWAY INC"],
        ["SPXX", "Nuveen S&P 500 Dynamic Overwrite Fund"],
        ["NCIQ", "Hashdex Nasdaq CME Crypto Index ETF"],
        ["APLE", "Apple Hospitality REIT, Inc."],
    ]
    path.write_text(json.dumps({"companies": companies}), encoding="utf-8")
    return path


@pytest.fixture
def directory(sec_file):
    return build_directory(get_catalog(), sec_file)


def labels(matches):
    return [m.label for m in matches]


def test_company_names_find_their_tickers(directory):
    assert directory.search("Apple")[0].label == "AAPL · Apple Inc. · Stock"
    assert directory.search("apple")[1].ticker == "APLE"  # name prefix, smaller company
    assert directory.search("coca cola")[0].ticker == "KO"  # a leading "The" is ignored


def test_index_names_find_the_index_and_the_funds_that_track_it(directory):
    found = labels(directory.search("S&P 500", limit=4))
    assert found[0] == "^GSPC · S&P 500 Index · Index"
    assert {"VOO · Vanguard S&P 500 ETF · ETF", "SPY · SPDR S&P 500 ETF Trust · ETF"} <= set(found)


def test_tickers_match_exactly_first_then_by_prefix(directory):
    assert directory.search("aapl")[0].ticker == "AAPL"
    assert directory.search("$nvda")[0].ticker == "NVDA"
    assert directory.search("VT")[0].ticker == "VT"
    assert all(m.ticker.startswith("VT") for m in directory.search("VT", limit=3))


def test_share_classes_written_two_ways_are_one_entry(directory):
    berkshire = directory.search("berkshire")
    assert [m.ticker for m in berkshire] == ["BRK.B"]  # Finnie's entry wins over SEC's BRK-B
    assert directory.get("BRK-B").ticker == "BRK.B"
    assert directory.get("ZZZZ") is None


def test_sec_entries_are_labelled_by_kind(directory):
    assert directory.get("SPXX").kind == "Fund"
    assert directory.get("NCIQ").kind == "ETF"
    assert directory.get("NVDA").kind == "Stock"
    assert directory.get("VFIFX").kind == "Mutual fund"
    assert directory.get("^RUT").kind == "Index"


def test_nothing_or_nonsense_finds_nothing(directory):
    assert directory.search("") == [] and directory.search("   ") == []
    assert directory.search("qqqqzzzz") == []


def test_missing_sec_file_leaves_indexes_and_funds(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    assert load_sec_companies(tmp_path / "missing.json") == []
    assert "run scripts/update_sec_tickers.py" in caplog.text
    directory = build_directory(get_catalog(), tmp_path / "missing.json")
    assert directory.get("^GSPC") is not None and directory.get("VTI") is not None
    assert len(directory) == len(INDEXES) + sum(
        1 for t in get_catalog().tickers() if get_catalog().get(t).type != "cash"
    )


def test_the_stored_sec_list_loads():
    directory = get_symbol_directory()
    assert len(directory) > 10_000
    assert directory.get("MSFT").name


def test_yahoo_search_maps_quotes():
    def fake_search(query, **kwargs):
        assert kwargs["news_count"] == 0 and kwargs["raise_errors"] is False
        return SimpleNamespace(
            quotes=[
                {"symbol": "nsrgy", "longname": "Nestlé S.A.", "quoteType": "EQUITY"},
                {"symbol": "VWRL.L", "shortname": "Vanguard FTSE All-World", "quoteType": "ETF"},
                {"symbol": "BTC-USD", "quoteType": "CRYPTOCURRENCY"},
                {"symbol": "", "shortname": "no symbol"},
                {"symbol": "X1", "shortname": "Something", "quoteType": "FUTURE"},
            ]
        )

    found = yahoo_search("Nestle", search=fake_search)
    assert labels(found) == [
        "NSRGY · Nestlé S.A. · Stock",
        "VWRL.L · Vanguard FTSE All-World · ETF",
        "BTC-USD · BTC-USD · Crypto",
        "X1 · Something · Other",
    ]
    assert all(m.source == "yahoo" for m in found)


def test_yahoo_search_failures_mean_no_suggestions():
    def broken(query, **kwargs):
        raise ConnectionError("offline")

    assert yahoo_search("Nestle", search=broken) == []
    assert yahoo_search("  ") == []


def test_first_listing_of_a_ticker_wins():
    directory = SymbolDirectory(
        [
            SymbolMatch(ticker="SPY", name="Finnie name", kind="ETF", source="finnie"),
            SymbolMatch(ticker="SPY", name="SEC name", kind="Stock", source="sec"),
        ]
    )
    assert len(directory) == 1 and directory.get("SPY").name == "Finnie name"


def test_yahoo_search_uses_yfinance_by_default(monkeypatch):
    import yfinance

    class FakeSearch:
        def __init__(self, query, **kwargs):
            self.quotes = [{"symbol": "SAP", "shortname": "SAP SE", "quoteType": "EQUITY"}]

    monkeypatch.setattr(yfinance, "Search", FakeSearch)  # never the real network in tests
    assert labels(yahoo_search("SAP software")) == ["SAP · SAP SE · Stock"]
