from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from src.core.reference import (
    RiskProfile,
    SecurityInfo,
    get_catalog,
    get_risk_profiles,
    load_catalog,
    load_risk_profiles,
)


def test_catalog_loads_and_is_consistent():
    catalog = get_catalog()
    assert len(catalog) >= 80 and "VTI" in catalog and "ZZZZ" not in catalog
    assert catalog.last_reviewed == "2026-09-30"
    vti = catalog.get("VTI")
    assert vti.type == "etf" and vti.diversified and vti.expense_ratio == 0.0003
    assert not vti.is_single_issuer and catalog.get("AAPL").is_single_issuer
    assert catalog.get("AOR").look_through == {"equity": 0.6, "bond": 0.4}
    assert catalog.get("BND").look_through == {"bond": 1.0}
    sectors = {
        catalog.get(t).sector
        for t in ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    }
    assert len(sectors) == 11


def test_mock_market_tickers_are_all_classified():
    import json

    from src.data.mock_provider import DEFAULT_MOCK_PATH

    mock = json.loads(DEFAULT_MOCK_PATH.read_text(encoding="utf-8"))["securities"]
    assert set(mock) <= {t for t in mock if t in get_catalog()}


@pytest.mark.parametrize(
    ("asset_type", "expected_type", "diversified", "sector", "expense_ratio"),
    [
        ("ETF", "etf", True, "Broad Market", None),  # a fund with an unknown fee
        ("Common Stock", "stock", False, "Unknown", 0.0),  # stocks have no fund fee
        ("Equity", "stock", False, "Unknown", 0.0),
        ("Mutual Fund", "mutual_fund", True, "Broad Market", None),
        (None, "other", False, "Unknown", None),
    ],
)
def test_classify_unknown_ticker(asset_type, expected_type, diversified, sector, expense_ratio):
    info = get_catalog().classify("NEWCO", name="New Co", asset_type=asset_type)
    assert (info.type, info.diversified, info.sector) == (expected_type, diversified, sector)
    assert info.known is False and info.asset_class == "equity"
    assert info.expense_ratio == expense_ratio and info.fees is None
    assert info.name == "New Co"


def test_classify_known_ticker_ignores_overview():
    assert get_catalog().classify("VTI", asset_type="Common Stock").type == "etf"


def test_classify_uses_overview_sector():
    assert get_catalog().classify("X", asset_type="Equity", sector="Energy").sector == "Energy"


def test_security_allocation_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        SecurityInfo(
            ticker="X",
            name="X",
            type="etf",
            asset_class="equity",
            allocation={"equity": 0.5, "bond": 0.4},
            sector="Broad Market",
            diversified=True,
            risk=5,
        )


def test_risk_profiles():
    profiles = get_risk_profiles()
    assert set(profiles) == {"conservative", "moderate", "aggressive"}
    moderate = profiles["moderate"]
    assert moderate.allocation == {"equity": 0.6, "bond": 0.35, "cash": 0.05}
    returns = [profiles[p].expected_return for p in ("conservative", "moderate", "aggressive")]
    vols = [profiles[p].volatility for p in ("conservative", "moderate", "aggressive")]
    assert returns == sorted(returns) and vols == sorted(vols)


def test_risk_profile_validation(tmp_path):
    with pytest.raises(ValidationError, match="sum to 1"):
        RiskProfile(
            name="moderate",
            label="M",
            description="d",
            allocation={"equity": 0.5, "bond": 0.2},
            expected_return=0.05,
            volatility=0.1,
        )
    path = tmp_path / "p.yaml"
    path.write_text(
        "profiles:\n  moderate: {label: M, description: d, allocation: {equity: 1.0},"
        " expected_return: 0.05, volatility: 0.1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggressive, conservative"):
        load_risk_profiles(path)


def test_loaders_reject_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_catalog(path)


def test_expense_ratios_are_sourced():
    catalog = get_catalog()
    funds = [catalog.get(t) for t in ("VTI", "QQQ", "BIL", "SPY", "FXAIX", "SCHD", "VTSAX")]
    assert all(f.fees and f.fees.source_url.startswith("https://") for f in funds)
    # figures corrected by the 2026-09-30 verification against provider pages
    assert catalog.get("QQQ").expense_ratio == 0.0018
    assert catalog.get("BIL").expense_ratio == 0.001353
    assert catalog.get("VUG").expense_ratio == 0.0003
    assert catalog.get("VYM").expense_ratio == 0.0004
    vtsax = catalog.get("VTSAX").fees
    assert vtsax.verified and vtsax.as_of == date(2026, 4, 28)
    assert vtsax.verified_on == date(2026, 9, 30)
    assert catalog.get("IVV").fees.as_of is None  # iShares pages state no fee date
    assert catalog.unverified_expense_ratios() == ["SCHB", "SCHH", "SWTSX"]
    assert catalog.get("AAPL").fees is None and catalog.get("AAPL").expense_ratio == 0.0
    assert catalog.get("CASH").fees is None and catalog.get("CASH").expense_ratio == 0.0


def write_catalog(tmp_path, securities, fees):
    path = tmp_path / "s.yaml"
    path.write_text(
        yaml.safe_dump({"securities": securities, "expense_ratios": fees}), encoding="utf-8"
    )
    return path


FUND = {
    "name": "F",
    "type": "etf",
    "asset_class": "equity",
    "sector": "Broad Market",
    "diversified": True,
    "risk": 6,
}
FEE = {
    "ratio": 0.001,
    "status": "verified",
    "verified_on": "2026-09-30",
    "source_url": "https://example.com/f",
}


def test_catalog_requires_a_fee_for_every_fund(tmp_path):
    with pytest.raises(ValueError, match="funds missing expense ratios"):
        load_catalog(write_catalog(tmp_path, {"F": FUND}, {}))


def test_catalog_rejects_fees_for_non_funds(tmp_path):
    stock = FUND | {"type": "stock", "diversified": False}
    with pytest.raises(ValueError, match="non-funds"):
        load_catalog(write_catalog(tmp_path, {"F": stock}, {"F": FEE}))


def test_verified_fee_needs_date_and_https_source(tmp_path):
    undated = {k: v for k, v in FEE.items() if k != "verified_on"}
    with pytest.raises(ValidationError, match="verified_on"):
        load_catalog(write_catalog(tmp_path, {"F": FUND}, {"F": undated}))
    with pytest.raises(ValidationError, match="source_url"):
        load_catalog(write_catalog(tmp_path, {"F": FUND}, {"F": FEE | {"source_url": "ftp://x"}}))
    assert load_catalog(write_catalog(tmp_path, {"F": FUND}, {"F": FEE})).get("F").fees.verified
