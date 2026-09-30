import pytest
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
    ("asset_type", "expected_type", "diversified", "sector"),
    [
        ("ETF", "etf", True, "Broad Market"),
        ("Common Stock", "stock", False, "Unknown"),
        ("Equity", "stock", False, "Unknown"),
        ("Mutual Fund", "mutual_fund", True, "Broad Market"),
        (None, "other", False, "Unknown"),
    ],
)
def test_classify_unknown_ticker(asset_type, expected_type, diversified, sector):
    info = get_catalog().classify("NEWCO", name="New Co", asset_type=asset_type)
    assert (info.type, info.diversified, info.sector) == (expected_type, diversified, sector)
    assert info.known is False and info.asset_class == "equity" and info.expense_ratio is None
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
