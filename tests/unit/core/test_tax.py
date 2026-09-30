import pytest
import yaml
from pydantic import ValidationError

from src.core.reference import REFERENCE_DIR
from src.core.tax import (
    UNVERIFIED_NOTE,
    compare_accounts,
    get_tax_reference,
    illustrate_capital_gains,
    load_tax_reference,
    marginal_rate,
    tax_on_income,
)

REF = get_tax_reference(2026)
SINGLE = REF.ordinary_income.for_status("single")


def test_reference_loads_with_all_figures_pending_verification():
    assert REF.tax_year == 2026 and REF.jurisdiction == "US federal"
    assert get_tax_reference(2026) is REF
    assert len(REF.figures) == 17
    assert all(
        f.status == "VERIFY" and f.source_url.startswith("https://www.irs.gov/")
        for f in REF.figures.values()
    )
    pending = REF.unverified()
    assert len(pending) == 19  # 17 figures + 2 bracket tables
    assert (
        "ira_contribution_limit: Traditional and Roth IRA contribution limit (combined)" in pending
    )
    assert REF.figure("ira_contribution_limit").value == 7500
    with pytest.raises(KeyError, match="Unknown tax figure"):
        REF.figure("nope")


def test_every_account_type_is_described():
    assert set(REF.accounts) == {
        "traditional_401k",
        "roth_401k",
        "traditional_ira",
        "roth_ira",
        "hsa",
        "plan_529",
        "taxable_brokerage",
    }


def test_compare_accounts():
    rows = compare_accounts(REF, ["roth_ira", "traditional_ira"])
    assert [r.name for r in rows] == ["Roth IRA", "Traditional IRA"]
    assert rows[0].growth == "Tax-free" and rows[0].has_unverified_figures
    assert [f.key for f in rows[1].limits] == ["ira_contribution_limit", "ira_catch_up_50_plus"]
    assert len(compare_accounts(REF)) == 7
    with pytest.raises(KeyError, match="Unknown account type"):
        compare_accounts(REF, ["crypto_ira"])


def test_progressive_tax_hand_computed():
    # 12,400 * 10% + (50,400 - 12,400) * 12% + (60,000 - 50,400) * 22% = 1,240 + 4,560 + 2,112
    assert tax_on_income(SINGLE, 60_000) == pytest.approx(7_912)
    assert tax_on_income(SINGLE, 0) == 0
    assert marginal_rate(SINGLE, 60_000) == 0.22
    assert marginal_rate(SINGLE, 12_400) == 0.10
    assert marginal_rate(SINGLE, 10_000_000) == 0.37


def test_capital_gains_illustration_long_vs_short():
    # $10,000 gain on $40,000 taxable income (single):
    #   long-term: 49,450 - 40,000 = 9,450 at 0%, remaining 550 at 15% = 82.50
    #   short-term: all within the 12% bracket = 1,200
    long_term = illustrate_capital_gains(REF, gain=10_000, holding_days=400, taxable_income=40_000)
    assert long_term.long_term and long_term.tax_if_long_term == 82.5
    assert long_term.tax_if_short_term == 1_200 and long_term.applied_tax == 82.5
    assert long_term.difference == 1_117.5 and long_term.effective_rate_on_gain == 0.0083
    assert long_term.uses_unverified_figures and UNVERIFIED_NOTE in long_term.caveats

    short_term = illustrate_capital_gains(REF, gain=10_000, holding_days=365, taxable_income=40_000)
    assert not short_term.long_term and short_term.applied_tax == 1_200


def test_capital_gains_married_joint():
    result = illustrate_capital_gains(
        REF, gain=20_000, holding_days=800, taxable_income=60_000, filing_status="married_joint"
    )
    assert result.tax_if_long_term == 0  # 80,000 stays under the 98,900 0% threshold


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gain": 0, "holding_days": 1, "taxable_income": 1},
        {"gain": 10, "holding_days": -1, "taxable_income": 1},
        {"gain": 10, "holding_days": 1, "taxable_income": -5},
    ],
)
def test_capital_gains_input_validation(kwargs):
    with pytest.raises(ValueError):
        illustrate_capital_gains(REF, **kwargs)


def test_verified_figures_drop_the_caveat(tmp_path):
    data = yaml.safe_load((REFERENCE_DIR / "tax_2026.yaml").read_text(encoding="utf-8"))
    data["figures"]["long_term_holding_period_days"]["status"] = "verified"
    for table in data["brackets"].values():
        table["status"] = "verified"
    path = tmp_path / "tax.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    ref = load_tax_reference(path)
    result = illustrate_capital_gains(ref, gain=100, holding_days=400, taxable_income=0)
    assert not result.uses_unverified_figures and UNVERIFIED_NOTE not in result.caveats
    assert len(ref.unverified()) == 16


def load_modified(tmp_path, mutate):
    data = yaml.safe_load((REFERENCE_DIR / "tax_2026.yaml").read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "tax.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_tax_reference(path)


def test_brackets_must_ascend_and_end_open(tmp_path):
    def unsorted(data):
        data["brackets"]["ordinary_income"]["single"][1]["up_to"] = 1

    def closed(data):
        data["brackets"]["long_term_capital_gains"]["single"][-1]["up_to"] = 999_999_999

    for mutate in (unsorted, closed):
        with pytest.raises(ValidationError, match="ascend"):
            load_modified(tmp_path, mutate)


def test_accounts_must_reference_known_figures(tmp_path):
    def bad(data):
        data["accounts"]["hsa"]["limit_figures"] = ["made_up_limit"]

    with pytest.raises(ValidationError, match="unknown figures"):
        load_modified(tmp_path, bad)


def test_bad_status_rejected(tmp_path):
    def bad(data):
        data["figures"]["ira_contribution_limit"]["status"] = "probably fine"

    with pytest.raises(ValidationError):
        load_modified(tmp_path, bad)
