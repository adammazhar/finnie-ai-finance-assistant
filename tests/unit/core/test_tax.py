from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from src.core.reference import REFERENCE_DIR
from src.core.tax import (
    UNVERIFIED_NOTE,
    compare_accounts,
    exceptions_for,
    get_tax_reference,
    illustrate_capital_gains,
    is_long_term,
    load_tax_reference,
    marginal_rate,
    one_year_anniversary,
    rmd_age_for,
    rmd_age_for_current_age,
    tax_on_income,
)

REF = get_tax_reference(2026)
SINGLE = REF.ordinary_income.for_status("single")
RETIREMENT_URL = "https://www.irs.gov/newsroom/401k-limit-increases-to-24500-for-2026-ira-limit-increases-to-7500"
INFLATION_URL = (
    "https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-"
    "including-amendments-from-the-one-big-beautiful-bill"
)
OWNER_VERIFIED = {
    "employee_401k_deferral_limit": (24_500, RETIREMENT_URL),
    "employee_401k_catch_up_50_plus": (8_000, RETIREMENT_URL),
    "employee_401k_catch_up_60_to_63": (11_250, RETIREMENT_URL),
    "ira_contribution_limit": (7_500, RETIREMENT_URL),
    "ira_catch_up_50_plus": (1_100, RETIREMENT_URL),
    "roth_ira_phase_out_single": ([153_000, 168_000], RETIREMENT_URL),
    "roth_ira_phase_out_married_joint": ([242_000, 252_000], RETIREMENT_URL),
    "hsa_limit_self_only": (4_400, "https://www.irs.gov/publications/p969"),
    "hsa_limit_family": (8_750, "https://www.irs.gov/publications/p969"),
    "hsa_catch_up_55_plus": (1_000, "https://www.irs.gov/publications/p969"),
    "standard_deduction_single": (16_100, INFLATION_URL),
    "standard_deduction_married_joint": (32_200, INFLATION_URL),
    "capital_loss_deduction_limit": (3_000, "https://www.irs.gov/taxtopics/tc409"),
}


def test_owner_verified_figures():
    for key, (value, url) in OWNER_VERIFIED.items():
        figure = REF.figure(key)
        assert figure.value == value, key
        assert figure.verified and figure.verified_on == date(2026, 9, 30), key
        assert figure.source_url == url, key
    for table in (REF.ordinary_income, REF.long_term_capital_gains):
        assert table.verified and table.verified_on == date(2026, 9, 30)
        assert table.source_url == INFLATION_URL


OWNER_VERIFIED_LATER = {
    "gift_tax_annual_exclusion": (
        19_000,
        "https://www.irs.gov/faqs/interest-dividends-other-types-of-income/gifts-inheritances/"
        "gifts-inheritances-1",
    ),
    "long_term_holding_period_years": (1, "https://www.irs.gov/taxtopics/tc409"),
    "wash_sale_window_days": (30, "https://www.irs.gov/publications/p550"),
    "net_investment_income_tax_rate": (
        0.038,
        "https://www.irs.gov/individuals/net-investment-income-tax",
    ),
}


def test_every_figure_is_now_verified():
    assert REF.tax_year == 2026 and REF.jurisdiction == "US federal"
    assert get_tax_reference(2026) is REF
    assert REF.unverified() == []
    for key, (value, url) in OWNER_VERIFIED_LATER.items():
        figure = REF.figure(key)
        assert (figure.value, figure.source_url) == (value, url), key
        assert figure.verified and figure.verified_on == date(2026, 9, 30), key
    assert all(f.source_url.startswith("https://www.irs.gov/") for f in REF.figures.values())
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
    rows = compare_accounts(REF, ["roth_ira", "traditional_ira", "plan_529"])
    assert [r.name for r in rows] == ["Roth IRA", "Traditional IRA", "529 Education Savings Plan"]
    assert rows[0].growth == "Tax-free"
    assert not rows[0].has_unverified_figures  # IRA limits and Roth phase-outs are verified
    assert not any(row.has_unverified_figures for row in rows)
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


# ---- holding period: more than one year, by calendar date -------------------------------


@pytest.mark.parametrize(
    ("purchase", "sale", "long_term"),
    [
        (date(2025, 6, 15), date(2026, 6, 15), False),  # on the anniversary: still short-term
        (date(2025, 6, 15), date(2026, 6, 16), True),  # the day after: long-term
        (date(2025, 6, 15), date(2025, 6, 15), False),  # same-day sale
        # 366 days, because the year includes Feb 29, 2024, but sold ON the anniversary
        (date(2023, 3, 1), date(2024, 3, 1), False),
        (date(2023, 3, 1), date(2024, 3, 2), True),
        # bought Feb 29 (leap day): the anniversary is Feb 28 of the next year
        (date(2024, 2, 29), date(2025, 2, 28), False),
        (date(2024, 2, 29), date(2025, 3, 1), True),
        # bought Feb 28 in a year before a leap year: Feb 29 is after the anniversary
        (date(2023, 2, 28), date(2024, 2, 28), False),
        (date(2023, 2, 28), date(2024, 2, 29), True),
        # year-end purchase
        (date(2025, 12, 31), date(2026, 12, 31), False),
        (date(2025, 12, 31), date(2027, 1, 1), True),
    ],
)
def test_long_term_uses_calendar_anniversary(purchase, sale, long_term):
    assert is_long_term(purchase, sale) is long_term


def test_one_year_anniversary():
    assert one_year_anniversary(date(2025, 6, 15)) == date(2026, 6, 15)
    assert one_year_anniversary(date(2024, 2, 29)) == date(2025, 2, 28)
    assert one_year_anniversary(date(2023, 2, 28)) == date(2024, 2, 28)


def test_sale_before_purchase_rejected():
    with pytest.raises(ValueError, match="before the purchase"):
        is_long_term(date(2026, 1, 2), date(2026, 1, 1))


def test_capital_gains_illustration_long_vs_short():
    # $10,000 gain on $40,000 taxable income (single):
    #   long-term: 49,450 - 40,000 = 9,450 at 0%, remaining 550 at 15% = 82.50
    #   short-term: all within the 12% bracket = 1,200
    long_term = illustrate_capital_gains(
        REF,
        gain=10_000,
        purchase_date=date(2025, 3, 1),
        sale_date=date(2026, 3, 2),
        taxable_income=40_000,
    )
    assert long_term.long_term and long_term.tax_if_long_term == 82.5
    assert long_term.first_long_term_sale_date == date(2026, 3, 2)
    assert long_term.tax_if_short_term == 1_200 and long_term.applied_tax == 82.5
    assert long_term.difference == 1_117.5 and long_term.effective_rate_on_gain == 0.0083
    assert not long_term.uses_unverified_figures and UNVERIFIED_NOTE not in long_term.caveats

    on_anniversary = illustrate_capital_gains(
        REF,
        gain=10_000,
        purchase_date=date(2025, 3, 1),
        sale_date=date(2026, 3, 1),
        taxable_income=40_000,
    )
    assert not on_anniversary.long_term and on_anniversary.applied_tax == 1_200


def test_capital_gains_married_joint():
    result = illustrate_capital_gains(
        REF,
        gain=20_000,
        purchase_date=date(2020, 1, 1),
        sale_date=date(2026, 1, 1),
        taxable_income=60_000,
        filing_status="married_joint",
    )
    assert result.tax_if_long_term == 0  # 80,000 stays under the 98,900 0% threshold


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gain": 0, "taxable_income": 1},
        {"gain": 10, "taxable_income": -5},
        {"gain": 10, "taxable_income": 1, "sale_date": date(2019, 1, 1)},
    ],
)
def test_capital_gains_input_validation(kwargs):
    args = {"purchase_date": date(2020, 1, 1), "sale_date": date(2026, 1, 1)} | kwargs
    with pytest.raises(ValueError):
        illustrate_capital_gains(REF, **args)


def test_unverified_holding_rule_adds_the_caveat(tmp_path):
    def unverify_holding_rule(data):
        rule = data["figures"]["long_term_holding_period_years"]
        rule["status"] = "VERIFY"
        del rule["verified_on"]

    ref = load_modified(tmp_path, unverify_holding_rule)
    result = illustrate_capital_gains(
        ref,
        gain=100,
        purchase_date=date(2024, 1, 1),
        sale_date=date(2026, 1, 1),
        taxable_income=0,
    )
    assert result.uses_unverified_figures and UNVERIFIED_NOTE in result.caveats
    assert [item.split(":")[0] for item in ref.unverified()] == ["long_term_holding_period_years"]


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


def test_verified_items_need_a_date(tmp_path):
    def undated(data):
        del data["figures"]["ira_contribution_limit"]["verified_on"]

    with pytest.raises(ValidationError, match="verified_on"):
        load_modified(tmp_path, undated)


def test_sources_must_be_irs_pages(tmp_path):
    def elsewhere(data):
        data["figures"]["ira_contribution_limit"]["source_url"] = "https://example.com/ira"

    with pytest.raises(ValidationError, match=r"IRS\.gov"):
        load_modified(tmp_path, elsewhere)


def test_unverified_bracket_tables_are_listed(tmp_path):
    def unverify(data):
        table = data["brackets"]["long_term_capital_gains"]
        table["status"] = "VERIFY"
        del table["verified_on"]

    ref = load_modified(tmp_path, unverify)
    assert ref.unverified()[-1].startswith("brackets: Long-term capital gains")
    result = illustrate_capital_gains(
        ref, gain=100, purchase_date=date(2024, 1, 1), sale_date=date(2026, 1, 1), taxable_income=0
    )
    assert result.uses_unverified_figures


def test_rmd_age_by_birth_year():
    ref = get_tax_reference()
    assert rmd_age_for(ref, birth_year=1948).endswith("age 70½")
    assert rmd_age_for(ref, birth_year=1949).endswith("age 70½ or 72")  # split mid-year
    assert rmd_age_for(ref, birth_year=1950).endswith("age 72")
    assert rmd_age_for(ref, birth_year=1955).endswith("age 73")
    assert "age 73 (set by the IRS's 2024 proposed regulations" in rmd_age_for(ref, birth_year=1959)
    assert rmd_age_for(ref, birth_year=1968).endswith("age 75")


def test_rmd_age_from_current_age():
    ref = get_tax_reference()
    assert rmd_age_for_current_age(ref, age=58, year=2026) == (
        "Age 58 in 2026 means born in 1967 or 1968; RMDs start at age 75"
    )
    straddle = rmd_age_for_current_age(ref, age=66, year=2026)  # born 1959 or 1960
    assert "born in 1959: RMDs start at age 73" in straddle
    assert "born in 1960: RMDs start at age 75" in straddle


def test_early_withdrawal_exceptions_differ_by_account():
    ref = get_tax_reference()
    plans, iras = exceptions_for(ref, "plans"), exceptions_for(ref, "iras")
    assert any("Rule of 55" in e for e in plans) and not any("Rule of 55" in e for e in iras)
    assert any("First-time home" in e for e in iras)
    assert not any("First-time home" in e for e in plans)
    assert any("higher education" in e for e in iras)
    assert not any("higher education" in e for e in plans)
    assert ref.early_withdrawal_exceptions.verified and ref.rmd_ages.verified
