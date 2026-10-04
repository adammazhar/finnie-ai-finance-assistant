"""US federal tax reference data and illustrative calculations, for education only.

Figures come from ``data/reference/tax_<year>.yaml``. Each carries a ``status``: figures
not yet confirmed against IRS.gov are ``VERIFY`` and every output that uses one says so.
The calculations are simplified illustrations (federal only, no credits, deductions,
state tax, or NIIT) and never a tax estimate for a real person.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.reference import REFERENCE_DIR

FilingStatus = Literal["single", "married_joint"]
Status = Literal["verified", "VERIFY"]
UNVERIFIED_NOTE = (
    "This figure hasn't been confirmed against IRS.gov yet. Check the linked IRS page."
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Sourced(_Model):
    """Anything backed by an IRS.gov page. Verified items must say when they were checked."""

    source_url: str
    status: Status
    verified_on: date | None = None

    @model_validator(mode="after")
    def _verified_has_date(self) -> _Sourced:
        if self.status == "verified" and self.verified_on is None:
            raise ValueError("verified items need a verified_on date")
        if not self.source_url.startswith("https://www.irs.gov/"):
            raise ValueError("source_url must be an IRS.gov page")
        return self

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class TaxFigure(_Sourced):
    """One sourced tax figure, such as a contribution limit, keyed by ``key`` in the YAML."""

    key: str
    label: str
    value: float | list[float]


class Bracket(_Model):
    """A tax bracket: ``rate`` (a fraction) applies to income up to ``up_to`` dollars."""

    rate: float = Field(ge=0, le=1)
    up_to: float | None = Field(default=None, gt=0)


class BracketTable(_Sourced):
    """Brackets for each filing status; limits must ascend and the top bracket is open-ended."""

    label: str
    single: list[Bracket]
    married_joint: list[Bracket]

    @model_validator(mode="after")
    def _ordered(self) -> BracketTable:
        for table in (self.single, self.married_joint):
            limits = [b.up_to for b in table[:-1]]
            if table[-1].up_to is not None or None in limits or limits != sorted(limits):  # type: ignore[type-var]
                raise ValueError(f"{self.label}: brackets must ascend and end open-ended")
        return self

    def for_status(self, status: FilingStatus) -> list[Bracket]:
        """The brackets for ``status``."""
        return self.single if status == "single" else self.married_joint


class AccountType(_Model):
    """A tax-advantaged account type and how it's taxed, in plain text."""

    key: str
    name: str
    contributions: str
    growth: str
    withdrawals: str
    limit_figures: list[str]
    early_withdrawal: str
    required_distributions: str
    notes: str


Applies = Literal["yes", "no", "n/a"]


class WithdrawalException(_Model):
    """One exception to the 10% early-withdrawal tax, and whether it applies to workplace
    plans (401(k), 403(b)) and to IRAs."""

    key: str
    label: str
    plans: Applies
    iras: Applies


class WithdrawalExceptions(_Sourced):
    """The IRS table of early-withdrawal exceptions by account type."""

    items: list[WithdrawalException]


class RmdAge(_Model):
    """The RMD starting age for people born in ``[born_from, born_before)``."""

    born_from: date | None = None
    born_before: date | None = None
    age: float
    note: str = ""

    def covers(self, born: date) -> bool:
        """True when someone born on ``born`` falls in this row."""
        after = self.born_from is None or born >= self.born_from
        before = self.born_before is None or born < self.born_before
        return after and before


class RmdAges(_Sourced):
    """When required minimum distributions start, by date of birth (SECURE 2.0)."""

    first_rmd_due: str
    schedule: list[RmdAge]


class TaxReference(_Model):
    """One tax year's reference data: figures, brackets, account types, early-withdrawal
    exceptions, and RMD ages."""

    tax_year: int
    jurisdiction: str
    last_reviewed: str
    figures: dict[str, TaxFigure]
    ordinary_income: BracketTable
    long_term_capital_gains: BracketTable
    accounts: dict[str, AccountType]
    early_withdrawal_exceptions: WithdrawalExceptions
    rmd_ages: RmdAges

    @model_validator(mode="after")
    def _account_figures_exist(self) -> TaxReference:
        for account in self.accounts.values():
            missing = set(account.limit_figures) - set(self.figures)
            if missing:
                raise ValueError(f"{account.key} references unknown figures: {sorted(missing)}")
        return self

    def unverified(self) -> list[str]:
        """Every item still marked VERIFY, for the owner's review list."""
        items = [f"{f.key}: {f.label}" for f in self.figures.values() if not f.verified]
        for table in (self.ordinary_income, self.long_term_capital_gains):
            if not table.verified:
                items.append(f"brackets: {table.label}")
        return items

    def figure(self, key: str) -> TaxFigure:
        """The figure named ``key``; raises ``KeyError`` with a readable message if unknown."""
        try:
            return self.figures[key]
        except KeyError:
            raise KeyError(f"Unknown tax figure '{key}'") from None


def load_tax_reference(path: Path) -> TaxReference:
    """Load and validate a ``tax_<year>.yaml`` file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    brackets = data.get("brackets") or {}
    return TaxReference(
        tax_year=data["tax_year"],
        jurisdiction=data["jurisdiction"],
        last_reviewed=str(data["last_reviewed"]),
        figures={k: TaxFigure(key=k, **v) for k, v in (data.get("figures") or {}).items()},
        ordinary_income=BracketTable(**brackets["ordinary_income"]),
        long_term_capital_gains=BracketTable(**brackets["long_term_capital_gains"]),
        accounts={k: AccountType(key=k, **v) for k, v in (data.get("accounts") or {}).items()},
        early_withdrawal_exceptions=WithdrawalExceptions(**data["early_withdrawal_exceptions"]),
        rmd_ages=RmdAges(**data["rmd_ages"]),
    )


@cache
def get_tax_reference(year: int = 2026) -> TaxReference:
    """The reference data for ``year``, loaded once per year per process."""
    return load_tax_reference(REFERENCE_DIR / f"tax_{year}.yaml")


# ---- account comparison ---------------------------------------------------------------


class AccountComparisonRow(BaseModel):
    """One account type's row in :func:`compare_accounts`, with its limit figures resolved."""

    key: str
    name: str
    contributions: str
    growth: str
    withdrawals: str
    early_withdrawal: str
    required_distributions: str
    limits: list[TaxFigure]
    notes: str
    has_unverified_figures: bool


def compare_accounts(
    reference: TaxReference, keys: list[str] | None = None
) -> list[AccountComparisonRow]:
    """Side-by-side rows for the requested account types (all when ``keys`` is None)."""
    selected = keys or list(reference.accounts)
    unknown = [k for k in selected if k not in reference.accounts]
    if unknown:
        raise KeyError(
            f"Unknown account type(s): {', '.join(unknown)}. Known: {', '.join(reference.accounts)}"
        )
    rows = []
    for key in selected:
        account = reference.accounts[key]
        limits = [reference.figures[f] for f in account.limit_figures]
        rows.append(
            AccountComparisonRow(
                key=key,
                name=account.name,
                contributions=account.contributions,
                growth=account.growth,
                withdrawals=account.withdrawals,
                early_withdrawal=account.early_withdrawal,
                required_distributions=account.required_distributions,
                limits=limits,
                notes=account.notes,
                has_unverified_figures=any(not f.verified for f in limits),
            )
        )
    return rows


def _age_text(age: float) -> str:
    return "70½" if age == 70.5 else f"{age:g}"


def rmd_age_for(reference: TaxReference, *, birth_year: int) -> str:
    """The RMD starting age for a birth year, in words, with the rule's note if any.

    Two rows can share a birth year (1949 was split mid-year); then both are given.
    """
    rows = [
        row
        for row in reference.rmd_ages.schedule
        if row.covers(date(birth_year, 1, 1)) or row.covers(date(birth_year, 12, 31))
    ]
    ages = " or ".join(dict.fromkeys(_age_text(r.age) for r in rows))
    notes = "; ".join(r.note for r in rows if r.note)
    return f"born in {birth_year}: RMDs start at age {ages}" + (f" ({notes})" if notes else "")


def rmd_age_for_current_age(reference: TaxReference, *, age: int, year: int) -> str:
    """The RMD age for someone who is ``age`` in ``year``. Their birth year is one of two
    (it depends on their birthday), so both are checked and given when they differ."""
    later, earlier = year - age, year - age - 1
    first, second = (
        rmd_age_for(reference, birth_year=earlier),
        rmd_age_for(reference, birth_year=later),
    )
    if first.split(": ", 1)[1] == second.split(": ", 1)[1]:
        return (
            f"Age {age} in {year} means born in {earlier} or {later}; " + second.split(": ", 1)[1]
        )
    return f"Age {age} in {year} means born in {earlier} or {later}: {first}; {second}"


def exceptions_for(reference: TaxReference, account: Literal["plans", "iras"]) -> list[str]:
    """Labels of the early-withdrawal exceptions that apply to ``plans`` or ``iras``."""
    return [
        item.label
        for item in reference.early_withdrawal_exceptions.items
        if getattr(item, account) == "yes"
    ]


# ---- illustrative calculations --------------------------------------------------------


def marginal_rate(brackets: list[Bracket], taxable_income: float) -> float:
    """The rate of the bracket ``taxable_income`` falls in, as a fraction."""
    for bracket in brackets:
        if bracket.up_to is None or taxable_income <= bracket.up_to:
            return bracket.rate
    raise ValueError("Bracket table must end open-ended")  # pragma: no cover - validated on load


def tax_on_income(brackets: list[Bracket], taxable_income: float) -> float:
    """Progressive tax: each slice of income is taxed at its bracket's rate."""
    tax, lower = 0.0, 0.0
    for bracket in brackets:
        upper = bracket.up_to if bracket.up_to is not None else float("inf")
        if taxable_income > lower:
            tax += (min(taxable_income, upper) - lower) * bracket.rate
        lower = upper
    return tax


def stacked_gains_tax(brackets: list[Bracket], ordinary_income: float, gain: float) -> float:
    """Long-term gains 'stack' on top of ordinary income to find which rates apply."""
    return tax_on_income(brackets, ordinary_income + gain) - tax_on_income(
        brackets, ordinary_income
    )


def one_year_anniversary(purchase: date) -> date:
    """The same calendar date one year later. A Feb 29 purchase's anniversary is Feb 28."""
    try:
        return purchase.replace(year=purchase.year + 1)
    except ValueError:  # Feb 29 in a leap year
        return date(purchase.year + 1, 2, 28)


def is_long_term(purchase: date, sale: date) -> bool:
    """Long-term means held more than one year: sold after the one-year anniversary.

    Selling on the anniversary itself is still short-term. This is a calendar rule, not a
    day count: buying Mar 1, 2023 and selling Mar 1, 2024 is 366 days but short-term.
    """
    if sale < purchase:
        raise ValueError("The sale date can't be before the purchase date.")
    return sale > one_year_anniversary(purchase)


class CapitalGainsIllustration(BaseModel):
    """Result of :func:`illustrate_capital_gains`.

    Taxes are dollars; ``effective_rate_on_gain`` is a fraction of the gain.
    """

    gain: float
    purchase_date: date
    sale_date: date
    long_term: bool
    first_long_term_sale_date: date
    filing_status: FilingStatus
    taxable_income_before_gain: float
    tax_if_short_term: float
    tax_if_long_term: float
    applied_tax: float
    effective_rate_on_gain: float
    difference: float = Field(description="Short-term tax minus long-term tax")
    tax_year: int
    uses_unverified_figures: bool
    caveats: list[str]


def illustrate_capital_gains(
    reference: TaxReference,
    *,
    gain: float,
    purchase_date: date,
    sale_date: date,
    taxable_income: float,
    filing_status: FilingStatus = "single",
) -> CapitalGainsIllustration:
    """Compare federal tax on a gain held short-term vs long-term (simplified)."""
    if gain <= 0:
        raise ValueError("Enter a positive gain; losses are handled differently.")
    if taxable_income < 0:
        raise ValueError("Taxable income can't be negative.")
    long_term = is_long_term(purchase_date, sale_date)

    holding_rule = reference.figure("long_term_holding_period_years")
    ordinary = reference.ordinary_income.for_status(filing_status)
    ltcg = reference.long_term_capital_gains.for_status(filing_status)

    short_tax = tax_on_income(ordinary, taxable_income + gain) - tax_on_income(
        ordinary, taxable_income
    )
    long_tax = stacked_gains_tax(ltcg, taxable_income, gain)
    applied = long_tax if long_term else short_tax
    unverified = not (
        holding_rule.verified
        and reference.ordinary_income.verified
        and reference.long_term_capital_gains.verified
    )
    caveats = [
        "Simplified federal illustration only: it ignores state taxes, the net investment "
        "income tax, credits, and other deductions.",
        "Taxable income here means income after deductions, before adding this gain.",
    ]
    if unverified:
        caveats.append(UNVERIFIED_NOTE)
    return CapitalGainsIllustration(
        gain=gain,
        purchase_date=purchase_date,
        sale_date=sale_date,
        long_term=long_term,
        first_long_term_sale_date=one_year_anniversary(purchase_date) + timedelta(days=1),
        filing_status=filing_status,
        taxable_income_before_gain=taxable_income,
        tax_if_short_term=round(short_tax, 2),
        tax_if_long_term=round(long_tax, 2),
        applied_tax=round(applied, 2),
        effective_rate_on_gain=round(applied / gain, 4),
        difference=round(short_tax - long_tax, 2),
        tax_year=reference.tax_year,
        uses_unverified_figures=unverified,
        caveats=caveats,
    )
