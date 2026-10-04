"""Reference data: security classifications and risk-profile assumptions.

Both come from YAML files in ``data/reference/`` and are validated on load.
"""

from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.config import PROJECT_ROOT
from src.core.models import RiskTolerance

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

SecurityType = Literal["stock", "etf", "mutual_fund", "money_market", "cash", "other"]
AssetClass = Literal["equity", "bond", "cash", "real_estate", "commodity", "crypto", "unknown_mix"]
# How each asset class (and the "other" comparison group) is named in charts, tables, and chat.
# "unknown_mix" is a fund Finnie has no breakdown for: it isn't counted as stocks or bonds.
ASSET_CLASS_LABELS: dict[str, str] = {
    "equity": "Stocks",
    "bond": "Bonds",
    "cash": "Cash",
    "real_estate": "Real estate",
    "commodity": "Commodities",
    "crypto": "Crypto",
    "unknown_mix": "Mix unknown",
    "other": "Other",
}


def asset_class_label(asset_class: str) -> str:
    """The display name of an asset class, e.g. ``unknown_mix`` -> "Mix unknown"."""
    return ASSET_CLASS_LABELS.get(asset_class, asset_class.replace("_", " ").capitalize())


_OVERVIEW_TYPES: dict[str, SecurityType] = {
    "equity": "stock",
    "common stock": "stock",
    "etf": "etf",
    "mutual fund": "mutual_fund",
}


FUND_TYPES = frozenset({"etf", "mutual_fund", "money_market"})


class ExpenseRatio(BaseModel):
    """A fund's expense ratio and where it was read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ratio: float = Field(ge=0, le=0.05)
    status: Literal["verified", "VERIFY"]
    as_of: date | None = Field(default=None, description="Date the provider states, if any")
    verified_on: date | None = None
    source_url: str = Field(pattern=r"^https://")

    @model_validator(mode="after")
    def _verified_has_date(self) -> ExpenseRatio:
        if self.status == "verified" and self.verified_on is None:
            raise ValueError("verified expense ratios need a verified_on date")
        return self

    @property
    def verified(self) -> bool:
        """True when the ratio was confirmed on the provider's site."""
        return self.status == "verified"


class SecurityInfo(BaseModel):
    """How one security is classified for analysis: type, asset class, sector, fees, and risk.

    ``allocation`` is the asset-class mix of a mixed fund (fractions summing to 1);
    ``risk`` is a 1-10 rating. ``known`` is false when the entry was inferred rather than
    taken from the catalog.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    name: str
    type: SecurityType
    asset_class: AssetClass
    allocation: dict[AssetClass, float] = Field(default_factory=dict)
    sector: str
    diversified: bool
    fees: ExpenseRatio | None = None
    risk: int = Field(ge=1, le=10)
    known: bool = True

    @model_validator(mode="after")
    def _check_allocation(self) -> SecurityInfo:
        if self.allocation and abs(sum(self.allocation.values()) - 1) > 1e-6:
            raise ValueError(f"{self.ticker}: allocation must sum to 1")
        return self

    @property
    def expense_ratio(self) -> float | None:
        """Annual fund fee as a fraction. Stocks and cash have none; unknown funds are None."""
        if self.type in ("stock", "cash"):
            return 0.0
        return self.fees.ratio if self.fees else None

    @property
    def look_through(self) -> dict[AssetClass, float]:
        """Asset-class mix inside this security (a balanced fund is part equity, part bond)."""
        return dict(self.allocation) or {self.asset_class: 1.0}

    @property
    def is_single_issuer(self) -> bool:
        """True when the whole position depends on one company (e.g. a stock)."""
        return not self.diversified


class SecurityCatalog:
    """The securities from ``securities.yaml``, keyed by ticker."""

    def __init__(self, securities: dict[str, SecurityInfo], last_reviewed: str) -> None:
        self._securities = securities
        self.last_reviewed = last_reviewed

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._securities

    def __len__(self) -> int:
        return len(self._securities)

    def tickers(self) -> list[str]:
        """Every ticker in the catalog, in file order."""
        return list(self._securities)

    def get(self, ticker: str) -> SecurityInfo | None:
        """The catalog entry for ``ticker``, or ``None`` if it isn't listed."""
        return self._securities.get(ticker)

    def unverified_expense_ratios(self) -> list[str]:
        """Funds whose expense ratio couldn't be confirmed on the provider's site."""
        return sorted(t for t, s in self._securities.items() if s.fees and not s.fees.verified)

    def classify(
        self,
        ticker: str,
        *,
        name: str | None = None,
        asset_type: str | None = None,
        sector: str | None = None,
    ) -> SecurityInfo:
        """Known securities come from the catalog; others are inferred from overview data.

        Inferred entries have ``known=False`` so analysis can say the classification is a guess.
        """
        known = self._securities.get(ticker)
        if known:
            return known
        security_type = _OVERVIEW_TYPES.get((asset_type or "").strip().lower(), "other")
        is_fund = security_type in ("etf", "mutual_fund")
        return SecurityInfo(
            ticker=ticker,
            name=name or ticker,
            type=security_type,
            # A fund's stock/bond split can't be guessed from its name: a target-date fund
            # holds both. Say so rather than counting it as 100% stocks.
            asset_class="unknown_mix" if is_fund else "equity",
            sector=sector or ("Broad Market" if is_fund else "Unknown"),
            diversified=is_fund,
            risk=6 if is_fund else 8,
            known=False,
        )


class RiskProfile(BaseModel):
    """A risk tolerance's target allocation and long-run return/volatility assumptions.

    The allocation is fractions summing to 1; ``expected_return`` (nominal) and
    ``volatility`` are annual fractions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: RiskTolerance
    label: str
    description: str
    allocation: dict[Literal["equity", "bond", "cash"], float]
    expected_return: float = Field(ge=-0.1, le=0.3)
    volatility: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check_allocation(self) -> RiskProfile:
        if abs(sum(self.allocation.values()) - 1) > 1e-6:
            raise ValueError(f"{self.name}: allocation must sum to 1")
        return self


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_catalog(path: Path = REFERENCE_DIR / "securities.yaml") -> SecurityCatalog:
    """Load securities and attach each fund's sourced expense ratio.

    Every fund must have an ``expense_ratios`` entry and every entry must match a fund,
    so a ratio can't silently go missing or refer to a ticker that isn't classified.
    """
    data = _load_yaml(path)
    raw = data.get("securities") or {}
    fees = {t: ExpenseRatio(**f) for t, f in (data.get("expense_ratios") or {}).items()}
    funds = {t for t, f in raw.items() if f.get("type") in FUND_TYPES}
    if funds - set(fees):
        raise ValueError(f"{path}: funds missing expense ratios: {sorted(funds - set(fees))}")
    if set(fees) - funds:
        raise ValueError(f"{path}: expense ratios for non-funds: {sorted(set(fees) - funds)}")
    securities = {
        ticker: SecurityInfo(ticker=ticker, fees=fees.get(ticker), **fields)
        for ticker, fields in raw.items()
    }
    return SecurityCatalog(securities, last_reviewed=str(data.get("last_reviewed", "unknown")))


def load_risk_profiles(
    path: Path = REFERENCE_DIR / "risk_profiles.yaml",
) -> dict[RiskTolerance, RiskProfile]:
    """Load the risk profiles; raises ``ValueError`` if any of the three tolerances is missing."""
    data = _load_yaml(path)
    profiles = {
        name: RiskProfile(name=name, **fields)
        for name, fields in (data.get("profiles") or {}).items()
    }
    missing = {"conservative", "moderate", "aggressive"} - set(profiles)
    if missing:
        raise ValueError(f"{path} is missing risk profiles: {', '.join(sorted(missing))}")
    return profiles


@cache
def get_catalog() -> SecurityCatalog:
    """The security catalog, loaded once per process."""
    return load_catalog()


@cache
def get_risk_profiles() -> dict[RiskTolerance, RiskProfile]:
    """The risk profiles, loaded once per process."""
    return load_risk_profiles()
