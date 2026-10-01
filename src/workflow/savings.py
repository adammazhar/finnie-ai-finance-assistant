"""How much of a saved portfolio counts toward a savings goal.

When a portfolio is saved and a goal question doesn't say what the user has saved, Finnie
asks once per goal: all of the portfolio, part of it (a dollar amount), or none. The
answer is kept per goal in the conversation, so follow-ups about the same goal don't ask
again. Savings stated in the question (an amount, or the holdings themselves) are used
directly.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

DEFAULT_GOAL = "goal"
SHORT_REPLY_WORDS = 8  # a longer unparseable reply is treated as a new question

_AMOUNT = r"(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|m|million)?\b"
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}
_PER_PERIOD = re.compile(r"^\s*(a|per|each|every|/)\s*(month|mo|year|yr|week)\b|^\s*monthly", re.I)
STATED = [
    re.compile(
        r"\$\s?" + _AMOUNT + r"\s+(?:already\s+)?(?:saved|in savings|set aside|put away|invested)",
        re.I,
    ),
    re.compile(
        r"\b(?:i(?:'ve| have)\s+(?:already\s+|currently\s+)?(?:saved|got|have|put away)?|"
        r"savings of|current savings (?:of|are|is))\s*\$\s?" + _AMOUNT,
        re.I,
    ),
]
NOTHING_SAVED = re.compile(
    r"\b(nothing saved|no savings|zero saved|starting from (zero|scratch|\$0))\b", re.I
)
REPLY_AMOUNT = re.compile(r"\$?\s?" + _AMOUNT + r"\s*(%|percent)?", re.I)
REPLY_HALF = re.compile(r"\bhalf\b", re.I)
REPLY_NONE = re.compile(r"\b(none|nothing|zero|no|don'?t|do not|exclude|skip|leave it out)\b", re.I)
REPLY_ALL = re.compile(r"\b(all|everything|whole|entire|full)\b", re.I)
REPLY_NEGATION = re.compile(r"\b(not|no|n't)\b|n't\b", re.I)


class GoalSavings(BaseModel):
    """The user's choice for one goal."""

    choice: Literal["all", "none", "amount"]
    amount: float | None = None  # for "amount"
    portfolio_value: float | None = None  # the value when the choice was made
    source: Literal["asked", "stated"] = "asked"


def goal_key(label: str | None) -> str:
    cleaned = re.sub(r"\s+", " ", (label or "").strip().lower())
    return cleaned or DEFAULT_GOAL


def _to_dollars(number: str, unit: str | None) -> float:
    return float(number.replace(",", "")) * _MULTIPLIERS.get((unit or "").lower(), 1.0)


def stated_savings(text: str) -> float | None:
    """Savings the user states in a goal question ("with $20,000 saved"), else ``None``."""
    if NOTHING_SAVED.search(text):
        return 0.0
    for pattern in STATED:
        for match in pattern.finditer(text):
            if _PER_PERIOD.match(text[match.end() :]):
                continue  # "$500 a month" is a contribution, not savings
            return _to_dollars(match.group(1), match.group(2))
    return None


NUMBER = re.compile(_AMOUNT, re.I)


def grounded_savings(amount: float | None, text: str) -> float | None:
    """The router model's savings figure, kept only when the question supports it.

    Models sometimes fill in 0 (or a number from elsewhere) when no savings were stated,
    which would skip the savings question. A 0 needs words like "nothing saved"; any other
    amount must appear in the text.
    """
    if amount is None:
        return None
    if amount == 0:
        return 0.0 if NOTHING_SAVED.search(text) else None
    for match in NUMBER.finditer(text):
        if abs(_to_dollars(match.group(1), match.group(2)) - amount) <= 0.01 * amount:
            return amount
    return None


def parse_savings_reply(text: str, portfolio_value: float) -> GoalSavings | None:
    """Read the answer to the savings question: all, none, or a dollar amount."""
    if REPLY_HALF.search(text):
        return GoalSavings(
            choice="amount", amount=round(portfolio_value / 2, 2), portfolio_value=portfolio_value
        )
    match = REPLY_AMOUNT.search(text)
    if match:
        if match.group(3):  # a percentage of the portfolio
            share = min(float(match.group(1).replace(",", "")), 100.0) / 100
            amount = round(portfolio_value * share, 2)
        else:
            amount = _to_dollars(match.group(1), match.group(2))
        if amount == 0:
            return GoalSavings(choice="none", portfolio_value=portfolio_value)
        return GoalSavings(choice="amount", amount=amount, portfolio_value=portfolio_value)
    if REPLY_ALL.search(text):
        if REPLY_NEGATION.search(text):
            return None  # "not all of it": ask for an amount
        return GoalSavings(choice="all", portfolio_value=portfolio_value)
    if REPLY_NONE.search(text):
        return GoalSavings(choice="none", portfolio_value=portfolio_value)
    return None


def is_short_reply(text: str) -> bool:
    return len(text.split()) <= SHORT_REPLY_WORDS


def savings_question(portfolio_value: float) -> str:
    return (
        f"You have a saved portfolio worth ${portfolio_value:,.2f}. Should I count all of it, "
        "part of it, or none toward this goal?\n\n"
        'You can reply "all", "none", or an amount such as $10,000.'
    )


def savings_reprompt(portfolio_value: float) -> str:
    return (
        f"Sorry, I didn't catch that. Should I count all of your ${portfolio_value:,.2f} "
        'portfolio, part of it, or none toward this goal? Reply "all", "none", or an amount '
        "such as $10,000."
    )


def savings_fact(savings: GoalSavings, portfolio_value: float | None) -> str:
    """Guidance for the goal agent: what to use as current savings, and why."""
    if savings.choice == "none":
        return (
            "The user chose not to count their saved portfolio toward this goal. Use 0 as "
            "current_balance unless they mention other savings, and say so."
        )
    if savings.choice == "amount" and savings.amount is not None:
        how = (
            "said they have"
            if savings.source == "stated"
            else "chose to count, from their saved portfolio,"
        )
        return (
            f"The user {how} ${savings.amount:,.2f} toward this goal. Use {savings.amount:.2f} "
            "as current_balance, and say so."
        )
    value = portfolio_value if portfolio_value is not None else savings.portfolio_value
    if value is None:
        return (
            "The user wants their whole portfolio counted toward this goal, but its value "
            "couldn't be fetched just now. Say so, and use 0 as current_balance."
        )
    return (
        f"The user counts their whole portfolio, currently worth ${value:,.2f}, toward this "
        f"goal. Use {value:.2f} as current_balance, and say so."
    )
