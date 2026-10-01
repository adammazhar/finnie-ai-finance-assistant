import pytest

from src.workflow.savings import (
    GoalSavings,
    goal_key,
    grounded_savings,
    is_short_reply,
    parse_savings_reply,
    savings_fact,
    savings_question,
    savings_reprompt,
    stated_savings,
)

VALUE = 30_000.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I'm 30 with $20,000 saved. Could I reach $500,000 by 55?", 20_000),
        ("I've saved $15k so far. Can I retire at 60?", 15_000),
        ("I have $30,000 and want $100,000 in 10 years", 30_000),
        ("current savings of $40,000", 40_000),
        ("with $1.5 million invested, am I on track?", 1_500_000),
        ("I have nothing saved. Can I get to $50k?", 0),
    ],
)
def test_stated_savings(text, expected):
    assert stated_savings(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Am I on track to have $1 million by 65 if I save $500 a month?",  # the target
        "I have $500 a month to invest. Can I reach $100k?",  # a contribution
        "I've saved $300 per month for years; can I retire?",
        "Am I on track for retirement?",
    ],
)
def test_no_stated_savings(text):
    assert stated_savings(text) is None


@pytest.mark.parametrize(
    "reply, choice, amount",
    [
        ("all", "all", None),
        ("All of it, please", "all", None),
        ("count everything", "all", None),
        ("none", "none", None),
        ("No", "none", None),
        ("don't count it", "none", None),
        ("$0", "none", None),
        ("$10,000", "amount", 10_000),
        ("10k of it", "amount", 10_000),
        ("1.2 million", "amount", 1_200_000),
        ("half", "amount", 15_000),
        ("25%", "amount", 7_500),
        ("150 percent", "amount", 30_000),  # capped at the whole portfolio
    ],
)
def test_parse_savings_reply(reply, choice, amount):
    savings = parse_savings_reply(reply, VALUE)
    assert savings is not None
    assert (savings.choice, savings.amount, savings.portfolio_value) == (choice, amount, VALUE)


@pytest.mark.parametrize("reply", ["not all of it", "hmm, not sure", "maybe"])
def test_unclear_replies(reply):
    assert parse_savings_reply(reply, VALUE) is None


def test_goal_key_normalizes():
    assert goal_key("  House   Down Payment ") == "house down payment"
    assert goal_key(None) == goal_key("") == "goal"


def test_short_reply():
    assert is_short_reply("not sure what you mean")
    assert not is_short_reply("Actually, what is an ETF and how does it work for beginners?")


def test_question_wording():
    assert savings_question(29121.4).startswith(
        "You have a saved portfolio worth $29,121.40. Should I count all of it, part of it, "
        "or none toward this goal?"
    )
    assert "$29,121.40" in savings_reprompt(29121.4)


def test_savings_facts():
    none = savings_fact(GoalSavings(choice="none"), None)
    assert "not to count" in none and "0 as current_balance" in none
    part = savings_fact(GoalSavings(choice="amount", amount=1500), None)
    assert "chose to count, from their saved portfolio, $1,500.00" in part
    stated = savings_fact(GoalSavings(choice="amount", amount=20000, source="stated"), None)
    assert "said they have $20,000.00" in stated
    everything = GoalSavings(choice="all", portfolio_value=3000)
    assert "currently worth $3,500.00" in savings_fact(everything, 3500)  # today's value wins
    assert "currently worth $3,000.00" in savings_fact(everything, None)  # else the stored one
    assert "couldn't be fetched" in savings_fact(GoalSavings(choice="all"), None)


@pytest.mark.parametrize(
    "amount, text, expected",
    [
        (None, "With $20,000 saved", None),
        (20000, "With $20,000 saved", 20000),
        (20000, "I've put away 20k", 20000),
        (20000, "Retire with $400,000? I have $20,000 saved.", 20000),
        (20000, "Am I on track to retire?", None),  # not in the question: ignored
        (0, "Am I on track to retire?", None),  # "not mentioned" isn't zero savings
        (0, "I have nothing saved yet", 0),
    ],
)
def test_grounded_savings(amount, text, expected):
    assert grounded_savings(amount, text) == expected
