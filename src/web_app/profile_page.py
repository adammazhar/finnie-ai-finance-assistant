"""Profile: the onboarding screen on a first visit, and the settings page after that."""

from __future__ import annotations

from typing import get_args

import streamlit as st

from src.core.models import KnowledgeLevel, RiskTolerance, UserProfile
from src.web_app import state
from src.web_app.quiz import QUESTIONS, tolerance_from_scores

LEVELS: list[str] = list(get_args(KnowledgeLevel))
RISKS: list[str] = list(get_args(RiskTolerance))
LEVEL_LABELS = {
    "beginner": "New to investing",
    "intermediate": "Some experience",
    "advanced": "Experienced",
}
RISK_LABELS = {"conservative": "Conservative", "moderate": "Moderate", "aggressive": "Aggressive"}
RISK_HELP = [
    "Steadier value, slower growth",
    "A balance of growth and stability",
    "More growth potential, bigger swings",
]
KEYS = {
    "knowledge_level": "profile_level",
    "risk_tolerance": "profile_risk",
    "age": "profile_age",
    "investment_horizon_years": "profile_horizon",
}
QUIZ_RESULT = "quiz_result"
CONFIRM_DELETE = "confirm_delete_my_data"
STORAGE_NOTE = (
    "What Finnie stores: your profile, portfolio, and conversations are saved on the server "
    "running Finnie, so they're here when you come back. They're linked to a random ID kept "
    "in a cookie in this browser, not to your name or email; there's no login. Clearing "
    "cookies or using another browser starts fresh. Delete my data on the Profile page "
    "removes everything."
)


def _load(current: UserProfile) -> None:
    """Fill the form from a saved profile. The optional numbers take their value from the
    widgets' ``value=`` argument, so their keys are cleared rather than set."""
    st.session_state[KEYS["knowledge_level"]] = current.knowledge_level
    st.session_state[KEYS["risk_tolerance"]] = current.risk_tolerance
    st.session_state.pop(KEYS["age"], None)
    st.session_state.pop(KEYS["investment_horizon_years"], None)


def open_profile() -> None:
    """Sidebar button callback: show the saved profile for editing."""
    _load(state.profile())
    st.session_state.pop(QUIZ_RESULT, None)
    state.go(state.PROFILE_PAGE)


def _save() -> None:
    age = st.session_state.get(KEYS["age"])
    horizon = st.session_state.get(KEYS["investment_horizon_years"])
    state.set_profile(
        UserProfile(
            knowledge_level=st.session_state[KEYS["knowledge_level"]],
            risk_tolerance=st.session_state[KEYS["risk_tolerance"]],
            age=int(age) if age is not None else None,
            investment_horizon_years=int(horizon) if horizon is not None else None,
        )
    )
    state.go("Chat")


def _score_quiz() -> None:
    scores = []
    for i, (_, options) in enumerate(QUESTIONS):
        choice = st.session_state.get(f"quiz_{i}")
        if choice is None:
            st.session_state[QUIZ_RESULT] = None
            return
        scores.append(dict(options)[choice])
    tolerance = tolerance_from_scores(scores)
    st.session_state[KEYS["risk_tolerance"]] = tolerance
    st.session_state[QUIZ_RESULT] = tolerance


def _quiz() -> None:
    with st.expander("Not sure? Take a 5-question quiz", expanded=False):
        for i, (question, options) in enumerate(QUESTIONS):
            st.radio(question, [label for label, _ in options], index=None, key=f"quiz_{i}")
        st.button("See my result", on_click=_score_quiz, key="quiz_score")
        if QUIZ_RESULT in st.session_state:
            result = st.session_state[QUIZ_RESULT]
            if result is None:
                st.warning("Answer all five questions to see a result.")
            else:
                st.success(
                    f"Your answers suggest a **{result}** risk tolerance, and it's now selected. "
                    "You can change it. This quiz is a starting point for learning, not advice."
                )


def render(first_visit: bool) -> None:
    if KEYS["knowledge_level"] not in st.session_state:
        _load(state.profile())
    _, middle, _ = st.columns([1, 3, 1])
    with middle:
        if first_visit:
            st.markdown("## Welcome to Finnie")
            st.markdown(
                "Finnie helps you learn about investing: concepts, your own portfolio, markets, "
                "savings goals, and taxes. A few quick questions so answers fit you. You can "
                "change these any time from **Profile** in the sidebar."
            )
        else:
            st.markdown("## Your profile")
            st.caption("Finnie uses this to pitch explanations and to set projection assumptions.")
        with st.container(border=True):
            st.radio(
                "How much do you know about investing?",
                LEVELS,
                format_func=LEVEL_LABELS.__getitem__,
                horizontal=True,
                key=KEYS["knowledge_level"],
            )
            st.radio(
                "Risk tolerance",
                RISKS,
                format_func=RISK_LABELS.__getitem__,
                captions=RISK_HELP,
                horizontal=True,
                key=KEYS["risk_tolerance"],
            )
            _quiz()
            left, right = st.columns(2)
            saved = state.profile()
            left.number_input(
                "Age (optional)", min_value=13, max_value=120, value=saved.age, key=KEYS["age"]
            )
            right.number_input(
                "Investment horizon in years (optional)",
                min_value=0,
                max_value=80,
                value=saved.investment_horizon_years,
                key=KEYS["investment_horizon_years"],
            )
        buttons = st.columns([1, 1, 2])
        buttons[0].button(
            "Get started" if first_visit else "Save profile",
            type="primary",
            on_click=_save,
            key="profile_save",
            width="stretch",
        )
        if not first_visit:
            buttons[1].button(
                "Cancel", on_click=state.go, args=("Chat",), key="profile_cancel", width="stretch"
            )
        st.caption(STORAGE_NOTE)
        if not first_visit:
            _delete_my_data()
        st.caption(
            "Finnie provides educational information only, not financial, investment, tax, or "
            "legal advice."
        )


def _ask_delete() -> None:
    st.session_state[CONFIRM_DELETE] = True


def _cancel_delete() -> None:
    st.session_state.pop(CONFIRM_DELETE, None)


def _delete_my_data() -> None:
    st.divider()
    if not st.session_state.get(CONFIRM_DELETE):
        st.button(
            "Delete my data",
            icon=":material/delete:",
            key="delete_my_data",
            on_click=_ask_delete,
        )
        return
    with st.container(border=True):
        st.markdown(
            "Delete your profile, portfolio, and **all conversations** saved for this browser? "
            "This can't be undone, and Finnie will start fresh."
        )
        confirm, cancel = st.columns(2)
        confirm.button(
            "Delete everything",
            type="primary",
            key="delete_my_data_confirm",
            on_click=state.delete_my_data,
            width="stretch",
        )
        cancel.button(
            "Cancel", key="delete_my_data_cancel", on_click=_cancel_delete, width="stretch"
        )
