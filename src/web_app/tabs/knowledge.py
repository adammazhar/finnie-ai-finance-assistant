"""Knowledge page: semantic search, browse by category, read an article, glossary A to Z.

The three views are a segmented control (not ``st.tabs``) so other pages can open a
specific article or glossary term here.
"""

from __future__ import annotations

import streamlit as st

from src.rag.knowledge_base import CATEGORIES, Article
from src.web_app import services, state
from src.web_app.formatting import match_label, md, snippet

VIEWS = ["Search", "Browse", "Glossary"]
DIFFICULTY = {"beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced"}


def _search() -> None:
    query = st.text_input(
        "Search the knowledge base",
        placeholder="e.g. why do bond prices fall when rates rise?",
        key="kb_query",
    )
    if not query.strip():
        return
    retriever = services.context().retriever
    if retriever is None:
        st.warning(
            "Search is unavailable because the knowledge base index isn't built. "
            "Run `python scripts/build_index.py` once."
        )
        return
    found = retriever.retrieve(query, k=5)
    if not found.chunks:
        st.info("Nothing in the knowledge base matches that closely. Try asking in Chat.")
        return
    for rank, item in enumerate(sorted(found.chunks, key=lambda c: c.score, reverse=True)):
        chunk = item.chunk
        glossary = chunk.kind == "glossary"
        where = "Glossary" if glossary else CATEGORIES.get(chunk.category, chunk.category)
        with st.container(border=True):
            heading = md(chunk.title) if glossary else f"{md(chunk.title)} · {md(chunk.section)}"
            st.markdown(f"**{heading}**")
            st.caption(f"{match_label(item.score)} · {where}")
            st.markdown(md(snippet(chunk.text)))
            if glossary:
                st.button(
                    "Open in Glossary",
                    key=f"kb_hit_{rank}",
                    on_click=state.open_glossary,
                    args=(chunk.title,),
                )
            else:
                st.button(
                    "Read article",
                    key=f"kb_hit_{rank}",
                    on_click=state.open_article,
                    args=(chunk.article_id, chunk.category),
                )


def _reader(article: Article) -> None:
    meta = article.meta
    st.markdown(f"## {md(meta.title)}")
    st.caption(
        f"{CATEGORIES[meta.category]} · {DIFFICULTY[meta.difficulty]} · "
        f"{article.word_count} words · last reviewed {meta.last_reviewed:%b %d, %Y}"
    )
    st.markdown(md(article.body))
    st.markdown("**Sources**")
    st.markdown("\n".join(f"- [{md(s.name)}]({s.url})" for s in meta.sources))


def _browse() -> None:
    articles = services.articles()
    category = st.selectbox(
        "Category", list(CATEGORIES), format_func=CATEGORIES.__getitem__, key="kb_category"
    )
    in_category = sorted(
        (a for a in articles if a.meta.category == category), key=lambda a: a.meta.id
    )
    if not in_category:
        st.info("No articles in this category yet.")
        return
    by_id = {a.meta.id: a for a in in_category}
    if st.session_state.get("kb_article") not in by_id:
        st.session_state["kb_article"] = in_category[0].meta.id
    chosen = st.selectbox(
        "Article",
        list(by_id),
        format_func=lambda i: f"{by_id[i].meta.title} ({DIFFICULTY[by_id[i].meta.difficulty]})",
        key="kb_article",
    )
    _reader(by_id[chosen])


def _glossary() -> None:
    glossary = services.glossary()
    text = st.text_input("Find a term", key="kb_term", placeholder="e.g. expense ratio")
    terms = sorted(glossary.terms, key=lambda t: t.term.lower())
    if text.strip():
        needle = text.strip().lower()
        terms = [t for t in terms if needle in t.term.lower() or needle in t.definition.lower()]
    st.caption(f"{len(terms)} of {len(glossary.terms)} terms")
    letter = ""
    for term in terms:
        first = term.term[0].upper()
        if first != letter:
            letter = first
            st.markdown(f"#### {letter}")
        related = f" *Related: {', '.join(term.related)}.*" if term.related else ""
        st.markdown(md(f"**{term.term}**: {term.definition}{related}"))
    st.markdown("**Glossary sources**")
    st.markdown("\n".join(f"- [{md(s.name)}]({s.url})" for s in glossary.sources))


def _keep_view() -> None:
    """Clicking the open view deselects it; keep showing that view instead."""
    if st.session_state.get("kb_view") is None:
        st.session_state["kb_view"] = st.session_state.get("kb_view_last", "Search")


def render() -> None:
    """Draw the Knowledge page and the selected view (Search, Browse, or Glossary)."""
    st.subheader("Learn")
    if st.session_state.get("kb_view") not in VIEWS:
        st.session_state["kb_view"] = "Search"
    st.session_state["kb_view_last"] = st.session_state["kb_view"]
    view = st.segmented_control(
        "View", VIEWS, key="kb_view", on_change=_keep_view, label_visibility="collapsed"
    )
    {"Search": _search, "Browse": _browse, "Glossary": _glossary}[view or "Search"]()
