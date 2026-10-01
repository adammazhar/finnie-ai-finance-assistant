from src.rag.chunking import Chunk
from src.rag.citations import (
    build_context,
    check_citations,
    cited_sources,
    render_sources,
)
from src.rag.knowledge_base import SourceRef
from src.rag.retriever import RetrievedChunk


def retrieved(
    article: str, section: str, text: str, score: float = 0.7, category="taxes", sources=True
) -> RetrievedChunk:
    refs = (
        [SourceRef(name="IRS Topic 409", url="https://www.irs.gov/taxtopics/tc409")]
        if sources
        else []
    )
    return RetrievedChunk(
        chunk=Chunk(
            id=f"{article}#0",
            kind="article",
            article_id=article,
            title=f"Title {article}",
            category=category,
            section=section,
            text=text,
            sources=refs,
        ),
        score=score,
    )


CHUNKS = [
    retrieved("taxes-002", "Overview", "Long-term means more than one year."),
    retrieved("taxes-002", "Example", "Sold after the anniversary.", score=0.6),
    retrieved("taxes-006", "Rule", "Wash sales disallow losses.", score=0.5),
]


def test_build_context_numbers_blocks():
    text, blocks = build_context(CHUNKS)
    assert [b.number for b in blocks] == [1, 2, 3]
    assert text.startswith("[1] Title taxes-002 > Overview\nLong-term means more than one year.")
    assert "\n\n[3] Title taxes-006 > Rule\n" in text
    source = blocks[0].source
    assert (source.title, source.kind, source.category) == (
        "Title taxes-002",
        "knowledge_base",
        "Taxes",
    )
    assert source.url == "https://www.irs.gov/taxtopics/tc409" and source.score == 0.7


def test_check_citations_keeps_valid_and_strips_invented():
    answer = "Held over a year [1]. After the anniversary [2][1]. Also [7] and [0]."
    result = check_citations(answer, block_count=3)
    assert result.cited == [1, 2]
    assert result.removed == [7, 0]
    assert result.text == "Held over a year [1]. After the anniversary [2][1]. Also and."


def test_check_citations_without_any():
    result = check_citations("No citations here.", block_count=2)
    assert (result.text, result.cited, result.removed) == ("No citations here.", [], [])


def test_cited_sources_one_per_article_in_order():
    _, blocks = build_context(CHUNKS)
    sources = cited_sources(blocks, [3, 1, 2])
    assert [s.article_id for s in sources] == ["taxes-006", "taxes-002"]


def test_render_sources():
    _, blocks = build_context(CHUNKS)
    rendered = render_sources(cited_sources(blocks, [1, 3]))
    assert rendered.splitlines() == [
        "**Sources**",
        "1. Title taxes-002 (Taxes) - [reference](https://www.irs.gov/taxtopics/tc409)",
        "2. Title taxes-006 (Taxes) - [reference](https://www.irs.gov/taxtopics/tc409)",
    ]
    assert render_sources([]) == ""


def test_source_without_url_or_category():
    chunk = retrieved("glossary:x", "Definition", "A term.", category="glossary", sources=False)
    _, blocks = build_context([chunk])
    source = blocks[0].source
    assert source.url is None and source.category == "Glossary"
    no_category = source.model_copy(update={"category": None})
    assert render_sources([no_category]) == "**Sources**\n1. Title glossary:x"
