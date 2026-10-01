"""The real knowledge base must always pass the validator (no network needed)."""

from src.rag.knowledge_base import MIN_ARTICLES, load_articles, validate_knowledge_base


def test_knowledge_base_passes_validation():
    report = validate_knowledge_base()
    assert report.ok, "\n".join(report.problems)
    assert report.articles >= MIN_ARTICLES
    assert report.glossary_terms >= 150
    assert all(count > 0 for count in report.per_category.values())


def test_every_article_cites_an_https_source_and_is_reviewed():
    for article in load_articles():
        assert article.meta.sources, article.path
        assert article.meta.last_reviewed.year >= 2026, article.path
