from pathlib import Path

import pytest
import yaml

from src.rag.knowledge_base import (
    ArticleError,
    load_articles,
    load_glossary,
    main,
    parse_article,
    validate_knowledge_base,
)

BODY = (
    "Opening paragraph that explains the idea in plain language. " * 25
    + "\n\n## How it works\n\n"
    + "More detail about the concept with a simple hypothetical example. " * 20
    + "\n\n## Key takeaways\n\n- One\n- Two\n- Three\n"
)


def front(**overrides):
    meta = {
        "id": "stocks-001",
        "title": "What Is a Stock?",
        "category": "stocks",
        "difficulty": "beginner",
        "tags": ["basics"],
        "sources": [{"name": "Investor.gov - Stocks", "url": "https://www.investor.gov/stocks"}],
        "last_reviewed": "2026-09-30",
    } | overrides
    return "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---\n"


def write(root: Path, category: str, name: str, text: str) -> Path:
    path = root / category / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def glossary(root: Path, terms=None, **overrides):
    terms = (
        terms
        if terms is not None
        else [
            {
                "term": "Stock",
                "definition": "A share of ownership in a company that can rise or fall in value.",
                "related": ["Bond"],
            },
            {
                "term": "Bond",
                "definition": "A loan an investor makes to a borrower in exchange for interest.",
            },
        ]
    )
    data = {
        "sources": [{"name": "Investor.gov glossary", "url": "https://www.investor.gov/glossary"}],
        "last_reviewed": "2026-09-30",
        "terms": terms,
    } | overrides
    (root / "glossary.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_parse_valid_article(tmp_path):
    path = write(tmp_path, "stocks", "stocks-001", front() + BODY)
    article = parse_article(path)
    assert article.meta.title == "What Is a Stock?" and 400 <= article.word_count <= 900
    assert article.sections == ["How it works", "Key takeaways"]
    assert load_articles(tmp_path)[0].meta.id == "stocks-001"


def test_curly_apostrophes_count_as_one_word(tmp_path):
    text = front() + BODY.replace("Opening", "It" + chr(0x2019) + "s")
    article = parse_article(write(tmp_path, "stocks", "stocks-001", text))
    plain = parse_article(write(tmp_path, "stocks", "stocks-002", front(id="stocks-002") + BODY))
    assert article.word_count == plain.word_count


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (BODY, "missing front matter"),
        ("---\nid: [unclosed\n---\n" + BODY, "unreadable front matter"),
        (front(category="astrology") + BODY, "category"),
        (front(sources=[]) + BODY, "sources"),
        (front(sources=[{"name": "X site", "url": "http://insecure.gov"}]) + BODY, "sources.0.url"),
        (front(extra_field=1) + BODY, "extra_field"),
        (front(id="Stocks 1") + BODY, "id"),
    ],
)
def test_parse_errors(tmp_path, text, message):
    with pytest.raises(ArticleError, match=message):
        parse_article(write(tmp_path, "stocks", "stocks-001", text))


def test_valid_knowledge_base(tmp_path):
    write(tmp_path, "stocks", "stocks-001", front() + BODY)
    glossary(tmp_path)
    report = validate_knowledge_base(tmp_path, min_articles=0, min_glossary_terms=2)
    assert report.ok and report.articles == 1 and report.glossary_terms == 2
    assert report.per_category["stocks"] == 1 and report.per_category["taxes"] == 0


def test_article_rule_violations(tmp_path):
    write(tmp_path, "taxes", "stocks-001", front() + BODY)  # wrong folder
    write(tmp_path, "stocks", "wrong-name", front(id="stocks-002") + BODY)
    write(
        tmp_path,
        "stocks",
        "stocks-003",
        front(id="stocks-003", category="stocks", title="Short article")
        + "Too short.\n\n## Only one\n",
    )
    write(
        tmp_path,
        "stocks",
        "stocks-004",
        front(id="stocks-004", title="What Is a Stock?") + "# Title again\n\n" + BODY,
    )
    write(
        tmp_path,
        "stocks",
        "stocks-005",
        front(id="stocks-005", title="Hot Tips Here")
        + BODY
        + "\nThis stock is a sure thing, you should buy it. TODO: finish.\n",
    )
    write(
        tmp_path,
        "stocks",
        "stocks-006",
        front(
            id="stocks-006",
            title="Duplicate sources",
            sources=[
                {"name": "A one", "url": "https://www.sec.gov/a"},
                {"name": "A two", "url": "https://www.sec.gov/a"},
            ],
        )
        + BODY,
    )
    write(
        tmp_path,
        "bonds_fixed_income",
        "stocks-007",
        front(id="stocks-007", category="bonds_fixed_income", title="Mismatched id prefix") + BODY,
    )
    problems = "\n".join(
        validate_knowledge_base(tmp_path, min_articles=0, require_glossary=False).problems
    )
    assert "taxes/stocks-001.md: category 'stocks' doesn't match its folder" in problems
    assert "wrong-name.md: file name doesn't match id 'stocks-002'" in problems
    assert "stocks-003.md" in problems and "words (expected 400-900)" in problems
    assert "needs at least 2 '## ' sections" in problems
    assert "don't repeat the title" in problems
    assert "duplicate title: 'what is a stock?'" in problems
    assert "advice/guarantee language: 'sure thing'" in problems
    assert "placeholder text: 'TODO'" in problems
    assert "duplicate source URLs" in problems
    assert "id 'stocks-007' must start with 'bonds_fixed_income-'" in problems


def test_buy_now_pay_later_is_not_advice():
    from src.rag.knowledge_base import FORBIDDEN_PHRASES

    assert not FORBIDDEN_PHRASES.search("Buy now, pay later plans split a purchase.")
    assert FORBIDDEN_PHRASES.search("Buy now before prices rise!")


def test_duplicate_ids_and_unreadable_files(tmp_path):
    write(tmp_path, "stocks", "stocks-001", front() + BODY)
    write(tmp_path, "stocks", "stocks-009", front(title="Another title") + BODY)
    write(tmp_path, "stocks", "stocks-010", "no front matter")
    problems = validate_knowledge_base(tmp_path, min_articles=0, require_glossary=False).problems
    assert any("duplicate id: 'stocks-001'" in p for p in problems)
    assert any("stocks-010.md: missing front matter" in p for p in problems)


def test_minimum_count_and_empty_categories(tmp_path):
    write(tmp_path, "stocks", "stocks-001", front() + BODY)
    problems = validate_knowledge_base(tmp_path, min_articles=5, require_glossary=False).problems
    assert "only 1 articles (expected at least 5)" in problems
    assert "category 'taxes' has no articles" in problems


def test_glossary_rules(tmp_path):
    glossary(
        tmp_path,
        terms=[
            {"term": "Stock", "definition": "Too short."},
            {
                "term": "stock",
                "definition": "A duplicate term that differs only in capitalization here.",
            },
            {
                "term": "Bond",
                "definition": "A loan to a borrower that pays interest and is a sure thing.",
                "related": ["Nonexistent"],
            },
        ],
    )
    report = validate_knowledge_base(tmp_path, min_articles=0, min_glossary_terms=10)
    problems = "\n".join(report.problems)
    assert "glossary: 3 terms (expected at least 10)" in problems
    assert "duplicate term 'stock'" in problems
    assert "'Stock' definition has 2 words" in problems
    assert "links to unknown term 'Nonexistent'" in problems
    assert "'Bond' contains 'sure thing'" in problems


def test_invalid_or_missing_glossary(tmp_path):
    (tmp_path / "glossary.yaml").write_text("terms: 5\n", encoding="utf-8")
    assert any(
        p.startswith("glossary: invalid")
        for p in validate_knowledge_base(tmp_path, min_articles=0).problems
    )
    (tmp_path / "glossary.yaml").unlink()
    assert "missing glossary.yaml" in validate_knowledge_base(tmp_path, min_articles=0).problems


def test_load_glossary(tmp_path):
    glossary(tmp_path)
    assert [t.term for t in load_glossary(tmp_path).terms] == ["Stock", "Bond"]


def test_cli(tmp_path, capsys):
    write(tmp_path, "stocks", "stocks-001", front() + BODY)
    assert main(["--root", str(tmp_path), "--min-articles", "0", "--no-glossary"]) == 0
    assert "Knowledge base OK" in capsys.readouterr().out
    assert main(["--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "only 1 articles" in err and "missing glossary.yaml" in err
