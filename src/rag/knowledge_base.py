"""Knowledge base schema, loading, and validation.

Articles are Markdown files in ``data/knowledge_base/<category>/<id>.md`` with YAML front
matter; the glossary is ``data/knowledge_base/glossary.yaml``. ``validate_knowledge_base``
enforces the content rules, and ``python scripts/validate_kb.py`` runs it as a gate.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import frontmatter
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.config import PROJECT_ROOT

KB_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
GLOSSARY_FILE = "glossary.yaml"

CATEGORIES: dict[str, str] = {
    "investing_basics": "Investing Basics",
    "stocks": "Stocks",
    "bonds_fixed_income": "Bonds and Fixed Income",
    "funds_etfs": "Funds and ETFs",
    "portfolio_management": "Portfolio Management",
    "retirement_planning": "Retirement Planning",
    "taxes": "Taxes",
    "personal_finance": "Personal Finance",
    "market_economics": "Markets and the Economy",
    "risk_behavioral": "Risk and Investor Behavior",
    "financial_planning_goals": "Financial Planning and Goals",
}
Category = Literal[
    "investing_basics",
    "stocks",
    "bonds_fixed_income",
    "funds_etfs",
    "portfolio_management",
    "retirement_planning",
    "taxes",
    "personal_finance",
    "market_economics",
    "risk_behavioral",
    "financial_planning_goals",
]

MIN_ARTICLES = 100
MIN_WORDS, MAX_WORDS = 400, 900
MIN_SECTIONS = 2
MIN_GLOSSARY_TERMS = 150
DEFINITION_WORDS = (8, 90)

# Education, not advice: phrases that promise returns or tell readers what to trade.
FORBIDDEN_PHRASES = re.compile(
    r"\b(guaranteed (?:return|profit)s?|can'?t lose|risk-free profit|you should (?:buy|sell)|"
    r"buy now(?!,? pay later)|sure thing)\b",
    re.IGNORECASE,
)
WORD = re.compile(r"\b[\w'" + chr(0x2019) + r"-]+\b")  # counts curly apostrophes too
PLACEHOLDER_TEXT = re.compile(r"\b(TODO|TBD|lorem ipsum|placeholder)\b", re.IGNORECASE)


class SourceRef(BaseModel):
    """A cited reference (name and https URL) in article front matter or the glossary."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3)
    url: str = Field(pattern=r"^https://[^\s]+$")


class ArticleMeta(BaseModel):
    """The validated YAML front matter of a knowledge base article."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z_]+-\d{3}$")
    title: str = Field(min_length=8, max_length=120)
    category: Category
    difficulty: Literal["beginner", "intermediate", "advanced"]
    tags: list[str] = Field(min_length=1, max_length=8)
    sources: list[SourceRef] = Field(min_length=1)
    last_reviewed: date


class Article(BaseModel):
    """A parsed knowledge base article: front matter, Markdown body, and file path."""

    meta: ArticleMeta
    body: str
    path: Path

    @property
    def word_count(self) -> int:
        """Words in the body; contractions and hyphenated words count once."""
        return len(WORD.findall(self.body))

    @property
    def sections(self) -> list[str]:
        """Titles of the body's ``##`` sections, in order."""
        return re.findall(r"^##\s+(.+?)\s*$", self.body, re.MULTILINE)


class ArticleError(ValueError):
    """An article file that can't be parsed; the message starts with its path."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{path.as_posix()}: {message}")
        self.path = path


def parse_article(path: Path) -> Article:
    """Read one article file.

    Raises ``ArticleError`` for unreadable, missing, or invalid front matter.
    """
    try:
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ArticleError(path, f"unreadable front matter ({exc})") from exc
    if not post.metadata:
        raise ArticleError(path, "missing front matter")
    try:
        meta = ArticleMeta(**post.metadata)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ArticleError(path, f"invalid front matter ({details})") from exc
    return Article(meta=meta, body=post.content.strip(), path=path)


def article_paths(root: Path = KB_ROOT) -> list[Path]:
    """Every ``<category>/<id>.md`` file under ``root``, sorted."""
    return sorted(p for p in root.glob("*/*.md") if p.is_file())


def load_articles(root: Path = KB_ROOT) -> list[Article]:
    """Load every article. Raises ``ArticleError`` on the first unreadable file."""
    return [parse_article(path) for path in article_paths(root)]


class GlossaryTerm(BaseModel):
    """One glossary entry, with links to related terms."""

    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=2)
    definition: str
    related: list[str] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        """Words in the definition."""
        return len(self.definition.split())


class Glossary(BaseModel):
    """The parsed ``glossary.yaml``: its sources, review date, and terms."""

    model_config = ConfigDict(extra="forbid")

    sources: list[SourceRef] = Field(min_length=1)
    last_reviewed: date
    terms: list[GlossaryTerm]


def load_glossary(root: Path = KB_ROOT) -> Glossary:
    """Load and validate ``glossary.yaml``.

    Raises ``OSError``, ``yaml.YAMLError``, or ``pydantic.ValidationError`` if it is missing or
    invalid.
    """
    data = yaml.safe_load((root / GLOSSARY_FILE).read_text(encoding="utf-8"))
    return Glossary(**data)


# ---- validation -----------------------------------------------------------------------


def _article_problems(article: Article) -> list[str]:
    meta, where = article.meta, article.path.as_posix()
    problems = []
    if article.path.parent.name != meta.category:
        problems.append(f"{where}: category '{meta.category}' doesn't match its folder")
    if article.path.stem != meta.id:
        problems.append(f"{where}: file name doesn't match id '{meta.id}'")
    if not meta.id.startswith(f"{meta.category}-"):
        problems.append(f"{where}: id '{meta.id}' must start with '{meta.category}-'")
    words = article.word_count
    if not MIN_WORDS <= words <= MAX_WORDS:
        problems.append(f"{where}: {words} words (expected {MIN_WORDS}-{MAX_WORDS})")
    if len(article.sections) < MIN_SECTIONS:
        problems.append(f"{where}: needs at least {MIN_SECTIONS} '## ' sections")
    if article.body.lstrip().startswith("# "):
        problems.append(f"{where}: don't repeat the title as a '# ' heading (it's in front matter)")
    for pattern, label in (
        (FORBIDDEN_PHRASES, "advice/guarantee language"),
        (PLACEHOLDER_TEXT, "placeholder text"),
    ):
        match = pattern.search(article.body)
        if match:
            problems.append(f"{where}: contains {label}: '{match.group(0)}'")
    urls = [s.url for s in meta.sources]
    if len(urls) != len(set(urls)):
        problems.append(f"{where}: duplicate source URLs")
    return problems


def _glossary_problems(glossary: Glossary, min_terms: int) -> list[str]:
    problems = []
    names = [t.term.lower() for t in glossary.terms]
    if len(glossary.terms) < min_terms:
        problems.append(f"glossary: {len(glossary.terms)} terms (expected at least {min_terms})")
    for name, count in Counter(names).items():
        if count > 1:
            problems.append(f"glossary: duplicate term '{name}'")
    known = set(names)
    low, high = DEFINITION_WORDS
    for term in glossary.terms:
        if not low <= term.word_count <= high:
            problems.append(
                f"glossary: '{term.term}' definition has {term.word_count} words "
                f"(expected {low}-{high})"
            )
        for related in term.related:
            if related.lower() not in known:
                problems.append(f"glossary: '{term.term}' links to unknown term '{related}'")
        match = FORBIDDEN_PHRASES.search(term.definition)
        if match:
            problems.append(f"glossary: '{term.term}' contains '{match.group(0)}'")
    return problems


class ValidationReport(BaseModel):
    """Outcome of ``validate_knowledge_base``: article and term counts plus every problem found."""

    articles: int
    per_category: dict[str, int]
    glossary_terms: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        """True when no problems were found."""
        return not self.problems


def validate_knowledge_base(
    root: Path = KB_ROOT,
    *,
    min_articles: int = MIN_ARTICLES,
    min_glossary_terms: int = MIN_GLOSSARY_TERMS,
    require_glossary: bool = True,
) -> ValidationReport:
    """Check every article and the glossary against the content rules.

    Problems are collected in the report, not raised. Covers front matter, word and section counts,
    advice or placeholder wording, duplicate ids and titles, empty categories, and glossary size,
    definitions, and related-term links. ``min_articles=0`` skips the article-count and
    empty-category checks.
    """
    problems: list[str] = []
    articles: list[Article] = []
    for path in article_paths(root):
        try:
            articles.append(parse_article(path))
        except ArticleError as exc:
            problems.append(str(exc))
    for article in articles:
        problems.extend(_article_problems(article))

    for label, values in (
        ("id", [a.meta.id for a in articles]),
        ("title", [a.meta.title.lower() for a in articles]),
    ):
        for value, count in Counter(values).items():
            if count > 1:
                problems.append(f"duplicate {label}: '{value}' ({count} articles)")

    per_category = {c: 0 for c in CATEGORIES}
    for article in articles:
        per_category[article.meta.category] += 1
    if len(articles) < min_articles:
        problems.append(f"only {len(articles)} articles (expected at least {min_articles})")
    if min_articles:
        for category, count in per_category.items():
            if count == 0:
                problems.append(f"category '{category}' has no articles")

    glossary_terms = 0
    if (root / GLOSSARY_FILE).is_file():
        try:
            glossary = load_glossary(root)
            glossary_terms = len(glossary.terms)
            problems.extend(_glossary_problems(glossary, min_glossary_terms))
        except (ValidationError, yaml.YAMLError, TypeError) as exc:
            problems.append(f"glossary: invalid ({exc})")
    elif require_glossary:
        problems.append(f"missing {GLOSSARY_FILE}")

    return ValidationReport(
        articles=len(articles),
        per_category=per_category,
        glossary_terms=glossary_terms,
        problems=problems,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``scripts/validate_kb.py``.

    Prints a summary and any problems (to stderr); returns 1 if there are problems, else 0.
    """
    parser = argparse.ArgumentParser(description="Validate the Finnie knowledge base.")
    parser.add_argument("--root", type=Path, default=KB_ROOT)
    parser.add_argument(
        "--min-articles",
        type=int,
        default=MIN_ARTICLES,
        help="0 skips the article-count and empty-category checks",
    )
    parser.add_argument("--no-glossary", action="store_true", help="skip glossary checks")
    args = parser.parse_args(argv)

    report = validate_knowledge_base(
        args.root, min_articles=args.min_articles, require_glossary=not args.no_glossary
    )
    print(f"Articles: {report.articles}  Glossary terms: {report.glossary_terms}")
    for category, count in report.per_category.items():
        print(f"  {category:<26} {count}")
    if report.ok:
        print("Knowledge base OK")
        return 0
    print(f"\n{len(report.problems)} problem(s):", file=sys.stderr)
    for problem in report.problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1
