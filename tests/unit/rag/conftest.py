from pathlib import Path

import pytest
import yaml

from src.core.config import RAGConfig
from tests.fakes.embeddings import FakeEmbeddings


def article_text(
    article_id: str, category: str, title: str, sections: dict[str, str], intro: str = ""
) -> str:
    meta = {
        "id": article_id,
        "title": title,
        "category": category,
        "difficulty": "beginner",
        "tags": ["test"],
        "sources": [{"name": f"{title} source", "url": f"https://www.investor.gov/{article_id}"}],
        "last_reviewed": "2026-09-30",
    }
    body = intro + "\n\n" + "\n\n".join(f"## {h}\n\n{t}" for h, t in sections.items())
    return "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---\n" + body.strip() + "\n"


def write_article(
    root: Path,
    article_id: str,
    category: str,
    title: str,
    sections: dict[str, str],
    intro: str = "",
) -> Path:
    path = root / category / f"{article_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(article_text(article_id, category, title, sections, intro), encoding="utf-8")
    return path


@pytest.fixture
def kb(tmp_path) -> Path:
    """A tiny knowledge base with clearly separable topics."""
    root = tmp_path / "kb"
    write_article(
        root,
        "bonds_fixed_income-001",
        "bonds_fixed_income",
        "What Is a Bond?",
        {
            "Bond basics": "A bond is a loan to a government or company that pays "
            "interest coupons until maturity.",
            "Bond prices and rates": "When interest rates rise, existing bond prices "
            "fall. Duration measures rate sensitivity.",
        },
        intro="Bonds are loans investors make to issuers.",
    )
    write_article(
        root,
        "stocks-001",
        "stocks",
        "What Is a Stock?",
        {
            "Ownership": "A stock is a share of ownership in a company.",
            "Dividends": "Some companies pay dividends to stockholders from profits.",
        },
    )
    write_article(
        root,
        "taxes-001",
        "taxes",
        "Capital Gains Tax",
        {
            "Holding period": "A capital gain is long-term when shares are held more "
            "than one year before the sale.",
            "Wash sale": "A wash sale disallows a loss when you buy the same security "
            "within thirty days.",
        },
    )
    glossary = {
        "sources": [{"name": "Investor.gov glossary", "url": "https://www.investor.gov/glossary"}],
        "last_reviewed": "2026-09-30",
        "terms": [
            {
                "term": "Coupon",
                "definition": "The interest payment a bond makes to its holder.",
                "related": ["Bond"],
            },
            {"term": "Bond", "definition": "A loan an investor makes to a borrower for interest."},
        ],
    }
    (root / "glossary.yaml").write_text(yaml.safe_dump(glossary), encoding="utf-8")
    return root


@pytest.fixture
def rag_config() -> RAGConfig:
    return RAGConfig(
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
        fetch_k=10,
        score_threshold=0.2,
        mmr_lambda=0.7,
        max_chunks_per_article=2,
    )


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
