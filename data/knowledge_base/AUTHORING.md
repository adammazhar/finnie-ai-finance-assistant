# Knowledge Base Authoring Guide

Rules for every article in `data/knowledge_base/`. The validator (`python scripts/validate_kb.py`) and link checker (`python scripts/check_kb_links.py`) enforce most of them. The knowledge base is not committed until both pass and the owner has spot-checked a sample.

## Audience and voice

- Readers are **beginners**. Use plain language, short paragraphs, and define any jargon the first time it appears.
- Write about **education, not advice**. Explain how things work and the trade-offs. Never tell the reader what to buy, sell, or do with their money. Don't write "you should buy/sell", "guaranteed returns", or "can't lose".
- Be conservative and accurate. Use only widely accepted principles. Avoid current market levels, prices, or rates that will go stale. When describing history, use hedged, well-established figures such as "stocks have historically returned more than bonds over long periods".

## Originality

All text is written fresh for Finnie. Do not copy or closely paraphrase any source. Sources are listed so readers can **verify** facts, not as text to reuse.

## Format

File: `data/knowledge_base/<category>/<id>.md`, where `id` is `<category>-NNN`.

```markdown
---
id: stocks-004
title: The Price-to-Earnings (P/E) Ratio
category: stocks
difficulty: beginner            # beginner | intermediate | advanced
tags: [valuation, ratios]       # 1-8 lowercase tags
sources:
  - name: "Investor.gov - Price-to-earnings ratio"
    url: https://www.investor.gov/...
last_reviewed: 2026-09-30
---
Opening paragraph (no "# Title" heading; the title lives in front matter).

## First section
...

## Key takeaways
- ...
```

- **Length:** 450-800 words of body text. The validator accepts 400-900.
- **Structure:** at least 2 `##` sections. End with `## Key takeaways`, 3-5 bullets.
- **Sources:** 1-3 per article (occasionally 4 when a claim needs its own primary source). Use **primary sources**: investor.gov, sec.gov, irs.gov, federalreserve.gov, federalreservehistory.org, treasurydirect.gov, consumerfinance.gov, fdic.gov, ssa.gov, dol.gov, bls.gov, bea.gov, nber.org, fund and index providers. Investopedia is not cited. Each source must be a real page you have opened and confirmed covers the topic. The full allow-list is in `src/rag/link_check.py`.
- **Fetching sources:** identify honestly. Use the project User-Agent (`FinnieLinkChecker/1.0`), never a browser User-Agent, and never work around a site that refuses automated access. If a site blocks checks, choose another primary source, or add the URL to `link_allowlist.yaml` with a written reason. Space requests to the same site at least 2 seconds apart.

## Tax figures (tax year 2026)

Use only these figures, which were verified against IRS.gov on 2026-09-30. Label them as 2026 figures and note that limits change each year.

| Item | 2026 |
|---|---|
| 401(k)/403(b) employee contribution limit | $24,500 |
| 401(k) catch-up, age 50+ | $8,000 |
| 401(k) higher catch-up, ages 60-63 | $11,250 |
| IRA contribution limit (traditional + Roth combined) | $7,500 |
| IRA catch-up, age 50+ | $1,100 |
| Roth IRA phase-out, single (MAGI) | $153,000-$168,000 |
| Roth IRA phase-out, married filing jointly | $242,000-$252,000 |
| HSA limit, self-only / family | $4,400 / $8,750 |
| HSA catch-up, age 55+ | $1,000 |
| Standard deduction, single / married filing jointly | $16,100 / $32,200 |
| Long-term capital gains 0% rate up to taxable income of (single / MFJ) | $49,450 / $98,900 |
| 15% rate up to (single / MFJ) | $545,500 / $613,700 |
| Net capital loss deductible against ordinary income | $3,000 per year |

These are also verified (2026-09-30): the annual gift tax exclusion is $19,000 per recipient, the wash sale window is 30 days before or after the sale, and the net investment income tax rate is 3.8%.

Don't state other dollar thresholds. Describe those qualitatively and point readers to IRS.gov.

**Holding period:** long-term means held **more than one year**, meaning sold after the one-year anniversary of the purchase date. Don't describe it as "365 days".

**RMD age (SECURE 2.0):** 73 for people born 1951-1959 and 75 for people born in 1960 or later. The first RMD is due by April 1 of the year after you reach your RMD age. Don't cite IRS pages or charts that still show age 72 as the current rule.
