# Demo Video Script (about 8 minutes)

The problem statement asks for a 5–10 minute video. It must show:
- multi-turn conversations with different agents
- portfolio analysis
- market data integration
- goal planning
- how the agents work together

This script covers all of them, plus guardrails, the MCP server, and saved data. Each prompt below is typed exactly as written.

## Before recording

- Start the app: `python -m src.web_app` (or `docker compose up`) and open http://localhost:8501 in a **private window**, so it starts at onboarding.
- Record during U.S. market hours if possible, so quotes show **Live** rather than "Last close".
- Optional, for scene 8: Claude Desktop with Finnie configured ([MCP.md](MCP.md)).
- Close other tabs and notifications. Use a 1440×900 or larger window, at 100% zoom.

## Scenes

| # | Time | Show | Say (in short) |
|---|---|---|---|
| 1 | 0:00–0:40 | **Onboarding.** Click **Not sure? Take a 5-question quiz**, answer it, and click **Get started**. | Finnie adapts to the user's knowledge level and risk tolerance. The quiz sets them for a beginner. |
| 2 | 0:40–2:00 | **Multi-turn chat with context.** Ask *"What is an index fund?"* Point out the progress line ("Consulting the financial concepts specialist…"), the streamed answer, the numbered citations, and the **Sources** list. Then ask the follow-up *"How is that different from an ETF?"* | Answers are grounded in a 112-article knowledge base with sources. The follow-up says "that", and Finnie resolves it from the conversation. That's the LangGraph memory. |
| 3 | 2:00–3:20 | **Portfolio analysis.** Open **Portfolio**, choose **Three-fund beginner**, then click **Load sample** and **Save holdings**. Scroll through the metrics, allocation donut, sector bars, back-test against SPY, and the comparison with a typical moderate mix. | Value, allocation, diversification score, risk level, the fund expense ratio, and past-year volatility, drawdown, beta, and Sharpe. All are computed locally from live prices. |
| 4 | 3:20–4:20 | **Agents working together.** Back in **Chat**, ask *"How diversified is my portfolio, and what does its expense ratio mean for me?"* Point out that two specialists run (portfolio and financial concepts), and that the answer merges both with shared citations. | The router chose two specialists. They ran in parallel, and their answers were merged into one with unified citations. |
| 5 | 4:20–5:20 | **Market data.** Ask *"How is the stock market doing today?"*, then open **Markets**. Show the index cards with the price time and whether the market is open, the sector chart, and a lookup of **AAPL** (trend, moving averages, RSI, news). | Live data from yfinance, with Alpha Vantage as backup. It's cached with freshness labels, and falls back to stale or mock data, clearly labelled, if a provider fails. |
| 6 | 5:20–6:40 | **Goal planning with risk appetite.** In chat, ask *"I want $50,000 for a house down payment in 8 years. I can add $400 a month."* Finnie asks how much of the saved portfolio counts toward this goal; answer *"none of it"*. Then open **Goals**, set the same numbers, and click **Run projection**. Change **Risk level** to *aggressive* and run it again. | A Monte Carlo simulation (10,000 paths) gives the chance of reaching the goal and the monthly amount for an 80% chance, and the assumptions follow the risk tolerance. Finnie asks before counting the portfolio. |
| 7 | 6:40–7:20 | **Other specialists and guardrails.** Ask *"What's the difference between a Roth IRA and a traditional IRA?"* (tax, with this year's IRS limits). Then ask *"Should I buy Tesla stock right now?"* | Six specialists in all. The second question gets education, not advice: the guardrail reframes it, and every answer carries the disclaimer. |
| 8 | 7:20–7:50 | **Optional: MCP in Claude Desktop.** Ask Claude *"Use Finnie to get a VTI quote and project $50k in 10 years with $300 a month."* | The same tools are available to Claude through the MCP server, over stdio or token-protected HTTP. |
| 9 | 7:50–8:10 | **Saved data.** Refresh the page: the conversation, profile, and portfolio are still there. Open the conversation's **⋯** menu to show Rename and Delete. | Data is kept per browser without a login, and **Delete my data** removes it all. |

## Closing line

"Finnie: six LangGraph agents, a sourced knowledge base, live market data, and goal simulations, built to teach beginners and never to give advice. The code, the 1,014 tests, the benchmarks, and Docker setup are in the repository."

## Notes

- If an answer takes longer than about 15 seconds, keep talking about the progress line; the benchmarks put multi-specialist turns at a 12.9 s median.
- After recording, delete the demo data with **Profile → Delete my data**.
