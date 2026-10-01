---
id: financial_planning_goals-005
title: Understanding Monte Carlo Projections
category: financial_planning_goals
difficulty: intermediate
tags: [monte-carlo, projections, probability, planning, uncertainty]
sources:
  - name: "Fidelity - Retirement Income Planner Detailed Methodology (Monte Carlo simulation)"
    url: https://www.fidelity.com/planning/retirement/pdf/rip_methodology.pdf
  - name: "Investor.gov - What is Risk?"
    url: https://www.investor.gov/introduction-investing/investing-basics/what-risk
last_reviewed: 2026-09-30
---
Many planning tools, including Finnie's goal projections, show results from a **Monte Carlo simulation**. Instead of a single prediction, you see a probability and a range of outcomes. Knowing how to read these results, and what they can't tell you, helps you use them wisely.

## Why not just use an average return?

A simple calculator might assume your investments earn exactly 6% every year. Real markets never behave that way. Returns bounce around: some years are up sharply, some are down. The *order* of returns matters too. A big drop shortly before you need the money can do more damage than the same drop decades earlier.

A single average hides all of that uncertainty. Monte Carlo methods were designed to show it.

## How a Monte Carlo simulation works

The method is named after the famous casino in Monaco, because it relies on randomness. In general, a planning tool:

1. **Sets assumptions:** a starting balance, contributions, a time horizon, and an expected average return and volatility (how much returns typically vary) for a mix of investments.
2. **Generates a random path:** it draws a plausible return for each period, consistent with those assumptions, and tracks the balance over time.
3. **Repeats this many times,** each path with a different random sequence of good and bad periods.
4. **Summarizes the results:** how often the goal was met and how widely the outcomes spread.

## How Finnie's goal projections work

Finnie's version follows that recipe with these specific choices:

- **Monthly steps.** Each month, your contribution is added and then the balance grows or shrinks by that month's random return.
- **Fat-tailed returns.** Monthly returns are drawn from a Student-t distribution rather than a normal "bell curve." This produces more extreme months, both good and bad, than a bell curve would. Real markets have shown such months more often than a bell curve predicts.
- **10,000 paths by default,** each with its own random sequence of months.
- **Saving toward a target.** The projection models building up savings toward a goal amount. It does not model withdrawals, so it isn't a test of whether retirement income will last.
- **Assumptions from your risk profile.** The expected return and volatility come from simplified, illustrative long-run assumptions for a conservative, moderate, or aggressive mix.
- **Optional inflation adjustment.** You can view the target and results in today's dollars, which lowers future values to account for assumed inflation.

## Reading probability of success

The **probability of success** is the share of simulated paths that reached the goal. In Finnie, that means the balance reached your target amount by the end of your time horizon.

Here is a hypothetical example. If 8,500 of 10,000 paths reach the target, the probability of success is 85%. Under the model's assumptions, the plan worked in 85% of the imagined futures. It does not mean you will end up with 85% of your target.

Finnie can also estimate the monthly contribution that would reach a chosen probability, such as 80%, under the same assumptions.

## Reading percentile bands

Finnie shows **percentile bands** for each year of the projection, from the 10th to the 90th percentile (P10 to P90):

- **P10:** a rough "bad luck" case. 90% of paths did better.
- **P50 (median):** the middle outcome. Half the paths did better, half worse.
- **P90:** a "good luck" case. Only 10% of paths did better.

The band between P10 and P90 is usually wide, and that width is the point: it shows how uncertain the future is. Planning around the median while checking that the P10 outcome would still be workable is one common way to use these ranges.

## Assumptions are not forecasts

The most important limitation: **a Monte Carlo simulation is only as good as its assumptions.** The results are hypothetical and are not predictions or guarantees.

- Expected returns and volatility are estimates, and the future may differ from the past.
- Finnie treats each month's return as independent of the last, which real markets don't strictly follow.
- Fees, taxes, and changes to your contributions over time aren't modeled.
- No model can anticipate life changes such as a job loss, an inheritance, or a health event.

Small changes to the assumptions, such as lowering the expected return by one percentage point, can change the probability noticeably. Trying a few scenarios shows how sensitive a plan is.

## Key takeaways
- Monte Carlo simulations run thousands of random market paths instead of one average.
- Finnie uses monthly steps, fat-tailed returns, and 10,000 paths to project savings toward a target.
- Probability of success is the share of paths that reached the target under the model's assumptions.
- P10-P90 bands show bad-luck to good-luck outcomes, with the median in the middle.
- Results are hypothetical and depend heavily on assumptions; they are not forecasts or promises.
