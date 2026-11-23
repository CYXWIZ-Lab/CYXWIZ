# Portfolio weights (Portfolio Weights node)

**Graph:** `portfolio.cyxgraph`. **Docs:** [Optimization and calculus](../../../docs/usage/optimization_calculus.md).

`returns.csv` holds 120 months (2016-01 to 2025-12) of returns of five
assets: `equity`, `growth`, `credit`, `bonds` and `gold`. The data is made up
for this example (seeded draws with a fixed correlation), so no download is
needed. The graph finds two mixes of the five, both without short positions:

```
Returns -+-> Least risky mix (Portfolio Weights, no target)
         +-> Earning 0.5 % a month (Portfolio Weights, target_return 0.005)
```

Each node works out the means and the covariance of the returns and solves
the quadratic program: minimize ½ w′C w with the weights summing to 1, and
for the second node a mean return of at least 0.005 a month.

## Run it

1. Open the graph (File > Open, or `cyxwiz-engine <project> examples/cyxgraph/portfolio/portfolio.cyxgraph`).
2. Press **Execute Pipeline** (the green arrow in the Studio toolbar).
3. In Data Studio, click **Least risky mix** under Node results and run the
   query. Then do the same for **Earning 0.5 % a month**.

Expected result (`weight`; each node also gives `mean_return`, `sd` and
`risk_share`):

| asset | Least risky mix | Earning 0.5 % a month |
| --- | --- | --- |
| equity | 0.0575 | 0.0847 |
| growth | 0.0060 | 0.3125 |
| credit | 0 | 0.2029 |
| bonds | 0.9365 | 0.4000 |
| gold | 0 | 0 |

The least risky mix moves 1.09 % a month (its sd). Earning 0.5 % a month
takes growth and credit in and halves bonds, at 2.65 % a month. scipy's
SLSQP gives the same weights on this file (see the Python example below).

To fit on part of the months only, put a Data Split before the nodes: they
then read its Train rows.

## The same from the window

Tools > Optimization and Calculus > **Quadratic**, **From a returns table**:
pick Returns (the five assets are ticked; `month` is text, so not offered), and leave
**Target return a period** empty or type 0.005. **Keep as a node: Portfolio weights** adds the
node this graph has.

The same portfolios from Python, checked against scipy, together with a
production plan, an integral and two optimizer paths:
`examples/python/optimization_calculus.py`.
