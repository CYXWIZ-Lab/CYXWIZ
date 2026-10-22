"""Reference answers for cyxwiz::StatisticalTests (TOFIX129 D3): scipy for the
distributions and tests, statsmodels for Welch's ANOVA and the linear models.

Run with a Python that has numpy, scipy and statsmodels, for example
    python -m venv statsvenv && statsvenv/Scripts/pip install numpy scipy statsmodels
    statsvenv/Scripts/python generate_statistical_tests_fixtures.py
Writes ../fixtures/statistical_tests_scipy.json. Inputs are stored at full
precision (json round-trips doubles exactly).
"""
import json
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "statistical_tests_scipy.json"


def vec(a):
    return [float(v) for v in np.asarray(a, dtype=float).ravel()]


def rng(seed):
    return np.random.default_rng(seed)


def main():
    cases = []

    def add(op, label, **fields):
        cases.append({"op": op, "label": label, **{k: (vec(v) if isinstance(v, np.ndarray) else v) for k, v in fields.items()}})

    # ---------------------------------------------------------------- distributions
    for t, df in [(0.3, 5), (2.1, 12), (8.55, 323.9), (30.0, 10), (-1.7, 40), (45.0, 300)]:
        add("t_sf", f"t{t}_df{df}", args=[t, df], want=float(stats.t.sf(t, df)))
    for q, df in [(0.975, 10), (0.975, 341), (0.995, 3), (0.025, 25), (0.9999, 7)]:
        add("t_ppf", f"q{q}_df{df}", args=[q, df], want=float(stats.t.ppf(q, df)))
    for f, d1, d2 in [(1.2, 3, 20), (343.6, 2, 339), (5.0, 1, 4), (0.4, 10, 12)]:
        add("f_sf", f"f{f}_{d1}_{d2}", args=[f, d1, d2], want=float(stats.f.sf(f, d1, d2)))
    for x, df in [(3.0, 4), (299.55, 4), (0.5, 1), (60.0, 30), (12.0, 2.5)]:
        add("chi2_sf", f"x{x}_df{df}", args=[x, df], want=float(stats.chi2.sf(x, df)))
    for z in [0.0, 1.96, 8.0, -2.5, 20.0]:
        add("normal_sf", f"z{z}", args=[z], want=float(stats.norm.sf(z)))
    for q, k, df in [(3.5, 3, 20), (2.0, 4, 60), (1.1, 3, 339), (5.0, 5, 12), (6.5, 3, 100)]:
        add("tukey_sf", f"q{q}_k{k}_df{df}", args=[q, k, df], want=float(stats.studentized_range.sf(q, k, df)))

    # ---------------------------------------------------------------- data
    normal = rng(1).normal(50, 8, 60)
    shifted = rng(2).normal(54, 12, 45)
    small = np.array([3.1, 4.7, 2.2, 5.9, 4.4, 3.8])
    paired_b = normal[:40] + rng(3).normal(1.5, 2.0, 40)
    ties = np.round(rng(4).normal(10, 3, 50))
    ties2 = np.round(rng(5).normal(11, 3, 35))
    groups = [rng(6).normal(10, 2, 30), rng(7).normal(11, 2, 25), rng(8).normal(13, 4, 40)]
    groups_ties = [np.round(g) for g in groups]

    for label, x, mu in [("normal_vs_48", normal, 48.0), ("small_vs_4", small, 4.0), ("shifted_vs_54", shifted, 54.0)]:
        r = stats.ttest_1samp(x, mu)
        ci = r.confidence_interval(0.95)
        ci90 = r.confidence_interval(0.90)
        add("one_sample_t", label, x=x, mu=mu, statistic=float(r.statistic), df=float(r.df), p=float(r.pvalue),
            estimate=float(x.mean()), ci=[ci.low, ci.high], ci90=[ci90.low, ci90.high],
            effect=float((x.mean() - mu) / x.std(ddof=1)))

    for label, a, b in [("normal_shifted", normal, shifted), ("small_normal", small, normal[:9])]:
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        for eq in (True, False):
            r = stats.ttest_ind(a, b, equal_var=eq)
            ci = r.confidence_interval(0.95)
            add("two_sample_t", f"{label}_{'student' if eq else 'welch'}", a=a, b=b, equal_var=eq,
                statistic=float(r.statistic), df=float(r.df), p=float(r.pvalue), estimate=float(a.mean() - b.mean()),
                ci=[ci.low, ci.high], effect=float((a.mean() - b.mean()) / sp))

    a, b = normal[:40], paired_b
    r = stats.ttest_rel(a, b)
    ci = r.confidence_interval(0.95)
    d = a - b
    add("paired_t", "normal_40", a=a, b=b, statistic=float(r.statistic), df=float(r.df), p=float(r.pvalue),
        estimate=float(d.mean()), ci=[ci.low, ci.high], effect=float(d.mean() / d.std(ddof=1)))

    for label, gs in [("three_normal", groups), ("three_ties", groups_ties)]:
        f = stats.f_oneway(*gs)
        allv = np.concatenate(gs)
        ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in gs)
        sst = ((allv - allv.mean()) ** 2).sum()
        add("anova", label, groups=[vec(g) for g in gs], welch=False, statistic=float(f.statistic),
            df=float(len(gs) - 1), df2=float(len(allv) - len(gs)), p=float(f.pvalue), effect=float(ssb / sst))
        w = anova_oneway(gs, use_var="unequal", welch_correction=True)
        add("anova", label + "_welch", groups=[vec(g) for g in gs], welch=True, statistic=float(w.statistic),
            df=float(w.df_num), df2=float(w.df_denom), p=float(w.pvalue), effect=float(ssb / sst))
        tk = stats.tukey_hsd(*gs)
        ci = tk.confidence_interval(0.95)
        pairs = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                pairs.append([i, j, float(tk.statistic[i, j]), float(ci.low[i, j]), float(ci.high[i, j]),
                              float(tk.pvalue[i, j])])
        add("tukey", label, groups=[vec(g) for g in gs], pairs=pairs)
        lv = stats.levene(*gs, center="median")
        add("levene", label, groups=[vec(g) for g in gs], statistic=float(lv.statistic), p=float(lv.pvalue))
        kw = stats.kruskal(*gs)
        add("kruskal", label, groups=[vec(g) for g in gs], statistic=float(kw.statistic), p=float(kw.pvalue))

    for label, x, y in [("normal_shifted", normal, shifted), ("ties", ties, ties2), ("small", small, normal[:7])]:
        mw = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=True)
        add("mann_whitney", label, a=x, b=y, statistic=float(mw.statistic), p=float(mw.pvalue))

    for label, table in [("penguins_species_island", [[44, 56, 52], [0, 68, 0], [124, 0, 0]]),
                         ("two_by_three", [[12, 30, 18], [20, 22, 9]]), ("two_by_two", [[15, 25], [30, 10]])]:
        c = stats.chi2_contingency(np.array(table), correction=False)
        n = np.array(table).sum()
        add("chi_square", label, table=table, statistic=float(c.statistic), df=float(c.dof), p=float(c.pvalue),
            effect=float(np.sqrt(c.statistic / (n * (min(np.array(table).shape) - 1)))),
            expected=[vec(r) for r in c.expected_freq], min_expected=float(c.expected_freq.min()))

    # ---------------------------------------------------------------- linear models (statsmodels OLS)
    def ols_case(label, columns, y, names, intercept=True):
        x = sm.add_constant(np.column_stack(columns)) if intercept else np.column_stack(columns)
        m = sm.OLS(y, x).fit()
        ci = m.conf_int(0.05)
        add("ols", label, columns=[vec(c) for c in columns], y=y, names=names, intercept=intercept,
            coef=vec(m.params), se=vec(m.bse),
            t=vec(m.tvalues), p=vec(m.pvalues), ci_low=vec(ci[:, 0]), ci_high=vec(ci[:, 1]), r2=float(m.rsquared),
            adj_r2=float(m.rsquared_adj), f=float(m.fvalue), f_p=float(m.f_pvalue), aic=float(m.aic),
            rse=float(np.sqrt(m.scale)), rmse=float(np.sqrt(m.ssr / m.nobs)), mae=float(np.abs(m.resid).mean()),
            residuals=vec(m.resid))

    g = rng(9)
    x1 = g.normal(200, 14, 80)
    x2 = g.normal(44, 5, 80)
    x3 = g.normal(17, 2, 80)
    y = -6000 + 50 * x1 + 4 * x2 + 20 * x3 + g.normal(0, 390, 80)
    ols_case("one_predictor", [x1], y, ["x1"])
    ols_case("three_predictors", [x1, x2, x3], y, ["x1", "x2", "x3"])
    ols_case("no_intercept", [x1, x2], y, ["x1", "x2"], intercept=False)
    # Polynomials in raw powers (statsmodels on the raw basis), x centred near 0 so statsmodels itself stays exact.
    xp = g.uniform(-2, 3, 60)
    yp = 1.5 - 0.7 * xp + 0.4 * xp ** 2 + 0.05 * xp ** 3 + g.normal(0, 0.3, 60)
    for deg in (2, 3):
        x = sm.add_constant(np.column_stack([xp ** k for k in range(1, deg + 1)]))
        m = sm.OLS(yp, x).fit()
        ci = m.conf_int(0.05)
        add("poly", f"degree{deg}", x=xp, y=yp, degree=deg, coef=vec(m.params), se=vec(m.bse), p=vec(m.pvalues),
            ci_low=vec(ci[:, 0]), ci_high=vec(ci[:, 1]), r2=float(m.rsquared), aic=float(m.aic))
    # A penguins-like scale (x ~ 200): the raw basis is ill-conditioned for statsmodels at higher degree,
    # so the reference is statsmodels on the centred, scaled basis mapped back (the same model).
    xr = g.normal(200, 14, 120)
    yr = 16770 - 173.2 * xr + 0.548 * xr ** 2 + g.normal(0, 380, 120)
    z = (xr - xr.mean()) / xr.std(ddof=1)
    m = sm.OLS(yr, sm.add_constant(np.column_stack([z, z ** 2]))).fit()
    mu, s = xr.mean(), xr.std(ddof=1)
    T = np.array([[1, -mu / s, mu * mu / s ** 2], [0, 1 / s, -2 * mu / s ** 2], [0, 0, 1 / s ** 2]])
    coef = T @ m.params
    cov = T @ m.cov_params() @ T.T
    se = np.sqrt(np.diag(cov))
    tc = stats.t.ppf(0.975, m.df_resid)
    add("poly", "degree2_scale200", x=xr, y=yr, degree=2, coef=vec(coef), se=vec(se),
        p=vec(2 * stats.t.sf(np.abs(coef / se), m.df_resid)), ci_low=vec(coef - tc * se), ci_high=vec(coef + tc * se),
        r2=float(m.rsquared), aic=float(m.aic))

    # ---------------------------------------------------------------- refusals (text the error must contain)
    add("refuse_one_sample", "constant", x=[5.0, 5.0, 5.0, 5.0], message="same")
    add("refuse_ols", "constant_column", columns=[vec(x1[:20]), [3.0] * 20], y=vec(y[:20]), names=["x1", "flat"],
        message="flat has the same value")
    add("refuse_ols", "collinear", columns=[vec(x1[:20]), vec(2 * x1[:20] + 1)], y=vec(y[:20]), names=["a", "b"],
        message="b is a combination")
    add("refuse_poly", "too_few_rows", x=[1.0, 2.0, 3.0, 4.0], y=[1.0, 4.0, 9.0, 16.0], degree=4, message="needs at least 6 rows")

    OUT.write_text(json.dumps({"schema_version": 1, "scipy_version": scipy.__version__,
                               "statsmodels_version": statsmodels.__version__, "cases": cases}, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
