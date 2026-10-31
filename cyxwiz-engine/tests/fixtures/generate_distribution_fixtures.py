"""Reference values for test_data_analyzer_distributions (scipy is the oracle).

Prints C++ initialisers; paste them into the test when cases change.
    python generate_distribution_fixtures.py
"""
import math

import numpy as np
from scipy import stats

NORMAL_CDF = [(-1.5, 0.0, 1.0), (0.3, 0.0, 1.0), (2.0, 1.0, 0.5), (-8.0, 0.0, 1.0)]

# Distribution fits on a fixed right-skewed positive sample (log-normal draw).
FIT_SAMPLE = [float(v) for v in np.round(np.random.default_rng(7).lognormal(3.5, 0.9, 40), 3)]


def show(name, rows):
    print("const Case %s[] = {" % name)
    for args, value in rows:
        print("    {{%s}, %.17g}," % (", ".join(repr(float(a)) for a in args), value))
    print("};")


show("kNormalCdf", [(c, stats.norm.cdf(c[0], c[1], c[2])) for c in NORMAL_CDF])

# Fits: maximum-likelihood parameters, AIC / BIC, the KS statistic, and the KS
# p-value from the asymptotic Kolmogorov distribution with Stephens' correction.
x = np.array(FIT_SAMPLE)
n = len(x)


def fit_row(name, dist, args, k, params):
    ll = dist.logpdf(x, *args).sum()
    d = stats.kstest(x, dist.cdf, args).statistic
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    print("    {%s, {%s}, %.17g, %.17g, %.17g, %.17g}," % (
        name, ", ".join("%.17g" % p for p in params), 2 * k - 2 * ll, k * math.log(n) - 2 * ll, d,
        stats.kstwobign.sf(lam)))


print("const double kFitSample[] = {%s};" % ", ".join(repr(v) for v in FIT_SAMPLE))
print("// {type, parameters, aic, bic, ks statistic, ks p-value}")
mu, sd = stats.norm.fit(x)
fit_row("Normal", stats.norm, (mu, sd), 2, (mu, sd))
fit_row("Uniform", stats.uniform, (x.min(), x.max() - x.min()), 2, (x.min(), x.max()))
fit_row("Exponential", stats.expon, (0, x.mean()), 1, (1 / x.mean(),))
s_, _, scale = stats.lognorm.fit(x, floc=0)
fit_row("LogNormal", stats.lognorm, (s_, 0, scale), 2, (math.log(scale), s_))

