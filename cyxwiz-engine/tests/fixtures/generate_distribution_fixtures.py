"""Reference values for test_data_analyzer_distributions (scipy is the oracle).

Prints C++ initialisers; paste them into the test when cases change.
    python generate_distribution_fixtures.py
"""
import numpy as np
from scipy import stats

T_CDF = [(0.5, 3), (1.96, 10), (2.5, 30), (-1.2, 5), (4.0, 2), (0.0, 7), (-3.5, 120)]
F_CDF = [(2.5, 3, 20), (0.8, 5, 10), (10.0, 1, 246), (1.0, 12, 4), (4.2, 2, 60)]
CHI2_CDF = [(3.84, 1), (10.0, 4), (30.0, 10), (250.0, 200), (0.5, 3), (180.0, 150)]
T_PPF = [(0.05, 1), (0.05, 5), (0.01, 30), (0.05, 247), (0.10, 12)]

# One-sample t-test on a fixed sample (H0 mu = 4).
SAMPLE = [4.2, 5.1, 3.9, 6.3, 4.8, 5.5, 4.1, 5.9, 4.4, 5.0, 3.7, 6.1]
MU0 = 4.0


def show(name, rows):
    print("const Case %s[] = {" % name)
    for args, value in rows:
        print("    {{%s}, %.17g}," % (", ".join(repr(float(a)) for a in args), value))
    print("};")


show("kTCdf", [(c, stats.t.cdf(c[0], c[1])) for c in T_CDF])
show("kFCdf", [(c, stats.f.cdf(c[0], c[1], c[2])) for c in F_CDF])
show("kChi2Cdf", [(c, stats.chi2.cdf(c[0], c[1])) for c in CHI2_CDF])
show("kTInv", [(c, stats.t.ppf(1 - c[0] / 2, c[1])) for c in T_PPF])

res = stats.ttest_1samp(SAMPLE, MU0)
ci = res.confidence_interval(0.95)
print("// ttest_1samp: t, p, ci_low, ci_high")
print("%.17g, %.17g, %.17g, %.17g" % (res.statistic, res.pvalue, ci.low, ci.high))
