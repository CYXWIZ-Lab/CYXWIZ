#include "cyxwiz/statistical_tests.h"

#include "arrayfire_host_values.h"
#include "arrayfire_indexed.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace cyxwiz {

namespace {

constexpr double kTiny = 1e-300;
constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------- special functions

// Continued fraction of the regularized incomplete beta (modified Lentz).
double BetaFraction(double a, double b, double x) {
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kTiny) d = kTiny;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 1000; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double step = d * c;
        h *= step;
        if (std::fabs(step - 1.0) < 1e-16) break;
    }
    return h;
}

// I_x(a, b), from the side that does not subtract from 1 when it is small.
// The tails below pass the transformed argument, so they need no upper form.
double BetaFront(double a, double b, double x) {
    return std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x));
}
double IncBetaLower(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (x < (a + 1.0) / (a + b + 2.0)) return BetaFront(a, b, x) * BetaFraction(a, b, x) / a;
    return 1.0 - BetaFront(b, a, 1.0 - x) * BetaFraction(b, a, 1.0 - x) / b;
}

// Regularized upper incomplete gamma Q(a, z), direct in both regions.
double GammaQ(double a, double z) {
    if (z <= 0.0) return 1.0;
    const double front = std::exp(-z + a * std::log(z) - std::lgamma(a));
    if (z < a + 1.0) {
        double term = 1.0 / a, sum = term;
        for (int n = 1; n <= 2000; ++n) {
            term *= z / (a + n);
            sum += term;
            if (std::fabs(term) < std::fabs(sum) * 1e-17) break;
        }
        return (std::max)(0.0, 1.0 - sum * front);
    }
    double b = z + 1.0 - a, c = 1.0 / kTiny, d = 1.0 / b, h = d;
    for (int i = 1; i <= 2000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = b + an / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double step = d * c;
        h *= step;
        if (std::fabs(step - 1.0) < 1e-17) break;
    }
    return (std::min)(1.0, front * h);
}

double PNorm(double x, double mu) { return StatisticalTests::NormalCdf(x - mu); }

// R's wprob / ptukey (nmath/ptukey.c, Copenhaver and Holland 1988): the
// probability that the studentized range of cc means is below w.
double WProb(double w, double rr, double cc) {
    static const double xleg[6] = {0.981560634246719250690549090149, 0.904117256370474856678465866119,
                                   0.769902674194304687036893833213, 0.587317954286617447296702418941,
                                   0.367831498998180193752691536644, 0.125233408511468915472441369464};
    static const double aleg[6] = {0.047175336386511827194615961485, 0.106939325995318430960254718194,
                                   0.160078328543346226334652529543, 0.203167426723065921749064455810,
                                   0.233492536538354808760849898925, 0.249147045813402785000562436043};
    const double qsqz = w * 0.5;
    if (qsqz >= 8.0) return 1.0;
    double pr_w = 2.0 * PNorm(qsqz, 0.0) - 1.0;
    pr_w = pr_w >= std::exp(-50.0 / cc) ? std::pow(pr_w, cc) : 0.0;
    const double wincr = w > 3.0 ? 2.0 : 3.0;
    double blb = qsqz;
    const double binc = (8.0 - qsqz) / wincr;
    double bub = blb + binc, einsum = 0.0;
    const double cc1 = cc - 1.0;
    for (double wi = 1; wi <= wincr; ++wi) {
        double elsum = 0.0;
        const double a = 0.5 * (bub + blb), b = 0.5 * (bub - blb);
        for (int jj = 1; jj <= 12; ++jj) {
            int j;
            double xx;
            if (6 < jj) {
                j = 12 - jj + 1;
                xx = xleg[j - 1];
            } else {
                j = jj;
                xx = -xleg[j - 1];
            }
            const double ac = a + b * xx, qexpo = ac * ac;
            if (qexpo > 60.0) break;
            const double pplus = 2.0 * PNorm(ac, 0.0), pminus = 2.0 * PNorm(ac, w);
            double rinsum = pplus * 0.5 - pminus * 0.5;
            if (rinsum >= std::exp(-30.0 / cc1)) {
                rinsum = aleg[j - 1] * std::exp(-0.5 * qexpo) * std::pow(rinsum, cc1);
                elsum += rinsum;
            }
        }
        elsum *= 2.0 * b * cc / std::sqrt(2.0 * kPi);
        einsum += elsum;
        blb = bub;
        bub += binc;
    }
    pr_w += einsum;
    if (pr_w <= std::exp(-30.0 / rr)) return 0.0;
    pr_w = std::pow(pr_w, rr);
    return (std::min)(1.0, pr_w);
}

double PTukey(double q, double cc, double df) {
    static const double xlegq[8] = {0.989400934991649932596154173450, 0.944575023073232576077988415535,
                                    0.865631202387831743880467897712, 0.755404408355003033895101194847,
                                    0.617876244402643748446671764049, 0.458016777657227386342419442984,
                                    0.281603550779258913230460501460, 0.950125098376374401853193354250e-1};
    static const double alegq[8] = {0.271524594117540948517805724560e-1, 0.622535239386478928628438369944e-1,
                                    0.951585116824927848099251076022e-1, 0.124628971255533872052476282192,
                                    0.149595988816576732081501730547, 0.169156519395002538189312079030,
                                    0.182603415044923588866763667969, 0.189450610455068496285396723208};
    if (q <= 0.0) return 0.0;
    if (!std::isfinite(q)) return 1.0;
    if (df > 25000.0) return WProb(q, 1.0, cc);
    const double f2 = df * 0.5;
    double f2lf = f2 * std::log(df) - df * std::log(2.0) - std::lgamma(f2);
    const double f21 = f2 - 1.0, ff4 = df * 0.25;
    const double ulen = df <= 100.0 ? 1.0 : df <= 800.0 ? 0.5 : df <= 5000.0 ? 0.25 : 0.125;
    f2lf += std::log(ulen);
    double ans = 0.0;
    for (int i = 1; i <= 50; ++i) {
        double otsum = 0.0;
        const double twa1 = (2 * i - 1) * ulen;
        for (int jj = 1; jj <= 16; ++jj) {
            int j;
            double t1;
            if (8 < jj) {
                j = jj - 8 - 1;
                t1 = f2lf + f21 * std::log(twa1 + xlegq[j] * ulen) - (xlegq[j] * ulen + twa1) * ff4;
            } else {
                j = jj - 1;
                t1 = f2lf + f21 * std::log(twa1 - xlegq[j] * ulen) + (xlegq[j] * ulen - twa1) * ff4;
            }
            if (t1 >= -30.0) {
                const double qsqz = 8 < jj ? q * std::sqrt((xlegq[j] * ulen + twa1) * 0.5)
                                           : q * std::sqrt((-(xlegq[j] * ulen) + twa1) * 0.5);
                otsum += WProb(qsqz, 1.0, cc) * alegq[j] * std::exp(t1);
            }
        }
        if (i * ulen >= 1.0 && otsum <= 1e-14) break;
        ans += otsum;
    }
    return (std::min)(1.0, ans);
}

// ---------------------------------------------------------------- device helpers

// float64 where the device has it, float32 otherwise (Iris Xe).
af::dtype ComputeType() { return DeviceHasDouble() ? f64 : f32; }
af::array Device(const std::vector<double>& v) { return HostValuesToDevice(v); }

struct Moments {
    double n = 0, mean = NAN, var = NAN;  // var: sample (ddof 1)
};

Moments MomentsOf(const af::array& x) {
    Moments m;
    m.n = static_cast<double>(x.elements());
    if (m.n == 0) return m;
    m.mean = af::mean<double>(x);
    if (m.n > 1) {
        const af::array d = x - m.mean;
        m.var = af::sum<double>(d * d) / (m.n - 1.0);
    }
    return m;
}

TestResult Fail(const std::string& why) {
    TestResult r;
    r.error = why;
    return r;
}

double TwoSidedT(double t, double df) { return IncBetaLower(df / 2.0, 0.5, df / (df + t * t)); }

double CriticalT(double confidence, double df) {
    return StatisticalTests::StudentTPpf(1.0 - (1.0 - confidence) / 2.0, df);
}

// Average ranks (1 .. N, ties averaged) of `values` and the tie sum
// sum(t^3 - t) over tied runs, on the device.
af::array AverageRanks(const af::array& values, double& tie_sum) {
    const dim_t n = values.elements();
    af::array sorted, order;
    SortWithIndex(sorted, order, values);  // sort-by-key: the oneAPI-safe argsort
    af::array starts = af::constant(1, n, s32);
    if (n > 1) {
        const double last = static_cast<double>(n - 1);
        starts(af::seq(1, last)) = (sorted(af::seq(1, last)) != sorted(af::seq(0, last - 1))).as(s32);
    }
    const af::array run = af::accum(starts) - 1;  // run id per sorted value
    const af::array position = af::range(af::dim4(n), 0, ComputeType()) + 1.0;
    af::array keys, sums, keys2, counts;
    af::sumByKey(keys, sums, run, position, 0);
    af::countByKey(keys2, counts, run, af::constant(1, n, ComputeType()), 0);
    const af::array c = counts.as(ComputeType());
    tie_sum = af::sum<double>(c * c * c - c);
    const af::array mean_rank = sums / c;
    af::array ranks = af::constant(0.0, n, ComputeType());
    ranks(order) = mean_rank(run);
    return ranks;
}


}  // namespace

// ================================================================ distributions

double StatisticalTests::NormalCdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
double StatisticalTests::NormalSf(double z) { return 0.5 * std::erfc(z / std::sqrt(2.0)); }

double StatisticalTests::StudentTSf(double t, double df) {
    if (!(df > 0.0)) return NAN;
    const double half_tail = 0.5 * IncBetaLower(df / 2.0, 0.5, df / (df + t * t));
    return t >= 0.0 ? half_tail : 1.0 - half_tail;
}

double StatisticalTests::StudentTCdf(double t, double df) { return StudentTSf(-t, df); }

double StatisticalTests::StudentTPpf(double q, double df) {
    if (!(q > 0.0 && q < 1.0) || !(df > 0.0)) return NAN;
    if (q == 0.5) return 0.0;
    const bool upper = q > 0.5;
    const double tail = upper ? 1.0 - q : q;  // P(T > t) = tail for t > 0
    double lo = 0.0, hi = 1.0;
    while (StudentTSf(hi, df) > tail && hi < 1e300) hi *= 2.0;
    for (int i = 0; i < 400 && hi - lo > 1e-15 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        (StudentTSf(mid, df) > tail ? lo : hi) = mid;
    }
    const double t = 0.5 * (lo + hi);
    return upper ? t : -t;
}

double StatisticalTests::FSf(double f, double df1, double df2) {
    if (!(df1 > 0.0 && df2 > 0.0)) return NAN;
    if (f <= 0.0) return 1.0;
    return IncBetaLower(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f));
}

double StatisticalTests::ChiSquareSf(double x, double df) {
    if (!(df > 0.0)) return NAN;
    return GammaQ(df / 2.0, x / 2.0);
}

double StatisticalTests::StudentizedRangeSf(double q, double k, double df) {
    if (!(k >= 2.0 && df >= 2.0)) return NAN;
    return (std::max)(0.0, 1.0 - PTukey(q, k, df));
}

// ================================================================ tests

TestResult StatisticalTests::OneSampleT(const std::vector<double>& x, double mu0, double confidence) {
    if (x.size() < 2) return Fail("needs at least 2 values");
    try {
        const Moments m = MomentsOf(Device(x));
        const double sd = std::sqrt(m.var);
        if (!(sd > 0.0)) return Fail("every value is the same, so the test cannot be computed");
        TestResult r;
        const double se = sd / std::sqrt(m.n);
        r.df = m.n - 1.0;
        r.statistic = (m.mean - mu0) / se;
        r.p = TwoSidedT(r.statistic, r.df);
        r.estimate = m.mean;
        const double tc = CriticalT(confidence, r.df);
        r.ci_low = m.mean - tc * se;
        r.ci_high = m.mean + tc * se;
        r.effect = (m.mean - mu0) / sd;
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::OneSampleT", e));
    }
}

TestResult StatisticalTests::TwoSampleT(const std::vector<double>& a, const std::vector<double>& b, bool equal_var,
                                        double confidence) {
    if (a.size() < 2 || b.size() < 2) return Fail("each group needs at least 2 values");
    try {
        const Moments ma = MomentsOf(Device(a)), mb = MomentsOf(Device(b));
        const double pooled = ((ma.n - 1.0) * ma.var + (mb.n - 1.0) * mb.var) / (ma.n + mb.n - 2.0);
        if (!(pooled > 0.0)) return Fail("every value is the same in both groups, so the test cannot be computed");
        TestResult r;
        double se;
        if (equal_var) {
            se = std::sqrt(pooled * (1.0 / ma.n + 1.0 / mb.n));
            r.df = ma.n + mb.n - 2.0;
        } else {
            const double va = ma.var / ma.n, vb = mb.var / mb.n;
            se = std::sqrt(va + vb);
            r.df = (va + vb) * (va + vb) / (va * va / (ma.n - 1.0) + vb * vb / (mb.n - 1.0));
        }
        r.estimate = ma.mean - mb.mean;
        r.statistic = r.estimate / se;
        r.p = TwoSidedT(r.statistic, r.df);
        const double tc = CriticalT(confidence, r.df);
        r.ci_low = r.estimate - tc * se;
        r.ci_high = r.estimate + tc * se;
        r.effect = r.estimate / std::sqrt(pooled);
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::TwoSampleT", e));
    }
}

TestResult StatisticalTests::PairedT(const std::vector<double>& a, const std::vector<double>& b, double confidence) {
    if (a.size() != b.size()) return Fail("the two columns need the same rows");
    if (a.size() < 2) return Fail("needs at least 2 rows with both values");
    try {
        const Moments m = MomentsOf(Device(a) - Device(b));
        const double sd = std::sqrt(m.var);
        if (!(sd > 0.0)) return Fail("every difference is the same, so the test cannot be computed");
        TestResult r;
        const double se = sd / std::sqrt(m.n);
        r.df = m.n - 1.0;
        r.estimate = m.mean;
        r.statistic = m.mean / se;
        r.p = TwoSidedT(r.statistic, r.df);
        const double tc = CriticalT(confidence, r.df);
        r.ci_low = m.mean - tc * se;
        r.ci_high = m.mean + tc * se;
        r.effect = m.mean / sd;
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::PairedT", e));
    }
}

TestResult StatisticalTests::OneWayAnova(const std::vector<std::vector<double>>& groups, bool welch) {
    if (groups.size() < 2) return Fail("needs at least 2 groups");
    for (const auto& g : groups)
        if (g.size() < 2) return Fail("each group needs at least 2 values");
    try {
        std::vector<Moments> m;
        double n = 0, total = 0;
        for (const auto& g : groups) {
            m.push_back(MomentsOf(Device(g)));
            n += m.back().n;
            total += m.back().mean * m.back().n;
        }
        const double k = static_cast<double>(groups.size()), grand = total / n;
        double ssb = 0, ssw = 0;
        for (const auto& g : m) {
            ssb += g.n * (g.mean - grand) * (g.mean - grand);
            ssw += (g.n - 1.0) * g.var;
        }
        if (!(ssw > 0.0)) return Fail("every group's values are all the same, so the test cannot be computed");
        TestResult r;
        r.effect = ssb / (ssb + ssw);
        if (!welch) {
            r.df = k - 1.0;
            r.df2 = n - k;
            r.statistic = (ssb / r.df) / (ssw / r.df2);
        } else {
            double wsum = 0, wmean = 0;
            for (const auto& g : m) {
                if (!(g.var > 0.0)) return Fail("a group has the same value in every row; Welch's test needs some spread");
                wsum += g.n / g.var;
                wmean += g.n / g.var * g.mean;
            }
            wmean /= wsum;
            double a = 0, tmp = 0;
            for (const auto& g : m) {
                const double w = g.n / g.var;
                a += w * (g.mean - wmean) * (g.mean - wmean);
                tmp += (1.0 - w / wsum) * (1.0 - w / wsum) / (g.n - 1.0);
            }
            a /= k - 1.0;
            const double bterm = 1.0 + 2.0 * (k - 2.0) / (k * k - 1.0) * tmp;
            r.statistic = a / bterm;
            r.df = k - 1.0;
            r.df2 = (k * k - 1.0) / (3.0 * tmp);
        }
        r.p = FSf(r.statistic, r.df, r.df2);
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::OneWayAnova", e));
    }
}

std::vector<TukeyPair> StatisticalTests::TukeyHsd(const std::vector<std::vector<double>>& groups, double confidence,
                                                  std::string* error) {
    std::vector<TukeyPair> pairs;
    if (groups.size() < 2) return pairs;
    std::vector<Moments> m;
    double n = 0, ssw = 0;
    try {
        for (const auto& g : groups) {
            if (g.size() < 2) return {};
            m.push_back(MomentsOf(Device(g)));
            n += m.back().n;
            ssw += (m.back().n - 1.0) * m.back().var;
        }
    } catch (const af::exception& e) {
        if (error) *error = DeviceError("StatisticalTests::TukeyHsd", e);
        return {};
    }
    const double k = static_cast<double>(groups.size()), df = n - k, mse = ssw / df;
    // The critical studentized range for the intervals, by bisection on its upper tail.
    const double alpha = 1.0 - confidence;
    double lo = 0.0, hi = 1.0;
    while (StudentizedRangeSf(hi, k, df) > alpha && hi < 1e6) hi *= 2.0;
    for (int i = 0; i < 200 && hi - lo > 1e-12 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        (StudentizedRangeSf(mid, k, df) > alpha ? lo : hi) = mid;
    }
    const double qcrit = 0.5 * (lo + hi);
    for (size_t i = 0; i < m.size(); ++i)
        for (size_t j = i + 1; j < m.size(); ++j) {
            TukeyPair p;
            p.a = i;
            p.b = j;
            p.difference = m[i].mean - m[j].mean;
            const double se = std::sqrt(mse / 2.0 * (1.0 / m[i].n + 1.0 / m[j].n));
            p.p = StudentizedRangeSf(std::fabs(p.difference) / se, k, df);
            p.ci_low = p.difference - qcrit * se;
            p.ci_high = p.difference + qcrit * se;
            pairs.push_back(p);
        }
    return pairs;
}

TestResult StatisticalTests::Levene(const std::vector<std::vector<double>>& groups) {
    if (groups.size() < 2) return Fail("needs at least 2 groups");
    try {
        std::vector<af::array> z;
        std::vector<Moments> mz;
        double n = 0, total = 0;
        for (const auto& g : groups) {
            if (g.size() < 2) return Fail("each group needs at least 2 values");
            const af::array x = Device(g);
            z.push_back(af::abs(x - af::median<double>(x)));
            mz.push_back(MomentsOf(z.back()));
            n += mz.back().n;
            total += mz.back().mean * mz.back().n;
        }
        const double k = static_cast<double>(groups.size()), grand = total / n;
        double num = 0, den = 0;
        for (const auto& g : mz) {
            num += g.n * (g.mean - grand) * (g.mean - grand);
            den += (g.n - 1.0) * g.var;
        }
        if (!(den > 0.0)) return Fail("no spread inside the groups, so the test cannot be computed");
        TestResult r;
        r.df = k - 1.0;
        r.df2 = n - k;
        r.statistic = (n - k) / (k - 1.0) * num / den;
        r.p = FSf(r.statistic, r.df, r.df2);
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::Levene", e));
    }
}

TestResult StatisticalTests::MannWhitneyU(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty()) return Fail("each group needs at least 1 value");
    try {
        const double n1 = static_cast<double>(a.size()), n2 = static_cast<double>(b.size()), n = n1 + n2;
        double tie_sum = 0;
        const af::array ranks = AverageRanks(af::join(0, Device(a), Device(b)), tie_sum);
        const double r1 = af::sum<double>(ranks(af::seq(0, static_cast<double>(a.size()) - 1)));
        TestResult r;
        r.statistic = r1 - n1 * (n1 + 1.0) / 2.0;  // U of the first group
        const double u = (std::max)(r.statistic, n1 * n2 - r.statistic);
        const double mu = n1 * n2 / 2.0;
        const double s = std::sqrt(n1 * n2 / 12.0 * ((n + 1.0) - tie_sum / (n * (n - 1.0))));
        if (!(s > 0.0)) return Fail("every value is the same, so the test cannot be computed");
        const double z = (u - mu - 0.5) / s;
        r.p = (std::min)(1.0, 2.0 * NormalSf(z));
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::MannWhitneyU", e));
    }
}

TestResult StatisticalTests::KruskalWallis(const std::vector<std::vector<double>>& groups) {
    if (groups.size() < 2) return Fail("needs at least 2 groups");
    try {
        std::vector<double> all;
        std::vector<size_t> sizes;
        for (const auto& g : groups) {
            if (g.empty()) return Fail("each group needs at least 1 value");
            all.insert(all.end(), g.begin(), g.end());
            sizes.push_back(g.size());
        }
        double tie_sum = 0;
        const af::array ranks = AverageRanks(Device(all), tie_sum);
        const double n = static_cast<double>(all.size());
        double h = 0, start = 0;
        for (const size_t s : sizes) {
            const double rs = af::sum<double>(ranks(af::seq(start, start + static_cast<double>(s) - 1)));
            h += rs * rs / static_cast<double>(s);
            start += static_cast<double>(s);
        }
        h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
        const double correction = 1.0 - tie_sum / (n * n * n - n);
        if (!(correction > 0.0)) return Fail("every value is the same, so the test cannot be computed");
        TestResult r;
        r.statistic = h / correction;
        r.df = static_cast<double>(groups.size()) - 1.0;
        r.p = ChiSquareSf(r.statistic, r.df);
        r.ok = true;
        return r;
    } catch (const af::exception& e) {
        return Fail(DeviceError("StatisticalTests::KruskalWallis", e));
    }
}

ChiSquareResult StatisticalTests::ChiSquareIndependence(const std::vector<std::vector<double>>& counts) {
    ChiSquareResult out;
    const size_t rows = counts.size(), cols = rows ? counts[0].size() : 0;
    if (rows < 2 || cols < 2) {
        out.test = Fail("needs at least 2 values in each column");
        return out;
    }
    for (const auto& row : counts)
        if (row.size() != cols) {
            out.test = Fail("every row needs the same number of columns");
            return out;
        }
    try {
        std::vector<double> flat;  // column-major for ArrayFire
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r) flat.push_back(counts[r][c]);
        const af::array o = af::moddims(Device(flat), static_cast<dim_t>(rows), static_cast<dim_t>(cols));
        const af::array row_sum = af::sum(o, 1), col_sum = af::sum(o, 0);
        const double total = af::sum<double>(o);
        if (!(af::min<double>(row_sum) > 0.0 && af::min<double>(col_sum) > 0.0)) {
            out.test = Fail("a value has no rows in the other column");
            return out;
        }
        const af::array e = af::matmul(row_sum, col_sum) / total;
        const af::array d = o - e;
        TestResult r;
        r.statistic = af::sum<double>(d * d / e);
        r.df = static_cast<double>((rows - 1) * (cols - 1));
        r.p = ChiSquareSf(r.statistic, r.df);
        r.effect = std::sqrt(r.statistic / (total * static_cast<double>((std::min)(rows, cols) - 1)));
        r.ok = true;
        out.test = r;
        out.min_expected = af::min<double>(e);
        const std::vector<double> ev = DeviceToHostValues(e, "StatisticalTests::ChiSquareIndependence");
        out.expected.assign(rows, std::vector<double>(cols));
        for (size_t c = 0; c < cols; ++c)
            for (size_t rr = 0; rr < rows; ++rr) out.expected[rr][c] = ev[c * rows + rr];
    } catch (const af::exception& ex) {
        out.test = Fail(DeviceError("StatisticalTests::ChiSquareIndependence", ex));
    }
    return out;
}

// ================================================================ linear models

namespace {

LinearFit FitFail(const std::string& why) {
    LinearFit f;
    f.error = why;
    return f;
}

// OLS on the device: R from an in-place QR, then corrected semi-normal
// equations (R'R b = X'y and one refinement step), which are as accurate as
// a full QR solve without forming the m x m Q. `x` holds the intercept column.
bool SolveOls(const af::array& x, const af::array& y, af::array& beta, af::array& r_factor, size_t& bad_column) {
    const dim_t p = x.dims(1);
    af::array a = x.copy(), tau;
    af::qrInPlace(tau, a);
    r_factor = af::upper(a(af::seq(0, static_cast<double>(p - 1)), af::span));
    const std::vector<double> diag = DeviceToHostValues(af::abs(af::diag(r_factor)), "StatisticalTests::FitLinear");
    const double largest = *std::max_element(diag.begin(), diag.end());
    for (size_t j = 0; j < diag.size(); ++j)
        if (!(diag[j] > 1e-10 * largest)) {
            bad_column = j;
            return false;
        }
    const auto solve_normal = [&](const af::array& rhs) {
        const af::array z = af::solve(r_factor.T(), rhs, AF_MAT_LOWER);
        return af::solve(r_factor, z, AF_MAT_UPPER);
    };
    beta = solve_normal(af::matmul(x.T(), y));
    const af::array residual = y - af::matmul(x, beta);
    beta += solve_normal(af::matmul(x.T(), residual));
    return true;
}

LinearFit Summarise(const af::array& x, const af::array& y, const af::array& beta, const af::array& r_factor,
                    std::vector<std::string> names, double confidence, bool intercept = true) {
    LinearFit f;
    const double n = static_cast<double>(x.dims(0)), p = static_cast<double>(x.dims(1));
    const double k = intercept ? p - 1.0 : p;  // model degrees of freedom
    const af::array fitted = af::matmul(x, beta);
    const af::array residual = y - fitted;
    const double sse = af::sum<double>(residual * residual);
    const double ymean = intercept ? af::mean<double>(y) : 0.0;  // uncentred without an intercept
    const af::array dy = y - ymean;
    const double sst = af::sum<double>(dy * dy);
    f.n = static_cast<int64_t>(n);
    f.df_model = static_cast<int>(k);
    f.df_resid = static_cast<int>(n - p);
    const double s2 = sse / (n - p);
    // (X'X)^-1 = R^-1 R^-T
    const af::array r_inv = af::solve(r_factor, af::identity(r_factor.dims(0), r_factor.dims(1), r_factor.type()),
                                      AF_MAT_UPPER);
    const af::array cov = af::matmul(r_inv, r_inv.T()) * s2;
    f.coef = DeviceToHostValues(beta, "StatisticalTests::FitLinear");
    const std::vector<double> var = DeviceToHostValues(af::diag(cov), "StatisticalTests::FitLinear");
    const double tc = CriticalT(confidence, n - p);
    for (size_t j = 0; j < f.coef.size(); ++j) {
        const double se = std::sqrt((std::max)(0.0, var[j]));
        f.se.push_back(se);
        const double t = se > 0.0 ? f.coef[j] / se : (f.coef[j] == 0.0 ? 0.0 : INFINITY);
        f.t.push_back(t);
        f.p.push_back(se > 0.0 ? TwoSidedT(t, n - p) : (f.coef[j] == 0.0 ? 1.0 : 0.0));
        f.ci_low.push_back(f.coef[j] - tc * se);
        f.ci_high.push_back(f.coef[j] + tc * se);
    }
    f.r2 = sst > 0.0 ? 1.0 - sse / sst : NAN;
    f.adj_r2 = 1.0 - (1.0 - f.r2) * (n - (intercept ? 1.0 : 0.0)) / (n - p);
    if (k > 0.0 && sse > 0.0) {
        f.f = (f.r2 / k) / ((1.0 - f.r2) / (n - p));
        f.f_p = StatisticalTests::FSf(f.f, k, n - p);
    }
    f.rse = std::sqrt(s2);
    f.rmse = std::sqrt(sse / n);
    f.mae = af::mean<double>(af::abs(residual));
    const double loglik = -n / 2.0 * (std::log(2.0 * kPi) + std::log(sse / n) + 1.0);
    f.aic = 2.0 * p - 2.0 * loglik;
    f.fitted = DeviceToHostValues(fitted, "StatisticalTests::FitLinear");
    f.residuals = DeviceToHostValues(residual, "StatisticalTests::FitLinear");
    f.names = std::move(names);
    f.ok = true;
    return f;
}

}  // namespace

LinearFit StatisticalTests::FitLinear(const std::vector<std::vector<double>>& columns, const std::vector<double>& y,
                                      const std::vector<std::string>& names, bool intercept, double confidence) {
    if (columns.empty()) return FitFail("needs at least one column to predict from");
    const size_t first = intercept ? 1 : 0;  // the design column of columns[0]
    const size_t n = y.size(), p = columns.size() + first;
    for (const auto& c : columns)
        if (c.size() != n) return FitFail("every column needs the same rows");
    if (n < p + 1) return FitFail("needs at least " + std::to_string(p + 1) + " rows with every value; " + std::to_string(n) + " have them");
    try {
        std::vector<double> flat(first * n, 1.0);  // column-major, the intercept first when there is one
        for (const auto& c : columns) flat.insert(flat.end(), c.begin(), c.end());
        const af::dtype type = ComputeType();
        const af::array x = af::moddims(Device(flat), static_cast<dim_t>(n), static_cast<dim_t>(p)).as(type);
        const af::array yy = Device(y).as(type);
        af::array beta, r_factor;
        size_t bad = 0;
        if (!SolveOls(x, yy, beta, r_factor, bad)) {
            const bool is_column = bad >= first && bad - first < columns.size();
            const std::string name = is_column && bad - first < names.size() ? names[bad - first] : "a column";
            const af::array column = is_column ? Device(columns[bad - first]) : Device(y);
            const bool constant = af::min<double>(column) == af::max<double>(column);
            return FitFail(!constant ? name + " is a combination of the other columns, so its effect cannot be told apart"
                                       : name + " has the same value in every row used, so a line through it cannot be fitted");
        }
        std::vector<std::string> all;
        if (intercept) all.push_back("intercept");
        all.insert(all.end(), names.begin(), names.end());
        return Summarise(x, yy, beta, r_factor, std::move(all), confidence, intercept);
    } catch (const af::exception& e) {
        return FitFail(DeviceError("StatisticalTests::FitLinear", e));
    }
}

LinearFit StatisticalTests::FitPolynomial(const std::vector<double>& x, const std::vector<double>& y, int degree,
                                          const std::string& name, double confidence) {
    if (degree < 1 || degree > 6) return FitFail("the degree must be 1 to 6");
    if (x.size() != y.size()) return FitFail("x and y need the same rows");
    const size_t need = static_cast<size_t>(degree) + 2;
    if (x.size() < need)
        return FitFail("degree " + std::to_string(degree) + " needs at least " + std::to_string(need) + " rows with both values; " +
                       std::to_string(x.size()) + " have them");
    try {
        const Moments m = MomentsOf(Device(x));
        const double scale = std::sqrt(m.var);
        if (!(scale > 0.0)) return FitFail(name + " has the same value in every row used, so a curve through it cannot be fitted");
        // Fit on z = (x - mean) / sd: powers of z stay near 1, so the solve stays accurate.
        const af::dtype type = ComputeType();
        const af::array z = (Device(x).as(type) - m.mean) / scale;
        af::array design = af::constant(1.0, static_cast<dim_t>(x.size()), degree + 1, type);
        af::array power = af::constant(1.0, static_cast<dim_t>(x.size()), type);
        for (int j = 1; j <= degree; ++j) {
            power = power * z;
            design(af::span, j) = power;
        }
        const af::array yy = Device(y).as(type);
        af::array beta, r_factor;
        size_t bad = 0;
        if (!SolveOls(design, yy, beta, r_factor, bad))
            return FitFail("degree " + std::to_string(degree) + " is too high for " + std::to_string(x.size()) +
                           " rows with this few distinct values of " + name);
        std::vector<std::string> names{"intercept", name};
        for (int j = 2; j <= degree; ++j) names.push_back(name + "^" + std::to_string(j));
        LinearFit fz = Summarise(design, yy, beta, r_factor, names, confidence);
        // Back to the original units: z^k = ((x - m) / s)^k, so the raw
        // coefficients are T b with T[j][k] = C(k, j) (-m)^(k-j) / s^k, and
        // their covariance T C T'.
        const int p = degree + 1;
        std::vector<double> t(static_cast<size_t>(p * p), 0.0);  // column-major p x p
        for (int k = 0; k < p; ++k)
            for (int j = 0; j <= k; ++j) {
                double binom = 1.0;
                for (int i = 1; i <= j; ++i) binom = binom * (k - j + i) / i;
                t[static_cast<size_t>(k * p + j)] = binom * std::pow(-m.mean, k - j) / std::pow(scale, k);
            }
        const af::array tm = af::moddims(Device(t), p, p).as(type);
        const double s2 = fz.rse * fz.rse;
        const af::array r_inv = af::solve(r_factor, af::identity(p, p, type), AF_MAT_UPPER);
        const af::array cov_z = af::matmul(r_inv, r_inv.T()) * s2;
        const af::array beta_raw = af::matmul(tm, beta);
        const af::array cov_raw = af::matmul(af::matmul(tm, cov_z), tm.T());
        fz.coef = DeviceToHostValues(beta_raw, "StatisticalTests::FitPolynomial");
        const std::vector<double> var = DeviceToHostValues(af::diag(cov_raw), "StatisticalTests::FitPolynomial");
        const double df = static_cast<double>(fz.df_resid), tc = CriticalT(confidence, df);
        for (int j = 0; j < p; ++j) {
            const double se = std::sqrt((std::max)(0.0, var[static_cast<size_t>(j)]));
            fz.se[static_cast<size_t>(j)] = se;
            fz.t[static_cast<size_t>(j)] = se > 0.0 ? fz.coef[static_cast<size_t>(j)] / se : NAN;
            fz.p[static_cast<size_t>(j)] = se > 0.0 ? TwoSidedT(fz.t[static_cast<size_t>(j)], df) : NAN;
            fz.ci_low[static_cast<size_t>(j)] = fz.coef[static_cast<size_t>(j)] - tc * se;
            fz.ci_high[static_cast<size_t>(j)] = fz.coef[static_cast<size_t>(j)] + tc * se;
        }
        return fz;
    } catch (const af::exception& e) {
        return FitFail(DeviceError("StatisticalTests::FitPolynomial", e));
    }
}

}  // namespace cyxwiz
