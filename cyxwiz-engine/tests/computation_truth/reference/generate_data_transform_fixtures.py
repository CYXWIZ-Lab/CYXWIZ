"""PyTorch (float64) fixtures for DataTransform (TOFIX140, data transforms on the device).

The references are written in torch from the library definitions the transforms follow:
scikit-learn MinMaxScaler / StandardScaler / RobustScaler / MaxAbsScaler / QuantileTransformer
(fitted on the whole column, no subsampling) and scipy.stats boxcox / yeojohnson / normaltest.
Where scipy has the same function the script cross-checks against it before writing.

    py -3.12 generate_data_transform_fixtures.py
"""

import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "data_transform_pytorch.json"
torch.set_default_dtype(torch.float64)


def vec(t):
    return [round(float(v), 15) for v in t.tolist()]


def gen(seed):
    return torch.Generator().manual_seed(seed)


def percentile(x, fractions):
    return torch.quantile(x, fractions, interpolation="linear")


def column_stats(x):
    q = percentile(x, torch.tensor([0.25, 0.5, 0.75]))
    d = x - x.mean()
    # Moment skewness m3 / m2^1.5 (scipy.stats.skew, bias=True).
    skewness = float((d ** 3).mean() / (d ** 2).mean() ** 1.5)
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std_dev": float(x.std(unbiased=False)), "q1": float(q[0]), "median": float(q[1]),
            "q3": float(q[2]), "iqr": float(q[2] - q[0]), "skewness": skewness, "count": x.numel()}


def power(x, lam, family):
    if family == "boxcox":
        return torch.log(x) if abs(lam) < 1e-12 else torch.expm1(lam * torch.log(x)) / lam
    lp = torch.log1p(x.abs())
    pos = lp if abs(lam) < 1e-12 else torch.expm1(lam * lp) / lam
    other = 2.0 - lam
    neg = -(lp if abs(other) < 1e-12 else torch.expm1(other * lp) / other)
    return torch.where(x >= 0, pos, neg)


def llf(x, lam, family):
    y = power(x, lam, family)
    jac = torch.log(x).sum() if family == "boxcox" else (torch.sign(x) * torch.log1p(x.abs())).sum()
    return float((lam - 1.0) * jac - 0.5 * x.numel() * torch.log(y.var(unbiased=False)))


def optimal_lambda(x, family, low=-5.0, high=5.0):
    # Dense scan, then golden-section search to 1e-13 (the likelihood is unimodal here).
    grid = torch.linspace(low, high, 2001)
    best = max(range(len(grid)), key=lambda i: llf(x, float(grid[i]), family))
    a, b = float(grid[max(best - 1, 0)]), float(grid[min(best + 1, len(grid) - 1)])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    while b - a > 1e-13:
        c, d = b - phi * (b - a), a + phi * (b - a)
        if llf(x, c, family) > llf(x, d, family):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


def interp(x, xp, fp):
    """np.interp (rightmost bracket, exact hits return fp[j])."""
    out = torch.empty_like(x)
    for i, v in enumerate(x.tolist()):
        if v < xp[0]:
            out[i] = fp[0]
            continue
        if v >= xp[-1]:
            out[i] = fp[-1]
            continue
        j = int(torch.searchsorted(xp, torch.tensor([v]), right=True)) - 1
        if xp[j] == v:
            out[i] = fp[j]
        else:
            out[i] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (v - xp[j]) + fp[j]
    return out


def exact_quantiles(sorted_x, size):
    """Linear quantiles at i / (size - 1), i = 0 .. size - 1, with the sorted position
    i * (n - 1) / (size - 1) split in integers so no rounding moves it off a data point
    (rounded positions make the result depend on ulps wherever values repeat)."""
    n = sorted_x.numel()
    numer = torch.arange(size, dtype=torch.int64) * (n - 1)
    lower = numer // (size - 1)
    t = (numer % (size - 1)).double() / (size - 1)
    a, b = sorted_x[lower], sorted_x[torch.clamp(lower + 1, max=n - 1)]
    return torch.where(t >= 0.5, b - (b - a) * (1.0 - t), a + (b - a) * t)


def quantile_transform(x, distribution, n_quantiles):
    size = min(n_quantiles, x.numel())
    refs = torch.arange(size).double() / (size - 1)
    q = torch.cummax(exact_quantiles(torch.sort(x).values, size), 0).values
    y = 0.5 * (interp(x, q, refs) - interp(-x, -q.flip(0), -refs.flip(0)))
    bound = 1e-7
    if distribution == "normal":
        upper, lower = x + bound > q[-1], x - bound < q[0]
    else:
        upper, lower = x == q[-1], x == q[0]
    y[upper] = 1.0
    y[lower] = 0.0
    if distribution == "normal":
        clip = bound - 2.220446049250313e-16
        y = torch.special.ndtri(y.clamp(clip, 1.0 - clip))
    return y


def normaltest(x):
    n = float(x.numel())
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    y = (m3 / m2 ** 1.5) * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1))
    y = float(y) if float(y) != 0 else 1.0
    z_skew = delta * math.log(y / alpha + math.sqrt((y / alpha) ** 2 + 1))
    b2 = float(m4 / m2 ** 2)
    e = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    xs = (b2 - e) / math.sqrt(varb2)
    sb1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    a = 6.0 + 8.0 / sb1 * (2.0 / sb1 + math.sqrt(1 + 4.0 / sb1 ** 2))
    denom = 1 + xs * math.sqrt(2 / (a - 4.0))
    term2 = math.copysign(1.0, denom) * ((1 - 2.0 / a) / abs(denom)) ** (1 / 3.0)
    z_kurt = ((1 - 2 / (9.0 * a)) - term2) / math.sqrt(2 / (9.0 * a))
    k2 = z_skew ** 2 + z_kurt ** 2
    return k2, math.exp(-k2 / 2)


def cross_check(cases):
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        print("scipy not installed: written without the scipy cross-check")
        return
    for c in cases:
        x = np.array(c["inputs"][0]) if c["inputs"] else None
        if c["op"] == "stats":
            assert abs(stats.skew(x) - c["stats"]["skewness"]) < 1e-10, c["label"]
        elif c["op"] == "boxcox_auto":
            assert abs(stats.boxcox(x)[1] - c["lambda"]) < 1e-6, c["label"]
        elif c["op"] == "yeojohnson_auto":
            assert abs(stats.yeojohnson(x)[1] - c["lambda"]) < 1e-6, c["label"]
        elif c["op"] in ("boxcox", "yeojohnson"):
            ref = (stats.boxcox if c["op"] == "boxcox" else stats.yeojohnson)(x, lmbda=c["lambda_in"])
            assert np.allclose(ref, c["outputs"][0], rtol=1e-10, atol=1e-12), c["label"]
        elif c["op"] == "normaltest":
            k2, p = stats.normaltest(x)
            assert abs(k2 - c["statistic"]) < 1e-9 * (1 + k2) and abs(p - c["p_value"]) < 1e-9, c["label"]
        elif c["op"] == "quantile":
            # scikit-learn's QuantileTransformer._transform_col, in numpy, on the same exact
            # quantiles (scikit-learn's np.percentile(x, refs * 100) moves some by ulps; they
            # agree with these to 1e-12, but an ulp flips which tied values match exactly).
            size = min(c["n_quantiles"], len(x))
            refs = np.arange(size) / (size - 1)
            exact = exact_quantiles(torch.sort(torch.tensor(x)).values, size).numpy()
            assert np.allclose(exact, np.percentile(x, refs * 100), rtol=1e-12, atol=1e-12), c["label"]
            q = np.maximum.accumulate(exact)
            y = 0.5 * (np.interp(x, q, refs) - np.interp(-x, -q[::-1], -refs[::-1]))
            if c["distribution"] == "normal":
                upper, lower = x + 1e-7 > q[-1], x - 1e-7 < q[0]
            else:
                upper, lower = x == q[-1], x == q[0]
            y[upper], y[lower] = 1.0, 0.0
            if c["distribution"] == "normal":
                y = np.clip(stats.norm.ppf(y), stats.norm.ppf(1e-7 - np.spacing(1)),
                            stats.norm.ppf(1 - (1e-7 - np.spacing(1))))
            assert np.allclose(y, c["outputs"][0], rtol=1e-10, atol=1e-12), c["label"]
        elif c["op"] in ("boxcox_lambda_profile", "yeojohnson_lambda_profile"):
            llf_fn = stats.boxcox_llf if c["op"].startswith("boxcox") else stats.yeojohnson_llf
            for lam, ll in zip(c["lambdas"], c["log_likelihoods"]):
                assert abs(llf_fn(lam, x) - ll) < 1e-8 * (1 + abs(ll)), (c["label"], lam)
    print("scipy cross-check passed")


def main():
    normal = torch.randn(400, generator=gen(1)) * 2.0 + 5.0
    lognormal = torch.exp(torch.randn(300, generator=gen(2)) * 0.7 + 1.0)
    skewed = torch.rand(250, generator=gen(3)) ** 3 * 10.0 - 2.0          # negatives too
    ties = torch.randint(0, 12, (180,), generator=gen(4)).double() * 0.5   # repeated values
    small = torch.tensor([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0])
    exponential = -torch.log(torch.rand(500, generator=gen(5))) * 3.0 + 0.1

    cases = []

    def add(op, label, inputs, **expected):
        cases.append({"op": op, "label": label, "inputs": [vec(c) for c in inputs], **expected})

    for label, x in [("normal", normal), ("ties", ties), ("small", small), ("skewed", skewed),
                     ("exponential", exponential)]:
        add("stats", label, [x], stats=column_stats(x))

    cols = [normal, skewed, ties]
    add("normalize", "default", cols, outputs=[vec((c - c.min()) / (c.max() - c.min())) for c in cols])
    add("normalize", "range", cols, range_min=-2.0, range_max=3.0,
        outputs=[vec((c - c.min()) / (c.max() - c.min()) * 5.0 - 2.0) for c in cols])
    add("standardize", "columns", cols, outputs=[vec((c - c.mean()) / c.std(unbiased=False)) for c in cols])
    add("standardize", "no_mean", cols, with_mean=False, outputs=[vec(c / c.std(unbiased=False)) for c in cols])
    add("standardize", "no_std", cols, with_std=False, outputs=[vec(c - c.mean()) for c in cols])
    constant = torch.full((10,), 3.0)
    # scikit-learn's _handle_zeros_in_scale: a zero scale divides by 1.
    add("standardize", "constant", [constant], outputs=[vec(constant - 3.0)])
    add("normalize", "constant", [constant], range_min=-1.0, range_max=2.0, outputs=[vec(torch.full((10,), -1.0))])
    add("maxabs_scale", "zeros", [torch.zeros(6)], outputs=[vec(torch.zeros(6))])
    centers, scales = [1.5, -2.0, 0.25], [2.0, 0.5, 4.0]
    add("scale_columns", "given", cols, centers=centers, scales=scales,
        outputs=[vec((c - m) / d) for c, m, d in zip(cols, centers, scales)])
    for label, x, method, t in [("iqr_skewed", skewed, "iqr", 1.5), ("iqr_ties", ties, "iqr", 1.0),
                                ("zscore_normal", normal, "zscore", 2.0), ("zscore_skewed", skewed, "zscore", 2.5)]:
        if method == "iqr":
            q = percentile(x, torch.tensor([0.25, 0.75]))
            flags = (x < q[0] - t * (q[1] - q[0])) | (x > q[1] + t * (q[1] - q[0]))
        else:
            flags = (x - x.mean()).abs() / x.std(unbiased=False) > t
        add("outliers", label, [x], method=method, threshold=t, flags=[int(f) for f in flags.tolist()])
    def robust(c, q_min=25.0, q_max=75.0, center=True, scale=True):
        """scikit-learn RobustScaler; a spread below 10 eps scales by 1 (_handle_zeros_in_scale)."""
        mid = percentile(c, torch.tensor(0.5)) if center else 0.0
        spread = float(percentile(c, torch.tensor(q_max / 100)) - percentile(c, torch.tensor(q_min / 100)))
        return vec((c - mid) / (spread if scale and spread >= 10 * 2.220446049250313e-16 else 1.0))

    add("robust_scale", "columns", cols, outputs=[robust(c) for c in cols])
    constant = torch.full((12,), 4.0)
    for label, options in [("q10_q90", dict(q_min=10.0, q_max=90.0)), ("no_centering", dict(center=False)),
                           ("no_scaling", dict(scale=False))]:
        add("robust_scale", label, cols, quantile_min=options.get("q_min", 25.0), quantile_max=options.get("q_max", 75.0),
            with_centering=options.get("center", True), with_scaling=options.get("scale", True),
            outputs=[robust(c, **options) for c in cols])
    add("robust_scale", "constant_column", [constant, cols[0]], outputs=[robust(constant), robust(cols[0])])
    add("maxabs_scale", "columns", cols, outputs=[vec(c / c.abs().max()) for c in cols])

    for base, divisor in [("natural", 1.0), ("log10", math.log(10.0)), ("log2", math.log(2.0))]:
        add("log", base + "_log", [lognormal, exponential], base=base, use_log1p=False,
            outputs=[vec(torch.log(c) / divisor) for c in (lognormal, exponential)])
        add("log", base + "_log1p", [skewed + 1.5, ties], base=base, use_log1p=True,
            outputs=[vec(torch.log1p(c) / divisor) for c in (skewed + 1.5, ties)])

    for lam in (0.0, 0.5, -1.0, 2.3):
        add("boxcox", f"lambda_{lam}", [lognormal], lambda_in=lam, outputs=[vec(power(lognormal, lam, "boxcox"))])
    for lam in (0.0, 2.0, 1.3, -0.7):
        add("yeojohnson", f"lambda_{lam}", [skewed], lambda_in=lam, outputs=[vec(power(skewed, lam, "yeojohnson"))])

    for label, x in [("lognormal", lognormal), ("exponential", exponential)]:
        lam = optimal_lambda(x, "boxcox")
        add("boxcox_auto", label, [x], **{"lambda": lam, "log_likelihood": llf(x, lam, "boxcox"),
                                          "outputs": [vec(power(x, lam, "boxcox"))]})
    for label, x in [("skewed", skewed), ("normal_shifted", normal - 5.0)]:
        lam = optimal_lambda(x, "yeojohnson")
        add("yeojohnson_auto", label, [x], **{"lambda": lam, "log_likelihood": llf(x, lam, "yeojohnson"),
                                              "outputs": [vec(power(x, lam, "yeojohnson"))]})
    grid = torch.linspace(-5.0, 5.0, 101).tolist()
    add("boxcox_lambda_profile", "lognormal", [lognormal], lambdas=grid,
        log_likelihoods=[llf(lognormal, g, "boxcox") for g in grid])
    add("yeojohnson_lambda_profile", "skewed", [skewed], lambdas=grid,
        log_likelihoods=[llf(skewed, g, "yeojohnson") for g in grid])

    for distribution in ("uniform", "normal"):
        for label, x, nq in [("normal_q100", normal, 100), ("ties_q1000", ties, 1000), ("small_q5", small, 5),
                             ("skewed_q1000", skewed, 1000)]:
            add("quantile", f"{distribution}_{label}", [x], distribution=distribution, n_quantiles=nq,
                outputs=[vec(quantile_transform(x, distribution, nq))])

    for label, x in [("normal", normal), ("lognormal", lognormal), ("exponential", exponential),
                     ("ties", ties)]:
        k2, p = normaltest(x)
        add("normaltest", label, [x], statistic=k2, p_value=p)

    # Refusals: the message must contain the given text.
    add("refuse_boxcox", "non_positive", [small], message="strictly positive")
    add("refuse_log", "non_positive", [small], message="non-positive")
    add("refuse_log1p", "minus_one", [torch.tensor([0.5, -1.0, 2.0])], message="<= -1")
    add("refuse_normaltest", "too_few", [small], message="at least 20")

    cross_check(cases)
    OUT.write_text(json.dumps({"schema_version": 1, "torch_version": torch.__version__, "cases": cases}, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
