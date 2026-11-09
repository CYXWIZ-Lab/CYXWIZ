# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
# TOFIX129 D11.
"""pycyxwiz.model against PyTorch and scikit-learn.

The cases are in cyxwiz-engine/tests/computation_truth/fixtures/:
model_python_torch.json (generate_model_python_fixtures.py: an MLP's summary,
saliency and Integrated Gradients, a self-attention module's count, a range
test as torch runs it), learning_rate_range_pytorch.json (the Engine's range
test cases) and threshold_calibration_sklearn.json (with
cyxwiz-engine/tests/fixtures/evaluation_sklearn.json: the calibration fits).
numpy is not needed (ctest's Python has none). Run with the Python that
pycyxwiz was built for:

    python test_python_model.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
cyx.initialize()
check = Checks()
mt = cyx.model

here = os.path.dirname(os.path.abspath(__file__))
engine_tests = os.path.join(here, "..", "..", "..", "cyxwiz-engine", "tests")
fixtures = os.path.join(engine_tests, "computation_truth", "fixtures")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


torch_truth = load(os.path.join(fixtures, "model_python_torch.json"))["cases"]
range_truth = load(os.path.join(fixtures, "learning_rate_range_pytorch.json"))["cases"]
calibration_truth = load(os.path.join(fixtures, "threshold_calibration_sklearn.json"))
source = load(os.path.join(engine_tests, "fixtures", "evaluation_sklearn.json"))


def flat(values):
    if isinstance(values, (list, tuple)):
        return [v for item in values for v in flat(item)]
    return [values]


def worst(a, b):
    a, b = flat(a), flat(b)
    if len(a) != len(b):
        return float("inf")
    return max(abs(x - y) for x, y in zip(a, b))


def tensor(values, shape):
    t = cyx.Tensor.zeros(shape)
    flat_view = t.reshape([t.num_elements()])
    for i, v in enumerate(values):
        flat_view.set(i, v)
    return flat_view.reshape(shape)


def values_of(t):
    n = t.num_elements()
    view = t.reshape([n])
    return [view.at(i) for i in range(n)]


def rows(values, shape):
    width = shape[1]
    return [values[i * width:(i + 1) * width] for i in range(shape[0])]


def mlp(weights):
    model = cyx.Sequential()
    model.add_linear(6, 16)
    model.add_relu()
    model.add_linear(16, 3)
    model.set_parameters({k: tensor(v["values"], v["shape"]) for k, v in weights.items()})
    return model


def raises(call, words):
    try:
        call()
    except (ValueError, RuntimeError) as e:
        return words in str(e)
    return False


# ---- Summary: the shared counter, as torch.utils.flop_counter / 2 ----------------
c = torch_truth["mlp"]
model = mlp(c["weights"])
s = mt.summary(model, [6])
check([r["parameters"] for r in s["rows"]] == c["summary"]["parameters"], "mlp parameters by module %s" % [r["parameters"] for r in s["rows"]])
check([r["multiply_adds"] for r in s["rows"]] == c["summary"]["multiply_adds"],
      "mlp multiply-adds by module %s (torch %s)" % ([r["multiply_adds"] for r in s["rows"]], c["summary"]["multiply_adds"]))
check(s["parameters"] == 163 and s["multiply_adds"] == 144 and s["all_counted"] and s["parameter_bytes"] == 652, "mlp totals")
check(s["rows"][0]["output"] == [16] and s["rows"][2]["output"] == [3] and s["rows"][0]["formula"] == "6 · 16 = 96",
      "mlp shapes and formula: %s" % s["rows"][0]["formula"])
a = torch_truth["attention"]
attention = cyx.Sequential()
attention.add(cyx.AttentionModule(a["width"], a["heads"]))
s = mt.summary(attention, [a["length"], a["width"]])
check(s["parameters"] == a["parameters"] and s["multiply_adds"] == a["multiply_adds"],
      "self-attention %d parameters, %d multiply-adds (torch %d, %d)" % (s["parameters"], s["multiply_adds"], a["parameters"], a["multiply_adds"]))


class Twice(cyx.Module):
    def forward(self, x):
        return x * 2.0

    def backward(self, g):
        return g * 2.0

    def get_name(self):
        return "Twice"


python_made = cyx.Sequential()
python_made.add_linear(6, 4)
python_made.add(Twice())
s = mt.summary(python_made, [6])
check(not s["rows"][1]["counted"] and "Python module" in s["rows"][1]["formula"] and not s["all_counted"],
      "a Python module is not counted, and says so")
check(raises(lambda: mt.summary(cyx.Sequential(), [6]), "no modules"), "an empty model is refused")

# ---- Explain: saliency and Integrated Gradients (midpoint, 64 steps) --------------
x = c["input"]
sal = mt.saliency(model, x)
check(sal["target"] == c["target"], "saliency explains the predicted class %d" % sal["target"])
check(worst(sal["gradients"], c["saliency"]) < 1e-5, "saliency d score / d x (max diff %.2e)" % worst(sal["gradients"], c["saliency"]))
check(worst(sal["absolute"], [abs(v) for v in c["saliency"]]) < 1e-5, "absolute saliency")
ig = mt.integrated_gradients(model, x, steps=64)
check(worst(ig["attributions"], c["integrated"]) < 1e-5, "Integrated Gradients (max diff %.2e)" % worst(ig["attributions"], c["integrated"]))
check(abs(ig["score"] - c["score"]) < 1e-5 and abs(ig["baseline_score"] - c["baseline_score"]) < 1e-5, "scores at x and the baseline")
check(abs(ig["attribution_sum"] - (ig["score"] - ig["baseline_score"])) < 2e-3, "completeness: the attributions add up to the score's move")
other = mt.integrated_gradients(model, x, baseline=[0.0] * 6, target=0)
check(other["target"] == 0, "a chosen class is explained")
sg = mt.smooth_grad(model, x, samples=8, noise=0.0)
check(worst(sg["gradients"], c["saliency"]) < 1e-5, "SmoothGrad without noise is the saliency")

# ---- Learning rate range test ------------------------------------------------------
# The rules: the Engine's torch cases replayed (their losses in, the same curve out).
for name, r in range_truth.items():
    test = mt.LearningRateRange(r["start"], r["end"], r["steps"], r["smoothing"], r["stop_factor"])
    for loss in r["losses"]:
        if test.finished:
            break
        test.record(loss)
    pts = test.points
    an = test.analyse()
    check(len(pts) == len(r["rates"]) and worst([p["learning_rate"] for p in pts], r["rates"]) < 1e-12 * max(r["rates"]),
          "%s: %d rates" % (name, len(pts)))
    check(worst([p["smoothed"] for p in pts], r["smoothed"]) < 1e-9, "%s: smoothed losses" % name)
    check(an["lowest"] == r["lowest"] and an["steepest"] == r["steepest"] and abs(an["steepest_rate"] - r["steepest_rate"]) < 1e-12,
          "%s: lowest %d, steepest %d at %.4g" % (name, an["lowest"], an["steepest"], an["steepest_rate"]))
    check(worst(an["slopes"], r["slopes"]) < 1e-9, "%s: slopes as numpy.gradient" % name)

# A real run, step for step against torch (SGD, mean squared error, batches in order).
r = torch_truth["range"]
model = mlp(r["weights"])
before = {k: values_of(v) for k, v in model.get_parameters().items()}
sgd = cyx.SGD(learning_rate=0.5)
out = mt.lr_range_test(model, cyx.MSELoss(), sgd, rows(r["x"]["values"], r["x"]["shape"]),
                       rows(r["y"]["values"], r["y"]["shape"]),
                       start=r["start"], end=r["end"], steps=r["steps"], batch_size=r["batch"])
losses = [p["loss"] for p in out["points"]]
n = len(losses)
check(n >= 20, "the run took %d steps (%s)" % (n, out["stop_reason"]))
check(worst(losses, r["losses"][:n]) < 1e-4, "each step's loss as torch's (max diff %.2e)" % worst(losses, r["losses"][:n]))
check(worst([p["learning_rate"] for p in out["points"]], r["rates"][:n]) < 1e-12, "each step's rate")
after = {k: values_of(v) for k, v in model.get_parameters().items()}
check(all(before[k] == after[k] for k in before), "the model's weights are put back")
check(abs(sgd.get_learning_rate() - 0.5) < 1e-12, "the optimizer's rate is put back")
check(out["analysis"]["ok"] and 0 <= out["analysis"]["steepest"] <= out["analysis"]["lowest"], "the run's analysis")
check(raises(lambda: mt.lr_range_test(model, cyx.MSELoss(), cyx.SGD(learning_rate=0.1), [[0.0] * 6] * 4,
                                      [[0.0] * 3] * 3), "same number of rows"), "x and y must match")
check(raises(lambda: mt.LearningRateRange(1.0, 0.5), "start > 0"), "a falling range is refused")

# ---- Calibration (scikit-learn / scipy) -------------------------------------------
f = calibration_truth
b = source["binary"]
male = [1 if t == "MALE" else 0 for t in b["truth"]]
score = b["score_male"]
rel = mt.reliability(score, male)
check(abs(rel["ece"] - f["binary_reliability_all"]["ece"]) < 1e-9 and len(rel["bins"]) == 10, "yes / no reliability ECE %.6f" % rel["ece"])
even = list(range(0, len(score), 2))
odd = list(range(1, len(score), 2))
pa, pb = mt.fit_platt([score[i] for i in even], [male[i] for i in even])
check(abs(pa - f["platt"]["a"]) < 1e-5 and abs(pb - f["platt"]["b"]) < 1e-5, "Platt a = %.6f, b = %.6f" % (pa, pb))
check(worst(mt.apply_platt(pa, pb, [score[i] for i in odd]), f["platt"]["check"]) < 1e-5, "Platt on the odd rows")
ix, iy = mt.fit_isotonic([score[i] for i in even], [male[i] for i in even])
check(worst(mt.apply_isotonic(ix, iy, [score[i] for i in odd]), f["isotonic"]["check"]) < 1e-6, "isotonic on the odd rows")
probabilities = f["multiclass_probabilities"]
truth = f["multiclass_truth"]
fit = mt.fit_temperature(probabilities, truth, rows=list(range(0, len(truth), 2)))
check(abs(fit["temperature"] - f["temperature"]["T"]) < 1e-5, "temperature %.6f (scipy %.6f)" % (fit["temperature"], f["temperature"]["T"]))
check(abs(fit["nll_before"] - f["temperature"]["nll_before"]) < 1e-6 and abs(fit["nll_after"] - f["temperature"]["nll_after"]) < 1e-6,
      "NLL before and after")
scaled = mt.with_temperature(probabilities, fit["temperature"])


def top(row):
    return max(range(len(row)), key=lambda k: row[k])


check(len(scaled) == len(probabilities) and all(abs(sum(row) - 1.0) < 1e-5 for row in scaled)
      and all(top(a) == top(b) for a, b in zip(scaled, probabilities)), "a temperature keeps every prediction")
check(raises(lambda: mt.fit_temperature(probabilities, truth[:-1]), "one class per row"), "truth must match the rows")

check.finish("pycyxwiz.model")
