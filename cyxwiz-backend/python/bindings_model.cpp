#include "bindings_model.h"

#include <pybind11/stl.h>

#include "bindings_module.h"
#include "cyxwiz/calibration.h"
#include "cyxwiz/learning_rate_range.h"
#include "cyxwiz/losses/loss_base.h"
#include "cyxwiz/model_cost.h"
#include "cyxwiz/model_interpretability.h"
#include "cyxwiz/optimizers/optimizer_base.h"
#include "cyxwiz/sequential.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pycyxwiz_bindings {
namespace {

size_t Elements(const std::vector<size_t>& shape);

// Numbers from Python with their shape: anything with the buffer protocol
// (a numpy array, array.array, memoryview) or nested lists / tuples. numpy
// is not needed.
struct Values {
    std::vector<float> data;
    std::vector<size_t> shape;
    bool integral = true;  // whole numbers given as integers (class indices)
};

template <typename T>
void ReadStrided(const py::buffer_info& info, std::vector<float>& out) {
    std::vector<py::ssize_t> index(static_cast<size_t>(info.ndim), 0);
    const size_t n = static_cast<size_t>(info.size);
    out.reserve(n);
    const char* base = static_cast<const char*>(info.ptr);
    for (size_t k = 0; k < n; ++k) {
        py::ssize_t offset = 0;
        for (py::ssize_t d = 0; d < info.ndim; ++d) offset += index[static_cast<size_t>(d)] * info.strides[static_cast<size_t>(d)];
        out.push_back(static_cast<float>(*reinterpret_cast<const T*>(base + offset)));
        for (py::ssize_t d = info.ndim - 1; d >= 0; --d) {
            if (++index[static_cast<size_t>(d)] < info.shape[static_cast<size_t>(d)]) break;
            index[static_cast<size_t>(d)] = 0;
        }
    }
}

void ReadNested(const py::handle& obj, size_t depth, Values& v) {
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
        const auto seq = py::reinterpret_borrow<py::sequence>(obj);
        const size_t n = py::len(seq);
        if (depth == v.shape.size()) v.shape.push_back(n);
        else if (v.shape[depth] != n) throw py::value_error("the nested lists are not rectangular");
        for (const auto& item : seq) ReadNested(item, depth + 1, v);
        return;
    }
    if (depth != v.shape.size()) throw py::value_error("the nested lists are not rectangular");
    v.integral = v.integral && py::isinstance<py::int_>(obj) && !py::isinstance<py::bool_>(obj);
    v.data.push_back(obj.cast<float>());
}

Values ValuesOf(const py::handle& obj) {
    Values v;
    if (PyObject_CheckBuffer(obj.ptr())) {
        const py::buffer_info info = py::reinterpret_borrow<py::buffer>(obj).request();
        for (py::ssize_t d : info.shape) v.shape.push_back(static_cast<size_t>(d));
        const std::string& f = info.format;
        const char code = f.empty() ? '?' : f.back();
        v.integral = code != 'f' && code != 'd';
        switch (code) {
            case 'f': ReadStrided<float>(info, v.data); break;
            case 'd': ReadStrided<double>(info, v.data); break;
            case 'i': ReadStrided<int32_t>(info, v.data); break;
            case 'l': ReadStrided<long>(info, v.data); break;
            case 'q': ReadStrided<long long>(info, v.data); break;
            case 'B': ReadStrided<uint8_t>(info, v.data); break;
            default: throw py::value_error("unsupported number format '" + f + "'");
        }
        return v;
    }
    ReadNested(obj, 0, v);
    return v;
}

cyxwiz::Tensor TensorOf(const Values& v, const std::vector<size_t>& shape) {
    cyxwiz::Tensor tensor(shape, cyxwiz::DataType::Float32);
    if (!v.data.empty()) std::memcpy(tensor.Data(), v.data.data(), v.data.size() * sizeof(float));
    return tensor;
}

// One sample as a batch of one: [features] -> [1, features]; values that
// already start with a batch of 1 stay.
cyxwiz::Tensor OneSample(const py::handle& x) {
    const Values v = ValuesOf(x);
    auto shape = v.shape;
    if (shape.empty() || v.data.empty()) throw py::value_error("x must hold one sample's values");
    if (!(shape.size() >= 2 && shape[0] == 1)) shape.insert(shape.begin(), 1);
    return TensorOf(v, shape);
}

// Values as nested lists of `shape` (a leading batch of one dropped).
py::object Nested(const float* data, const std::vector<size_t>& shape, size_t depth = 0) {
    if (depth == shape.size()) return py::float_(*data);
    py::list out;
    size_t step = 1;
    for (size_t d = depth + 1; d < shape.size(); ++d) step *= shape[d];
    for (size_t i = 0; i < shape[depth]; ++i) out.append(Nested(data + i * step, shape, depth + 1));
    return out;
}

py::object ListOf(const std::vector<float>& values, std::vector<size_t> shape) {
    if (shape.size() >= 2 && shape[0] == 1) shape.erase(shape.begin());
    if (values.size() != Elements(shape)) throw py::value_error("internal: values do not fill their shape");
    return Nested(values.data(), shape);
}

size_t Elements(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

double ScalarOf(const cyxwiz::Tensor& t) {
    if (t.NumElements() == 0) throw py::value_error("the loss gave no value");
    return static_cast<double>(t.ReadData<float>()[0]);
}

// Puts the model in eval mode for an inspection and back afterwards.
class EvalMode {
public:
    explicit EvalMode(cyxwiz::SequentialModel& model) : model_(model) {
        const cyxwiz::Module* first = model.GetModule(0);
        training_ = first && first->IsTraining();
        model_.SetTraining(false);
    }
    ~EvalMode() { model_.SetTraining(training_); }

private:
    cyxwiz::SequentialModel& model_;
    bool training_ = false;
};

py::dict Summary(cyxwiz::SequentialModel& model, const std::vector<size_t>& input_shape, size_t batch) {
    if (model.Size() == 0) throw py::value_error("the model has no modules");
    if (input_shape.empty() || Elements(input_shape) == 0) throw py::value_error("input_shape is one sample's shape, e.g. (784,)");
    batch = std::max<size_t>(1, batch);
    std::vector<size_t> shape{batch};
    shape.insert(shape.end(), input_shape.begin(), input_shape.end());
    cyxwiz::Tensor input = cyxwiz::Tensor::Zeros(shape, cyxwiz::DataType::Float32);

    // Each module's output for the batch, by index; a sample is the batch-first slice.
    std::map<size_t, std::vector<size_t>> outputs;
    {
        EvalMode eval(model);
        model.SetForwardObserver([&](size_t index, const cyxwiz::Module&, const cyxwiz::Tensor& out) {
            auto s = out.Shape();
            if (!s.empty() && s[0] == batch) s.erase(s.begin());
            outputs[index] = s;
        });
        try {
            model.Forward(input);
        } catch (...) {
            model.SetForwardObserver({});
            throw;
        }
        model.SetForwardObserver({});
    }

    py::list rows;
    long long total_parameters = 0, total_multiply_adds = 0;
    bool all_counted = true;
    std::vector<size_t> previous = input_shape;
    for (size_t i = 0; i < model.Size(); ++i) {
        cyxwiz::Module* module = model.GetModule(i);
        const cyxwiz::Module& real = HeldModule(*module);
        const auto out = outputs.count(i) ? outputs[i] : std::vector<size_t>{};
        long long parameters = 0;
        for (const auto& [name, tensor] : module->GetParameters()) parameters += static_cast<long long>(tensor.NumElements());
        cyxwiz::LayerCost cost;
        if (IsPythonModule(*module)) {
            cost = {false, 0, "A Python module: its cost is its own code's."};
        } else {
            cost = cyxwiz::CountMultiplyAdds(cyxwiz::ModuleCostInput(real, previous, out));
        }
        all_counted = all_counted && cost.counted;
        total_parameters += parameters;
        total_multiply_adds += cost.value;
        rows.append(py::dict("index"_a = i, "name"_a = module->GetName(), "output"_a = out, "parameters"_a = parameters,
                             "trainable"_a = module->IsTrainable(), "multiply_adds"_a = cost.value,
                             "counted"_a = cost.counted, "formula"_a = cost.formula));
        previous = out;
    }
    return py::dict("rows"_a = rows, "parameters"_a = total_parameters, "multiply_adds"_a = total_multiply_adds,
                    "all_counted"_a = all_counted, "parameter_bytes"_a = total_parameters * 4);
}

py::dict AnalysisDict(const std::vector<cyxwiz::LearningRateRangePoint>& points) {
    const auto a = cyxwiz::AnalyseLearningRateRange(points);
    if (!a.ok) return py::dict("ok"_a = false, "error"_a = a.error);
    return py::dict("ok"_a = true, "lowest"_a = a.lowest, "steepest"_a = a.steepest, "lowest_rate"_a = a.lowest_rate,
                    "lowest_loss"_a = a.lowest_loss, "steepest_rate"_a = a.steepest_rate,
                    "lowest_over_ten"_a = a.lowest_over_ten, "slopes"_a = a.slopes);
}

py::list PointList(const std::vector<cyxwiz::LearningRateRangePoint>& points) {
    py::list out;
    for (const auto& p : points)
        out.append(py::dict("learning_rate"_a = p.learning_rate, "loss"_a = p.loss, "smoothed"_a = p.smoothed));
    return out;
}

cyxwiz::LearningRateRangeSettings RangeSettings(double start, double end, int steps, double smoothing, double stop_factor) {
    if (!(start > 0.0) || !(end > start)) throw py::value_error("the rates run from start > 0 up to end > start");
    if (steps < 2) throw py::value_error("steps must be at least 2");
    cyxwiz::LearningRateRangeSettings s;
    s.enabled = true;
    s.start = start;
    s.end = end;
    s.steps = steps;
    s.smoothing = smoothing;
    s.stop_factor = stop_factor;
    return s;
}

py::dict RangeTest(cyxwiz::SequentialModel& model, cyxwiz::Loss& loss, cyxwiz::Optimizer& optimizer, const py::object& x_in,
                   const py::object& y_in, double start, double end, int steps, size_t batch_size, double smoothing,
                   double stop_factor) {
    const Values x = ValuesOf(x_in), y = ValuesOf(y_in);
    const auto& x_shape = x.shape;
    const auto& y_shape = y.shape;
    if (x_shape.empty() || y_shape.empty() || x_shape[0] != y_shape[0] || x_shape[0] == 0)
        throw py::value_error("x and y need the same number of rows (axis 0)");
    batch_size = std::max<size_t>(1, std::min(batch_size, x_shape[0]));
    cyxwiz::LearningRateRangeTest test(RangeSettings(start, end, steps, smoothing, stop_factor));

    // The weights come back afterwards; the optimizer keeps its state (pass a new one).
    std::map<std::string, cyxwiz::Tensor> saved;
    for (const auto& [name, tensor] : model.GetParameters()) saved.emplace(name, tensor.Clone());
    const double rate_before = optimizer.GetLearningRate();
    const size_t x_row = Elements(x_shape) / x_shape[0], y_row = Elements(y_shape) / y_shape[0];
    const size_t rows = x_shape[0];
    model.SetTraining(true);
    std::string error;
    try {
        size_t at = 0;
        while (!test.Finished()) {
            if (PyErr_CheckSignals() != 0) throw py::error_already_set();
            const size_t n = std::min(batch_size, rows - at);
            auto xs = x_shape, ys = y_shape;
            xs[0] = ys[0] = n;
            // Integer targets (class indices) stay integers, as CrossEntropyLoss reads them.
            cyxwiz::Tensor xb(xs, cyxwiz::DataType::Float32);
            cyxwiz::Tensor yb(ys, y.integral ? cyxwiz::DataType::Int32 : cyxwiz::DataType::Float32);
            std::memcpy(xb.Data(), x.data.data() + at * x_row, n * x_row * sizeof(float));
            if (y.integral) {
                auto* target = static_cast<int32_t*>(yb.Data());
                for (size_t k = 0; k < n * y_row; ++k) target[k] = static_cast<int32_t>(y.data[at * y_row + k]);
            } else {
                std::memcpy(yb.Data(), y.data.data() + at * y_row, n * y_row * sizeof(float));
            }
            at = at + n >= rows ? 0 : at + n;
            optimizer.SetLearningRate(test.CurrentLearningRate());
            const cyxwiz::Tensor prediction = model.Forward(xb);
            const double value = ScalarOf(loss.Forward(prediction, yb));
            model.Backward(loss.Backward(prediction, yb));
            model.UpdateParameters(&optimizer);
            test.Record(value);
        }
    } catch (const py::error_already_set&) {
        model.SetParameters(saved);
        optimizer.SetLearningRate(rate_before);
        throw;
    } catch (const std::exception& e) {
        error = e.what();
    }
    model.SetParameters(saved);
    optimizer.SetLearningRate(rate_before);
    if (!error.empty()) throw py::value_error(error);
    return py::dict("points"_a = PointList(test.Points()), "stop_reason"_a = test.StopReason(),
                    "analysis"_a = AnalysisDict(test.Points()));
}

py::dict SaliencyDict(const cyxwiz::SaliencyMap& s, bool smooth) {
    if (!s.success) throw py::value_error(s.error_message);
    const auto& values = smooth ? s.smoothgrad : s.gradients;
    return py::dict("gradients"_a = ListOf(values, s.shape), "absolute"_a = ListOf(s.absolute_gradients, s.shape),
                    "target"_a = s.target_class);
}

std::vector<size_t> RowsOrAll(const std::optional<std::vector<size_t>>& rows, size_t n) {
    if (rows) return *rows;
    std::vector<size_t> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = i;
    return all;
}

}  // namespace

void BindModelTools(py::module_& m) {
    auto mm = m.def_submodule("model", "The model's summary, learning rate range test, explanations and calibration "
                                       "(the Engine's Tools > Model and Results > Test > Calibration)");

    mm.def("summary", &Summary, py::arg("model"), py::arg("input_shape"), py::arg("batch") = 1,
           "Each module's output shape, parameters and multiply-adds for one sample (as PyTorch's "
           "torch.utils.flop_counter counts FLOPs, halved). input_shape is one sample's shape. Returns a dict: "
           "rows (index, name, output, parameters, trainable, multiply_adds, counted, formula), parameters, "
           "multiply_adds, all_counted, parameter_bytes (float32).");

    py::class_<cyxwiz::LearningRateRangeTest>(mm, "LearningRateRange",
        "The range test's rules without training: the rate of each step (start x (end / start)^(step / (steps - 1))), "
        "the smoothed loss, the stop once it passes stop_factor x its lowest. record() each step's loss.")
        .def(py::init([](double start, double end, int steps, double smoothing, double stop_factor) {
                 return cyxwiz::LearningRateRangeTest(RangeSettings(start, end, steps, smoothing, stop_factor));
             }),
             py::arg("start") = 1e-7, py::arg("end") = 10.0, py::arg("steps") = 100, py::arg("smoothing") = 0.05,
             py::arg("stop_factor") = 4.0,
             "Rates from start to end over steps steps; smoothing weighs the newest loss in the running average")
        .def("learning_rate_at", &cyxwiz::LearningRateRangeTest::LearningRateAt, py::arg("step"), "The rate of a step (0 = start)")
        .def_property_readonly("current_learning_rate", &cyxwiz::LearningRateRangeTest::CurrentLearningRate,
                               "The rate for the next step")
        .def("record", &cyxwiz::LearningRateRangeTest::Record, py::arg("loss"),
             "The loss of the step just taken at current_learning_rate; True when the test is over")
        .def_property_readonly("finished", &cyxwiz::LearningRateRangeTest::Finished, "True once the steps ran out or the loss blew up")
        .def_property_readonly("stop_reason", &cyxwiz::LearningRateRangeTest::StopReason, "Why the test ended, in words")
        .def_property_readonly("points", [](const cyxwiz::LearningRateRangeTest& t) { return PointList(t.Points()); },
                               "Each step: learning_rate, loss, smoothed")
        .def("analyse", [](const cyxwiz::LearningRateRangeTest& t) { return AnalysisDict(t.Points()); },
             "Where to train: lowest (index, rate, loss), steepest (the most negative slope of the smoothed loss "
             "against log10(rate) up to the lowest, numpy.gradient), lowest_over_ten, slopes");

    mm.def("lr_range_test", &RangeTest, py::arg("model"), py::arg("loss"), py::arg("optimizer"), py::arg("x"),
           py::arg("y"), py::arg("start") = 1e-7, py::arg("end") = 10.0, py::arg("steps") = 100,
           py::arg("batch_size") = 64, py::arg("smoothing") = 0.05, py::arg("stop_factor") = 4.0,
           "A learning rate range test (Smith 2017): trains on batches of x, y (in order, round again at the end; "
           "integer y, such as class indices for CrossEntropyLoss, stays integer) "
           "raising the optimizer's rate every step, then puts the model's weights and the optimizer's rate back "
           "(its moments stay changed: pass a new optimizer). Returns points, stop_reason and analysis.");

    mm.def("saliency", [](cyxwiz::SequentialModel& model, const py::object& x, int target) {
               EvalMode eval(model);
               return SaliencyDict(cyxwiz::ModelInterpretability::ComputeSaliencyMap(
                                       cyxwiz::ModelInterpretability::FunctionsOf(model), OneSample(x), target),
                                   false);
           },
           py::arg("model"), py::arg("x"), py::arg("target") = -1,
           "d score / d input for one sample (target -1 = the predicted class). Returns gradients, absolute "
           "(|gradients|, Captum's Saliency) and target, in x's shape with a batch of one.");

    mm.def("smooth_grad", [](cyxwiz::SequentialModel& model, const py::object& x, int target, int samples, double noise,
                             unsigned long long seed) {
               EvalMode eval(model);
               return SaliencyDict(cyxwiz::ModelInterpretability::ComputeSmoothGrad(
                                       cyxwiz::ModelInterpretability::FunctionsOf(model), OneSample(x), target,
                                       samples, static_cast<float>(noise), seed),
                                   true);
           },
           py::arg("model"), py::arg("x"), py::arg("target") = -1, py::arg("samples") = 50, py::arg("noise") = 0.1,
           py::arg("seed") = 0ULL, "Saliency averaged over noisy copies of x (gradients is the average).");

    mm.def("integrated_gradients", [](cyxwiz::SequentialModel& model, const py::object& x, const py::object& baseline,
                                      int target, int steps) {
               EvalMode eval(model);
               const cyxwiz::Tensor input = OneSample(x);
               const cyxwiz::Tensor base = !baseline.is_none() ? OneSample(baseline)
                                                               : cyxwiz::Tensor::Zeros(input.Shape(), cyxwiz::DataType::Float32);
               if (base.Shape() != input.Shape()) throw py::value_error("baseline must have x's shape");
               const auto r = cyxwiz::ModelInterpretability::ComputeIntegratedGradients(
                   cyxwiz::ModelInterpretability::FunctionsOf(model), input, base, target, steps);
               if (!r.success) throw py::value_error(r.error_message);
               return py::dict("attributions"_a = ListOf(r.attributions, r.shape), "target"_a = r.target_class,
                               "steps"_a = r.steps, "score"_a = r.score, "baseline_score"_a = r.baseline_score,
                               "attribution_sum"_a = r.attribution_sum);
           },
           py::arg("model"), py::arg("x"), py::arg("baseline") = py::none(), py::arg("target") = -1,
           py::arg("steps") = 64,
           "Integrated Gradients from baseline (zeros by default) in steps midpoint steps, all in one batch. "
           "attribution_sum is about score - baseline_score (the completeness check).");

    mm.def("reliability", [](const std::vector<double>& confidence, const std::vector<int>& correct, int bins) {
               const auto r = cyxwiz::ReliabilityOf(confidence, correct, bins);
               py::list out;
               for (const auto& b : r.bins)
                   out.append(py::dict("low"_a = b.low, "high"_a = b.high, "count"_a = b.count,
                                       "confidence"_a = b.confidence, "accuracy"_a = b.accuracy));
               return py::dict("bins"_a = out, "ece"_a = r.ece, "rows"_a = r.rows);
           },
           py::arg("confidence"), py::arg("correct"), py::arg("bins") = 10,
           "How often the model is right when it says p: equal bands of [0, 1] and the expected calibration error.");

    mm.def("fit_temperature", [](const py::object& probabilities, const std::vector<int>& truth,
                                 std::optional<std::vector<size_t>> rows) {
               const Values p = ValuesOf(probabilities);
               if (p.shape.size() != 2) throw py::value_error("probabilities is [rows, classes]");
               const size_t n = p.shape[0];
               const int classes = static_cast<int>(p.shape[1]);
               if (truth.size() != n) throw py::value_error("truth needs one class per row");
               const auto fit = cyxwiz::FitTemperature(p.data, classes, truth, RowsOrAll(rows, n));
               return py::dict("temperature"_a = fit.temperature, "nll_before"_a = fit.nll_before,
                               "nll_after"_a = fit.nll_after);
           },
           py::arg("probabilities"), py::arg("truth"), py::arg("rows") = py::none(),
           "One temperature T minimizing the mean negative log-likelihood of softmax(log p / T) over rows (all by "
           "default). It changes no prediction.");

    mm.def("with_temperature", [](const py::object& probabilities, double temperature) {
               const Values p = ValuesOf(probabilities);
               if (p.shape.empty() || p.shape.size() > 2) throw py::value_error("probabilities is a row or [rows, classes]");
               const int classes = static_cast<int>(p.shape.back());
               const size_t n = p.data.size() / static_cast<size_t>(std::max(1, classes));
               std::vector<float> out;
               out.reserve(p.data.size());
               for (size_t r = 0; r < n; ++r)
                   for (double v : cyxwiz::WithTemperature(p.data.data() + r * classes, classes, temperature))
                       out.push_back(static_cast<float>(v));
               return Nested(out.data(), p.shape);
           },
           py::arg("probabilities"), py::arg("temperature"), "Probabilities (a row or [rows, classes]) after a temperature.");

    mm.def("fit_platt", [](const std::vector<double>& scores, const std::vector<int>& labels) {
               const auto fit = cyxwiz::FitPlatt(scores, labels);
               return py::make_tuple(fit.a, fit.b);
           },
           py::arg("scores"), py::arg("labels"),
           "Platt's sigmoid 1 / (1 + exp(a s + b)) of a yes / no score, as scikit-learn's _sigmoid_calibration: (a, b).");

    mm.def("apply_platt", [](double a, double b, const std::vector<double>& scores) {
               std::vector<double> out(scores.size());
               for (size_t i = 0; i < out.size(); ++i) out[i] = cyxwiz::ApplyPlatt({a, b}, scores[i]);
               return out;
           },
           py::arg("a"), py::arg("b"), py::arg("scores"), "Platt's probabilities of the scores, from fit_platt's (a, b).");

    mm.def("fit_isotonic", [](const std::vector<double>& scores, const std::vector<int>& labels) {
               const auto fit = cyxwiz::FitIsotonic(scores, labels);
               return py::make_tuple(fit.x, fit.y);
           },
           py::arg("scores"), py::arg("labels"),
           "Isotonic regression of the label on the score, as scikit-learn's IsotonicRegression(out_of_bounds='clip'): "
           "the fitted points (x, y).");

    mm.def("apply_isotonic", [](const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& scores) {
               if (x.size() != y.size()) throw py::value_error("x and y are the fit's points");
               const cyxwiz::IsotonicFit fit{x, y};
               std::vector<double> out(scores.size());
               for (size_t i = 0; i < out.size(); ++i) out[i] = cyxwiz::ApplyIsotonic(fit, scores[i]);
               return out;
           },
           py::arg("x"), py::arg("y"), py::arg("scores"),
           "The isotonic fit's probabilities of the scores: linear between fit_isotonic's points, clipped outside them.");
}

}  // namespace pycyxwiz_bindings
