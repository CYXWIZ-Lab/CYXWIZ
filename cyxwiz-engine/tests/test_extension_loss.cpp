// Loss nodes from extensions (TOFIX125 P7 steps 7.2 and 7.3): a loss
// registered in ExtensionNodeRegistry is the graph's training objective. It
// compiles as the loss (not a layer), says what its targets are, is built by
// its factory inside a guard, and trains from a Data Input:
//   Data Input -> Split -> Loader -> Dense 16 -> ReLU -> Dense k -> <loss> -> Adam
// The stand-in losses here compute what the built-in cross entropy and MSE
// compute, so each run must match the built-in run epoch by epoch.
// No GUI, no Python: the provider is this file.
#include "../src/core/execution_placement_plan.h"
#include "../src/core/extension_node_metadata.h"
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_training_job.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <cyxwiz/losses/loss_base.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kProvider = "test.losses";
constexpr int kLossNodeId = 7;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

std::vector<float> HostValues(const cyxwiz::Tensor& tensor) {
    const float* data = tensor.ReadData<float>();
    return std::vector<float>(data, data + tensor.NumElements());
}

cyxwiz::Tensor Scalar(float value) { return cyxwiz::Tensor(std::vector<size_t>{1}, &value); }

// Mean over rows of -sum(t * log softmax(p)); dL/dp = (softmax(p) - t) / rows.
class SoftmaxCrossEntropy : public cyxwiz::Loss {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& p, const cyxwiz::Tensor& t) override {
        const auto probabilities = Softmax(p);
        const auto targets = HostValues(t);
        const size_t width = p.Shape().back();
        double total = 0.0;
        for (size_t i = 0; i < probabilities.size(); ++i) {
            total -= targets[i] * std::log(std::max(probabilities[i], 1e-12f));
        }
        return Scalar(static_cast<float>(total / (probabilities.size() / width)));
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& p, const cyxwiz::Tensor& t) override {
        auto gradient = Softmax(p);
        const auto targets = HostValues(t);
        const float rows = static_cast<float>(gradient.size() / p.Shape().back());
        for (size_t i = 0; i < gradient.size(); ++i) gradient[i] = (gradient[i] - targets[i]) / rows;
        return cyxwiz::Tensor(p.Shape(), gradient.data());
    }
    std::string GetName() const override { return "SoftmaxCrossEntropy"; }

private:
    static std::vector<float> Softmax(const cyxwiz::Tensor& p) {
        auto values = HostValues(p);
        const size_t width = p.Shape().back();
        for (size_t row = 0; row < values.size() / width; ++row) {
            float* r = values.data() + row * width;
            float top = r[0];
            for (size_t j = 1; j < width; ++j) top = std::max(top, r[j]);
            double sum = 0.0;
            for (size_t j = 0; j < width; ++j) sum += (r[j] = std::exp(r[j] - top));
            for (size_t j = 0; j < width; ++j) r[j] = static_cast<float>(r[j] / sum);
        }
        return values;
    }
};

// mean((p - t)^2); dL/dp = 2 (p - t) / n.
class MeanSquaredError : public cyxwiz::Loss {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& p, const cyxwiz::Tensor& t) override {
        const auto a = HostValues(p);
        const auto b = HostValues(t);
        double total = 0.0;
        for (size_t i = 0; i < a.size(); ++i) total += (a[i] - b[i]) * (a[i] - b[i]);
        return Scalar(static_cast<float>(total / a.size()));
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& p, const cyxwiz::Tensor& t) override {
        auto a = HostValues(p);
        const auto b = HostValues(t);
        for (size_t i = 0; i < a.size(); ++i) a[i] = 2.0f * (a[i] - b[i]) / static_cast<float>(a.size());
        return cyxwiz::Tensor(p.Shape(), a.data());
    }
    std::string GetName() const override { return "MeanSquaredError"; }
};

// Its gradient drops the last column: not the predictions' shape.
class WrongGradient : public SoftmaxCrossEntropy {
public:
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& p, const cyxwiz::Tensor&) override {
        auto shape = p.Shape();
        shape.back() -= 1;
        return cyxwiz::Tensor(shape);
    }
};

// Gives the predictions back instead of one value.
class NotOneValue : public SoftmaxCrossEntropy {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& p, const cyxwiz::Tensor&) override { return p; }
};

enum class Kind { CrossEntropy, MSE, WrongGradient, NotOneValue };

class Factory : public cyxwiz::ITensorLossFactory {
public:
    explicit Factory(Kind kind) : kind_(kind) {}
    std::unique_ptr<cyxwiz::Loss> CreateLoss(const std::map<std::string, std::string>&,
                                             std::string&) const override {
        switch (kind_) {
            case Kind::CrossEntropy: return std::make_unique<SoftmaxCrossEntropy>();
            case Kind::MSE: return std::make_unique<MeanSquaredError>();
            case Kind::WrongGradient: return std::make_unique<WrongGradient>();
            case Kind::NotOneValue: return std::make_unique<NotOneValue>();
        }
        return nullptr;
    }

private:
    Kind kind_;
};

void Register(const std::string& type_name, Kind kind, cyxwiz::ExtensionLossTarget target) {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    auto& d = registration.descriptor;
    d.provider_id = kProvider;
    d.type_name = type_name;
    d.version = "1.0.0";
    d.kind = cyxwiz::ExtensionNodeKind::Loss;
    d.loss_target = target;
    d.metadata.name = "My " + type_name;
    cyxwiz::SetExtensionLossPorts(d.metadata);
    registration.loss_factory = std::make_shared<Factory>(kind);
    std::string error;
    if (!cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error)) {
        std::cerr << "register failed: " << error << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// Rows of four features: classes 0..2 (one feature high per class), and a
// target y = x0 + 2 x1 - x2 for regression.
bool WriteRows(const fs::path& path, int rows) {
    arrow::DoubleBuilder features[4];
    arrow::Int64Builder labels;
    arrow::DoubleBuilder targets;
    uint32_t state = 3u;
    const auto noise = [&state] {
        state = state * 1664525u + 1013904223u;
        return (static_cast<double>(state >> 8) / 16777216.0 - 0.5) * 0.6;
    };
    for (int row = 0; row < rows; ++row) {
        const int label = row % 3;
        double x[4];
        for (int column = 0; column < 4; ++column) {
            x[column] = (column == label ? 2.0 : 0.0) + noise();
            if (!features[column].Append(x[column]).ok()) return false;
        }
        if (!labels.Append(label).ok() || !targets.Append(x[0] + 2.0 * x[1] - x[2]).ok()) return false;
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(6);
    for (int column = 0; column < 4; ++column) {
        if (!features[column].Finish(&arrays[column]).ok()) return false;
    }
    if (!labels.Finish(&arrays[4]).ok() || !targets.Finish(&arrays[5]).ok()) return false;
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("x0", arrow::float64()), arrow::field("x1", arrow::float64()),
                       arrow::field("x2", arrow::float64()), arrow::field("x3", arrow::float64()),
                       arrow::field("class", arrow::int64()), arrow::field("y", arrow::float64())}),
        arrays);
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 100).ok() &&
           (*out)->Close().ok();
}

// The dataset carries both targets; each graph keeps the one it learns.
nlohmann::json Graph(bool regression, const std::string& extension_loss, gui::NodeType builtin_loss) {
    using gui::NodeType;
    nlohmann::json nodes = nlohmann::json::array();
    const auto add = [&nodes](int id, NodeType type, const std::string& name, nlohmann::json parameters) {
        nodes.push_back({{"id", id}, {"type", static_cast<int>(type)}, {"name", name}, {"parameters", parameters}});
    };
    add(1, NodeType::DataInput, "Rows",
        {{"dataset_name", "loss_rows"}, {"file_category", "tabular"}, {"file_type", "parquet"},
         {"label_column", regression ? "y" : "class"},
         {"feature_columns", "x0,x1,x2,x3"}});
    add(2, NodeType::DataSplit, "Split",
        {{"seed", "42"}, {"stratified", "false"}, {"train_ratio", "0.8"}, {"val_ratio", "0.1"}, {"test_ratio", "0.1"}});
    add(3, NodeType::DataLoader, "Loader",
        {{"batch_size", "32"}, {"epochs", "4"}, {"shuffle", "true"}, {"seed", "11"}, {"model_seed", "7"}});
    add(10, NodeType::Dense, "Dense 16", {{"units", "16"}});
    add(11, NodeType::ReLU, "ReLU", nlohmann::json::object());
    add(12, NodeType::Dense, "Head", {{"units", regression ? "1" : "3"}});
    if (extension_loss.empty()) {
        add(kLossNodeId, builtin_loss, "Built-in loss", nlohmann::json::object());
    } else {
        nodes.push_back(
            {{"id", kLossNodeId}, {"type", static_cast<int>(NodeType::PluginCustom)},
             {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", "My " + extension_loss},
             {"parameters", nlohmann::json::object()},
             {"extension",
              {{"contract", 1}, {"type_id", std::string(kProvider) + ":" + extension_loss}, {"version", "1.0.0"},
               {"content_hash", ""},
               {"inputs", nlohmann::json::array({{{"name", "Predictions"}, {"type", "Tensor"}},
                                                  {{"name", "Targets"}, {"type", "Labels"}}})},
               {"outputs", nlohmann::json::array({{{"name", "Loss"}, {"type", "Loss"}}})}}}});
    }
    add(8, NodeType::Adam, "Adam", {{"learning_rate", "0.01"}});
    add(9, NodeType::Output, "Output", regression ? nlohmann::json::object() : nlohmann::json{{"num_classes", "3"}});
    nlohmann::json links = nlohmann::json::array();
    int link_id = 1;
    const auto link = [&](int from, int from_index, int to, int to_index) {
        links.push_back({{"id", link_id}, {"from_node", from}, {"from_pin", 100 + link_id}, {"from_pin_index", from_index},
                         {"to_node", to}, {"to_pin", 200 + link_id}, {"to_pin_index", to_index}, {"link_type", 0}});
        ++link_id;
    };
    link(1, 0, 2, 0);
    link(2, 0, 3, 0);
    link(3, 0, 10, 0);
    link(10, 0, 11, 0);
    link(11, 0, 12, 0);
    link(12, 0, kLossNodeId, 0);
    link(3, 1, kLossNodeId, 1);
    link(kLossNodeId, 0, 8, 0);
    link(8, 0, 9, 0);
    return {{"version", "1.0"}, {"data_boundary_version", 2}, {"nodes", nodes}, {"links", links}};
}

cyxwiz::TrainingConfiguration Compile(const nlohmann::json& graph) {
    cyxwiz::GraphDocument document;
    std::string error;
    if (!cyxwiz::ParseGraphDocument(graph.dump(), document, error)) {
        Check(false, "the graph loads: " + error);
        return {};
    }
    return cyxwiz::GraphCompiler{}.Compile(document.nodes, document.links, true);
}

std::string Issues(const cyxwiz::TrainingConfiguration& config) {
    std::string text;
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error) text += " [" + issue.message + "]";
    }
    return text;
}

struct Outcome {
    bool ok = false;
    std::string error;
    std::vector<float> losses;
    float last_accuracy = 0.0f;
};

Outcome Train(const nlohmann::json& graph, const fs::path& parquet, const fs::path& checkpoints) {
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = graph.dump();
    request.dataset_files["loss_rows"] = parquet.string();
    request.checkpoint_dir_override = checkpoints.string();
    Outcome outcome;
    cyxwiz::GraphTrainingJobCallbacks callbacks;
    callbacks.on_epoch = [&outcome](int, float train_loss, float train_accuracy, float, float, float) {
        outcome.losses.push_back(train_loss);
        outcome.last_accuracy = train_accuracy;
    };
    const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
    outcome.ok = result.ok;
    outcome.error = result.error;
    return outcome;
}

std::string Describe(const Outcome& outcome) {
    if (!outcome.ok) return "failed: " + outcome.error;
    std::string text = "losses";
    for (float loss : outcome.losses) text += " " + std::to_string(loss);
    return text + ", train accuracy " + std::to_string(outcome.last_accuracy);
}

// The extension run follows the built-in run: same model, same batches, the
// same loss value and gradient up to float rounding.
void CheckSameRun(const Outcome& extension, const Outcome& builtin, const std::string& what) {
    bool same = extension.ok && builtin.ok && extension.losses.size() == builtin.losses.size() &&
                !extension.losses.empty();
    float worst = 0.0f;
    for (size_t i = 0; same && i < extension.losses.size(); ++i) {
        worst = std::max(worst, std::fabs(extension.losses[i] - builtin.losses[i]));
    }
    same = same && worst < 1e-3f;
    Check(same, what + " (largest epoch-loss difference " + std::to_string(worst) + "); extension: " +
                    Describe(extension) + "; built-in: " + Describe(builtin));
}

}  // namespace

int main() {
    const fs::path work = fs::temp_directory_path() / "cyxwiz_extension_loss_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "rows.parquet";
    if (!WriteRows(parquet, 600)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    using Target = cyxwiz::ExtensionLossTarget;
    Register("CrossEntropy", Kind::CrossEntropy, Target::Categorical);
    Register("MSE", Kind::MSE, Target::Continuous);
    Register("WrongGradient", Kind::WrongGradient, Target::Categorical);
    Register("NotOneValue", Kind::NotOneValue, Target::Categorical);

    std::cout << "an extension loss compiles as the graph's loss\n";
    {
        const auto config = Compile(Graph(false, "CrossEntropy", gui::NodeType::CrossEntropyLoss));
        Check(config.is_valid, "the graph is valid" + Issues(config));
        Check(config.loss_type == gui::NodeType::PluginCustom && config.loss_node_id == kLossNodeId &&
                  config.loss_extension_type_id == "test.losses:CrossEntropy" &&
                  config.loss_extension_version == "1.0.0",
              "the configuration names the extension loss");
        Check(config.GetLossName() == "My CrossEntropy", "its name is the node's: " + config.GetLossName());
        Check(config.target.value_kind == cyxwiz::TargetValueKind::Categorical &&
                  !cyxwiz::UsesContinuousTargetMetrics(config),
              "categorical targets: classification metrics");
        bool layer = false;
        for (const auto& compiled : config.layers) layer = layer || compiled.node_id == kLossNodeId;
        Check(!layer, "the loss node is not a layer of the model");
        Check(config.output_size == 3, "the head's width is the output size");
        Check(cyxwiz::IsSupportedTrainingMetricContract(config), "the metric contract is supported");
        cyxwiz::ExecutionDeviceContext context;
        const auto plan = cyxwiz::BuildExecutionPlacementPlan(config, context);
        bool placed = false;
        for (const auto& entry : plan.entries) {
            placed = placed || (entry.node_id == kLossNodeId && entry.node_type == "Loss.test.losses:CrossEntropy" &&
                                entry.reason_code == cyxwiz::BackendPlacementReason::ExtensionModuleCode);
        }
        Check(placed, "the placement plan reports it as the extension's own code");
        Check(plan.fatal_blockers.empty(), "nothing in the plan stops training");

        const auto regression = Compile(Graph(true, "MSE", gui::NodeType::MSELoss));
        Check(regression.is_valid, "a continuous extension loss compiles" + Issues(regression));
        Check(regression.target.value_kind == cyxwiz::TargetValueKind::Continuous &&
                  cyxwiz::UsesContinuousTargetMetrics(regression),
              "continuous targets: regression metrics");
    }

    std::cout << "code export refuses an extension loss, naming it\n";
    {
        cyxwiz::GraphDocument document;
        std::string error;
        cyxwiz::ParseGraphDocument(Graph(false, "CrossEntropy", gui::NodeType::CrossEntropyLoss).dump(), document,
                                   error);
        std::optional<std::string> refusal;
        for (const auto& node : document.nodes) {
            if (node.id == kLossNodeId) refusal = cyxwiz::ExtensionCodeExportRefusal(node);
        }
        Check(refusal && refusal->find("My CrossEntropy") != std::string::npos &&
                  refusal->find("own code") != std::string::npos,
              "refused: " + refusal.value_or("not refused"));
    }

    std::cout << "a missing extension loss is refused on the node\n";
    {
        auto graph = Graph(false, "CrossEntropy", gui::NodeType::CrossEntropyLoss);
        for (auto& node : graph["nodes"]) {
            if (node["id"] == kLossNodeId) node["extension"]["type_id"] = "test.losses:NotInstalled";
        }
        const auto config = Compile(graph);
        bool reported = false;
        for (const auto& issue : config.issues) {
            reported = reported || (issue.node_id == kLossNodeId && issue.level == cyxwiz::IssueLevel::Error &&
                                    issue.message.find("not installed") != std::string::npos);
        }
        Check(!config.is_valid && reported, "not valid, and the node says it is not installed" + Issues(config));
    }

    std::cout << "a categorical extension loss trains as the built-in cross entropy does\n";
    {
        const auto builtin = Train(Graph(false, "", gui::NodeType::CrossEntropyLoss), parquet, work / "ck_ce");
        const auto extension = Train(Graph(false, "CrossEntropy", gui::NodeType::CrossEntropyLoss), parquet,
                                     work / "ck_ext_ce");
        Check(extension.ok && extension.losses.size() == 4 && extension.losses.back() < extension.losses.front() &&
                  extension.last_accuracy > 0.9f,
              "it learns the three classes: " + Describe(extension));
        CheckSameRun(extension, builtin, "the same losses as the built-in cross entropy");
    }

    std::cout << "a continuous extension loss trains as the built-in MSE does\n";
    {
        const auto builtin = Train(Graph(true, "", gui::NodeType::MSELoss), parquet, work / "ck_mse");
        const auto extension = Train(Graph(true, "MSE", gui::NodeType::MSELoss), parquet, work / "ck_ext_mse");
        Check(extension.ok && extension.losses.size() == 4 && extension.losses.back() < extension.losses.front(),
              "the loss goes down: " + Describe(extension));
        CheckSameRun(extension, builtin, "the same losses as the built-in MSE");
    }

    std::cout << "a loss that breaks its contract stops training at the node\n";
    {
        const auto wrong = Train(Graph(false, "WrongGradient", gui::NodeType::CrossEntropyLoss), parquet,
                                 work / "ck_wrong");
        Check(!wrong.ok && wrong.error.find("My WrongGradient") != std::string::npos &&
                  wrong.error.find("expected the predictions' shape") != std::string::npos,
              "a gradient of another shape: " + wrong.error);
        const auto many = Train(Graph(false, "NotOneValue", gui::NodeType::CrossEntropyLoss), parquet,
                                work / "ck_many");
        Check(!many.ok && many.error.find("My NotOneValue") != std::string::npos &&
                  many.error.find("must return one value") != std::string::npos,
              "more than one value: " + many.error);
    }

    cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider(kProvider);
    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_extension_loss: all checks passed\n";
    return 0;
}
