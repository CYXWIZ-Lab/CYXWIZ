// One group-training round of the Engine against the PyTorch reference
// (tests/computation_truth/reference/round_training_reference.py): three
// workers start from the same state, train 7, 3 and 1 steps with Adam and
// return their updates.

#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/round_training_session.h"
#include "../src/core/round_update_codec.h"
#include "../src/core/graph_node_factory.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
    std::cout << "  ok   " << message << "\n";
}

void SelectArrayFireCpu() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(devices.begin(), devices.end(), [](const cyxwiz::DeviceInfo& device) {
        return device.type == cyxwiz::DeviceType::CPU;
    });
    if (cpu == devices.end()) {
        std::cerr << "FAIL: an ArrayFire CPU route is required\n";
        std::exit(1);
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot(devices);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(cyxwiz::DeviceType::CPU, cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
}

// A worker's batches as a Parquet file: 8 float features and the class, in batch order.
bool WriteBatches(const fs::path& path, const json& batches, int repeat = 1) {
    arrow::FloatBuilder features[8];
    arrow::Int64Builder labels;
    for (int pass = 0; pass < repeat; ++pass) {
        for (const json& batch : batches) {
            const std::vector<float> x = batch.at("x").at("values").get<std::vector<float>>();
            const std::vector<int> y = batch.at("y").get<std::vector<int>>();
            for (size_t row = 0; row < y.size(); ++row) {
                for (size_t column = 0; column < 8; ++column) {
                    if (!features[column].Append(x[row * 8 + column]).ok()) return false;
                }
                if (!labels.Append(y[row]).ok()) return false;
            }
        }
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(9);
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int column = 0; column < 8; ++column) {
        if (!features[column].Finish(&arrays[column]).ok()) return false;
        fields.push_back(arrow::field("x" + std::to_string(column), arrow::float32()));
    }
    if (!labels.Finish(&arrays[8]).ok()) return false;
    fields.push_back(arrow::field("class", arrow::int64()));
    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 1024).ok() &&
           (*out)->Close().ok();
}

// Data Input -> Split -> Loader -> Dense 144 -> ReLU -> Dense 4 -> cross entropy -> Adam -> Output.
std::string Graph(int hidden, const nlohmann::json& optimizer_extra = nlohmann::json::object(), bool step_schedule = false) {
    using gui::NodeType;
    json nodes = json::array();
    const auto add = [&nodes](int id, NodeType type, const char* name, json parameters) {
        nodes.push_back({{"id", id}, {"type", static_cast<int>(type)}, {"name", name}, {"parameters", parameters}});
    };
    add(1, NodeType::DataInput, "Rows",
        {{"dataset_name", "round_rows"}, {"file_category", "tabular"}, {"file_type", "parquet"}, {"label_column", "class"}});
    add(2, NodeType::DataSplit, "Split",
        {{"seed", "42"}, {"stratified", "false"}, {"train_ratio", "0.8"}, {"val_ratio", "0.1"}, {"test_ratio", "0.1"}});
    add(3, NodeType::DataLoader, "Loader",
        {{"batch_size", "8"}, {"epochs", "2"}, {"shuffle", "true"}, {"seed", "11"}, {"model_seed", "7"}});
    add(4, NodeType::Dense, "Hidden", {{"units", std::to_string(hidden)}});
    add(5, NodeType::ReLU, "ReLU", json::object());
    add(6, NodeType::Dense, "Classes", {{"units", "4"}});
    add(7, NodeType::CrossEntropyLoss, "Cross entropy", json::object());
    json adam = {{"learning_rate", "0.01"}};
    adam.update(optimizer_extra);
    add(8, NodeType::Adam, "Adam", adam);
    add(9, NodeType::Output, "Output", {{"num_classes", "4"}});
    if (step_schedule) add(10, NodeType::StepLR, "Step", {{"step_size", "2"}, {"gamma", "0.5"}});
    json links = json::array();
    int link_id = 1;
    const auto link = [&](int from, int from_index, int to, int to_index) {
        links.push_back({{"id", link_id}, {"from_node", from}, {"from_pin", 100 + link_id}, {"from_pin_index", from_index},
                         {"to_node", to}, {"to_pin", 200 + link_id}, {"to_pin_index", to_index}, {"link_type", 0}});
        ++link_id;
    };
    link(1, 0, 2, 0);
    link(2, 0, 3, 0);
    link(3, 0, 4, 0);
    link(4, 0, 5, 0);
    link(5, 0, 6, 0);
    link(6, 0, 7, 0);
    link(3, 1, 7, 1);
    link(7, 0, 8, 0);
    link(8, 0, 9, 0);
    if (step_schedule) link(8, 0, 10, 0);
    return json{{"version", "1.0"}, {"data_boundary_version", 2}, {"nodes", nodes}, {"links", links}}.dump();
}

// The reference lists fc1.weight, fc1.bias, fc2.weight, fc2.bias; the Engine
// orders by layer, then parameter name. Where each reference value sits in
// the Engine's exchanged state.
std::vector<size_t> ReferenceToEngine(const json& reference_layout, const cyxwiz::ModelStateLayout& layout) {
    std::vector<std::string> engine_layers;
    for (const auto& segment : layout.segments) {
        const std::string layer = segment.name.substr(0, segment.name.find('.'));
        if (engine_layers.empty() || engine_layers.back() != layer) engine_layers.push_back(layer);
    }
    std::vector<std::string> reference_layers;
    std::vector<size_t> position;
    for (const json& reference : reference_layout) {
        const std::string name = reference.at("name").get<std::string>();
        const std::string layer = name.substr(0, name.find('.'));
        if (reference_layers.empty() || reference_layers.back() != layer) reference_layers.push_back(layer);
        const std::string wanted = engine_layers.at(reference_layers.size() - 1) + name.substr(name.find('.'));
        const auto segment = std::find_if(layout.segments.begin(), layout.segments.end(),
                                          [&](const auto& candidate) { return candidate.name == wanted; });
        if (segment == layout.segments.end() || segment->count != reference.at("count").get<size_t>()) {
            std::cerr << "FAIL: the Engine has no segment " << wanted << " of the reference's size\n";
            std::exit(1);
        }
        for (size_t i = 0; i < segment->count; ++i) position.push_back(segment->exchanged_offset + i);
    }
    return position;
}

// The exchanged values of a full state.
std::vector<float> Exchanged(const cyxwiz::ModelStateLayout& layout, const std::vector<float>& full) {
    std::vector<float> values(layout.exchanged_count);
    for (const auto& segment : layout.segments) {
        if (segment.role == cyxwiz::ModelStateRole::Frozen) continue;
        std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(segment.offset), segment.count,
                    values.begin() + static_cast<std::ptrdiff_t>(segment.exchanged_offset));
    }
    return values;
}

// A language model's round on token windows: opens on `sample`, loads `start`
// when given, trains one round over `leases`.
struct LanguageRound {
    std::vector<float> start;       // the session's own starting state
    cyxwiz::ModelStateLayout layout;
    cyxwiz::RoundOutcome outcome;
};
LanguageRound TrainLanguageRound(const std::string& graph, const fs::path& sample, const cyxwiz::RoundData& leases,
                                 const std::vector<float>* start, const std::string& what, uint32_t worker = 0) {
    LanguageRound run;
    SelectArrayFireCpu();
    cyxwiz::RoundFailure failure;
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = graph;
    request.dataset_files["tiny_causal_lm_tokens"] = sample.string();
    auto session = cyxwiz::RoundTrainingSession::Open(request, {worker, 0}, failure);
    Check(session != nullptr, what + ": the session opens " + failure.text);
    run.layout = session->Layout();
    run.start = session->InitialState();
    std::string error;
    if (start) Check(session->LoadInitialState(*start, error), what + ": the starting state is accepted " + error);
    const std::vector<float> theta = Exchanged(run.layout, start ? *start : run.start);
    cyxwiz::RoundBudget budget;
    budget.max_steps = 1000;
    SelectArrayFireCpu();
    Check(session->TrainRound({1, 0, 0}, theta, leases, budget, {}, run.outcome, failure), what + ": the round trains " + failure.text);
    return run;
}

cyxwiz::GraphTrainingJobRequest Request(const std::string& graph, const fs::path& file) {
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = graph;
    request.dataset_files["round_rows"] = file.string();
    return request;
}

}  // namespace

int main() {
    std::ifstream file(CYXWIZ_ROUND_TRAINING_FIXTURE);
    if (!file.good()) {
        std::cerr << "FAIL: the fixture should open: " << CYXWIZ_ROUND_TRAINING_FIXTURE << "\n";
        return 1;
    }
    const json fixture = json::parse(file);
    const json& round = fixture.at("rounds").at(0);   // float32 wire, no BatchNorm
    const double tolerance = fixture.at("tolerance").at("round").get<double>();
    const int hidden = round.at("layers").at(0).at("out").get<int>();
    const std::string graph = Graph(hidden);

    const fs::path work = fs::temp_directory_path() / "cyxwiz_round_training_session_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);

    std::cout << "one round against PyTorch\n";
    std::string layout_hash;
    std::vector<float> first_delta_of_fast;
    for (const json& worker : round.at("workers")) {
        const std::string name = worker.at("name").get<std::string>();
        const fs::path data = work / (name + ".parquet");
        Check(WriteBatches(data, worker.at("batches")), name + ": its batches are written");

        SelectArrayFireCpu();
        cyxwiz::RoundFailure failure;
        // The graph is compiled against one part of the job's data (here the
        // fast worker's file); the round's rows come from the worker's own.
        auto session = cyxwiz::RoundTrainingSession::Open(Request(graph, work / "fast.parquet"),
                                                          {static_cast<uint32_t>(layout_hash.empty() ? 0 : 1), 0}, failure);
        Check(session != nullptr, name + ": the session opens " + failure.text);
        const cyxwiz::ModelStateLayout& layout = session->Layout();
        if (layout_hash.empty()) layout_hash = layout.hash;
        Check(layout.hash == layout_hash, name + ": every worker builds the same layout");
        Check(session->InitialState().size() == layout.total_count && layout.total_count == layout.exchanged_count,
              name + ": the starting state covers the model");

        const std::vector<size_t> position = ReferenceToEngine(round.at("layout"), layout);
        const std::vector<float> reference_theta = round.at("theta").at("values").get<std::vector<float>>();
        Check(position.size() == reference_theta.size() && position.size() == layout.exchanged_count,
              name + ": the reference and the Engine exchange the same number of values");
        std::vector<float> theta(layout.exchanged_count);
        for (size_t i = 0; i < position.size(); ++i) theta[position[i]] = reference_theta[i];

        std::string error;
        Check(session->LoadInitialState(theta, error), name + ": the starting state is accepted " + error);
        cyxwiz::RoundData leases;
        leases.leases.push_back({data, 0, 0});
        cyxwiz::RoundBudget budget;
        budget.max_steps = worker.at("steps").get<uint64_t>();
        cyxwiz::RoundOutcome outcome;
        SelectArrayFireCpu();
        Check(session->TrainRound({1, 0, 0}, theta, leases, budget, {}, outcome, failure), name + ": the round trains " + failure.text);
        Check(outcome.steps == worker.at("steps").get<uint64_t>() && outcome.samples == worker.at("samples").get<uint64_t>(),
              name + ": " + std::to_string(outcome.steps) + " steps, " + std::to_string(outcome.samples) + " samples");
        Check(outcome.rows_done.size() == 1 && outcome.rows_done[0] == outcome.samples, name + ": the rows trained are reported");

        const std::vector<float> reference_delta = worker.at("delta").at("values").get<std::vector<float>>();
        double worst = 0.0;
        for (size_t i = 0; i < position.size(); ++i) {
            worst = std::max(worst, static_cast<double>(std::abs(outcome.delta[position[i]] - reference_delta[i])));
        }
        Check(worst <= tolerance, name + ": the update equals PyTorch's (largest difference " + std::to_string(worst) + ")");

        // The update goes through the wire codec unchanged in float32.
        cyxwiz::EncodedUpdate encoded;
        std::vector<float> decoded;
        Check(cyxwiz::EncodeUpdate(outcome.delta, layout, cyxwiz::UpdateWire::Float32, encoded, error) &&
                  cyxwiz::DecodeUpdate(encoded, layout, cyxwiz::UpdateWire::Float32, decoded, error) && decoded == outcome.delta,
              name + ": the update encodes and decodes");
        if (name == "fast") first_delta_of_fast = outcome.delta;
    }

    std::cout << "rounds on a kept model\n";
    {
        const json& fast = round.at("workers").at(0);
        const fs::path data = work / "many.parquet";
        Check(WriteBatches(data, fast.at("batches"), 400), "a larger file is written");   // 7 x 8 x 400 = 22,400 rows
        SelectArrayFireCpu();
        cyxwiz::RoundFailure failure;
        auto session = cyxwiz::RoundTrainingSession::Open(Request(graph, data), {}, failure);
        Check(session != nullptr, "the session opens " + failure.text);
        const cyxwiz::ModelStateLayout& layout = session->Layout();
        const std::vector<size_t> position = ReferenceToEngine(round.at("layout"), layout);
        const std::vector<float> reference_theta = round.at("theta").at("values").get<std::vector<float>>();
        std::vector<float> theta(layout.exchanged_count);
        for (size_t i = 0; i < position.size(); ++i) theta[position[i]] = reference_theta[i];

        cyxwiz::RoundBudget seven;
        seven.max_steps = 7;
        cyxwiz::RoundData first;
        first.leases.push_back({data, 0, 56});
        cyxwiz::RoundOutcome outcome;
        SelectArrayFireCpu();
        Check(session->TrainRound({1, 0, 0}, theta, first, seven, {}, outcome, failure), "round 1 trains " + failure.text);
        Check(outcome.delta == first_delta_of_fast, "a row lease of the first 56 rows gives the same update as the whole small file");

        // Round 2 from the same state: Adam's moments were kept, so the update differs.
        SelectArrayFireCpu();
        Check(session->TrainRound({2, 0, 0}, theta, first, seven, {}, outcome, failure), "round 2 trains " + failure.text);
        Check(outcome.delta != first_delta_of_fast, "the optimizer's state is kept between rounds");

        // Two leases, the data ends before the budget.
        cyxwiz::RoundData two;
        two.leases.push_back({data, 56, 24});
        two.leases.push_back({data, 1000, 16});
        cyxwiz::RoundBudget large;
        large.max_steps = 1000;
        SelectArrayFireCpu();
        Check(session->TrainRound({3, 0, 0}, theta, two, large, {}, outcome, failure), "a round over two leases trains " + failure.text);
        Check(outcome.steps == 5 && outcome.samples == 40 && outcome.rows_done == std::vector<uint64_t>({24, 16}),
              "it ends when its 40 rows are used: " + std::to_string(outcome.steps) + " steps");

        // A time budget cuts a round short; the rows trained are the first ones.
        cyxwiz::RoundData all;
        all.leases.push_back({data, 0, 0});
        cyxwiz::RoundBudget brief;
        brief.train_seconds = 0.25;
        SelectArrayFireCpu();
        Check(session->TrainRound({4, 0, 0}, theta, all, brief, {}, outcome, failure), "a timed round trains " + failure.text);
        Check(outcome.steps > 0 && outcome.samples == outcome.steps * 8 && outcome.samples < 22400 &&
                  outcome.rows_done == std::vector<uint64_t>({outcome.samples}),
              "it stops on time: " + std::to_string(outcome.steps) + " steps in " + std::to_string(outcome.train_seconds) + " s");

        // Stopped from outside.
        cyxwiz::RoundCallbacks stop;
        stop.should_stop = [] { return true; };
        SelectArrayFireCpu();
        Check(!session->TrainRound({5, 0, 0}, theta, all, large, stop, outcome, failure) &&
                  failure.kind == cyxwiz::TrainingFailureKind::Cancelled,
              "a stop request ends the round as cancelled");

        // Refusals.
        std::string error;
        std::vector<float> short_state(layout.total_count - 1, 0.0f);
        Check(!session->LoadInitialState(short_state, error), "a starting state of the wrong length is refused");
        Check(!session->TrainRound({6, 0, 0}, short_state, all, seven, {}, outcome, failure), "a round state of the wrong length is refused");
        cyxwiz::RoundData beyond;
        beyond.leases.push_back({data, 22390, 100});
        Check(!session->TrainRound({6, 0, 0}, theta, beyond, seven, {}, outcome, failure) &&
                  failure.kind == cyxwiz::TrainingFailureKind::DataError,
              "a lease past the end of its file is refused");
        Check(!session->TrainRound({6, 0, 0}, theta, all, cyxwiz::RoundBudget{}, {}, outcome, failure), "a round with no budget is refused");
    }

    std::cout << "the learning rate follows the group\n";
    {
        const fs::path data = work / "many.parquet";
        const auto rate_at = [&](cyxwiz::RoundTrainingSession& session, uint64_t round, uint64_t epoch, uint64_t step) {
            const std::vector<float> theta = Exchanged(session.Layout(), session.InitialState());
            cyxwiz::RoundData lease;
            lease.leases.push_back({data, 0, 8});
            cyxwiz::RoundBudget one_step;
            one_step.max_steps = 1;
            cyxwiz::RoundOutcome outcome;
            cyxwiz::RoundFailure failure;
            SelectArrayFireCpu();
            Check(session.TrainRound({round, epoch, step}, theta, lease, one_step, {}, outcome, failure),
                  "a round at epoch " + std::to_string(epoch) + ", step " + std::to_string(step) + " trains " + failure.text);
            return outcome.learning_rate;
        };
        const auto close_to = [](double actual, double expected) { return std::abs(actual - expected) <= 1e-9 + 1e-6 * expected; };

        // The optimizer's own schedule: 10 warm-up steps of 100, then a straight line down to a tenth.
        const std::string warmup = Graph(hidden, {{"lr_schedule", "warmup_linear"}, {"warmup_ratio", "0.1"}, {"min_lr_ratio", "0.1"}});
        cyxwiz::RoundFailure failure;
        SelectArrayFireCpu();
        Check(cyxwiz::RoundTrainingSession::Open(Request(warmup, data), {}, failure) == nullptr &&
                  failure.kind == cyxwiz::TrainingFailureKind::Refused,
              "a schedule by step is refused when the job gives no length");
        SelectArrayFireCpu();
        auto by_step = cyxwiz::RoundTrainingSession::Open(Request(warmup, data), {0, 100}, failure);
        Check(by_step != nullptr, "with the job's length it opens " + failure.text);
        const double peak = rate_at(*by_step, 1, 0, 10);
        Check(peak > 0.0098 && peak <= 0.01, "at the end of the warm-up the rate is at the optimizer's: " + std::to_string(peak));
        const double early = rate_at(*by_step, 2, 0, 5);
        Check(early > 0.0 && early < peak, "a position behind the clock is evaluated again: " + std::to_string(early));
        const double late = rate_at(*by_step, 3, 0, 100);
        Check(close_to(late, 0.001), "at the end of the job the rate is a tenth: " + std::to_string(late));
        Check(close_to(rate_at(*by_step, 4, 0, 5000), late), "past the end it stays there");

        // A scheduler node by epoch: halved every 2 epochs of the group.
        SelectArrayFireCpu();
        auto by_epoch = cyxwiz::RoundTrainingSession::Open(Request(Graph(hidden, json::object(), true), data), {}, failure);
        Check(by_epoch != nullptr, "a graph with a StepLR node opens " + failure.text);
        Check(close_to(rate_at(*by_epoch, 1, 0, 0), 0.01) && close_to(rate_at(*by_epoch, 2, 1, 0), 0.01),
              "in the group's first two epochs the rate is the optimizer's");
        Check(close_to(rate_at(*by_epoch, 3, 2, 0), 0.005) && close_to(rate_at(*by_epoch, 4, 5, 0), 0.0025),
              "it is halved after 2 epochs and again after 4, whatever this worker's own steps");
    }

    std::cout << "a language model on token windows\n";
    {
        const fs::path root = CYXWIZ_SOURCE_ROOT;
        const fs::path first = work / "windows_a.parquet";
        const fs::path second = work / "windows_b.parquet";
        Check(cyxwiz::test::WriteTokenWindows(root, first, 0, 8) && cyxwiz::test::WriteTokenWindows(root, second, 8, 2),
              "two parts of token windows are written");
        // The example has dropout and no model seed; a seed makes two workers comparable.
        json example = cyxwiz::test::LoadTokenWindowGraph(root);
        for (auto& node : example["nodes"]) {
            if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataLoader)) node["parameters"]["model_seed"] = "7";
        }
        const std::string language_graph = example.dump();

        cyxwiz::RoundData whole;
        whole.leases.push_back({first, 0, 0});
        const LanguageRound one = TrainLanguageRound(language_graph, first, whole, nullptr, "worker one");
        Check(one.layout.exchanged_count > 0 && one.outcome.delta.size() == one.layout.exchanged_count,
              "the update covers the model's " + std::to_string(one.layout.exchanged_count) + " exchanged values");
        Check(one.outcome.steps > 0 && one.outcome.samples == 8 && one.outcome.rows_done == std::vector<uint64_t>({8}),
              "every window of the lease is trained once: " + std::to_string(one.outcome.steps) + " steps");
        Check(std::any_of(one.outcome.delta.begin(), one.outcome.delta.end(), [](float value) { return value != 0.0f; }),
              "training moved the model");

        // The same worker again, from the same starting state, on the same rows: the same update.
        const LanguageRound again = TrainLanguageRound(language_graph, first, whole, &one.start, "worker one again");
        Check(again.layout.hash == one.layout.hash, "the layout is the same every time");
        Check(again.outcome.delta == one.outcome.delta, "one worker's round can be reproduced");

        // Another member of the group, same state and rows: its own dropout masks, so another update.
        const LanguageRound two = TrainLanguageRound(language_graph, first, whole, &one.start, "worker two", 1);
        Check(two.layout.hash == one.layout.hash, "both workers build the same layout");
        Check(two.outcome.delta != one.outcome.delta, "two workers do not draw the same dropout masks");

        // Rows from two parts in one round; the tokenizer travels with the rows.
        cyxwiz::RoundData mixed;
        mixed.leases.push_back({first, 4, 4});
        mixed.leases.push_back({second, 0, 0});
        const LanguageRound three = TrainLanguageRound(language_graph, first, mixed, &one.start, "worker three");
        Check(three.outcome.samples == 6 && three.outcome.rows_done == std::vector<uint64_t>({4, 2}),
              "a round over two parts trains their 6 windows");
        Check(three.outcome.delta != one.outcome.delta, "other rows give another update");
    }

    fs::remove_all(work, ec);
    std::cout << "round training session: all checks passed\n";
    return 0;
}
