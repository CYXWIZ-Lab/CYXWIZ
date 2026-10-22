// Cross-validation and learning curves that train the graph (TOFIX129 D4),
// against PyTorch training the same graph on the same folds
// (reference/generate_cross_validation_fixtures.py, several seeds a fold):
// each fold's validation accuracy and loss fall inside the spread of real
// training, the learning curve's training and validation accuracy at each
// share likewise, and the row counts are exact. Then the runner's own folds:
// the graph's test rows are left out, every row used is validated once, class
// shares stay, the same seed gives the same folds, and bad settings are
// refused.
#include "../../src/core/arrow_dataset.h"
#include "../../src/core/cross_validation_runner.h"
#include "../../src/core/graph_node_factory.h"
#include "../../src/core/graph_training_job.h"
#include "../route_qualification_test_fixture.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef CYXWIZ_CROSS_VALIDATION_FIXTURE
#error "CYXWIZ_CROSS_VALIDATION_FIXTURE must name cross_validation_pytorch.json"
#endif

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

std::string Num(double v) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.4f", v);
    return text;
}

bool WritePenguins(const nlohmann::json& fixture, const fs::path& path) {
    arrow::DoubleBuilder x[4];
    arrow::Int64Builder y;
    for (const auto& row : fixture.at("x"))
        for (int c = 0; c < 4; ++c)
            if (!x[c].Append(row.at(static_cast<size_t>(c)).get<double>()).ok()) return false;
    for (const auto& label : fixture.at("y"))
        if (!y.Append(label.get<int64_t>()).ok()) return false;
    std::vector<std::shared_ptr<arrow::Array>> arrays(5);
    for (int c = 0; c < 4; ++c)
        if (!x[c].Finish(&arrays[static_cast<size_t>(c)]).ok()) return false;
    if (!y.Finish(&arrays[4]).ok()) return false;
    const auto table = arrow::Table::Make(
        arrow::schema({arrow::field("bill_length", arrow::float64()), arrow::field("bill_depth", arrow::float64()),
                       arrow::field("flipper_length", arrow::float64()), arrow::field("body_mass", arrow::float64()),
                       arrow::field("species", arrow::int64())}),
        arrays);
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 1000).ok() && (*out)->Close().ok();
}

// Data Input -> [Split] -> Loader -> Dense 16 -> ReLU -> Dense 3 -> cross
// entropy -> Adam -> Output, as the oracle's graph.
nlohmann::json PenguinsGraph(bool with_split, int epochs) {
    using gui::NodeType;
    nlohmann::json nodes = nlohmann::json::array();
    const auto add = [&nodes](int id, NodeType type, const char* name, nlohmann::json parameters) {
        nodes.push_back({{"id", id}, {"type", static_cast<int>(type)}, {"name", name}, {"parameters", parameters}});
    };
    add(1, NodeType::DataInput, "Penguins",
        {{"dataset_name", "penguins"}, {"file_category", "tabular"}, {"file_type", "parquet"}, {"label_column", "species"}});
    add(2, NodeType::DataSplit, "Split",
        {{"seed", "42"}, {"stratified", "true"}, {"train_ratio", "0.7"}, {"val_ratio", "0.1"}, {"test_ratio", "0.2"}});
    add(3, NodeType::DataLoader, "Loader",
        {{"batch_size", "32"}, {"epochs", std::to_string(epochs)}, {"shuffle", "true"}, {"seed", "11"}, {"model_seed", "7"}});
    add(4, NodeType::Dense, "Dense 16", {{"units", "16"}});
    add(5, NodeType::ReLU, "ReLU", nlohmann::json::object());
    add(6, NodeType::Dense, "Classes", {{"units", "3"}});
    add(7, NodeType::CrossEntropyLoss, "Cross entropy", nlohmann::json::object());
    add(8, NodeType::Adam, "Adam", {{"learning_rate", "0.01"}});
    add(9, NodeType::Output, "Output", {{"num_classes", "3"}});
    nlohmann::json links = nlohmann::json::array();
    int link_id = 1;
    const auto link = [&](int from, int from_index, int to, int to_index) {
        links.push_back({{"id", link_id}, {"from_node", from}, {"from_pin", 100 + link_id}, {"from_pin_index", from_index},
                         {"to_node", to}, {"to_pin", 200 + link_id}, {"to_pin_index", to_index}, {"link_type", 0}});
        ++link_id;
    };
    if (with_split) {
        link(1, 0, 2, 0);
        link(2, 0, 3, 0);
    } else {
        link(1, 0, 3, 0);
    }
    link(3, 0, 4, 0);
    link(4, 0, 5, 0);
    link(5, 0, 6, 0);
    link(6, 0, 7, 0);
    link(3, 1, 7, 1);
    link(7, 0, 8, 0);
    link(8, 0, 9, 0);
    if (!with_split) nodes.erase(nodes.begin() + 1);  // no Split node
    return {{"version", "1.0"}, {"data_boundary_version", 2}, {"nodes", nodes}, {"links", links}};
}

struct Prepared {
    cyxwiz::PreparedGraphJob job;
    std::shared_ptr<cyxwiz::ArrowDataset> table;
    std::string error;
};

Prepared Prepare(const fs::path& parquet, bool with_split, int epochs) {
    Prepared p;
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = PenguinsGraph(with_split, epochs).dump();
    request.dataset_files["penguins"] = parquet.string();
    const auto refused = cyxwiz::PrepareGraphJob(request, p.job);
    if (!refused.error.empty()) {
        p.error = refused.error;
        return p;
    }
    const auto it = p.job.datasets.find("penguins");
    if (it != p.job.datasets.end()) p.table = it->second;
    else p.error = "the prepared job has no penguins table";
    return p;
}

double Accuracy(const cyxwiz::CrossValidationRun& run) { return run.val.accuracy; }
double TrainAccuracy(const cyxwiz::CrossValidationRun& run) { return run.train.accuracy; }

}  // namespace

int main() {
    std::ifstream in(CYXWIZ_CROSS_VALIDATION_FIXTURE);
    if (!in.good()) {
        std::cerr << "cannot open " << CYXWIZ_CROSS_VALIDATION_FIXTURE << "\n";
        return 1;
    }
    const nlohmann::json fixture = nlohmann::json::parse(in);
    // The executor's route preflight refuses unqualified devices.
    cyxwiz::test::InstallQualifiedRouteSnapshot();
    const fs::path work = fs::temp_directory_path() / "cyxwiz_cross_validation_truth";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "penguins.parquet";
    Check(WritePenguins(fixture, parquet), "penguins written (" + std::to_string(fixture.at("y").size()) + " rows)");
    const int epochs = fixture.at("graph").at("epochs").get<int>();
    const int batch = fixture.at("graph").at("batch_size").get<int>();
    const std::vector<int> folds = fixture.at("fold").get<std::vector<int>>();

    std::cout << "cross-validation on the oracle's folds\n";
    {
        const Prepared p = Prepare(parquet, false, epochs);
        Check(p.error.empty() && p.table, "graph prepared" + (p.error.empty() ? "" : ": " + p.error));
        if (p.table) {
            cyxwiz::CrossValidationSettings settings;
            settings.folds = 5;
            settings.fold_of_row = folds;
            const auto r = cyxwiz::RunCrossValidation(p.job.config, p.table, "species", epochs, batch, settings);
            Check(r.ok && r.runs.size() == 5, "five folds trained" + (r.error.empty() ? "" : ": " + r.error));
            Check(r.rows == static_cast<int64_t>(folds.size()) && r.test_rows_left_out == 0 && !r.regression,
                  "every row used, none held out (no Split)");
            for (size_t k = 0; k < r.runs.size() && k < 5; ++k) {
                const auto& want = fixture.at("folds").at(k);
                const auto& run = r.runs[k];
                const double mean = want.at("accuracy_mean").get<double>(), sd = want.at("accuracy_sd").get<double>();
                const double band = std::max(3.0 * sd, 0.03);
                Check(run.train_rows == want.at("train_rows").get<int64_t>() && run.val_rows == want.at("val_rows").get<int64_t>(),
                      "fold " + std::to_string(k + 1) + ": " + std::to_string(run.train_rows) + " / " + std::to_string(run.val_rows) + " rows");
                Check(std::abs(run.val.accuracy - mean) <= band,
                      "fold " + std::to_string(k + 1) + ": accuracy " + Num(run.val.accuracy) + " in PyTorch's " + Num(mean) + " +- " + Num(band));
                const double loss = want.at("loss_mean").get<double>(), loss_band = std::max(4.0 * want.at("loss_sd").get<double>(), 0.06);
                Check(std::abs(run.val.loss - loss) <= loss_band,
                      "fold " + std::to_string(k + 1) + ": loss " + Num(run.val.loss) + " in PyTorch's " + Num(loss) + " +- " + Num(loss_band));
                Check(run.epochs == epochs, "fold " + std::to_string(k + 1) + ": trained every epoch (no early stop)");
            }
            const auto summary = cyxwiz::SummariseRuns(r.runs, 1.0, &Accuracy);
            Check(summary.count == 5 && summary.low <= summary.mean && summary.mean <= summary.high, "summary: mean " + Num(summary.mean));
        }
    }

    std::cout << "learning curve on the oracle's folds\n";
    {
        const Prepared p = Prepare(parquet, false, epochs);
        if (p.table) {
            cyxwiz::CrossValidationSettings settings;
            settings.folds = 5;
            settings.fold_of_row = folds;
            for (const auto& point : fixture.at("learning_curve")) settings.train_shares.push_back(point.at("share").get<double>());
            const auto r = cyxwiz::RunCrossValidation(p.job.config, p.table, "species", epochs, batch, settings);
            Check(r.ok && r.runs.size() == 15, "three shares x five folds trained" + (r.error.empty() ? "" : ": " + r.error));
            for (const auto& point : fixture.at("learning_curve")) {
                const double share = point.at("share").get<double>();
                const auto sizes = point.at("train_rows").get<std::vector<int64_t>>();
                bool sizes_ok = true;
                for (const auto& run : r.runs)
                    if (std::abs(run.share - share) < 1e-9) sizes_ok = sizes_ok && run.train_rows == sizes.at(static_cast<size_t>(run.fold - 1));
                Check(sizes_ok, "share " + Num(share) + ": training rows as the oracle's");
                const auto val = cyxwiz::SummariseRuns(r.runs, share, &Accuracy);
                const auto train = cyxwiz::SummariseRuns(r.runs, share, &TrainAccuracy);
                const double val_mean = point.at("val_accuracy_mean").get<double>();
                const double val_band = std::max(2.0 * point.at("val_accuracy_sd").get<double>(), 0.04);
                const double train_mean = point.at("train_accuracy_mean").get<double>();
                const double train_band = std::max(2.0 * point.at("train_accuracy_sd").get<double>(), 0.04);
                Check(std::abs(val.mean - val_mean) <= val_band,
                      "share " + Num(share) + ": validation " + Num(val.mean) + " in PyTorch's " + Num(val_mean) + " +- " + Num(val_band));
                Check(std::abs(train.mean - train_mean) <= train_band,
                      "share " + Num(share) + ": training " + Num(train.mean) + " in PyTorch's " + Num(train_mean) + " +- " + Num(train_band));
            }
        }
    }

    std::cout << "the runner's own folds, with the graph's test split\n";
    {
        const Prepared p = Prepare(parquet, true, 2);
        Check(p.error.empty() && p.table, "graph with a Split prepared" + (p.error.empty() ? "" : ": " + p.error));
        if (p.table) {
            cyxwiz::CrossValidationSettings settings;
            settings.folds = 4;
            settings.seed = 3;
            const auto a = cyxwiz::RunCrossValidation(p.job.config, p.table, "species", 2, batch, settings);
            const auto b = cyxwiz::RunCrossValidation(p.job.config, p.table, "species", 2, batch, settings);
            Check(a.ok && b.ok, "trained twice" + (a.error.empty() ? "" : ": " + a.error));
            const int64_t n = static_cast<int64_t>(folds.size());
            Check(a.test_rows_left_out > 0 && std::abs(static_cast<double>(a.test_rows_left_out) - 0.2 * static_cast<double>(n)) <= 3.0 &&
                      a.rows + a.test_rows_left_out == n,
                  "the graph's 20% test rows are left out: " + std::to_string(a.test_rows_left_out) + " of " + std::to_string(n));
            int64_t validated = 0, smallest = n, largest = 0;
            bool same = a.runs.size() == b.runs.size();
            for (size_t k = 0; k < a.runs.size(); ++k) {
                validated += a.runs[k].val_rows;
                smallest = std::min(smallest, a.runs[k].val_rows);
                largest = std::max(largest, a.runs[k].val_rows);
                Check(a.runs[k].train_rows + a.runs[k].val_rows == a.rows, "fold " + std::to_string(k + 1) + ": the rest trains");
                same = same && a.runs[k].val_rows == b.runs[k].val_rows;
            }
            Check(validated == a.rows, "every row used is validated once (" + std::to_string(validated) + ")");
            Check(largest - smallest <= 3, "stratified folds of near-equal size (" + std::to_string(smallest) + " .. " + std::to_string(largest) + ")");
            Check(same, "the same seed gives the same folds");

            cyxwiz::CrossValidationSettings bad = settings;
            bad.folds = 1;
            Check(!cyxwiz::RunCrossValidation(p.job.config, p.table, "species", 2, batch, bad).ok, "one fold is refused");
            bad = settings;
            bad.fold_of_row = {1, 2, 3};
            const auto wrong = cyxwiz::RunCrossValidation(p.job.config, p.table, "species", 2, batch, bad);
            Check(!wrong.ok && wrong.error.find("given folds") != std::string::npos, "given folds of the wrong length: " + wrong.error);
            bad = settings;
            bad.train_shares = {0.0};
            Check(!cyxwiz::RunCrossValidation(p.job.config, p.table, "species", 2, batch, bad).ok, "a share of 0 is refused");
            int calls = 0;
            cyxwiz::CrossValidationCallbacks stop;
            stop.should_cancel = [&calls] { return ++calls > 3; };
            const auto cancelled = cyxwiz::RunCrossValidation(p.job.config, p.table, "species", 2, batch, settings, stop);
            Check(!cancelled.ok && cancelled.cancelled, "Cancel stops it");
        }
    }

    fs::remove_all(work, ec);
    std::cout << (g_failures == 0 ? "cross-validation truth: OK\n" : "cross-validation truth: FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
