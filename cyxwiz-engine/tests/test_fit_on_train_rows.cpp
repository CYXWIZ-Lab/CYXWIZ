// Fit on the Train rows (TOFIX178 S0-g2 / g3): a pipeline run for a training
// graph marks each Data Input row's partition ("__partition__": 0 train,
// 1 validation, 2 test) with the shared split rule, and the steps that learn
// from the data learn from the Train rows only, then transform every row,
// like scikit-learn's fit(X_train).transform(X). Expected values come from
// scikit-learn 1.8 fitted on the Train rows (StandardScaler, TfidfVectorizer
// with norm=None, CountVectorizer) and from the smoothing formula on the
// Train rows (TargetEncoder). Steps that rebuild the table row for row carry
// the partition; exports leave it out.

#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/node_executors/count_vectorizer_operator.h"
#include "../src/core/node_executors/preprocessing_operators.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/node_executors/tfidf_vectorizer_operator.h"
#include "../src/core/pipeline_executor.h"
#include "../src/core/sparse_feature_dataset.h"
#include "../src/core/sparse_feature_dataset_batcher.h"
#include "../src/core/sparse_feature_dataset_cache.h"
#include "../src/core/split_partitioning.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// Rows 0, 1, 2 and 5 are Train rows; row 3 validation, row 4 test. The
// validation and test rows hold what would change a fit on every row: a
// large a, the word "zebra", a blue row with y = 1.
std::shared_ptr<arrow::Table> Table(bool marked = true) {
    arrow::DoubleBuilder a, y;
    arrow::StringBuilder text, c;
    arrow::Int8Builder partition;
    const double av[] = {1, 2, 4, 10, 7, 3}, yv[] = {1, 0, 1, 1, 0, 1};
    const char* tv[] = {"cat dog", "dog fish", "cat cat bird", "zebra cat", "zebra dog", "fish"};
    const char* cv[] = {"red", "blue", "red", "blue", "green", "blue"};
    const int8_t pv[] = {0, 0, 0, 1, 2, 0};
    for (int i = 0; i < 6; ++i) {
        (void)a.Append(av[i]);
        (void)y.Append(yv[i]);
        (void)text.Append(tv[i]);
        (void)c.Append(cv[i]);
        (void)partition.Append(pv[i]);
    }
    std::shared_ptr<arrow::Array> aa, yy, tt, cc, pp;
    (void)a.Finish(&aa);
    (void)y.Finish(&yy);
    (void)text.Finish(&tt);
    (void)c.Finish(&cc);
    (void)partition.Finish(&pp);
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field("a", arrow::float64()), arrow::field("y", arrow::float64()),
                                                         arrow::field("text", arrow::utf8()), arrow::field("c", arrow::utf8())};
    std::vector<std::shared_ptr<arrow::Array>> columns = {aa, yy, tt, cc};
    if (marked) {
        fields.push_back(arrow::field(kSplitPartitionColumn, arrow::int8()));
        columns.push_back(pp);
    }
    return arrow::Table::Make(arrow::schema(fields), columns);
}

std::shared_ptr<arrow::Table> Run(IPipelineOperator& op, const std::map<std::string, std::string>& params,
                                  const std::shared_ptr<arrow::Table>& input) {
    std::string error;
    Check(op.Configure(params, error), op.GetName() + " configure: " + error);
    auto out = op.Apply(input);
    Check(out.ok(), op.GetName() + " apply: " + out.status().ToString());
    return *out;
}

std::vector<double> Column(const std::shared_ptr<arrow::Table>& t, const std::string& name) {
    std::vector<double> out;
    auto col = t->GetColumnByName(name);
    Check(col != nullptr, "column '" + name + "' present");
    for (const auto& chunk : col->chunks())
        for (int64_t i = 0; i < chunk->length(); ++i)
            out.push_back(std::static_pointer_cast<arrow::DoubleScalar>(*(*chunk->GetScalar(i))->CastTo(arrow::float64()))->value);
    return out;
}

void Expect(const std::vector<double>& got, const std::vector<double>& want, const std::string& what) {
    Check(got.size() == want.size(), what + ": row count");
    for (size_t i = 0; i < want.size(); ++i)
        Check(std::fabs(got[i] - want[i]) <= 2e-5 * std::max(1.0, std::fabs(want[i])),
              what + ": row " + std::to_string(i + 1) + " got " + std::to_string(got[i]) + ", want " + std::to_string(want[i]));
}

// The vectorizer's features are numbered (prefix0, prefix1, ...): there are as
// many as `want` has terms, and each term's scikit-learn column is one of them.
void ExpectFeatures(const std::shared_ptr<arrow::Table>& out, const std::string& prefix,
                    const std::map<std::string, std::vector<double>>& want, const std::string& what) {
    std::vector<std::vector<double>> features;
    for (const auto& field : out->schema()->fields())
        if (field->name().rfind(prefix, 0) == 0) features.push_back(Column(out, field->name()));
    Check(features.size() == want.size(), what + ": " + std::to_string(want.size()) + " features (none from validation / test rows only), got " +
                                              std::to_string(features.size()));
    for (const auto& [term, values] : want) {
        bool found = false;
        for (const auto& feature : features) {
            bool same = feature.size() == values.size();
            for (size_t i = 0; same && i < values.size(); ++i)
                same = std::fabs(feature[i] - values[i]) <= 2e-5 * std::max(1.0, std::fabs(values[i]));
            found = found || same;
        }
        Check(found, what + ": the scikit-learn column of '" + term + "'");
    }
}

void CheckPartitionCarried(const std::shared_ptr<arrow::Table>& out, const std::string& what) {
    const auto column = out->GetColumnByName(kSplitPartitionColumn);
    Check(column && column->type()->id() == arrow::Type::INT8, what + ": the rows' partition is carried");
    Check(TrainRowMask(out) == std::vector<bool>({true, true, true, false, false, true}), what + ": the same partition");
}

void CheckOperators() {
    const auto table = Table();
    Check(TrainRowMask(table) == std::vector<bool>({true, true, true, false, false, true}), "Train row mask");
    Check(TrainRowMask(Table(false)).empty(), "an unmarked table has no mask");
    Check((*TrainRowsOf(table))->num_rows() == 4, "TrainRowsOf keeps the four Train rows");
    Check((*TrainRowsOf(Table(false)))->num_rows() == 6, "TrainRowsOf keeps every row of an unmarked table");

    {
        StandardScalerOperator op;
        auto out = Run(op, {{"columns", "a"}}, table);
        Expect(Column(out, "a"), {-1.3416407864998738, -0.4472135954999579, 1.3416407864998738, 6.7082039324993685,
                                  4.024922359499621, 0.4472135954999579},
               "Standard scaler fitted on the Train rows");
    }
    {
        TargetEncoderOperator op;
        auto out = Run(op, {{"columns", "c"}, {"target_col", "y"}, {"smoothing", "1"}}, table);
        // Blue would be (3 * 2/3 + 0.8) / 4 = 0.7 with the validation row's y.
        Expect(Column(out, "c"), {0.9166666666666666, 0.5833333333333334, 0.9166666666666666, 0.5833333333333334, 0.75,
                                  0.5833333333333334},
               "Target encoder: category means from the Train rows; green (no Train row) gets the Train mean");
    }
    const std::map<std::string, std::string> text_params = {
        {"text_col", "text"}, {"label_col", ""}, {"norm", "none"}, {"ngram_range", "1,1"}, {"stop_words", "none"},
        {"output_format", "dense"}};
    {
        TFIDFVectorizerOperator op;
        auto params = text_params;
        params["use_idf"] = "true";
        params["smooth_idf"] = "true";
        params["min_df"] = "1";
        auto out = Run(op, params, table);
        ExpectFeatures(out, "tfidf_",
                       {{"bird", {0, 0, 1.916290731874155, 0, 0, 0}},
                        {"cat", {1.5108256237659907, 0, 3.0216512475319814, 1.5108256237659907, 0, 0}},
                        {"dog", {1.5108256237659907, 1.5108256237659907, 0, 0, 1.5108256237659907, 0}},
                        {"fish", {0, 1.5108256237659907, 0, 0, 0, 1.5108256237659907}}},
                       "TF-IDF");
        CheckPartitionCarried(out, "TF-IDF");

        // Sparse output: the dataset carries the partition, its cache file
        // keeps it, and the batcher serves the Train rows the vectorizer
        // learned from (it no longer splits the rows again).
        TFIDFVectorizerOperator sparse_op;
        params["output_format"] = "sparse";
        std::string error;
        Check(sparse_op.Configure(params, error), "sparse TF-IDF configure: " + error);
        auto sparse = sparse_op.ApplySparse(table, "fit_on_train_sparse");
        Check(sparse.ok(), "sparse TF-IDF: " + sparse.status().ToString());
        const std::vector<int8_t> want = {0, 0, 0, 1, 2, 0};
        Check((*sparse)->GetPartition() == want, "sparse TF-IDF carries the rows' partition");
        Check((*sparse)->GetNumFeatures() == 4, "sparse TF-IDF: 4 features (none from validation / test rows only)");
        const fs::path file = fs::temp_directory_path() / "cyxwiz_fit_on_train_rows.csr.arrow";
        Check(SparseFeatureDatasetCache::SaveAtomically(**sparse, file.string()).ok(), "save the sparse dataset");
        auto loaded = SparseFeatureDatasetCache::Load(file.string());
        Check(loaded.ok() && (*loaded)->GetPartition() == want, "the cache file keeps the partition");
        std::error_code ec;
        fs::remove(file, ec);
        for (const auto& [phase, rows] : std::vector<std::pair<BatcherPhase, size_t>>{
                 {BatcherPhase::Train, 4}, {BatcherPhase::Val, 1}, {BatcherPhase::Test, 1}}) {
            SparseFeatureDatasetBatcher batcher(*loaded, 2, false, 0.8f, phase == BatcherPhase::Train, phase, 0.1f);
            Check(batcher.GetNumSamples() == rows, "the sparse batcher serves the marked rows of each phase");
        }
    }
    {
        CountVectorizerOperator op;
        auto params = text_params;
        params["binary"] = "false";
        auto out = Run(op, params, table);
        ExpectFeatures(out, "count_",
                       {{"bird", {0, 0, 1, 0, 0, 0}}, {"cat", {1, 0, 2, 1, 0, 0}}, {"dog", {1, 1, 0, 0, 1, 0}},
                        {"fish", {0, 1, 0, 0, 0, 1}}},
                       "Count");
        CheckPartitionCarried(out, "Count");
    }
    {
        TextTokenizerOperator marked, every;
        const std::map<std::string, std::string> params = {{"text_col", "text"}, {"label_col", ""}};
        auto out = Run(marked, params, table);
        (void)Run(every, params, Table(false));
        Check(marked.GetLastVocabSize() + 1 == every.GetLastVocabSize(),
              "Tokenizer: the vocabulary leaves out 'zebra' (only in validation / test rows): " +
                  std::to_string(marked.GetLastVocabSize()) + " vs " + std::to_string(every.GetLastVocabSize()));
        CheckPartitionCarried(out, "Tokenizer");
    }
}

// The executor marks the rows (g2) and the scaler learns from the Train rows.
void CheckExecutor() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_fit_on_train_rows";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const fs::path csv_path = dir / "rows.csv";
    {
        std::ofstream csv(csv_path, std::ios::binary);
        csv << "a,y\n";
        for (int i = 1; i <= 20; ++i) csv << i * i << ',' << (i % 2 ? "odd" : "even") << '\n';
    }
    const auto pipeline = [&](bool split) {
        nlohmann::json j;
        j["nodes"] = nlohmann::json::array(
            {{{"id", 1}, {"type", "DataInput"}, {"name", "Rows"},
              {"parameters", {{"source_type", "file"}, {"file_path", csv_path.string()}, {"type", "csv"},
                              {"has_header", "true"}, {"delimiter", ","}, {"label_column", "y"},
                              {"force_disk_backed", "false"}}}},
             {{"id", 2}, {"type", "StandardScaler"}, {"name", "Scale"}, {"parameters", {{"columns", "a"}}}}});
        j["links"] = nlohmann::json::array({{{"start_node", 1}, {"end_node", 2}}});
        if (split)
            j["training_split"] = {{"label_column", "y"}, {"train", 0.8}, {"val", 0.1}, {"test", 0.1},
                                   {"seed", 42}, {"stratified", true}, {"shuffle", true}};
        return j.dump();
    };
    auto& registry = DataRegistry::Instance();
    const auto scaled = [&](PipelineExecutor& executor) {
        const auto it = executor.NodeResults().find(2);
        Check(it != executor.NodeResults().end(), "the scaler's result is named");
        const auto dataset = registry.GetArrowDataset(it->second);
        Check(dataset && dataset->GetArrowTable(), "the scaler's result is in memory");
        return dataset->GetArrowTable();
    };

    PipelineExecutor plain;
    Check(plain.ExecutePipeline(pipeline(false)), "a run without a split: " + plain.GetLastError());
    Check(scaled(plain)->schema()->GetFieldIndex(kSplitPartitionColumn) < 0, "no split: the rows are not marked");

    PipelineExecutor split;
    Check(split.ExecutePipeline(pipeline(true)), "a run with the training split: " + split.GetLastError());
    const auto out = scaled(split);
    const auto mask = TrainRowMask(out);
    Check(mask.size() == 20, "the scaler's rows carry their partition");
    // The same rows the training batcher takes (the shared rule, stratified on y).
    const auto source = registry.GetArrowDataset(split.NodeResults().at(1));
    auto expected = AddSplitPartitionColumn(
        source->GetArrowTable(), SplitPartitionOptions{"y", 0.8f, 0.1f, 0.1f, true, 42u, true, "test"});
    Check(expected.ok(), "the shared rule partitions the rows");
    Check(TrainRowMask(*expected) == mask, "the run's partition is the shared split rule's");
    const auto a = Column(out, "a");
    double sum = 0, squares = 0;
    int rows = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (mask[i]) sum += a[i], squares += a[i] * a[i], ++rows;
    Check(rows == 16, "16 Train rows of 20");
    const double mean = sum / rows, variance = squares / rows - mean * mean;
    Check(std::fabs(mean) < 1e-5 && std::fabs(variance - 1.0) < 1e-4,
          "the scaler learned from the Train rows: their mean " + std::to_string(mean) + ", variance " +
              std::to_string(variance));

    // An export writes the data, not the partition.
    const fs::path exported = dir / "out.csv";
    Check(registry.ExportTableCSV(split.NodeResults().at(2), exported.string()), "export the scaled rows");
    std::ifstream in(exported);
    std::string header;
    std::getline(in, header);
    Check(header.find(kSplitPartitionColumn) == std::string::npos, "the export leaves out the partition: " + header);
    in.close();
    fs::remove_all(dir, ec);
}
}  // namespace

int main() {
    CheckOperators();
    CheckExecutor();
    std::cout << "fit on Train rows: scaler, target encoder, TF-IDF, count and tokenizer learn from the Train rows "
                 "(scikit-learn values); the run marks rows with the shared split rule; exports leave the partition out. OK\n";
    return 0;
}
