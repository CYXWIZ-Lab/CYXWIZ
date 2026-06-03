#include "../src/core/pipeline_materializer.h"
#include "../src/core/arrow_dataset.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTextTable() {
    auto text = FinishStringArray({
        "Small text sample",
        "Another small sample",
        "Text pipelines should tokenize",
    });
    auto label = FinishStringArray({"positive", "positive", "negative"});

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {text, label}, 3);
}

gui::MLNode MakeDataInputNode() {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Data Input";
    return node;
}

gui::MLNode MakeTokenizerNode() {
    gui::MLNode node;
    node.id = 2;
    node.type = gui::NodeType::TextTokenizer;
    node.category = gui::NodeCategory::TextProcessing;
    node.name = "Text Tokenizer";
    node.parameters = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };
    return node;
}

} // namespace

namespace cyxwiz {

PipelineOperatorFactory& PipelineOperatorFactory::Instance() {
    static PipelineOperatorFactory instance;
    return instance;
}

PipelineOperatorFactory::PipelineOperatorFactory() = default;

std::unique_ptr<IPipelineOperator> PipelineOperatorFactory::Create(
    gui::NodeType type) const {
    if (type == gui::NodeType::TextTokenizer) {
        return std::make_unique<TextTokenizerOperator>();
    }
    return nullptr;
}

bool PipelineOperatorFactory::HasOperator(gui::NodeType type) const {
    return type == gui::NodeType::TextTokenizer;
}

void PipelineOperatorFactory::RegisterCreator(gui::NodeType, Creator) {}

std::vector<gui::NodeType> PipelineOperatorFactory::GetSupportedTypes() const {
    return {gui::NodeType::TextTokenizer};
}

} // namespace cyxwiz

int main() {
    std::vector<gui::MLNode> nodes = {
        MakeDataInputNode(),
        MakeTokenizerNode(),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };

    auto result = cyxwiz::PipelineMaterializer::MaterializeTable(
        nodes, links, MakeTextTable());
    Check(result.success, result.error_message);
    Check(result.operators_applied == 1, "expected one tokenizer operator");

    auto table = result.table;
    Check(table != nullptr, "materialized Arrow table should not be null");
    Check(table->num_rows() == 3, "expected 3 materialized rows");
    Check(table->num_columns() == 5, "expected 4 token columns plus y");
    Check(table->GetColumnByName("tok_0") != nullptr, "missing tok_0");
    Check(table->GetColumnByName("tok_3") != nullptr, "missing tok_3");
    Check(table->GetColumnByName("y") != nullptr, "missing y label column");
    Check(table->GetColumnByName("text") == nullptr,
          "raw text column should not remain after tokenization");

    const auto parquet_path =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_arrow_materializer.parquet";
    std::remove(parquet_path.string().c_str());

    cyxwiz::ArrowDataset tokenized_dataset(table, "tokenized_text");
    Check(tokenized_dataset.ExportParquet(parquet_path.string()),
          "tokenized table should export to Parquet");

    auto reloaded = cyxwiz::ArrowDataset::FromParquet(
        parquet_path.string(), "tokenized_text_reloaded");
    Check(reloaded != nullptr, "exported Parquet should reload");
    auto reloaded_table = reloaded->GetArrowTable();
    Check(reloaded_table != nullptr, "reloaded table should not be null");
    Check(reloaded_table->num_rows() == table->num_rows(),
          "reloaded row count should match");
    Check(reloaded_table->num_columns() == table->num_columns(),
          "reloaded column count should match");
    Check(reloaded_table->GetColumnByName("tok_0") != nullptr,
          "reloaded table missing tok_0");
    Check(reloaded_table->GetColumnByName("y") != nullptr,
          "reloaded table missing y");

    auto batcher_dataset = std::make_shared<cyxwiz::ArrowDataset>(
        table, "tokenized_text_train");
    cyxwiz::ArrowDatasetBatcher batcher(
        batcher_dataset,
        "y",
        /*batch_size=*/2,
        /*shuffle=*/false,
        /*train_split=*/1.0f,
        /*is_training=*/true);
    batcher.SetOneHotEncoding(2);

    auto batch = batcher.GetNextBatch();
    Check(batch.IsValid(), "training batch should be valid");
    Check(batch.size == 2, "training batch should contain 2 samples");
    Check(batch.data.Shape().size() == 2, "feature tensor should be 2D");
    Check(batch.data.Shape()[0] == 2, "feature batch dimension should be 2");
    Check(batch.data.Shape()[1] == 4, "feature width should equal max_length");
    Check(batch.labels.Shape().size() == 2, "label tensor should be 2D");
    Check(batch.labels.Shape()[0] == 2, "label batch dimension should be 2");
    Check(batch.labels.Shape()[1] == 2, "label width should equal num_classes");

    const float* label_data = batch.labels.Data<float>();
    for (size_t r = 0; r < batch.size; ++r) {
        const float row_sum = label_data[r * 2] + label_data[r * 2 + 1];
        Check(row_sum == 1.0f, "one-hot label row should sum to 1");
    }

    std::remove(parquet_path.string().c_str());

    std::cout << "Text Arrow materializer path passed\n";
    return 0;
}
