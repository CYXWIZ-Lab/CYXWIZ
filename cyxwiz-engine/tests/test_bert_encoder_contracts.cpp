#include "../src/core/bert_encoder_contract.h"
#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cyxwiz/tensor.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

DataLoader* GetByBackendTag(int) {
    return nullptr;
}

const std::vector<DataLoader*>& All() {
    static const std::vector<DataLoader*> loaders;
    return loaders;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::NodePin Pin(int id,
                 gui::PinType type,
                 const std::string& name,
                 bool is_input) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    return pin;
}

gui::MLNode Node(int id,
                 gui::NodeType type,
                 const std::string& name,
                 std::vector<gui::NodePin> inputs,
                 std::vector<gui::NodePin> outputs) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
}

gui::NodeLink Link(int id, int from_node, int from_pin, int to_node, int to_pin) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.from_pin = from_pin;
    link.to_node = to_node;
    link.to_pin = to_pin;
    return link;
}

bool HasIssueText(const cyxwiz::TrainingConfiguration& config,
                  const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

const cyxwiz::BackendPlacementEntry* FindPlacement(
    const cyxwiz::TrainingConfiguration& config,
    int node_id) {
    for (const auto& placement : config.backend_placements) {
        if (placement.node_id == node_id) {
            return &placement;
        }
    }
    return nullptr;
}

gui::MLNode DataNode() {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "BERT Tokens",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "bert_encoder_contract_dataset";
    data.parameters["shape"] = "[4]";
    data.parameters["model_family"] = "bert_encoder";
    data.parameters["token_column"] = "token_ids";
    data.parameters["create_attention_mask"] = "true";
    return data;
}

gui::MLNode EmbeddingNode() {
    auto node = Node(2,
                     gui::NodeType::Embedding,
                     "Token Embedding",
                     {Pin(201, gui::PinType::Tensor, "Indices", true)},
                     {Pin(202, gui::PinType::Tensor, "Embeddings", false)});
    node.parameters["num_embeddings"] = "32";
    node.parameters["embedding_dim"] = "4";
    return node;
}

gui::MLNode PositionalNode() {
    auto node = Node(3,
                     gui::NodeType::PositionalEncoding,
                     "Position",
                     {Pin(301, gui::PinType::Tensor, "Input", true)},
                     {Pin(302, gui::PinType::Tensor, "Output", false)});
    node.parameters["d_model"] = "4";
    node.parameters["max_sequence_length"] = "4";
    return node;
}

gui::MLNode EncoderNode() {
    auto node = Node(4,
                     gui::NodeType::TransformerEncoder,
                     "BERT Encoder",
                     {Pin(401, gui::PinType::Tensor, "Input", true)},
                     {Pin(402, gui::PinType::Tensor, "Output", false)});
    node.parameters["d_model"] = "4";
    node.parameters["num_heads"] = "2";
    node.parameters["dim_feedforward"] = "8";
    node.parameters["dropout"] = "0";
    node.parameters["norm_first"] = "false";
    return node;
}

gui::MLNode CrossEntropyNode(int id = 20) {
    return Node(id,
                gui::NodeType::CrossEntropyLoss,
                "Cross Entropy",
                {Pin(id * 100 + 1, gui::PinType::Tensor, "Predictions", true),
                 Pin(id * 100 + 2, gui::PinType::Labels, "Targets", true)},
                {Pin(id * 100 + 3, gui::PinType::Loss, "Loss", false)});
}

gui::MLNode AdamNode(int id = 21) {
    return Node(id,
                gui::NodeType::Adam,
                "Adam",
                {Pin(id * 100 + 1, gui::PinType::Loss, "Loss", true)},
                {});
}

void TestBertSequenceClassificationGraphContract() {
    cyxwiz::GraphCompiler compiler;

    auto cls_select = Node(5,
                           gui::NodeType::TensorIndexSelect,
                           "CLS Select",
                           {Pin(501, gui::PinType::Tensor, "Input", true)},
                           {Pin(502, gui::PinType::Tensor, "Output", false)});
    cls_select.parameters["dim"] = "0";
    cls_select.parameters["indices"] = "0";

    auto squeeze = Node(6,
                        gui::NodeType::Squeeze,
                        "CLS Squeeze",
                        {Pin(601, gui::PinType::Tensor, "Input", true)},
                        {Pin(602, gui::PinType::Tensor, "Output", false)});
    squeeze.parameters["dim"] = "0";

    auto dense = Node(7,
                      gui::NodeType::Dense,
                      "Classifier",
                      {Pin(701, gui::PinType::Tensor, "Input", true)},
                      {Pin(702, gui::PinType::Tensor, "Output", false)});
    dense.parameters["units"] = "2";

    auto output = Node(8,
                       gui::NodeType::Output,
                       "Class Output",
                       {Pin(801, gui::PinType::Tensor, "Input", true)},
                       {});
    output.parameters["classes"] = "2";

    auto loss = CrossEntropyNode();
    auto optimizer = AdamNode();

    std::vector<gui::MLNode> nodes = {
        DataNode(), EmbeddingNode(), PositionalNode(), EncoderNode(),
        cls_select, squeeze, dense, output, loss, optimizer,
    };
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 3, 301),
        Link(3, 3, 302, 4, 401),
        Link(4, 4, 402, 5, 501),
        Link(5, 5, 502, 6, 601),
        Link(6, 6, 602, 7, 701),
        Link(7, 7, 702, 8, 801),
        Link(8, 7, 702, 20, 2001),
        Link(9, 1, 102, 20, 2002),
        Link(10, 20, 2003, 21, 2101),
    };

    const auto config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "BERT CLS sequence classifier graph should compile: " +
              config.error_message);
    Check(config.bert_encoder_graph.detected,
          "BERT sequence graph contract should be detected");
    Check(config.bert_encoder_graph.supported,
          "BERT sequence graph contract should be supported");
    Check(config.bert_encoder_graph.task ==
              cyxwiz::BertEncoderTask::SequenceClassification,
          "BERT sequence graph should identify sequence classification");
    Check(config.bert_encoder_graph.input_kind ==
              cyxwiz::BertEncoderInputKind::TokenIds,
          "BERT sequence graph should identify token-id input");
    Check(config.bert_encoder_graph.has_cls_extraction,
          "BERT sequence graph should record CLS extraction");
    Check(config.bert_encoder_graph.output_contract ==
              "Float32[batch,classes]",
          "BERT sequence graph should expose classifier output contract");
    const auto* encoder_placement = FindPlacement(config, 4);
    Check(encoder_placement != nullptr,
          "BERT sequence graph should report TransformerEncoder placement");
    Check(encoder_placement->node_type == "TransformerEncoder",
          "BERT sequence placement should name TransformerEncoder");
    Check(encoder_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "BERT sequence TransformerEncoder should be reported CPU-backed");
    Check(encoder_placement->reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeCpuBacked,
          "BERT sequence TransformerEncoder should use CPU-backed reason");
}

void TestBertTokenClassificationGraphContract() {
    cyxwiz::GraphCompiler compiler;

    auto time_distributed = Node(
        5,
        gui::NodeType::TimeDistributed,
        "Token Classifier",
        {Pin(501, gui::PinType::Tensor, "Input", true)},
        {Pin(502, gui::PinType::Tensor, "Output", false)});
    time_distributed.parameters["units"] = "3";

    auto output = Node(6,
                       gui::NodeType::SequenceTagOutput,
                       "Tag Output",
                       {Pin(601, gui::PinType::Tensor, "Input", true)},
                       {});
    output.parameters["num_tags"] = "3";

    auto loss = CrossEntropyNode();
    auto optimizer = AdamNode();

    std::vector<gui::MLNode> nodes = {
        DataNode(), EmbeddingNode(), PositionalNode(), EncoderNode(),
        time_distributed, output, loss, optimizer,
    };
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 3, 301),
        Link(3, 3, 302, 4, 401),
        Link(4, 4, 402, 5, 501),
        Link(5, 5, 502, 6, 601),
        Link(6, 5, 502, 20, 2001),
        Link(7, 1, 102, 20, 2002),
        Link(8, 20, 2003, 21, 2101),
    };

    const auto config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "BERT token classifier graph should compile: " +
              config.error_message);
    Check(config.bert_encoder_graph.detected,
          "BERT token graph contract should be detected");
    Check(config.bert_encoder_graph.supported,
          "BERT token graph contract should be supported");
    Check(config.bert_encoder_graph.task ==
              cyxwiz::BertEncoderTask::TokenClassification,
          "BERT token graph should identify token classification");
    Check(config.bert_encoder_graph.output_contract ==
              "Float32[batch,seq,classes]",
          "BERT token graph should expose token-classifier output contract");
    const auto* encoder_placement = FindPlacement(config, 4);
    Check(encoder_placement != nullptr,
          "BERT token graph should report TransformerEncoder placement");
    Check(encoder_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "BERT token TransformerEncoder should be reported CPU-backed");
    const auto* head_placement = FindPlacement(config, 5);
    Check(head_placement != nullptr,
          "BERT token graph should report TimeDistributed placement");
    Check(head_placement->node_type == "TimeDistributed",
          "BERT token head placement should name TimeDistributed");
    Check(head_placement->status == cyxwiz::BackendPlacementStatus::Unknown,
          "BERT token TimeDistributed wrapper should remain explicit unknown");
    Check(head_placement->reason_code ==
              cyxwiz::BackendPlacementReason::TimeDistributedSequenceWrapper,
          "BERT token TimeDistributed should use wrapper reason");
    Check(HasIssueText(config, "TimeDistributed"),
          "BERT token TimeDistributed placement warning should surface");
}

void TestBertGraphRejectsUnsupportedSegmentIds() {
    cyxwiz::GraphCompiler compiler;

    auto data = DataNode();
    data.parameters["token_type_ids"] = "true";

    auto cls_select = Node(5,
                           gui::NodeType::TensorIndexSelect,
                           "CLS Select",
                           {Pin(501, gui::PinType::Tensor, "Input", true)},
                           {Pin(502, gui::PinType::Tensor, "Output", false)});
    cls_select.parameters["dim"] = "0";
    cls_select.parameters["indices"] = "0";

    auto squeeze = Node(6,
                        gui::NodeType::Squeeze,
                        "CLS Squeeze",
                        {Pin(601, gui::PinType::Tensor, "Input", true)},
                        {Pin(602, gui::PinType::Tensor, "Output", false)});
    squeeze.parameters["dim"] = "0";

    auto dense = Node(7,
                      gui::NodeType::Dense,
                      "Classifier",
                      {Pin(701, gui::PinType::Tensor, "Input", true)},
                      {Pin(702, gui::PinType::Tensor, "Output", false)});
    dense.parameters["units"] = "2";

    auto loss = CrossEntropyNode();
    auto optimizer = AdamNode();

    std::vector<gui::MLNode> nodes = {
        data, EmbeddingNode(), PositionalNode(), EncoderNode(),
        cls_select, squeeze, dense, loss, optimizer,
    };
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 3, 301),
        Link(3, 3, 302, 4, 401),
        Link(4, 4, 402, 5, 501),
        Link(5, 5, 502, 6, 601),
        Link(6, 6, 602, 7, 701),
        Link(7, 7, 702, 20, 2001),
        Link(8, 1, 102, 20, 2002),
        Link(9, 20, 2003, 21, 2101),
    };

    const auto config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "BERT graph requiring token_type_ids should fail closed");
    Check(config.bert_encoder_graph.detected,
          "rejected BERT graph should still expose contract details");
    Check(!config.bert_encoder_graph.supported,
          "rejected BERT graph should not be marked supported");
    Check(HasIssueText(config, "token_type/segment IDs"),
          "BERT token_type/segment blocker should be surfaced");
}

void TestBertTokenInputContract() {
    const std::vector<int64_t> ids = {1, 2, 3, 4};
    const cyxwiz::Tensor token_ids({2, 2}, ids.data(), cyxwiz::DataType::Int64);
    const cyxwiz::Tensor attention({2, 2}, ids.data(), cyxwiz::DataType::Int64);

    const auto ok =
        cyxwiz::ValidateBertEncoderTokenInput(token_ids, &attention, nullptr, 4);
    Check(ok.compatible,
          "BERT token input with matching attention mask should pass: " +
              ok.error);
    Check(ok.batch_size == 2 && ok.sequence_length == 2,
          "BERT token input should surface batch and sequence length");
    Check(ok.has_attention_mask,
          "BERT token input should surface attention-mask presence");

    const cyxwiz::Tensor bad_attention({1, 4},
                                       ids.data(),
                                       cyxwiz::DataType::Int64);
    const auto bad_mask =
        cyxwiz::ValidateBertEncoderTokenInput(token_ids, &bad_attention);
    Check(!bad_mask.compatible,
          "BERT attention mask shape mismatch should fail");
    Check(bad_mask.error.find("attention_mask shape") != std::string::npos,
          "BERT attention mask error should be clear");

    const cyxwiz::Tensor token_type_ids({2, 2},
                                        ids.data(),
                                        cyxwiz::DataType::Int64);
    const auto unsupported_segments =
        cyxwiz::ValidateBertEncoderTokenInput(token_ids,
                                              nullptr,
                                              &token_type_ids);
    Check(!unsupported_segments.compatible,
          "BERT token_type_ids should fail closed");
    Check(unsupported_segments.error.find("token_type/segment") !=
              std::string::npos,
          "BERT token_type_ids error should be clear");
}

void TestBertRuntimeOutputContract() {
    const std::vector<float> sequence_logits(2 * 3, 0.0f);
    const cyxwiz::Tensor sequence_output({2, 3},
                                         sequence_logits.data(),
                                         cyxwiz::DataType::Float32);
    const auto sequence_contract =
        cyxwiz::ValidateBertEncoderRuntimeOutput(
            sequence_output,
            cyxwiz::BertEncoderTask::SequenceClassification,
            2,
            0,
            3);
    Check(sequence_contract.compatible,
          "BERT sequence output should pass: " +
              sequence_contract.error);
    Check(sequence_contract.batch_size == 2 &&
              sequence_contract.class_count == 3,
          "BERT sequence output should surface [batch, classes]");

    const std::vector<float> token_logits(2 * 4 * 5, 0.0f);
    const cyxwiz::Tensor token_output({2, 4, 5},
                                      token_logits.data(),
                                      cyxwiz::DataType::Float32);
    const auto token_contract =
        cyxwiz::ValidateBertEncoderRuntimeOutput(
            token_output,
            cyxwiz::BertEncoderTask::TokenClassification,
            2,
            4,
            5);
    Check(token_contract.compatible,
          "BERT token output should pass: " + token_contract.error);
    Check(token_contract.sequence_length == 4 &&
              token_contract.class_count == 5,
          "BERT token output should surface [batch, seq, classes]");

    const auto wrong_rank =
        cyxwiz::ValidateBertEncoderRuntimeOutput(
            sequence_output,
            cyxwiz::BertEncoderTask::TokenClassification,
            2,
            4,
            5);
    Check(!wrong_rank.compatible,
          "BERT token output should reject rank-2 logits");
    Check(wrong_rank.error.find("rank 3") != std::string::npos,
          "BERT token output rank error should be clear");

    const auto wrong_classes =
        cyxwiz::ValidateBertEncoderRuntimeOutput(
            token_output,
            cyxwiz::BertEncoderTask::TokenClassification,
            2,
            4,
            6);
    Check(!wrong_classes.compatible,
          "BERT token output should reject class metadata mismatch");
    Check(wrong_classes.error.find("class count") != std::string::npos,
          "BERT class mismatch error should be clear");
}

} // namespace

int main() {
    TestBertSequenceClassificationGraphContract();
    TestBertTokenClassificationGraphContract();
    TestBertGraphRejectsUnsupportedSegmentIds();
    TestBertTokenInputContract();
    TestBertRuntimeOutputContract();

    std::cout << "BERT encoder contract test passed\n";
    return 0;
}
