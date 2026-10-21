#include "../src/core/transformer_configuration_policy.h"
#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

namespace cyxwiz {

std::string ReadTreeModelArtifactType(const std::string&, std::string*) {
    return {};
}

} // namespace cyxwiz

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

void CheckNear(float actual,
               float expected,
               float tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << ": expected=" << expected
                  << " actual=" << actual << "\n";
        std::exit(1);
    }
}

std::string BuildTransformerDecoderGraphJson(
    const std::map<std::string, std::string>& extra_decoder_parameters = {}) {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "Input"},
        {"parameters", {
            {"dataset_name", "transformer_decoder_roundtrip_dataset"},
            {"dataset", "transformer_decoder_roundtrip_dataset"},
            {"shape", "[4]"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::TransformerDecoder)},
        {"name", "Decoder"},
        {"parameters", {
            {"d_model", "4"},
            {"num_heads", "2"},
            {"dim_feedforward", "8"},
            {"dropout", "0"},
            {"norm_first", "false"}
        }}
    });
    for (const auto& [key, value] : extra_decoder_parameters) {
        graph["nodes"].back()["parameters"][key] = value;
    }
    graph["nodes"].push_back({
        {"id", 3},
        {"type", static_cast<int>(gui::NodeType::Flatten)},
        {"name", "Flatten"},
        {"parameters", nlohmann::json::object()}
    });
    graph["nodes"].push_back({
        {"id", 4},
        {"type", static_cast<int>(gui::NodeType::Dense)},
        {"name", "Head"},
        {"parameters", {{"units", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 5},
        {"type", static_cast<int>(gui::NodeType::Output)},
        {"name", "Output"},
        {"parameters", {{"num_classes", "2"}}}
    });
    graph["nodes"].push_back({
        {"id", 6},
        {"type", static_cast<int>(gui::NodeType::MSELoss)},
        {"name", "MSE"},
        {"parameters", json::object()}
    });
    graph["nodes"].push_back({
        {"id", 7},
        {"type", static_cast<int>(gui::NodeType::Adam)},
        {"name", "Adam"},
        {"parameters", {{"learning_rate", "0.001"}}}
    });

    graph["links"].push_back({{"id", 101}, {"from_node", 1}, {"to_node", 2}});
    graph["links"].push_back({{"id", 102}, {"from_node", 2}, {"to_node", 3}});
    graph["links"].push_back({{"id", 103}, {"from_node", 3}, {"to_node", 4}});
    graph["links"].push_back({{"id", 104}, {"from_node", 4}, {"to_node", 5}});
    graph["links"].push_back({{"id", 105}, {"from_node", 4}, {"to_node", 6}});
    graph["links"].push_back({{"id", 106}, {"from_node", 7}, {"to_node", 6}});

    return graph.dump();
}

void CheckTensorValues(const cyxwiz::Tensor& tensor,
                       const cyxwiz::Tensor& expected,
                       const std::string& name) {
    Check(tensor.Shape() == expected.Shape(), name + " shape mismatch");
    const float* actual_data = tensor.Data<float>();
    const float* expected_data = expected.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        CheckNear(actual_data[i], expected_data[i], 1e-6f, name);
    }
}

void RunRoundTrip(const std::string& label,
                  const std::map<std::string, std::string>& decoder_parameters,
                  const cyxwiz::TransformerBlockOptions& options,
                  bool norm_first) {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() / ("cyxwiz_cyxmodel_transformer_decoder_roundtrip_" + label);
    const fs::path package_path = root / "transformer_decoder.cyxmodel";
    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::SequentialModel source;
    source.Add<cyxwiz::TransformerDecoderModule>(4, 2, 8, 0.0f, norm_first, 0.0f, options);
    source.Add<cyxwiz::FlattenModule>();
    source.Add<cyxwiz::LinearModule>(16, 2, true);

    const auto source_params = source.GetParameters();
    Check(!source_params.empty(),
          "source TransformerDecoder model should expose parameters");

    cyxwiz::ExportOptions export_options;
    export_options.format = cyxwiz::ModelFormat::CyxModel;
    export_options.include_graph = true;
    export_options.include_training_history = false;
    export_options.include_optimizer_state = false;

    cyxwiz::ModelExporter exporter;
    const cyxwiz::ExportResult exported = exporter.ExportCyxModel(
        source,
        nullptr,
        nullptr,
        BuildTransformerDecoderGraphJson(decoder_parameters),
        package_path.string(),
        export_options);
    Check(exported.success,
          "TransformerDecoder .cyxmodel export failed: " +
              exported.error_message);

    cyxwiz::SequentialModel imported;
    cyxwiz::ImportOptions import_options;
    import_options.strict_mode = true;

    cyxwiz::ModelImporter importer;
    const cyxwiz::ImportResult imported_result = importer.ImportCyxModel(
        package_path.string(),
        imported,
        import_options);
    Check(imported_result.success,
          "TransformerDecoder .cyxmodel import failed: " +
              imported_result.error_message);
    Check(imported.Size() == 3,
          "TransformerDecoder .cyxmodel import should rebuild three modules");

    const auto imported_params = imported.GetParameters();
    Check(imported_params.size() == source_params.size(),
          "TransformerDecoder parameter count should round-trip");
    for (const auto& [name, tensor] : source_params) {
        Check(imported_params.count(name) == 1,
              "imported TransformerDecoder parameter missing: " + name);
        CheckTensorValues(imported_params.at(name), tensor, name);
    }

    const float input_values[] = {
        0.2f, -0.1f, 0.4f, 0.7f,
        -0.3f, 0.5f, 0.1f, -0.2f,
        0.6f, 0.2f, -0.4f, 0.3f,
        -0.5f, 0.8f, 0.2f, -0.1f,
    };
    const cyxwiz::Tensor input({1, 4, 4}, input_values,
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor source_output = source.Forward(input);
    const cyxwiz::Tensor imported_output = imported.Forward(input);
    CheckTensorValues(imported_output,
                      source_output,
                      "TransformerDecoder imported inference output");

    fs::remove_all(root);
    std::cout << "CyxModel TransformerDecoder round-trip passed: " << label << "\n";
}

}  // namespace

int main() {
    RunRoundTrip("classic", {}, cyxwiz::TransformerBlockOptions{}, false);

    // LLaMA-style block (tofix112): pre-norm, RMSNorm, SwiGLU, no FFN bias.
    // The graph node carries the options; import must rebuild the same block.
    cyxwiz::TransformerBlockOptions llama;
    llama.norm_type = cyxwiz::TransformerNormType::RMSNorm;
    llama.ffn_type = cyxwiz::TransformerFeedForwardType::Gated;
    llama.ffn_activation = cyxwiz::ActivationType::SiLU;
    llama.ffn_bias = false;
    llama.position_encoding = cyxwiz::TransformerPositionEncoding::Rope;
    RunRoundTrip("llama_style",
                 {{"norm_first", "true"}, {"norm_type", "rms_norm"}, {"ffn_type", "gated"},
                  {"ffn_activation", "silu"}, {"ffn_bias", "false"}, {"position_encoding", "rope"}},
                 llama, true);

    // Preset llama_style plus QK-norm and squared ReLU (tofix112 group 1):
    // attention biases absent, q/k norm gammas present, both must round-trip.
    cyxwiz::TransformerBlockOptions modern = llama;
    modern.attention_bias = false;
    modern.qk_norm = true;
    modern.ffn_activation = cyxwiz::ActivationType::SquaredReLU;
    RunRoundTrip("qk_norm_no_attention_bias",
                 {{"norm_first", "true"}, {"norm_type", "rms_norm"}, {"ffn_type", "gated"},
                  {"ffn_activation", "squared_relu"}, {"ffn_bias", "false"}, {"position_encoding", "rope"},
                  {"attention_bias", "false"}, {"qk_norm", "true"}},
                 modern, true);

    // Groups 2-4: grouped-query attention, ALiBi, sandwich norm, soft-cap,
    // sliding window, residual init scale; key/value weights are narrower.
    cyxwiz::TransformerBlockOptions gemma = modern;
    gemma.position_encoding = cyxwiz::TransformerPositionEncoding::Alibi;
    gemma.sandwich_norm = true;
    gemma.num_kv_heads = 1;
    gemma.attn_logit_softcap = 2.0f;
    gemma.sliding_window = 3;
    gemma.residual_init_scale = 0.5f;
    RunRoundTrip("gqa_alibi_sandwich",
                 {{"norm_first", "true"}, {"norm_type", "rms_norm"}, {"ffn_type", "gated"},
                  {"ffn_activation", "squared_relu"}, {"ffn_bias", "false"}, {"position_encoding", "alibi"},
                  {"attention_bias", "false"}, {"qk_norm", "true"}, {"sandwich_norm", "true"},
                  {"num_kv_heads", "1"}, {"attn_logit_softcap", "2"}, {"sliding_window", "3"},
                  {"residual_init_scale", "0.5"}},
                 gemma, true);
    cyxwiz::TransformerBlockOptions parallel = llama;
    parallel.block_layout = cyxwiz::TransformerBlockLayout::Parallel;
    RunRoundTrip("parallel",
                 {{"norm_first", "true"}, {"norm_type", "rms_norm"}, {"ffn_type", "gated"},
                  {"ffn_activation", "silu"}, {"ffn_bias", "false"}, {"position_encoding", "rope"},
                  {"block_layout", "parallel"}},
                 parallel, true);

    // residual_init_scale scales only the residual output projections at
    // construction: with 0.001 every W_o and FFN-down weight lies within
    // 0.001 x its initialization bound while W_q keeps its full range.
    {
        cyxwiz::TransformerBlockOptions scaled;
        scaled.residual_init_scale = 0.001f;
        cyxwiz::TransformerDecoderLayer layer(16, 4, 32, 0.0f, true, 0.0f, scaled);
        const auto params = layer.GetParameters();
        const auto max_abs = [&](const std::string& name) {
            const cyxwiz::Tensor& t = params.at(name);
            const float* data = t.Data<float>();
            float m = 0.0f;
            for (size_t i = 0; i < t.NumElements(); ++i) m = std::max(m, std::fabs(data[i]));
            return m;
        };
        cyxwiz::TransformerDecoderLayer reference(16, 4, 32, 0.0f, true, 0.0f, cyxwiz::TransformerBlockOptions{});
        const auto reference_params = reference.GetParameters();
        const auto reference_max = [&](const std::string& name) {
            const cyxwiz::Tensor& t = reference_params.at(name);
            const float* data = t.Data<float>();
            float m = 0.0f;
            for (size_t i = 0; i < t.NumElements(); ++i) m = std::max(m, std::fabs(data[i]));
            return m;
        };
        Check(max_abs("self_attn.W_o") < 0.01f * reference_max("self_attn.W_o"), "residual_init_scale must scale W_o");
        Check(max_abs("linear2.weights") < 0.01f * reference_max("linear2.weights"),
              "residual_init_scale must scale the FFN down projection");
        Check(max_abs("self_attn.W_q") > 0.1f * reference_max("self_attn.W_q"), "residual_init_scale must not scale W_q");
        Check(max_abs("linear1.weights") > 0.1f * reference_max("linear1.weights"),
              "residual_init_scale must not scale the FFN up projection");
    }

    // Policy: invalid choices fail closed with the field named; the encoder
    // rejects block options it does not implement instead of ignoring them.
    using cyxwiz::ResolveInvalidTransformerConfigurationReason;
    const auto reason = [](gui::NodeType type, std::map<std::string, std::string> params) {
        params.emplace("d_model", "4");
        params.emplace("num_heads", "2");
        return ResolveInvalidTransformerConfigurationReason(type, params);
    };
    Check(!reason(gui::NodeType::TransformerDecoder, {{"ffn_activation", "swish"}, {"norm_type", "RMSNorm"}}),
          "aliases swish and RMSNorm are accepted");
    const auto bad_activation = reason(gui::NodeType::TransformerDecoder, {{"ffn_activation", "softmax"}});
    Check(bad_activation && bad_activation->find("ffn_activation") != std::string::npos,
          "softmax is rejected as a feed-forward activation");
    const auto bad_eps = reason(gui::NodeType::TransformerDecoder, {{"norm_eps", "0"}});
    Check(bad_eps && bad_eps->find("norm_eps") != std::string::npos, "norm_eps must be positive");
    const auto encoder = reason(gui::NodeType::TransformerEncoder, {{"ffn_type", "gated"}});
    Check(encoder && encoder->find("TransformerDecoder") != std::string::npos,
          "the encoder rejects block options it does not implement");
    Check(!reason(gui::NodeType::TransformerEncoder, {{"ffn_type", "mlp"}, {"norm_type", "layer_norm"}}),
          "the encoder accepts explicit classic values");
    const auto odd_rope = reason(gui::NodeType::TransformerDecoder,
                                 {{"position_encoding", "rope"}, {"d_model", "6"}, {"num_heads", "2"}});
    Check(odd_rope && odd_rope->find("even head width") != std::string::npos,
          "rope needs an even head width");
    const auto low_base = reason(gui::NodeType::TransformerDecoder, {{"position_encoding", "rope"}, {"rope_base", "1"}});
    Check(low_base && low_base->find("rope_base") != std::string::npos, "rope_base must exceed 1");

    std::cout << "CyxModel TransformerDecoder round-trip test passed\n";
    return 0;
}
