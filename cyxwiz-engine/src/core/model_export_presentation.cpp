#include "model_export_presentation.h"

#include "compiled_node_presentation.h"

#include <filesystem>

namespace cyxwiz::model_export {

std::vector<FormatChoice> AvailableFormats(bool onnx_built) {
    std::vector<FormatChoice> out = {
        {ModelFormat::CyxModel, "CyxWiz package (.cyxmodel)",
         "Weights with the graph, configuration and training history, as you choose below. Its graph opens again "
         "in CyxWiz."},
        {ModelFormat::Safetensors, "Weights only (.safetensors)",
         "The parameter tensors only, readable by other frameworks; no graph or training package."},
    };
    if (onnx_built)
        out.push_back({ModelFormat::ONNX, "ONNX (.onnx)", "A model for inference in other runtimes."});
    return out;
}

bool IsAvailable(ModelFormat format, bool onnx_built) {
    for (const auto& choice : AvailableFormats(onnx_built))
        if (choice.format == format) return true;
    return false;
}

std::string DefaultPath(const std::string& project_root, const std::string& graph_path, ModelFormat format) {
    if (project_root.empty()) return {};
    std::string stem = graph_path.empty() ? std::string() : std::filesystem::path(graph_path).stem().string();
    if (stem.empty()) stem = "model";
    return (std::filesystem::path(project_root) / "exports" / (stem + GetFormatExtension(format))).string();
}

std::string WithExtension(const std::string& path, ModelFormat format) {
    if (path.empty()) return path;
    std::filesystem::path p(path);
    p.replace_extension(GetFormatExtension(format));
    return p.string();
}

std::string DefaultName(const std::string& trained_name, const std::string& graph_path) {
    if (!trained_name.empty()) return trained_name;
    return graph_path.empty() ? std::string() : std::filesystem::path(graph_path).stem().string();
}

std::vector<std::pair<std::string, std::string>> ResultRows(const ExportResult& r) {
    std::vector<std::pair<std::string, std::string>> rows;
    rows.emplace_back("File", r.output_path);
    rows.emplace_back("Size", FormatCompiledBytes(r.file_size_bytes));
    rows.emplace_back("Parameters", FormatCompiledCount(r.num_parameters) + " in " +
                                        FormatCompiledCount(r.num_layers) + (r.num_layers == 1 ? " layer" : " layers"));
    rows.emplace_back("Time", std::to_string(r.export_time_ms) + " ms");
    return rows;
}

}  // namespace cyxwiz::model_export
