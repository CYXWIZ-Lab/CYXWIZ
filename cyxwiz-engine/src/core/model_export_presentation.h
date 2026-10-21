#pragma once

// Export Model (TOFIX129 B5 E-2): the formats this build can write, the file
// a new export goes to, and the facts of a finished export. Pure data in and
// out (no ImGui).

#include "model_format.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::model_export {

struct FormatChoice {
    ModelFormat format = ModelFormat::CyxModel;
    std::string label;        // "CyxWiz package (.cyxmodel)"
    std::string description;
};

// The formats this build writes: the CyxWiz package and safetensors, and ONNX
// when the build includes the ONNX exporter. GGUF is not offered until built.
// ONNX is written from a stack of built-in layers: onnx_writable is false for
// a graph with branches or a model with extension nodes (TOFIX125 P6).
std::vector<FormatChoice> AvailableFormats(bool onnx_built, bool onnx_writable = true);

bool IsAvailable(ModelFormat format, bool onnx_built, bool onnx_writable = true);

// <project>/exports/<graph name><extension>; without a project, empty (the
// user picks a file).
std::string DefaultPath(const std::string& project_root, const std::string& graph_path, ModelFormat format);

// `path` with the extension of `format`.
std::string WithExtension(const std::string& path, ModelFormat format);

// The model name to prefill: the trained run's, else the graph's file name.
std::string DefaultName(const std::string& trained_name, const std::string& graph_path);

// The rows of a finished export: file, size, parameters, layers, time.
std::vector<std::pair<std::string, std::string>> ResultRows(const ExportResult& result);

}  // namespace cyxwiz::model_export
