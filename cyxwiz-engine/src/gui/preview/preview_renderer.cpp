#include "preview_renderer.h"
#include "text_preview.h"
#include "json_tree.h"
#include "table_preview.h"
#include "image_preview.h"
#include "markdown_preview.h"
#include "hex_preview.h"
#include "hdf5_preview.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>

namespace fs = std::filesystem;

namespace gui {

PreviewRenderer::PreviewRenderer()
    : text_renderer_(std::make_unique<TextPreviewRenderer>())
    , json_renderer_(std::make_unique<JSONTreeRenderer>())
    , table_renderer_(std::make_unique<TablePreviewRenderer>())
    , image_renderer_(std::make_unique<ImagePreviewRenderer>())
    , markdown_renderer_(std::make_unique<MarkdownPreviewRenderer>())
    , hex_renderer_(std::make_unique<HexPreviewRenderer>())
    , hdf5_renderer_(std::make_unique<HDF5PreviewRenderer>())
{}

PreviewRenderer::~PreviewRenderer() = default;

PreviewContentType PreviewRenderer::DetectFileType(const std::string& filepath) {
    if (filepath.empty()) return PreviewContentType::Unknown;

    std::string ext = fs::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Images
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
        ext == ".gif" || ext == ".tga" || ext == ".tiff" || ext == ".webp")
        return PreviewContentType::Image;

    // CSV/TSV
    if (ext == ".csv" || ext == ".tsv")
        return PreviewContentType::TableCSV;

    // JSON
    if (ext == ".json" || ext == ".geojson")
        return PreviewContentType::JSONTree;

    // Markdown
    if (ext == ".md" || ext == ".markdown")
        return PreviewContentType::Markdown;

    // Syntax-highlighted text
    if (ext == ".py" || ext == ".pyw" || ext == ".pyx" ||
        ext == ".cpp" || ext == ".c" || ext == ".h" || ext == ".hpp" || ext == ".cc" || ext == ".cxx" ||
        ext == ".xml" || ext == ".html" || ext == ".htm" || ext == ".svg" ||
        ext == ".yaml" || ext == ".yml" ||
        ext == ".css" || ext == ".scss" ||
        ext == ".sql" ||
        ext == ".sh" || ext == ".bash" || ext == ".bat" || ext == ".ps1" ||
        ext == ".js" || ext == ".ts" || ext == ".jsx" || ext == ".tsx" ||
        ext == ".java" || ext == ".cs" || ext == ".rs" || ext == ".go" ||
        ext == ".rb" || ext == ".lua" || ext == ".r" || ext == ".m" ||
        ext == ".toml" || ext == ".cmake")
        return PreviewContentType::SyntaxText;

    // Plain text
    if (ext == ".txt" || ext == ".log" || ext == ".cfg" || ext == ".conf" ||
        ext == ".ini" || ext == ".env" || ext == ".gitignore" || ext == ".dockerfile" ||
        ext == ".proto" || ext == ".diff" || ext == ".patch" || ext == ".rst" ||
        ext == "" /* no extension */)
        return PreviewContentType::PlainText;

    // HDF5 files (tree view with dataset info)
    if (ext == ".h5" || ext == ".hdf5" || ext == ".hdf")
        return PreviewContentType::HDF5;

    // Binary/unknown -> hex dump
    return PreviewContentType::HexDump;
}

const char* PreviewRenderer::GetTypeName(PreviewContentType type) {
    switch (type) {
        case PreviewContentType::Image:          return "Image";
        case PreviewContentType::TableCSV:       return "Table (CSV)";
        case PreviewContentType::JSONTree:       return "JSON";
        case PreviewContentType::PlainText:      return "Plain Text";
        case PreviewContentType::SyntaxText:     return "Source Code";
        case PreviewContentType::Markdown:       return "Markdown";
        case PreviewContentType::HexDump:        return "Binary";
        case PreviewContentType::HDF5:           return "HDF5";
        default:                                 return "Unknown";
    }
}

void PreviewRenderer::RenderHDF5Preview(const std::string& filepath) {
    if (!filepath.empty()) {
        hdf5_renderer_->LoadFile(filepath);
        hdf5_renderer_->Render();
    }
}

void PreviewRenderer::SetFilePath(const std::string& path) {
    if (path != current_path_) {
        current_path_ = path;
        content_loaded_ = false;
        current_type_ = DetectFileType(path);
    }
}

void PreviewRenderer::LoadFileContent(const std::string& filepath) {
    cached_content_.clear();

    if (!fs::exists(filepath)) {
        cached_content_ = "File not found: " + filepath;
        return;
    }

    auto file_size = fs::file_size(filepath);

    // For text files, read content (limit to 2MB)
    if (current_type_ != PreviewContentType::Image &&
        current_type_ != PreviewContentType::HexDump) {
        size_t max_read = std::min(file_size, static_cast<uintmax_t>(2 * 1024 * 1024));
        std::ifstream f(filepath, std::ios::binary);
        if (f) {
            cached_content_.resize(max_read);
            f.read(cached_content_.data(), max_read);
            cached_content_.resize(f.gcount());
            if (file_size > max_read) {
                cached_content_ += "\n\n--- Truncated (showing first 2 MB of "
                    + std::to_string(file_size / 1024 / 1024) + " MB) ---";
            }
        }
    }

    content_loaded_ = true;
}

void PreviewRenderer::EnsureContent(const std::string& filepath) {
    if (!content_loaded_ || filepath != current_path_) {
        SetFilePath(filepath);
        LoadFileContent(filepath);
    }
}

void PreviewRenderer::RenderFilePreview(const std::string& filepath) {
    EnsureContent(filepath);

    // Info bar
    auto fname = fs::path(filepath).filename().string();
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", fname.c_str());
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "(%s)", GetTypeName(current_type_));

    if (fs::exists(filepath)) {
        auto fsize = fs::file_size(filepath);
        ImGui::SameLine();
        if (fsize < 1024)
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "- %zu B", fsize);
        else if (fsize < 1024 * 1024)
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "- %.1f KB", fsize / 1024.0f);
        else
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "- %.1f MB", fsize / 1024.0f / 1024.0f);
    }

    ImGui::Separator();

    // Dispatch to sub-renderer
    switch (current_type_) {
        case PreviewContentType::Image:
            image_renderer_->RenderFile(filepath);
            break;

        case PreviewContentType::TableCSV: {
            char delim = (fs::path(filepath).extension() == ".tsv") ? '\t' : ',';
            table_renderer_->LoadFromFile(filepath, delim);
            table_renderer_->Render();
            break;
        }

        case PreviewContentType::JSONTree:
            json_renderer_->SetContent(cached_content_);
            json_renderer_->Render();
            break;

        case PreviewContentType::Markdown:
            markdown_renderer_->SetContent(cached_content_);
            markdown_renderer_->Render();
            break;

        case PreviewContentType::SyntaxText: {
            auto lang = TextPreviewRenderer::DetectLanguage(filepath);
            text_renderer_->SetContent(cached_content_, lang);
            text_renderer_->Render();
            break;
        }

        case PreviewContentType::PlainText:
            text_renderer_->SetContent(cached_content_, SyntaxLanguage::None);
            text_renderer_->Render();
            break;

        case PreviewContentType::HexDump:
            hex_renderer_->LoadFile(filepath);
            hex_renderer_->Render();
            break;

        case PreviewContentType::HDF5:
            hdf5_renderer_->LoadFile(filepath);
            hdf5_renderer_->Render();
            break;

        default:
            ImGui::TextWrapped("No preview available for this file type.");
            break;
    }
}

} // namespace gui
