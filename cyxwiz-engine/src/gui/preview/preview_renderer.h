#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include "../../core/data_registry.h"

namespace gui {

// Content type for preview dispatch
enum class PreviewContentType {
    Unknown,
    // Raw file types
    Image,          // PNG, JPG, BMP, etc.
    TableCSV,       // CSV/TSV with headers
    JSONTree,       // JSON file (tree view)
    PlainText,      // .txt, .log, .cfg with line numbers
    SyntaxText,     // .py, .cpp, .xml, .yaml - syntax highlighted
    Markdown,       // .md - rendered formatted
    HexDump,        // Binary/unknown files
    HDF5,           // HDF5 files (.h5, .hdf5) - tree view
    // Dataset types
    DatasetImage,   // Loaded image dataset
    DatasetTabular, // Loaded tabular dataset
    DatasetJSON,    // Loaded JSON dataset
    DatasetText,    // Loaded text dataset
    DatasetHDF5     // Loaded HDF5 dataset (tree view)
};

// Forward declarations for sub-renderers
class TextPreviewRenderer;
class JSONTreeRenderer;
class TablePreviewRenderer;
class ImagePreviewRenderer;
class MarkdownPreviewRenderer;
class HexPreviewRenderer;
class HDF5PreviewRenderer;

/**
 * Central preview dispatcher
 * Auto-detects content type and routes to appropriate sub-renderer
 */
class PreviewRenderer {
public:
    PreviewRenderer();
    ~PreviewRenderer();

    // For raw files (from Asset Browser)
    void RenderFilePreview(const std::string& filepath);

    // For loaded datasets (from Dataset Panel)
    void RenderDatasetPreview(const cyxwiz::DatasetHandle& handle,
                               const cyxwiz::DatasetInfo& info,
                               int sample_idx, int view_mode, float zoom,
                               int grid_cols, int grid_rows,
                               const std::vector<std::string>& class_names);

    // Content type detection
    static PreviewContentType DetectFileType(const std::string& filepath);
    static PreviewContentType DetectDatasetType(const cyxwiz::DatasetInfo& info);

    // Get human-readable type name
    static const char* GetTypeName(PreviewContentType type);

    // Render HDF5 file directly (for loaded HDF5 datasets)
    void RenderHDF5Preview(const std::string& filepath);

    // State for file preview
    void SetFilePath(const std::string& path);
    const std::string& GetCurrentPath() const { return current_path_; }

private:
    // Sub-renderers
    std::unique_ptr<TextPreviewRenderer> text_renderer_;
    std::unique_ptr<JSONTreeRenderer> json_renderer_;
    std::unique_ptr<TablePreviewRenderer> table_renderer_;
    std::unique_ptr<ImagePreviewRenderer> image_renderer_;
    std::unique_ptr<MarkdownPreviewRenderer> markdown_renderer_;
    std::unique_ptr<HexPreviewRenderer> hex_renderer_;
    std::unique_ptr<HDF5PreviewRenderer> hdf5_renderer_;

    // File cache
    std::string current_path_;
    PreviewContentType current_type_ = PreviewContentType::Unknown;
    std::string cached_content_;
    bool content_loaded_ = false;

    void LoadFileContent(const std::string& filepath);
    void EnsureContent(const std::string& filepath);
};

} // namespace gui
