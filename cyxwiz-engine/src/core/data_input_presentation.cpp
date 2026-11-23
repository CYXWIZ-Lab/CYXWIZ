#include "data_input_presentation.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace cyxwiz::data_input_view {

std::string FormatCount(int64_t value) {
    const bool negative = value < 0;
    std::string digits = std::to_string(negative ? -value : value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return negative ? "-" + digits : digits;
}

std::string FormatSize(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double size = static_cast<double>(bytes);
    int unit = 0;
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    char buffer[32];
    if (unit == 0) std::snprintf(buffer, sizeof(buffer), "%llu B", static_cast<unsigned long long>(bytes));
    else std::snprintf(buffer, sizeof(buffer), "%.1f %s", size, units[unit]);
    return buffer;
}

std::string FormatSeconds(double seconds) {
    char buffer[32];
    if (seconds < 1.0) std::snprintf(buffer, sizeof(buffer), "%.0f ms", seconds * 1000.0);
    else std::snprintf(buffer, sizeof(buffer), "%.2f s", seconds);
    return buffer;
}

namespace {

// What one row of each kind of data is called.
const char* RowWord(Backend backend) {
    switch (backend) {
        case Backend::ImageFolder: return "images";
        case Backend::AudioFolder: return "clips";
        case Backend::Text: return "samples";
        default: return "rows";
    }
}

std::string KindWord(Backend backend) {
    switch (backend) {
        case Backend::ImageFolder: return "images";
        case Backend::AudioFolder: return "audio";
        case Backend::Text: return "text";
        default: return "table";
    }
}

// How the loaded data is held.
std::string Holding(const StripInput& in) {
    switch (in.backend) {
        case Backend::ParquetDisk: return "kept on disk (Parquet, " + FormatSize(in.bytes) + "), read a row group at a time";
        case Backend::ImageFolder: return "decoded once at the Resize size into cache/images when training starts";
        case Backend::AudioFolder: return "decoded per batch by the feature node; nothing is held";
        default:
            if (in.bytes == 0) return {};
            return in.bytes_estimate ? "about " + FormatSize(in.bytes) + " in memory when fully read"
                                     : FormatSize(in.bytes) + " in memory";
    }
}

}  // namespace

Strip BuildStrip(const StripInput& in) {
    Strip s;
    s.title = in.source.empty() ? "No file or folder chosen"
                                : std::filesystem::path(in.source).filename().string();
    if (s.title.empty()) s.title = in.source;  // a path ending in a separator

    if (in.loading) {
        s.tone = LoadTone::Loading;
        s.state = in.loading_message.empty() ? "Loading" : "Loading: " + in.loading_message;
        s.detail = FormatSeconds(in.loading_seconds);
        s.note = "Runs in the background (Task View shows it). Close the dialog or switch tabs freely; OK and "
                 "Apply wait for it.";
        return s;
    }
    if (in.failed && !in.loaded) {
        s.tone = LoadTone::Failed;
        s.state = "Not loaded";
        s.detail = in.failure;
        s.note = in.source.empty() ? "Choose a file or folder in Source, then Apply."
                                   : "Fix the source in Source, then Apply.";
        return s;
    }
    if (!in.loaded && in.cloud) {
        s.tone = LoadTone::NotLoaded;
        s.state = "Not loaded";
        s.detail = "CyxCloud";
        if (!in.cloud_dataset.empty()) s.detail += " \xC2\xB7 " + in.cloud_dataset;
        if (!in.kind.empty()) s.detail += " \xC2\xB7 " + in.kind;
        if (in.file_size > 0) s.facts.push_back({FormatSize(in.file_size), "in CyxCloud"});
        s.facts.push_back({"no", "local copy"});
        s.note = "A node trains from CyxCloud without a copy here. Preview on this computer reads a sample "
                 "through the gateway.";
        return s;
    }
    if (!in.loaded) {
        s.tone = LoadTone::NotLoaded;
        s.state = "Not loaded";
        if (!in.kind.empty()) s.detail = in.kind;
        if (in.file_size > 0) s.facts.push_back({FormatSize(in.file_size), "on disk"});
        s.note = in.source.empty() ? "Choose a file or folder in Source." : "Apply loads it.";
        return s;
    }

    s.tone = LoadTone::Loaded;
    s.state = in.load_seconds >= 0.0 ? "Loaded in " + FormatSeconds(in.load_seconds) : "Loaded";
    if (!in.dataset_name.empty()) {
        const std::string kind = in.time_column.empty() ? KindWord(in.backend) : "time series by " + in.time_column;
        s.detail = "as " + in.dataset_name + " (" + kind + ")";
    }
    s.facts.push_back({FormatCount(in.rows), RowWord(in.backend)});
    if (in.classes >= 0) s.facts.push_back({FormatCount(in.classes), in.classes == 1 ? "class" : "classes"});
    if (in.cols > 1) s.facts.push_back({FormatCount(in.cols), "columns"});
    if (in.file_size > 0) s.facts.push_back({FormatSize(in.file_size), "on disk"});
    if (in.audit_errors > 0 || in.audit_warnings > 0)
        s.facts.push_back({std::to_string((std::max)(0, in.audit_errors)) + " / " +
                               std::to_string((std::max)(0, in.audit_warnings)),
                           "audit errors / warnings"});
    s.note = Holding(in);
    // An Apply that failed while an earlier load stays registered.
    if (in.failed) s.note = "The last Apply failed: " + in.failure;
    return s;
}

}  // namespace cyxwiz::data_input_view
