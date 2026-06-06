// DataInputDialog memory and audit tabs.

#include "node_config_dialog.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace gui {

void DataInputDialog::RenderMemoryTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Memory Management");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Current Status:");
    ImGui::SameLine(120);

    switch (data_load_state_) {
        case DataLoadState::InMemory: {
            const bool disk_backed = (loaded_backend_ == 2);
            if (disk_backed) {
                ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "Loaded via Parquet cache");
                ImGui::SameLine();
                ImGui::TextDisabled("- %s on disk", FormatBytes(loaded_memory_bytes_).c_str());
            } else if (loaded_memory_is_estimate_) {
                ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.4f, 1.0f), "Lazy-loaded");
                ImGui::SameLine();
                ImGui::TextDisabled("- ~%s if fully cached (estimated)",
                                    FormatBytes(loaded_memory_bytes_).c_str());
                ImGui::TextDisabled("  Current RAM is loader cache plus active batches; this is not reserved RAM.");
            } else {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "In Memory");
                ImGui::SameLine();
                ImGui::TextDisabled("- %s", FormatBytes(loaded_memory_bytes_).c_str());
            }

            ImGui::Text("Rows:");
            ImGui::SameLine(120);
            ImGui::Text("%lld", static_cast<long long>(loaded_rows_));
            ImGui::SameLine(200);
            ImGui::Text("Columns:");
            ImGui::SameLine(280);
            ImGui::Text("%lld", static_cast<long long>(loaded_cols_));

            if (disk_backed) {
                ImGui::TextDisabled("  Training reads pages lazily via memory-mapped I/O. "
                                    "RAM use bounded by the OS page cache, not the file size.");
            }

            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.4f, 0.4f, 1.0f));
            if (ImGui::Button("Unload from Memory", ImVec2(180, 0))) {
                UnloadDataset();
            }
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            ImGui::TextDisabled("Free RAM by removing cached data");
            break;
        }
        case DataLoadState::OnDisk:
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.0f, 1.0f), "On Disk (streaming)");
            break;
        case DataLoadState::NotLoaded:
        default:
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Not Loaded");
            ImGui::TextDisabled("Click Apply to load data");
            break;
    }

    if (file_size_ > 0 && data_load_state_ != DataLoadState::InMemory) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        UpdateRAMEstimate();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f),
                           "File size on disk: %.1f MB", estimated_ram_mb_);
        ImGui::TextDisabled("   This is file size, not a RAM reservation. Actual RAM depends on loader mode.");
        ImGui::TextDisabled("   In-memory tabular loads may shrink after integer compaction; lazy loaders decode batches on demand.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(accent, "Advanced");
    ImGui::Spacing();
    if (ImGui::Checkbox("Force disk-backed cache", &force_disk_backed_)) {
        has_changes_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Normally the engine loads small CSVs directly into RAM and spills\n"
            "larger-than-RAM CSVs to a Parquet cache on disk. This checkbox\n"
            "forces the disk-backed cache path even for small files - useful\n"
            "for testing or benchmarking the lazy load code path. Takes effect\n"
            "on the next Apply.");
    }
    ImGui::TextDisabled("   When on, the next Apply writes a Parquet cache in the system temp dir.");
}

namespace {

int ReadIntParam(const gui::MLNode* node, const char* key, int fallback = 0) {
    if (!node) return fallback;
    auto it = node->parameters.find(key);
    if (it == node->parameters.end() || it->second.empty()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

std::string TrimCopy(std::string value) {
    auto not_space = [](unsigned char c) {
        return std::isspace(c) == 0;
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::vector<std::string> SplitAuditExamples(const std::string& value) {
    std::vector<std::string> examples;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = TrimCopy(std::move(item));
        if (!item.empty()) {
            examples.push_back(std::move(item));
        }
    }
    return examples;
}

struct AuditIssueView {
    std::string severity = "info";
    std::string code = "issue";
    std::string message;
    std::vector<std::string> examples;
};

AuditIssueView ParseAuditIssueLine(const std::string& line) {
    AuditIssueView out;
    out.message = line;

    const size_t open = line.find(" [");
    const size_t close = line.find("]: ", open == std::string::npos ? 0 : open);
    if (open == std::string::npos || close == std::string::npos) {
        return out;
    }

    out.severity = line.substr(0, open);
    out.code = line.substr(open + 2, close - (open + 2));

    std::string detail = line.substr(close + 3);
    const std::string examples_marker = " Examples: ";
    const size_t examples_pos = detail.find(examples_marker);
    if (examples_pos != std::string::npos) {
        out.examples = SplitAuditExamples(detail.substr(examples_pos + examples_marker.size()));
        detail = detail.substr(0, examples_pos);
    }
    out.message = TrimCopy(std::move(detail));
    return out;
}

ImVec4 AuditSeverityColor(const std::string& severity,
                          const ImVec4& warn_color,
                          const ImVec4& err_color,
                          const ImVec4& ok_color) {
    if (severity == "error") return err_color;
    if (severity == "warning") return warn_color;
    return ok_color;
}

}  // namespace

void DataInputDialog::RenderAuditTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];
    const ImVec4 ok_color(0.35f, 0.85f, 0.45f, 1.0f);
    const ImVec4 warn_color(0.95f, 0.72f, 0.25f, 1.0f);
    const ImVec4 err_color(0.95f, 0.35f, 0.35f, 1.0f);

    ImGui::Spacing();
    ImGui::TextColored(accent, "DATASET AUDIT");
    ImGui::Separator();
    ImGui::Spacing();

    if (!node_) {
        ImGui::TextDisabled("No node is selected.");
        return;
    }

    const int errors = ReadIntParam(node_, "audit_errors");
    const int warnings = ReadIntParam(node_, "audit_warnings");
    const bool has_audit_result =
        errors > 0 || warnings > 0 || !audit_issue_lines_.empty();
    const bool loaded = data_load_state_ == DataLoadState::InMemory &&
                        !loaded_dataset_name_.empty();
    if (!loaded && !has_audit_result) {
        ImGui::TextDisabled("No loaded dataset to audit.");
        ImGui::TextDisabled("Click Apply to load data and run the audit.");
        return;
    }

    if (errors > 0) {
        ImGui::TextColored(err_color, "Failed");
    } else if (warnings > 0) {
        ImGui::TextColored(warn_color, "Warnings");
    } else {
        ImGui::TextColored(ok_color, "Passed");
    }

    ImGui::Spacing();
    if (!loaded && errors > 0) {
        ImGui::TextDisabled("Dataset was not accepted by the audit.");
    }
    ImGui::Text("Dataset: %s", loaded_dataset_name_.c_str());
    ImGui::Text("Rows / samples: %lld", static_cast<long long>(loaded_rows_));
    if (loaded_cols_ > 0) {
        ImGui::Text("Columns: %lld", static_cast<long long>(loaded_cols_));
    }
    if (loaded_backend_ > 0) {
        ImGui::Text("Backend: %d", loaded_backend_);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::BeginTable("DatasetAuditSummary", 2,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg,
        ImVec2(0, 0))) {
        ImGui::TableSetupColumn("Check", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Errors");
        ImGui::TableSetColumnIndex(1);
        if (errors > 0) ImGui::TextColored(err_color, "%d", errors);
        else ImGui::Text("%d", errors);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Warnings");
        ImGui::TableSetColumnIndex(1);
        if (warnings > 0) ImGui::TextColored(warn_color, "%d", warnings);
        else ImGui::Text("%d", warnings);

        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (errors == 0 && warnings == 0) {
        ImGui::TextDisabled("Metadata and table-level audit checks found no issues.");
    } else {
        ImGui::TextDisabled("Current-session issue details:");
        ImGui::Spacing();
        if (audit_issue_lines_.empty()) {
            ImGui::TextDisabled("No issue details are available for this session.");
        } else if (ImGui::BeginTable("DatasetAuditIssues", 3,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
            ImVec2(0, 240.0f))) {
            ImGui::TableSetupColumn("Severity", ImGuiTableColumnFlags_WidthFixed, 90.0f);
            ImGui::TableSetupColumn("Code", ImGuiTableColumnFlags_WidthFixed, 210.0f);
            ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < audit_issue_lines_.size(); ++i) {
                const AuditIssueView issue = ParseAuditIssueLine(audit_issue_lines_[i]);
                const ImVec4 severity_color =
                    AuditSeverityColor(issue.severity, warn_color, err_color, ok_color);

                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(severity_color, "%s", issue.severity.c_str());

                ImGui::TableSetColumnIndex(1);
                ImGui::TextWrapped("%s", issue.code.c_str());

                ImGui::TableSetColumnIndex(2);
                const bool open = ImGui::TreeNodeEx(
                    "issue",
                    ImGuiTreeNodeFlags_SpanAvailWidth |
                    ImGuiTreeNodeFlags_DefaultOpen,
                    "%s", issue.message.c_str());
                if (open) {
                    if (!issue.examples.empty()) {
                        ImGui::Spacing();
                        ImGui::TextDisabled("Examples");
                        for (const auto& example : issue.examples) {
                            ImGui::BulletText("%s", example.c_str());
                        }
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }
}

} // namespace gui
