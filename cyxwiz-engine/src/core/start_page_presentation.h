// Start page and Create Project presentation model (TOFIX129 piece A2).
//
// Pure data in, data out: grouping and filtering of recent projects, the
// project templates, and the Create dialog's validation and preview. The
// start page, File > New Project and their test all use this one model.
#pragma once

#include <ctime>
#include <string>
#include <vector>

namespace cyxwiz::startpage {

struct RecentEntry {
    std::string name;
    std::string path;          // .cyxwiz file
    std::time_t last_opened = 0;
};

struct RecentRow {
    std::string name;
    std::string path;
    std::string folder;        // folder that holds the project file
    std::string when;          // "Today, 09:14", "Yesterday, 18:02", "Mon, 11:40", "12 Sep", "3 Aug 2025"
};

struct RecentGroup {
    std::string key;           // week, month, older
    std::string label;         // This week, This month, Older
    std::vector<RecentRow> rows;
};

struct RecentView {
    std::vector<RecentGroup> groups;  // empty groups left out
    int total = 0;                    // entries before the search
    int shown = 0;                    // entries after the search
    bool no_projects = false;         // nothing recent at all
    bool no_match = false;            // recent projects exist, none match the search
};

// Search matches the name or the folder, case-insensitively.
RecentView BuildRecentView(const std::vector<RecentEntry>& entries, const std::string& query, std::time_t now);

// Short, local-time label for a recent project.
std::string WhenLabel(std::time_t when, std::time_t now);

struct ProjectTemplate {
    const char* name;
    const char* description;
    const char* default_project_name;
};
const std::vector<ProjectTemplate>& ProjectTemplates();

struct CreateInputs {
    std::string name;
    std::string location;
    bool target_exists = false;   // the caller checks the file system
};

struct CreateCheck {
    bool can_create = false;
    std::string reason;           // why Create is disabled; empty when it is not
    std::string warning;          // shown under the fields (exists, bad character)
    std::string preview;          // the folder that will be created, or a hint
    std::string target;           // location/name when both are present
};

// Name rules: not empty after trimming, no path separators or characters
// Windows rejects, not a reserved device name, no trailing dot or space.
CreateCheck CheckCreate(const CreateInputs& inputs);

}  // namespace cyxwiz::startpage
