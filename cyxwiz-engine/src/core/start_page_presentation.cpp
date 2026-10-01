#include "start_page_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace cyxwiz::startpage {

namespace {

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string Trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool Local(std::time_t t, std::tm& out) {
#ifdef _WIN32
    return localtime_s(&out, &t) == 0;
#else
    return localtime_r(&t, &out) != nullptr;
#endif
}

long DayNumber(const std::tm& tm) {
    // Days since a fixed origin, good enough to compare calendar days.
    return static_cast<long>(tm.tm_year) * 400 + tm.tm_yday;
}

const char* kMonths[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
const char* kDays[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

std::string Folder(const std::string& path) {
    std::filesystem::path p(path);
    return p.has_parent_path() ? p.parent_path().string() : path;
}

}  // namespace

std::string WhenLabel(std::time_t when, std::time_t now) {
    std::tm w{}, n{};
    if (when <= 0 || !Local(when, w) || !Local(now, n)) return "Unknown";
    char time_part[16];
    std::snprintf(time_part, sizeof(time_part), "%02d:%02d", w.tm_hour, w.tm_min);
    if (w.tm_year == n.tm_year && w.tm_yday == n.tm_yday) return std::string("Today, ") + time_part;
    const long days = (n.tm_year == w.tm_year) ? (n.tm_yday - w.tm_yday)
                      : static_cast<long>(std::difftime(now, when) / 86400.0) + 1;
    if (days == 1) return std::string("Yesterday, ") + time_part;
    if (days > 1 && days < 7) return std::string(kDays[w.tm_wday]) + ", " + time_part;
    char buf[32];
    if (w.tm_year == n.tm_year) std::snprintf(buf, sizeof(buf), "%d %s", w.tm_mday, kMonths[w.tm_mon]);
    else std::snprintf(buf, sizeof(buf), "%d %s %d", w.tm_mday, kMonths[w.tm_mon], w.tm_year + 1900);
    return buf;
}

RecentView BuildRecentView(const std::vector<RecentEntry>& entries, const std::string& query, std::time_t now) {
    RecentView view;
    view.total = static_cast<int>(entries.size());
    view.no_projects = entries.empty();
    const std::string q = Lower(Trim(query));
    std::tm n{};
    const bool have_now = Local(now, n);

    RecentGroup week{"week", "This week", {}};
    RecentGroup month{"month", "This month", {}};
    RecentGroup older{"older", "Older", {}};

    std::vector<const RecentEntry*> sorted;
    for (const auto& e : entries) sorted.push_back(&e);
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const RecentEntry* a, const RecentEntry* b) { return a->last_opened > b->last_opened; });

    for (const RecentEntry* e : sorted) {
        const std::string folder = Folder(e->path);
        if (!q.empty() && Lower(e->name).find(q) == std::string::npos && Lower(folder).find(q) == std::string::npos)
            continue;
        RecentRow row{e->name, e->path, folder, WhenLabel(e->last_opened, now)};
        std::tm w{};
        const double age = std::difftime(now, e->last_opened);
        if (age < 7.0 * 86400.0) week.rows.push_back(row);
        else if (have_now && Local(e->last_opened, w) && w.tm_year == n.tm_year && w.tm_mon == n.tm_mon) month.rows.push_back(row);
        else older.rows.push_back(row);
        ++view.shown;
    }
    for (auto* g : {&week, &month, &older})
        if (!g->rows.empty()) view.groups.push_back(std::move(*g));
    view.no_match = !view.no_projects && view.shown == 0;
    return view;
}

const std::vector<ProjectTemplate>& ProjectTemplates() {
    static const std::vector<ProjectTemplate> templates = {
        {"Blank project", "An empty project with the standard folders.", "New CyxWiz Project"},
        {"Classic ML workflow", "Tabular data, feature preparation and a classic model.", "Classic ML Project"},
        {"Deep Learning workflow", "A neural network trained from a dataset.", "Deep Learning Project"},
        {"Tabular project", "Rows and columns: CSV, Parquet, SQL.", "Tabular ML Project"},
        {"Vision project", "Images, labels and augmentation.", "Vision ML Project"},
        {"NLP project", "Text, tokenizers and sequence models.", "NLP Project"}};
    return templates;
}

namespace {

// Returns an empty string when `name` is usable as a file or folder name,
// else the reason. `raw` is the untrimmed text (to catch a trailing space).
std::string NameProblem(const std::string& name, const std::string& raw) {
    static const std::string bad = "<>:\"/\\|?*";
    for (char ch : name) {
        if (bad.find(ch) != std::string::npos || static_cast<unsigned char>(ch) < 32)
            return "The name contains a character that cannot be used in a file name.";
    }
    if (name.back() == '.' || (!raw.empty() && raw.back() == ' ')) return "A name cannot end with a dot or a space.";
    static const char* reserved[] = {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "lpt1", "lpt2", "lpt3"};
    std::string lower = Lower(name);
    const auto dot = lower.find('.');
    if (dot != std::string::npos) lower = lower.substr(0, dot);
    for (const char* r : reserved)
        if (lower == r) return "That name is reserved by Windows.";
    return {};
}

}  // namespace

CreateCheck CheckCreate(const CreateInputs& in) {
    CreateCheck c;
    const std::string name = Trim(in.name);
    const std::string location = Trim(in.location);
    if (!name.empty() && !location.empty()) {
        c.target = (std::filesystem::path(location) / name).string();
        c.preview = c.target;
    } else {
        c.preview = "The folder named above, once you enter a name and a location.";
    }
    if (name.empty()) {
        c.reason = "Enter a project name.";
        return c;
    }
    if (const std::string problem = NameProblem(name, in.name); !problem.empty()) {
        c.reason = problem;
        c.warning = problem + " Avoid < > : \" / \\ | ? * and a trailing dot or space.";
        return c;
    }
    if (location.empty()) {
        c.reason = "Choose a location.";
        return c;
    }
    if (in.target_exists) {
        c.reason = "A folder with this name already exists here.";
        c.warning = "A folder with this name already exists here. Choose another name or location.";
        return c;
    }
    c.can_create = true;
    return c;
}

std::string ScriptFileName(const ScriptInputs& in) {
    const std::string name = Trim(in.name);
    if (name.empty()) return {};
    const std::string lower = Lower(name);
    auto ends = [&lower](const char* ext) {
        const std::string e(ext);
        return lower.size() > e.size() && lower.compare(lower.size() - e.size(), e.size(), e) == 0;
    };
    if (ends(".py") || ends(".cyx")) return name;
    return name + (in.python ? ".py" : ".cyx");
}

CreateCheck CheckNewScript(const ScriptInputs& in) {
    CreateCheck c;
    const std::string file = ScriptFileName(in);
    const std::string folder = Trim(in.folder);
    if (!file.empty() && !folder.empty()) {
        c.target = (std::filesystem::path(folder) / file).string();
        c.preview = c.target;
    } else {
        c.preview = "The file named above, once you enter a name and a folder.";
    }
    if (file.empty()) {
        c.reason = "Enter a script name.";
        return c;
    }
    if (const std::string problem = NameProblem(Trim(in.name), in.name); !problem.empty()) {
        c.reason = problem;
        c.warning = problem + " Avoid < > : \" / \\ | ? * and a trailing dot or space.";
        return c;
    }
    if (folder.empty()) {
        c.reason = "Choose a folder.";
        return c;
    }
    if (in.target_exists) c.warning = "A file with this name already exists in that folder. Creating replaces it.";
    c.can_create = true;
    return c;
}

std::string ScriptTemplate(const std::string& file_name, bool python) {
    if (python) {
        return "# " + file_name + "\n\nimport pycyxwiz\n\n\ndef main():\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n";
    }
    return "# " + file_name + "\n# CyxWiz script: define your ML pipeline here.\n\n";
}

}  // namespace cyxwiz::startpage
