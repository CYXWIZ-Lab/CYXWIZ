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
    static const std::string bad = "<>:\"/\\|?*";
    for (char ch : name) {
        if (bad.find(ch) != std::string::npos || static_cast<unsigned char>(ch) < 32) {
            c.reason = "The name contains a character that cannot be used in a folder name.";
            c.warning = "Remove these characters from the name: < > : \" / \\ | ? *";
            return c;
        }
    }
    if (name.back() == '.' || in.name.back() == ' ') {
        c.reason = "A folder name cannot end with a dot or a space.";
        c.warning = c.reason;
        return c;
    }
    static const char* reserved[] = {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "lpt1", "lpt2", "lpt3"};
    const std::string lower = Lower(name);
    for (const char* r : reserved) {
        if (lower == r) {
            c.reason = "That name is reserved by Windows.";
            c.warning = c.reason + " Choose another name.";
            return c;
        }
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

}  // namespace cyxwiz::startpage
