// Start page and Create Project model (TOFIX129 piece A2): recent project
// grouping and search, time labels, templates, and Create validation.
#include "../src/core/start_page_presentation.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

using namespace cyxwiz::startpage;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::time_t At(int year, int month, int day, int hour, int minute) {
    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_isdst = -1;
    return std::mktime(&tm);
}

}  // namespace

int main() {
    const std::time_t now = At(2026, 10, 20, 14, 30);  // a Tuesday

    // Time labels.
    Check(WhenLabel(At(2026, 10, 20, 9, 14), now) == "Today, 09:14", "today");
    Check(WhenLabel(At(2026, 10, 19, 18, 2), now) == "Yesterday, 18:02", "yesterday");
    Check(WhenLabel(At(2026, 10, 17, 11, 40), now) == "Sat, 11:40", "this week uses the weekday");
    Check(WhenLabel(At(2026, 9, 12, 8, 0), now) == "12 Sep", "same year: day and month");
    Check(WhenLabel(At(2025, 8, 3, 8, 0), now) == "3 Aug 2025", "older year: day, month, year");
    Check(WhenLabel(0, now) == "Unknown", "unknown time");

    const std::vector<RecentEntry> entries = {
        {"mnist", "D:/projects/mnist/mnist.cyxwiz", At(2026, 10, 17, 11, 40)},
        {"Project Berean", "D:/dev/examples/thinking_llm/Project Berean/berean.cyxwiz", At(2026, 10, 20, 9, 14)},
        {"ner_sequence", "D:/projects/ner_sequence/ner.cyxwiz", At(2026, 10, 5, 8, 0)},
        {"speech", "D:/projects/speech/speech.cyxwiz", At(2026, 8, 3, 8, 0)},
        {"thinking_llm", "D:/dev/examples/thinking_llm/thinking.cyxwiz", At(2026, 10, 19, 18, 2)}};

    // Grouping: newest first, empty groups left out.
    const RecentView all = BuildRecentView(entries, "", now);
    Check(all.total == 5 && all.shown == 5 && !all.no_projects && !all.no_match, "all shown");
    Check(all.groups.size() == 3, "three groups");
    Check(all.groups[0].label == "This week" && all.groups[0].rows.size() == 3, "this week has 3");
    Check(all.groups[0].rows[0].name == "Project Berean", "newest first");
    Check(all.groups[1].label == "This month" && all.groups[1].rows.size() == 1, "this month has 1");
    Check(all.groups[2].label == "Older" && all.groups[2].rows[0].name == "speech", "older");
    Check(all.groups[0].rows[0].folder.find("Project Berean") != std::string::npos, "folder is the project folder");

    // Search covers the name and the folder.
    const RecentView by_folder = BuildRecentView(entries, "THINKING_llm", now);
    Check(by_folder.shown == 2, "folder search finds both projects under thinking_llm");
    const RecentView by_name = BuildRecentView(entries, "ner", now);
    Check(by_name.shown == 1 && by_name.groups.size() == 1 && by_name.groups[0].key == "month", "name search");
    const RecentView none = BuildRecentView(entries, "zzz", now);
    Check(none.no_match && !none.no_projects && none.groups.empty(), "no match is reported");
    const RecentView empty = BuildRecentView({}, "", now);
    Check(empty.no_projects && !empty.no_match, "no projects is reported");

    // Templates: six, each with a default name.
    Check(ProjectTemplates().size() == 6, "six templates");
    for (const auto& t : ProjectTemplates())
        Check(t.name && t.description && t.default_project_name && t.default_project_name[0], "template complete");

    // Create validation.
    {
        CreateCheck c = CheckCreate({"", "C:/Projects", false});
        Check(!c.can_create && c.reason == "Enter a project name.", "name required");
        c = CheckCreate({"mnist", "", false});
        Check(!c.can_create && c.reason == "Choose a location.", "location required");
        c = CheckCreate({"a/b", "C:/Projects", false});
        Check(!c.can_create && !c.warning.empty(), "slash rejected");
        c = CheckCreate({"what?", "C:/Projects", false});
        Check(!c.can_create, "question mark rejected");
        c = CheckCreate({"CON", "C:/Projects", false});
        Check(!c.can_create && c.reason.find("reserved") != std::string::npos, "reserved name rejected");
        c = CheckCreate({"draft.", "C:/Projects", false});
        Check(!c.can_create, "trailing dot rejected");
        c = CheckCreate({"mnist", "C:/Projects", true});
        Check(!c.can_create && c.warning.find("already exists") != std::string::npos, "existing folder rejected");
        c = CheckCreate({"  mnist  ", "C:/Projects", false});
        Check(!c.can_create, "trailing space rejected");
        c = CheckCreate({"mnist", "C:/Projects", false});
        Check(c.can_create && c.reason.empty() && c.warning.empty(), "valid name accepted");
        Check(c.target.find("mnist") != std::string::npos && c.preview == c.target, "preview is the target folder");
    }

    // New script checks.
    {
        Check(ScriptFileName({"train", "D:/p/scripts", true, false}) == "train.py", "py extension added");
        Check(ScriptFileName({"train", "D:/p/scripts", false, false}) == "train.cyx", "cyx extension added");
        Check(ScriptFileName({"train.PY", "D:/p/scripts", false, false}) == "train.PY", "existing extension kept");
        CreateCheck c = CheckNewScript({"", "D:/p/scripts", true, false});
        Check(!c.can_create && c.reason == "Enter a script name.", "script name required");
        c = CheckNewScript({"train", "", true, false});
        Check(!c.can_create && c.reason == "Choose a folder.", "folder required");
        c = CheckNewScript({"a:b", "D:/p/scripts", true, false});
        Check(!c.can_create && !c.warning.empty(), "colon rejected");
        c = CheckNewScript({"nul", "D:/p/scripts", true, false});
        Check(!c.can_create, "reserved name rejected");
        c = CheckNewScript({"train", "D:/p/scripts", true, true});
        Check(c.can_create && c.warning.find("replaces") != std::string::npos, "existing file warns but allows");
        c = CheckNewScript({"train", "D:/p/scripts", true, false});
        Check(c.can_create && c.warning.empty() && c.target.find("train.py") != std::string::npos, "valid script");
        Check(ScriptTemplate("train.py", true).find("import pycyxwiz") != std::string::npos, "python template");
        Check(ScriptTemplate("p.cyx", false).find("# p.cyx") == 0, "cyx template");
    }

    std::cout << "start page presentation: grouping, search, time labels, templates, create checks. OK\n";
    return 0;
}
