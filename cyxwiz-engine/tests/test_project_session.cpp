// Reopening the last project (TOFIX129 B9-3): the recent project to open and
// the graph a project had open, kept relative to the project.
#include "../src/core/project_session.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>

namespace {

int failures = 0;

void Same(const std::string& got, const std::string& want, const std::string& what) {
    if (got == want) return;
    std::printf("FAIL: %s\n  got:  %s\n  want: %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

using namespace cyxwiz::project_session;

}  // namespace

int main() {
    // The newest project that still exists
    const std::set<std::string> on_disk = {"D:/work/b/b.cyxwiz", "D:/work/c/c.cyxwiz"};
    const auto exists = [&on_disk](const std::string& p) { return on_disk.count(p) > 0; };
    Same(FirstExisting({"D:/gone/a.cyxwiz", "D:/work/b/b.cyxwiz", "D:/work/c/c.cyxwiz"}, exists), "D:/work/b/b.cyxwiz",
         "first existing: a moved project is skipped");
    Same(FirstExisting({"D:/gone/a.cyxwiz"}, exists), "", "first existing: none left");
    Same(FirstExisting({}, exists), "", "first existing: no recent projects");
    Same(FirstExisting({"", "D:/work/c/c.cyxwiz"}, exists), "D:/work/c/c.cyxwiz", "first existing: an empty entry is skipped");

    // The graph kept in the project file
    Same(StoredGraph("D:/work/b/cyxgraph/model.cyxgraph", "D:/work/b"), "cyxgraph/model.cyxgraph", "stored: inside, relative");
    Same(StoredGraph("D:/work/b/./cyxgraph/../model.cyxgraph", "D:/work/b/"), "model.cyxgraph", "stored: normalised");
    Same(StoredGraph("D:/elsewhere/model.cyxgraph", "D:/work/b"), "D:/elsewhere/model.cyxgraph", "stored: outside, absolute");
    Same(StoredGraph("D:/work/bb/model.cyxgraph", "D:/work/b"), "D:/work/bb/model.cyxgraph",
         "stored: a sibling folder with the same prefix is outside");
    Same(StoredGraph("", "D:/work/b"), "", "stored: no graph");

    // Back to a full path
    namespace fs = std::filesystem;
    Same(ResolvedGraph("cyxgraph/model.cyxgraph", "D:/work/b"), (fs::path("D:/work/b") / "cyxgraph/model.cyxgraph").lexically_normal().string(),
         "resolved: under the root");
    Same(ResolvedGraph("D:/elsewhere/model.cyxgraph", "D:/work/b"), fs::path("D:/elsewhere/model.cyxgraph").lexically_normal().string(),
         "resolved: absolute stays");
    Same(ResolvedGraph("", "D:/work/b"), "", "resolved: none");

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("Project session passed\n");
    return EXIT_SUCCESS;
}
