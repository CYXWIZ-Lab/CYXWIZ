// Approval of Python node bundles (TOFIX125 P4 step 4.2, D6): approvals are
// per bundle id + content hash, survive a restart (a new store reading the
// same file), and a changed hash is not approved.
#include "../src/core/python_node_approvals.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
    std::cout << "ok   " << message << '\n';
}

const std::string kHashA = "sha256:" + std::string(64, 'a');
const std::string kHashB = "sha256:" + std::string(64, 'b');

}  // namespace

int main() {
    const fs::path work = fs::temp_directory_path() / "cyxwiz_test_python_node_approvals";
    fs::remove_all(work);
    const fs::path file = work / "sub" / "node_approvals.json";

    {
        cyxwiz::PythonNodeApprovals store(file);
        std::string error;
        Check(store.Load(&error) && store.All().empty(), "a missing file is an empty store");
        Check(!store.IsApproved("com.example.acts", kHashA), "nothing is approved at first");
        Check(store.Approve({"com.example.acts", kHashA, "D:/p/nodes/acts.cyxnode", "2026-10-17", "approved"},
                            &error),
              "approve and save (the folder is created): " + error);
        Check(store.IsApproved("com.example.acts", kHashA), "the approved hash is approved");
        Check(!store.IsApproved("com.example.acts", kHashB), "an edited node.py (other hash) is not approved");
        Check(!store.IsApproved("com.example.other", kHashA), "another bundle id is not approved");
        Check(store.Approve({"com.example.acts", kHashA, "D:/q/nodes/acts.cyxnode", "2026-10-18", "approved"}),
              "approving the same code again");
        Check(store.All().size() == 1 && store.All()[0].folder == "D:/q/nodes/acts.cyxnode",
              "approving the same code again updates the record, no duplicate");
        Check(store.Approve({"com.example.acts", kHashB, "", "2026-10-18", "created"}), "a second version");
    }
    {
        cyxwiz::PythonNodeApprovals restarted(file);
        std::string error;
        Check(restarted.Load(&error), "the file reads back: " + error);
        Check(restarted.IsApproved("com.example.acts", kHashA) && restarted.IsApproved("com.example.acts", kHashB),
              "approvals survive a restart");
        const auto all = restarted.All();
        Check(all.size() == 2 && all[1].how == "created", "fields read back");
        Check(restarted.Revoke("com.example.acts") && restarted.All().empty(), "revoke removes every version");
    }
    {
        cyxwiz::PythonNodeApprovals again(file);
        Check(again.Load() && again.All().empty(), "a revoke is saved");
    }

    // Damaged or foreign files.
    {
        std::ofstream(file) << "{ not json";
        cyxwiz::PythonNodeApprovals store(file);
        std::string error;
        Check(!store.Load(&error) && error.find("not valid JSON") != std::string::npos && store.All().empty(),
              "a damaged file is reported and nothing is approved: " + error);
    }
    {
        std::ofstream(file) << R"({"version": 2, "approvals": []})";
        cyxwiz::PythonNodeApprovals store(file);
        std::string error;
        Check(!store.Load(&error) && error.find("version 1") != std::string::npos, "another version is refused");
    }
    {
        std::vector<cyxwiz::PythonNodeApproval> parsed;
        std::string error;
        Check(cyxwiz::ParsePythonNodeApprovals(
                  R"({"version": 1, "approvals": [{"bundle_id": "x", "content_hash": "md5:1"},
                                                  {"bundle_id": "", "content_hash": "sha256:1"},
                                                  {"bundle_id": "ok", "content_hash": "sha256:1"}]})",
                  parsed, error) &&
                  parsed.size() == 1 && parsed[0].bundle_id == "ok",
              "entries without an id or a sha256 hash are ignored");
    }
    Check(cyxwiz::PythonNodeApprovalsPath().filename() == "node_approvals.json", "the store's file name");

    fs::remove_all(work);
    std::cout << "test_python_node_approvals: all checks passed\n";
    return 0;
}
