#pragma once

// Approval of Python node bundles (TOFIX125 P4 step 4.2, decision D6): a
// bundle's code runs only after the user approved that exact code, recorded
// as bundle id + content hash. Editing node.py changes the hash, so the new
// code asks again. Bundles the user creates in the Engine are approved when
// they are written. Stored in %APPDATA%\cyxwiz\node_approvals.json
// (~/.cyxwiz/node_approvals.json elsewhere).

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz {

struct PythonNodeApproval {
    std::string bundle_id;     // "com.example.acts"
    std::string content_hash;  // "sha256:<hex>" of node.py
    std::string folder;        // where the approved bundle was (for the user's reference)
    std::string approved_on;   // ISO date
    std::string how;           // "approved" (asked) or "created" (written in the Engine)
};

class PythonNodeApprovals {
public:
    explicit PythonNodeApprovals(std::filesystem::path file);

    // The per-user store at PythonNodeApprovalsPath(), loaded on first use.
    static PythonNodeApprovals& User();

    // A missing file is an empty store. False with a reason for an unreadable
    // or invalid file; the store is then empty and is rewritten on the next
    // Approve.
    bool Load(std::string* error = nullptr);

    bool IsApproved(const std::string& bundle_id, const std::string& content_hash) const;
    // Records and saves. False with a reason when the file cannot be written
    // (the approval still holds for this session).
    bool Approve(PythonNodeApproval approval, std::string* error = nullptr);
    // Removes every approval of a bundle id; saves.
    bool Revoke(const std::string& bundle_id, std::string* error = nullptr);
    std::vector<PythonNodeApproval> All() const;

    const std::filesystem::path& File() const { return file_; }

private:
    bool SaveLocked(std::string* error) const;

    std::filesystem::path file_;
    mutable std::mutex mutex_;
    std::vector<PythonNodeApproval> approvals_;
};

std::filesystem::path PythonNodeApprovalsPath();

// Text form of the store (version 1), for tests and the file.
std::string SerializePythonNodeApprovals(const std::vector<PythonNodeApproval>& approvals);
bool ParsePythonNodeApprovals(const std::string& text, std::vector<PythonNodeApproval>& out,
                              std::string& error);

}  // namespace cyxwiz
