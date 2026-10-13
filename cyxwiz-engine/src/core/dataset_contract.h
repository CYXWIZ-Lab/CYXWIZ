#pragma once

// Column roles and the dataset contract (TOFIX134 P3 foundation,
// dashboard_architecture.md L3 with the owner's answers of 2026-10-02).
//
// Each column has one role, decided in this order:
//   1. contract: what the graph says (the Data Input label column is the
//      Target; training roles), never overridden here;
//   2. user: set in Data Studio's Profile tab, saved with the project
//      (column_role_store.h) for this file and its columns;
//   3. inferred: guessed from the column (unique per row: ID; few values:
//      Category; dates; long or mostly unique text: Text; file paths).
// A user role that contradicts the contract (changing the target) is kept
// and shown as a warning; it does not change training.
// Pure: no Arrow, no registry; the profiler supplies the column facts.

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class ColumnRole { Id, Target, Numeric, Category, DateTime, Text, FilePath, Weight, Ignore };
enum class RoleSource { Contract, User, Inferred };

const char* RoleLabel(ColumnRole role);         // "ID", "Target", "Numeric", ...
const char* RoleId(ColumnRole role);            // "id", "target", ... (saved)
std::optional<ColumnRole> RoleFromId(const std::string& id);
const char* RoleSourceLabel(RoleSource source); // "contract", "you", "inferred"
// A feature for models and dashboards: not an ID, the target, a weight or ignored.
bool IsFeatureRole(ColumnRole role);

// What the profiler knows about a column (the facts inference reads).
struct ColumnFacts {
    std::string name;
    enum class Type { Integer, Float, Boolean, Text, Temporal, Other } type = Type::Other;
    size_t rows = 0;         // rows looked at
    size_t non_null = 0;
    size_t distinct = 0;     // distinct non-null values
    double avg_length = 0;   // text: mean characters
    double date_share = 0;   // text: share of values that read as dates (YYYY-MM-DD, ISO date-time, ...)
    double path_share = 0;   // text: share of values that end in an image / audio file extension
};

struct ColumnContract {
    std::string name;
    std::string type;                 // "int", "float", "bool", "text", "date"
    ColumnRole role = ColumnRole::Numeric;
    RoleSource source = RoleSource::Inferred;
    std::string reason;               // why inferred ("unique per row", "3 values"), or "Data Input label"
    std::optional<ColumnRole> user_conflict;  // a user role that contradicts the contract (warning)
};

struct DatasetContract {
    std::string source_key;           // the file it came from (or the dataset name)
    std::string schema_fingerprint;   // names and types
    std::vector<ColumnContract> columns;
    // Saved user roles whose column is not in this table (schema changed); kept, re-applied when it returns.
    std::vector<std::string> unmatched;
    const ColumnContract* Find(const std::string& name) const;
    // The target column, if any.
    std::optional<std::string> Target() const;
};

// The inferred role of one column, with the reason in words.
ColumnRole InferRole(const ColumnFacts& facts, std::string* reason = nullptr);

// "name:type;name:type" -> a short hex fingerprint.
std::string SchemaFingerprint(const std::vector<ColumnFacts>& columns);

// contract > user > inferred, per column.
DatasetContract BuildContract(const std::string& source_key, const std::vector<ColumnFacts>& columns,
                              const std::map<std::string, ColumnRole>& contract_roles,
                              const std::map<std::string, ColumnRole>& user_roles);

// Whether a text reads as a date (2024-05-01, 2024/05/01, 2024-05-01T10:00, 01/05/2024).
bool LooksLikeDate(const std::string& text);
// Whether a text names an image or audio file (by extension).
bool LooksLikeMediaPath(const std::string& text);

}  // namespace cyxwiz
