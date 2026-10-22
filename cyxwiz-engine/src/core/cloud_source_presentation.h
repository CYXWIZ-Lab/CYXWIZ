#pragma once
// The Data Input dialog's CyxCloud source panel (TOFIX164 slice 4b, mockup
// approved 2026-10-22): what the account can pick, what was picked, how the
// connection stands and what Apply may do. Pure data in, words out; the
// dialog draws it.

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::cloud_source_view {

enum class Tone { Muted, Accent, Success, Warning, Error };

// A dataset the gateway lists for the account (the cloud browser's rows).
struct DatasetIn {
    std::string id;
    std::string name;
    int trust = 4;  // network::TrustLevel: 0 Self, 1 Signed, 2 Verified, 3 Attested, 4 Untrusted
    int64_t size_bytes = 0;
    int file_count = 0;
};

// A file of the chosen dataset.
struct FileIn {
    std::string id;
    std::string name;  // its name in the dataset; the extension says the format
    int64_t size_bytes = 0;
};

// What the graph remembers: the Data Input's cloud_* parameters.
struct Remembered {
    std::string dataset_name;
    std::string dataset_id;
    std::string file_name;
    std::string file_id;
    uint64_t size_bytes = 0;
    bool Any() const { return !file_id.empty(); }
};

struct Input {
    bool signed_in = false;
    bool browser_opened = false;  // the Cloud Browser created the Engine's gateway client
    bool connected = false;
    std::string gateway;        // host:port the Engine uses (servers.cloud_gateway or derived)
    std::string account;        // the signed-in e-mail
    std::string connect_error;  // the last connect failure's words, "" when none
    bool loading_datasets = false;
    std::string datasets_error;
    std::vector<DatasetIn> datasets;
    std::string selected_dataset_id;
    bool loading_files = false;
    std::vector<FileIn> files;  // of the selected dataset
    std::string selected_file_id;
    Remembered remembered;
};

struct DatasetRow {
    std::string id;
    std::string name;
    std::string trust;  // "Self", "Shared, verified", ...
    std::string meta;   // "3 files · 158.9 MB"
    bool selected = false;
};

struct FileRow {
    std::string id;
    std::string name;
    std::string meta;  // "Parquet · 149.6 MB" or "Text · 4.1 MB · not for remote training"
    bool trainable = false;
    bool selected = false;
};

struct Pair {
    std::string key;
    std::string value;
};

struct View {
    // The connection line beside the source control.
    std::string connection;
    Tone connection_tone = Tone::Muted;
    bool offer_sign_in = false;
    bool offer_try_again = false;
    bool offer_open_browser = false;

    // The dataset picker and the file list.
    std::string datasets_hint;  // under the picker: what the list holds, or why it is empty
    std::vector<DatasetRow> datasets;
    std::string files_hint;
    std::vector<FileRow> files;

    // The Detected line for the chosen (or remembered) file.
    std::string detected;
    std::string detected_note;

    // Without a connection: what the graph remembers, and why it matters.
    std::vector<Pair> remembered;
    std::string remembered_note;

    bool can_apply = false;
    std::string apply_blocked_reason;
};

View Build(const Input& in);

// "Self", "Signed", "Verified", "Attested", "Unverified".
std::string TrustWord(int trust);
// "Parquet · 149.6 MB", with " · not for remote training" for other formats.
std::string FileMeta(const FileIn& file);
// The status line Apply shows for a cloud source (slice 4b.1: the choice is
// saved; the sample read that feeds preview comes with 4b.2).
std::string ApplyMessage(const Remembered& chosen);
// The first 8 characters of an id, for prose.
std::string ShortId(const std::string& id);

}  // namespace cyxwiz::cloud_source_view
