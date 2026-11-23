#include "cloud_source_presentation.h"

#include "cloud_dataset_file_format.h"
#include "data_input_presentation.h"

namespace cyxwiz::cloud_source_view {

namespace {

using data_input_view::FormatSize;

const char* kNotTrainable = "not for remote training";

std::string FileWord(int count) { return count == 1 ? "1 file" : std::to_string(count) + " files"; }

}  // namespace

std::string TrustWord(int trust) {
    switch (trust) {
        case 0: return "Self";
        case 1: return "Signed";
        case 2: return "Verified";
        case 3: return "Attested";
        default: return "Unverified";
    }
}

std::string ShortId(const std::string& id) { return id.size() > 8 ? id.substr(0, 8) : id; }

std::string FileMeta(const FileIn& file) {
    std::string meta = CloudFileFormatWord(file.name) + " \xC2\xB7 " + FormatSize(static_cast<uint64_t>(file.size_bytes));
    if (!CloudDatasetFileFormatKnown(file.name)) meta += std::string(" \xC2\xB7 ") + kNotTrainable;
    return meta;
}

std::string ApplyMessage(const Remembered& chosen) {
    return "Source saved: CyxCloud file " + chosen.file_name + " in " + chosen.dataset_name +
           ". A node trains from it; preview on this computer comes with sample reads.";
}

View Build(const Input& in) {
    View v;

    // --- the connection line
    if (!in.signed_in) {
        v.connection = "Not connected: sign in to CyxWiz first (Account, top right)";
        v.connection_tone = Tone::Warning;
        v.offer_sign_in = true;
    } else if (!in.connected) {
        if (!in.browser_opened) {
            v.connection = "Not connected yet: open the Cloud Browser once (sidebar, Network); it connects to " +
                           in.gateway;
            v.connection_tone = Tone::Warning;
        } else if (in.connect_error.empty()) {
            v.connection = "Connecting to " + in.gateway + "...";
            v.connection_tone = Tone::Accent;
        } else {
            v.connection = "Gateway unreachable at " + in.gateway + " (" + in.connect_error +
                           "): check the server in Preferences > Servers, or start the gateway";
            v.connection_tone = Tone::Warning;
            v.offer_try_again = true;
        }
    } else {
        v.connection = "Connected to " + in.gateway + " as " + in.account;
        v.connection_tone = Tone::Success;
        v.offer_open_browser = true;
    }

    // --- the dataset picker and the files of the chosen dataset
    const DatasetIn* dataset = nullptr;
    const FileIn* file = nullptr;
    if (in.connected) {
        if (in.loading_datasets) {
            v.datasets_hint = "Loading your datasets...";
        } else if (!in.datasets_error.empty()) {
            v.datasets_hint = "Could not list datasets: " + in.datasets_error;
        } else if (in.datasets.empty()) {
            v.datasets_hint = "No datasets yet. Upload files in the Cloud browser, then Refresh.";
        } else {
            v.datasets_hint = "Your datasets and the ones shared with you; Refresh in the Cloud browser after an upload.";
        }
        for (const auto& d : in.datasets) {
            DatasetRow row;
            row.id = d.id;
            row.name = d.name;
            row.trust = TrustWord(d.trust);
            row.meta = FileWord(d.file_count) + " \xC2\xB7 " + FormatSize(static_cast<uint64_t>(d.size_bytes));
            row.selected = d.id == in.selected_dataset_id;
            if (row.selected) dataset = &d;
            v.datasets.push_back(std::move(row));
        }
        if (dataset) {
            if (in.loading_files) {
                v.files_hint = "Loading files...";
            } else if (in.files.empty()) {
                v.files_hint = "This dataset has no files.";
            }
            for (const auto& f : in.files) {
                FileRow row;
                row.id = f.id;
                row.name = f.name;
                row.meta = FileMeta(f);
                row.trainable = CloudDatasetFileFormatKnown(f.name);
                row.selected = f.id == in.selected_file_id;
                if (row.selected) file = &f;
                v.files.push_back(std::move(row));
            }
        } else if (!in.datasets.empty()) {
            v.files_hint = "Pick a dataset to see its files.";
        }
    }

    // --- Detected: the chosen file, or what the graph remembers
    std::string detected_name, detected_file_id, detected_dataset_id;
    uint64_t detected_size = 0;
    if (file) {
        detected_name = file->name;
        detected_file_id = file->id;
        detected_dataset_id = dataset ? dataset->id : std::string();
        detected_size = static_cast<uint64_t>(file->size_bytes);
    } else if (in.remembered.Any()) {
        // No live pick yet (or the listing failed): the graph's own record.
        detected_name = in.remembered.file_name;
        detected_file_id = in.remembered.file_id;
        detected_dataset_id = in.remembered.dataset_id;
        detected_size = in.remembered.size_bytes;
    }
    if (!detected_name.empty()) {
        const bool trainable = CloudDatasetFileFormatKnown(detected_name);
        if (trainable) {
            v.detected = CloudFileFormatWord(detected_name) + " table, " + FormatSize(detected_size) + ", file " +
                         ShortId(detected_file_id) + " in dataset " + ShortId(detected_dataset_id);
            v.detected_note =
                "Train on a node: the node streams it from CyxCloud; nothing is sent from this computer. "
                "Preview and training on this computer read a sample through the gateway.";
        } else {
            v.detected = CloudFileFormatWord(detected_name) + " file, " + FormatSize(detected_size) + ": " + kNotTrainable;
            v.detected_note = "Remote nodes train from Arrow IPC or Parquet files only; convert the file first.";
        }
    }

    // --- the graph's own record, until a live pick replaces it
    if (!file && in.remembered.Any()) {
        v.remembered = {{"Dataset", in.remembered.dataset_name},
                        {"Dataset id", in.remembered.dataset_id},
                        {"File", in.remembered.file_name},
                        {"File id", in.remembered.file_id},
                        {"Size", FormatSize(in.remembered.size_bytes)},
                        {"Gateway", in.gateway + " (from the server address)"}};
        v.remembered_note =
            in.connected
                ? "These are saved in the graph; pick a dataset and file above to change them."
                : "These are saved in the graph, so a node can still be sent the job once you are signed in. Without "
                  "a connection the Engine cannot list datasets, mint the node's access token, or read a sample.";
    }

    // --- Apply
    if (!in.signed_in) {
        v.apply_blocked_reason = "Sign in to CyxWiz to use a cloud dataset";
    } else if (!in.connected) {
        v.apply_blocked_reason = "Connect to CyxCloud first";
    } else if (!file) {
        v.apply_blocked_reason = dataset ? "Pick a file in the dataset" : "Pick a dataset and a file";
    } else if (!CloudDatasetFileFormatKnown(file->name)) {
        v.apply_blocked_reason = "Pick an Arrow IPC or Parquet file; remote nodes train from those only";
    } else {
        v.can_apply = true;
    }
    return v;
}

}  // namespace cyxwiz::cloud_source_view
