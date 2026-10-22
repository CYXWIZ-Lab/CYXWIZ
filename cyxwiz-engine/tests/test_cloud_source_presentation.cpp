// The Data Input dialog's CyxCloud source panel (TOFIX164 slice 4b), on the
// Berean raw files as uploaded to the local gateway.
#include "core/cloud_source_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace cyxwiz::cloud_source_view;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

Input Connected() {
    Input in;
    in.signed_in = true;
    in.connected = true;
    in.gateway = "192.168.1.116:50060";
    in.account = "e2e-user@local.test";
    in.datasets = {{"3b9d2c1e-5f60-4a71-8b92-c3d4e5f60718", "berean-raw-v2", 0, 166615611, 3},
                   {"f8ccb4ed-dda2-4b37-a693-61657fd6a257", "tiny-causal-lm", 2, 6088, 2}};
    in.selected_dataset_id = in.datasets[0].id;
    in.files = {{"2f1c7f8e-7d3a-4b4e-9c2d-1a2b3c4d5e6f", "simplewiki_20231101.parquet", 156885218},
                {"a1", "bsb.txt", 4331393},
                {"a2", "eng-kjv_vpl.zip", 5225899}};
    return in;
}

void CheckConnectedAndPicked() {
    Input in = Connected();
    in.selected_file_id = in.files[0].id;
    const View v = Build(in);
    Check(v.connection == "Connected to 192.168.1.116:50060 as e2e-user@local.test" && v.connection_tone == Tone::Success,
          "connection line: " + v.connection);
    Check(v.offer_open_browser && !v.offer_sign_in && !v.offer_try_again, "connected offers the Cloud browser only");
    Check(v.datasets.size() == 2 && v.datasets[0].selected && v.datasets[0].trust == "Self" &&
              v.datasets[0].meta == "3 files \xC2\xB7 158.9 MB" && v.datasets[1].trust == "Verified",
          "dataset rows: " + v.datasets[0].meta);
    Check(Contains(v.datasets_hint, "shared with you"), "datasets hint: " + v.datasets_hint);
    Check(v.files.size() == 3 && v.files[0].trainable && v.files[0].selected &&
              v.files[0].meta == "Parquet \xC2\xB7 149.6 MB",
          "parquet row: " + v.files[0].meta);
    Check(!v.files[1].trainable && v.files[1].meta == "Text \xC2\xB7 4.1 MB \xC2\xB7 not for remote training",
          "text row: " + v.files[1].meta);
    Check(!v.files[2].trainable && Contains(v.files[2].meta, "ZIP"), "zip row: " + v.files[2].meta);
    Check(v.detected == "Parquet table, 149.6 MB, file 2f1c7f8e in dataset 3b9d2c1e", "detected: " + v.detected);
    Check(Contains(v.detected_note, "nothing is sent from this computer"), "detected note");
    Check(v.can_apply && v.apply_blocked_reason.empty(), "Apply allowed");
    Check(v.remembered.empty(), "nothing to remember while connected");
}

void CheckPickingRules() {
    Input in = Connected();
    View v = Build(in);
    Check(!v.can_apply && v.apply_blocked_reason == "Pick a file in the dataset", "no file: " + v.apply_blocked_reason);
    Check(v.detected.empty(), "no Detected line without a file");

    in.selected_file_id = "a1";
    v = Build(in);
    Check(!v.can_apply && Contains(v.apply_blocked_reason, "Arrow IPC or Parquet"), "text file refused: " + v.apply_blocked_reason);
    Check(v.detected == "Text file, 4.1 MB: not for remote training", "detected for a text file: " + v.detected);

    in.selected_dataset_id.clear();
    in.files.clear();
    v = Build(in);
    Check(v.apply_blocked_reason == "Pick a dataset and a file" && v.files_hint == "Pick a dataset to see its files.",
          "no dataset: " + v.files_hint);

    in.datasets.clear();
    v = Build(in);
    Check(Contains(v.datasets_hint, "No datasets yet"), "empty account: " + v.datasets_hint);

    in.loading_datasets = true;
    Check(Build(in).datasets_hint == "Loading your datasets...", "loading hint");
    in.loading_datasets = false;
    in.datasets_error = "deadline exceeded";
    Check(Contains(Build(in).datasets_hint, "deadline exceeded"), "list error shown");
}

void CheckNotConnected() {
    Input in;
    in.gateway = "192.168.1.116:50060";
    in.remembered = {"berean-raw-v2", "3b9d2c1e-5f60-4a71-8b92-c3d4e5f60718", "simplewiki_20231101.parquet",
                     "2f1c7f8e-7d3a-4b4e-9c2d-1a2b3c4d5e6f", 156885218};
    View v = Build(in);
    Check(Contains(v.connection, "sign in to CyxWiz first") && v.connection_tone == Tone::Warning && v.offer_sign_in,
          "signed out: " + v.connection);
    Check(v.remembered.size() == 6 && v.remembered[0].key == "Dataset" && v.remembered[0].value == "berean-raw-v2" &&
              v.remembered[4].value == "149.6 MB" && Contains(v.remembered[5].value, "192.168.1.116:50060"),
          "remembered pairs");
    Check(Contains(v.remembered_note, "mint the node's access token"), "remembered note");
    Check(v.detected == "Parquet table, 149.6 MB, file 2f1c7f8e in dataset 3b9d2c1e", "detected from memory: " + v.detected);
    Check(!v.can_apply && v.apply_blocked_reason == "Sign in to CyxWiz to use a cloud dataset", "Apply off signed out");

    in.signed_in = true;
    v = Build(in);
    Check(Contains(v.connection, "open the Cloud Browser once") && v.connection_tone == Tone::Warning,
          "browser never opened: " + v.connection);
    in.browser_opened = true;
    v = Build(in);
    Check(v.connection == "Connecting to 192.168.1.116:50060..." && v.connection_tone == Tone::Accent, "connecting: " + v.connection);
    Check(v.apply_blocked_reason == "Connect to CyxCloud first", "Apply off while connecting");

    in.connect_error = "connection refused";
    v = Build(in);
    Check(Contains(v.connection, "Gateway unreachable at 192.168.1.116:50060 (connection refused)") && v.offer_try_again,
          "unreachable: " + v.connection);
    Check(v.datasets.empty() && v.files.empty() && v.datasets_hint.empty(), "no lists without a connection");

    Input fresh;
    fresh.signed_in = true;
    Check(Build(fresh).remembered.empty() && Build(fresh).detected.empty(), "nothing remembered on a new node");
}

void CheckWords() {
    Check(TrustWord(0) == "Self" && TrustWord(3) == "Attested" && TrustWord(4) == "Unverified" && TrustWord(9) == "Unverified",
          "trust words");
    Check(ShortId("2f1c7f8e-7d3a") == "2f1c7f8e" && ShortId("abc") == "abc", "short ids");
    Remembered chosen{"tiny-causal-lm", "f8", "tokens.parquet", "fa02", 3117};
    Check(ApplyMessage(chosen) ==
              "Source saved: CyxCloud file tokens.parquet in tiny-causal-lm. A node trains from it; preview on this "
              "computer comes with sample reads.",
          "apply message: " + ApplyMessage(chosen));
    Check(FileMeta({"x", "data.FEATHER", 1024}) == "Arrow IPC \xC2\xB7 1.0 KB", "feather meta: " + FileMeta({"x", "data.FEATHER", 1024}));
}

}  // namespace

int main() {
    CheckConnectedAndPicked();
    CheckPickingRules();
    CheckNotConnected();
    CheckWords();
    std::cout << "cloud source presentation: connection, picker, detected, remembered, apply. OK\n";
    return EXIT_SUCCESS;
}
