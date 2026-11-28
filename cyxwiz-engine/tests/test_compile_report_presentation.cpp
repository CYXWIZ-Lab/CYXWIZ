// The compile report's wording (TOFIX129 B3), on the sentiment graph's real
// compile (fasttext_check: 6 layers, data not loaded, then loaded), a blocked
// training, a placement that needs attention, and the memory-risk question.

#include "../src/core/compile_report_presentation.h"

#include <cyxwiz/error_codes.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace cyxwiz;
using namespace cyxwiz::compile_report;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

BackendPlacementEntry Gpu(int id, const std::string& name, const std::string& type) {
    BackendPlacementEntry p;
    p.node_id = id;
    p.node_name = name;
    p.node_type = type;
    p.expected_backend = "ArrayFire active backend";
    p.fallback_backend = "none";
    p.status = BackendPlacementStatus::Gpu;
    p.reason_code = BackendPlacementReason::ArrayFireTensorOpCapable;
    p.explanation = type + " runs on the active ArrayFire backend.";
    p.suggested_action = "No action needed.";
    return p;
}

Input SentimentNotLoaded() {
    Input in;
    in.mode = Mode::Compile;
    in.graph_name = "fasttext_check";
    in.compiled_ok = false;
    // Compile order today: the data-check note, the error, the Data Split note.
    in.issues = {
        {IssueLevel::Info, 1, "Sentiment CSV",
         "Check the data before a long training: Data Studio's Explore and Validate show missing values, class "
         "balance and column types",
         errors::Compiler::DataCheckSuggested},
        {IssueLevel::Error, 1, "Sentiment CSV", "Data is not loaded - open the node and click Apply",
         errors::Compiler::DataNotLoaded},
        {IssueLevel::Info, 4, "Split 80/10/10",
         "Data Split resolves its named Dataset inputs into one Train/Validation/Test partition contract for Data "
         "Loader.",
         errors::Compiler::GenericIssue},
    };
    in.placements = {Gpu(6, "Embedding (word2vec vectors)", "Embedding"), Gpu(8, "Dense 64", "Dense"),
                     Gpu(9, "ReLU", "ReLU"), Gpu(10, "Dropout", "Dropout"), Gpu(11, "Dense 7", "Dense")};
    BackendPlacementEntry lstm;
    lstm.node_id = 7;
    lstm.node_name = "LSTM (last state, placement-probed)";
    lstm.node_type = "LSTM";
    lstm.expected_backend = "native provider cyxwiz.nvidia-cublas-cell";
    lstm.fallback_backend = "ArrayFire";
    lstm.status = BackendPlacementStatus::Gpu;
    lstm.reason_code = BackendPlacementReason::NativeProviderSelected;
    lstm.observation_device = "NVIDIA GeForce GTX 1050 Ti";
    in.placements.insert(in.placements.begin() + 1, lstm);
    in.has_config = true;
    ConfigFacts& c = in.config;
    c.layers = 6;
    c.input_size = 64;
    c.input_shape = {64};
    c.output_size = 7;
    c.dataset = "Sentiment CSV";
    c.backing = Backing::NotLoaded;
    c.has_split = true;
    c.split_node = "Split 80/10/10";
    c.train_ratio = 0.8;
    c.val_ratio = 0.1;
    c.test_ratio = 0.1;
    c.has_loader = true;
    c.loader_node = "DataLoader";
    c.batch_size = 64;
    c.epochs = 1;
    c.shuffle = true;
    c.num_workers = 4;
    c.log_interval = 10;
    c.validation_freq = 1;
    c.seed = 42;
    c.loss = "CrossEntropy";
    c.optimizer = "AdamW";
    c.learning_rate = 0.001;
    c.one_hot = true;
    c.num_classes = 7;
    return in;
}

void CheckFailedCompile() {
    const Report r = BuildReport(SentimentNotLoaded());
    Check(r.title == "Compile: fasttext_check", "title: " + r.title);
    Check(r.verdict == "Not ready to train: 1 error" && r.verdict_tone == Tone::Error, "verdict: " + r.verdict);
    Check(!r.ready && !r.offer_train, "not ready, no Train");
    Check(r.chips.size() == 4 && r.chips[0].text == "1 error" && r.chips[1].text == "0 warnings" &&
              r.chips[2].text == "2 notes" && r.chips[3].text == "6 layers on the GPU",
          "count chips");
    // Errors first; the notes keep their compile order.
    Check(r.findings.size() == 3 && r.findings[0].level == IssueLevel::Error &&
              r.findings[1].code == errors::Compiler::DataCheckSuggested && r.findings[2].node_name == "Split 80/10/10",
          "errors first, then notes in compile order");
    Check(r.findings[0].open_dialog && !r.findings[0].open_data_studio, "data not loaded: open the node's dialog");
    Check(r.findings[1].open_data_studio && !r.findings[1].open_dialog, "data check: open Data Studio");
    Check(r.findings[2].code.empty() && r.findings[0].code == "CW-C-0201", "the generic code is hidden (C-1)");
    Check(r.placement_sentence ==
              "All 6 layers run on the GPU (NVIDIA GeForce GTX 1050 Ti). None falls back to the CPU.",
          "placement sentence: " + r.placement_sentence);
    const auto& lstm = r.placements[1];
    Check(lstm.layer == "LSTM (last state, placement-probed)" && lstm.runs_on == "GPU" &&
              lstm.engine == "native provider cyxwiz.nvidia-cublas-cell" &&
              Contains(lstm.why, "native provider is installed and selected"),
          "LSTM row in plain words: " + lstm.why);
    // What a provider failure does depends on the policy; the row says both
    // (a Strict run stops: it does not continue on ArrayFire)
    Check(lstm.why == "A native provider is installed and selected. If the provider fails, a Compatibility run continues "
                      "on ArrayFire and a Strict run stops.",
          "the provider row is true under both policies: " + lstm.why);
    Check(!Contains(lstm.why, "ArrayFire runs it if the provider fails"), "the old sentence, wrong under Strict, is gone");
    Check(r.placements[0].engine == "ArrayFire" && r.placements[0].why == "ArrayFire tensor operations on the selected device.",
          "ArrayFire row");
    bool code_in_details = false;
    for (const auto& [key, value] : r.placements[0].details)
        code_in_details = code_in_details || (key == "Reason code" && value == "arrayfire_tensor_op_capable");
    Check(code_in_details, "the reason code moves to Details");
    Check(r.config.size() == 3 && r.config[0].rows[0].second == "Sentiment CSV" &&
              Contains(r.config[0].rows[1].second, "not loaded yet"),
          "Data group says the dataset is not loaded yet");
    Check(r.config[0].rows[2].second == "train 0.8, validation 0.1, test 0.1 (from Split 80/10/10)", "split row");
    Check(r.config[1].title == "Training loop (from DataLoader)" &&
              r.config[1].rows[0].second == "batch size 64, shuffled, keep the last short batch",
          "loop group");
    Check(r.config[2].rows[1].second == "AdamW, learning rate 0.001" && r.config[2].rows[2].second == "one-hot, 7 classes",
          "learning group");
    Check(r.tiles.size() == 3, "no parameter tile while not counted");
    Check(Contains(r.copy_text, "Error: Sentiment CSV: Data is not loaded") && Contains(r.copy_text, "[CW-C-0201]") &&
              !Contains(r.copy_text, "CW-C-0001"),
          "copy text");
}

void CheckReady() {
    Input in = SentimentNotLoaded();
    in.compiled_ok = true;
    in.issues.erase(in.issues.begin() + 1);  // loaded: no error
    in.config.dataset = "sentiment_mental_health";
    in.config.backing = Backing::OtherLoaded;
    in.config.backing_label = "text, loaded lazily";
    in.config.rows = 52681;
    in.config.parameters = 519719;
    const Report r = BuildReport(in);
    Check(r.ready && r.verdict == "Ready to train" && r.verdict_tone == Tone::Ok && r.offer_train, "ready: " + r.verdict);
    Check(r.tiles.size() == 4 && r.tiles[3].value == "519,719" && r.tiles[3].label == "parameters, 1.98 MB",
          "parameter tile: " + (r.tiles.size() > 3 ? r.tiles[3].label : std::string()));
    Check(r.config[0].rows[1].second == "text, loaded lazily, 52,681 samples", "backing: " + r.config[0].rows[1].second);
    in.issues.push_back({IssueLevel::Warning, -1, "", "No Local Debug (F6) run on this graph yet.", "CW-U-0201"});
    Check(BuildReport(in).verdict == "Ready to train, with 1 warning", "ready with a warning");
}

// The training-step memory (TOFIX178 6.0d): a tile and a breakdown group,
// with the share of the device when its size is known.
void CheckTrainingMemory() {
    Input in = SentimentNotLoaded();
    in.compiled_ok = true;
    in.issues.erase(in.issues.begin() + 1);
    in.config.parameters = 1228097;
    in.config.optimizer = "Adam";
    ModelMemoryEstimate& m = in.config.memory;
    m.available = true;
    m.batch_size = 64;
    m.trained_elements = 1228097;
    m.state_elements = 2880;
    m.optimizer_tensors_per_parameter = 2;
    m.parameter_bytes = m.gradient_bytes = 1228097ull * 4;
    m.optimizer_bytes = 1228097ull * 8;
    m.state_bytes = 2880ull * 4;
    m.activation_bytes = 1400ull * 1024 * 1024;
    m.total_bytes = m.parameter_bytes * 4 + m.state_bytes + m.activation_bytes;
    Report r = BuildReport(in);
    Check(r.tiles.size() == 5 && r.tiles[4].value == "about 1.39 GB" &&
              r.tiles[4].label == "per training step, batch 64",
          "memory tile: " + (r.tiles.size() > 4 ? r.tiles[4].value + " / " + r.tiles[4].label : std::string()));
    const auto& group = r.config.back();
    Check(group.title == "Memory of a training step (estimate)" && group.rows.size() == 6, "memory group");
    Check(group.rows[0].first == "Weights" && Contains(group.rows[0].second, "1,228,097 parameters"), "weights row");
    Check(group.rows[2].first == "Optimizer state" && Contains(group.rows[2].second, "Adam keeps 2 per parameter"),
          "optimizer row: " + group.rows[2].second);
    Check(group.rows[3].first == "Kept state" && Contains(group.rows[3].second, "BatchNorm running statistics"),
          "kept state row");
    Check(group.rows[4].first == "Activations" && Contains(group.rows[4].second, "batch 64"), "activations row");
    Check(group.rows[5].first == "Total" && Contains(group.rows[5].second, "device runtime"), "total row");

    m.device_name = "NVIDIA GeForce GTX 1050 Ti";
    m.device_memory_bytes = 4ull * 1024 * 1024 * 1024;
    r = BuildReport(in);
    Check(r.config.back().rows.back().first == "Device" &&
              Contains(r.config.back().rows.back().second, "GTX 1050 Ti, 4.00 GB: the estimate is 35% of it"),
          "device row: " + r.config.back().rows.back().second);
    m.device_memory_bytes = 1ull * 1024 * 1024 * 1024;
    r = BuildReport(in);
    Check(r.tiles[4].label == "per training step: more than the device has" &&
              Contains(r.config.back().rows.back().second, "Lower the batch size"),
          "too large for the device: " + r.tiles[4].label);

    in.config.memory = {};
    Check(BuildReport(in).tiles.size() == 4, "no memory tile while not estimated");
}

void CheckBlocked() {
    Input in = SentimentNotLoaded();
    in.mode = Mode::TrainBlocked;
    in.graph_name = "missing_data";
    in.stop_reason = "the data could not be loaded";
    in.issues = {{IssueLevel::Error, 1, "Sentiment CSV",
                  "Could not load the data before training: Error loading text: file not found: D:/x.csv", ""}};
    const Report r = BuildReport(in);
    Check(r.title == "Cannot start training: missing_data", "blocked title");
    Check(r.verdict == "Training did not start: the data could not be loaded" &&
              r.subtitle == "Nothing was trained and nothing changed in the graph.",
          "the stop reason is the verdict (C-3)");
    Check(r.findings[0].open_dialog, "a load failure offers the node's dialog");
    Check(r.config.size() == 3, "the real configuration stays");

    in.mode = Mode::CheckpointBlocked;
    in.stop_reason.clear();
    const Report checkpoint = BuildReport(in);
    Check(checkpoint.title == "Cannot load the checkpoint: missing_data" &&
              checkpoint.verdict == "The checkpoint was not loaded: the open graph must compile first",
          "checkpoint wording names the checkpoint");
}

void CheckAttentionFirst() {
    Input in;
    in.compiled_ok = true;
    in.placements = {Gpu(1, "Dense 64", "Dense")};
    BackendPlacementEntry cpu;
    cpu.node_id = 2;
    cpu.node_type = "CustomOp";
    cpu.status = BackendPlacementStatus::Unknown;
    cpu.reason_code = BackendPlacementReason::BackendCapabilityUnclassified;
    cpu.suggested_action = "Classify this graph op before relying on any GPU claim.";
    in.placements.push_back(cpu);
    const Report r = BuildReport(in);
    Check(r.placements[0].layer == "CustomOp" && r.placements[0].attention && r.placements[0].runs_on == "Unknown",
          "a row that needs attention sorts first");
    Check(r.placements[0].why ==
              "No placement rule exists for this layer yet. Classify this graph op before relying on any GPU claim.",
          "its action is in Why: " + r.placements[0].why);
    Check(r.placement_sentence == "1 of 2 layers run on the GPU; the others are listed first with what to do.",
          "mixed placement sentence: " + r.placement_sentence);
    Check(r.placement_chips.size() == 6 && r.placement_chips[5].text == "Unknown 1", "all six placement counts");
}

void CheckMemoryRisk() {
    MemoryRiskInput in;
    in.node_name = "Scale features";
    in.node_id = 7;
    in.risk = "risky";
    in.estimated_bytes = 18ull * 1024 * 1024 * 1024;
    in.available_bytes = 24ull * 1024 * 1024 * 1024;
    in.budget_bytes = 20ull * 1024 * 1024 * 1024;
    in.detail = "Materializing Scale features needs about 18 GB.";
    const MemoryRiskView v = BuildMemoryRiskView(in);
    Check(v.verdict == "Preparing \"Scale features\" may use most of the free memory" && v.tone == Tone::Error,
          "memory verdict: " + v.verdict);
    Check(v.rows.size() == 5 && v.rows[0].second == "Scale features (node 7)" && v.rows[1].second == "about 18.00 GB",
          "memory rows: " + (v.rows.size() > 1 ? v.rows[1].second : std::string()));
    Check(v.estimate_share > 0.74 && v.estimate_share < 0.76, "estimate share of free memory");
    in.risk = "warning";
    Check(BuildMemoryRiskView(in).tone == Tone::Warning, "a warning-level estimate");
}

}  // namespace

void CheckTextNotTokenized() {
    Input in;
    in.mode = Mode::Compile;
    in.graph_name = "mental_health";
    in.compiled_ok = false;
    in.issues = {{IssueLevel::Error, 1, "Sentiment CSV",
                  "'Sentiment CSV' gives text, and nothing turns it into numbers before the model: add a Text "
                  "Tokenizer after it (or a TF-IDF / Count vectorizer)",
                  errors::Compiler::TextNotTokenized}};
    const Report r = BuildReport(in);
    Check(r.findings.size() == 1 && r.findings[0].add_node == AddNodeFix::TextTokenizer &&
              r.findings[0].code == "CW-C-0203" && !r.findings[0].open_dialog,
          "text not tokenized: the fix adds a Text Tokenizer after the Data Input");
    Check(BuildReport(SentimentNotLoaded()).findings[0].add_node == AddNodeFix::None, "other findings add nothing");
    in.issues = {{IssueLevel::Error, 1, "Cats", "'Cats' gives images, and no Resize sets their size",
                  errors::Compiler::ImageNotResized},
                 {IssueLevel::Error, 2, "Speech", "'Speech' gives audio, and no node turns it into features",
                  errors::Compiler::AudioFeaturesMissing}};
    const Report media = BuildReport(in);
    Check(media.findings.size() == 2 && media.findings[0].add_node == AddNodeFix::Resize &&
              media.findings[1].add_node == AddNodeFix::AudioFeatures,
          "images without Resize offer Add a Resize; audio without features offers the feature nodes");
}

// Missing values (TOFIX223 0.3): the card is the verdict; "Fix and train"
// while it is open, "Train" once it is fixed; Copy report lists the columns.
// A layer that asks a native provider when none serves the device: the row
// says the actual path (ArrayFire), the reason and what to do.
void CheckMissingProvider() {
    Input in;
    in.mode = Mode::Compile;
    in.compiled_ok = true;
    BackendPlacementEntry conv = Gpu(3, "Conv2D 32", "Conv2D");
    conv.missing_provider = "NVIDIA provider not installed. Install the CUDA GPU pack.";
    in.placements = {conv, Gpu(4, "Dense 10", "Dense")};
    const Report r = BuildReport(in);
    Check(r.placements.size() == 2, "two rows");
    Check(r.placements[0].engine == "ArrayFire" &&
              r.placements[0].why == "ArrayFire tensor operations on the selected device. No native provider runs it: "
                                     "NVIDIA provider not installed. Install the CUDA GPU pack.",
          "the row of a layer whose provider is missing: " + r.placements[0].why);
    Check(r.placements[1].why == "ArrayFire tensor operations on the selected device.",
          "a layer that asks no provider says nothing of one: " + r.placements[1].why);
    Check(Contains(r.copy_text, "No native provider runs it: NVIDIA provider not installed"), "and Copy carries it");
}

void CheckMissingValues() {
    namespace mv = cyxwiz::missing_values_fix;
    mv::Input card_in;
    card_in.input_name = "Patients CSV";
    card_in.total_rows = 1200;
    cyxwiz::MissingColumn age;
    age.name = "age";
    age.rows = 37;
    cyxwiz::MissingColumn label;
    label.name = "diagnosis";
    label.rows = 4;
    label.label = true;
    card_in.columns = {age, label};
    Input in;
    in.mode = Mode::TrainBlocked;
    in.compiled_ok = true;
    in.missing_values = {mv::BuildCard(card_in)};
    Report r = BuildReport(in);
    Check(r.verdict == "Training did not start: 2 columns have missing values" && r.verdict_tone == Tone::Error,
          "missing values verdict: " + r.verdict);
    Check(r.offer_fix_and_train && !r.offer_train && r.fix_and_train_blocked.empty(), "Fix and train is offered");
    Check(r.errors == 1 && r.chips.front().text == "1 error", "an open card is an error: " + r.chips.front().text);
    Check(r.copy_text.find("Missing values") != std::string::npos &&
              r.copy_text.find("feature age: 37 empty") != std::string::npos,
          "Copy report lists the columns");
    card_in.features = mv::Done::Filled;
    card_in.label = mv::Done::Dropped;
    in.missing_values = {mv::BuildCard(card_in)};
    r = BuildReport(in);
    Check(r.verdict == "Fixed: 2 steps added after Patients CSV" && r.verdict_tone == Tone::Ok && r.offer_train &&
              !r.offer_fix_and_train && r.errors == 0,
          "fixed: " + r.verdict);
    Check(r.missing_value_groups == 2, "the Findings tab counts the card's two groups");

    // A supplied Test input with only empty labels: no step reaches it, and
    // Fix and train says which input to fix (not an empty name).
    mv::Input supplied = mv::MakeInput({label}, 50, "Patients test CSV", "Test");
    in.missing_values = {mv::BuildCard(supplied)};
    r = BuildReport(in);
    Check(r.fix_and_train_blocked ==
              "The missing values of Patients test CSV cannot be fixed with a step here: fix them in its file.",
          "blocked: " + r.fix_and_train_blocked);
}

int main() {
    CheckFailedCompile();
    CheckReady();
    CheckTrainingMemory();
    CheckBlocked();
    CheckAttentionFirst();
    CheckMemoryRisk();
    CheckTextNotTokenized();
    CheckMissingProvider();
    CheckMissingValues();
    std::cout << "compile report presentation: failed, ready, training memory, blocked, placement, memory risk, text not "
                 "tokenized, missing provider, missing values. OK\n";
    return 0;
}
