#include "data_studio_results.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <regex>

namespace cyxwiz::studio {

namespace fs = std::filesystem;

namespace {

std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// The text between the first <tag> and </tag>, "" when absent.
std::string Tagged(const std::string& text, const std::string& tag) {
    const std::string open = "<" + tag + ">", close = "</" + tag + ">";
    const size_t a = text.find(open);
    if (a == std::string::npos) return "";
    const size_t b = text.find(close, a + open.size());
    if (b == std::string::npos) return "";
    return Trim(text.substr(a + open.size(), b - a - open.size()));
}

struct Ref {
    std::string book;
    int chapter = 0, verse = 0;
};

// The first "Book C:V" in the text ("1 Thessalonians 5:18", "Psalm 34:13").
bool FirstRef(const std::string& text, Ref& ref, std::string* shown = nullptr) {
    static const std::regex pattern(R"(((?:[1-3]\s)?[A-Z][a-z]+(?:\s(?:of\s)?[A-Z][a-z]+)*)\s(\d{1,3}):(\d{1,3}))");
    std::smatch m;
    if (!std::regex_search(text, m, pattern)) return false;
    ref.book = m[1].str();
    ref.chapter = std::stoi(m[2].str());
    ref.verse = std::stoi(m[3].str());
    if (shown) *shown = m[0].str();
    return true;
}

bool Loops(const std::string& text) {
    std::vector<std::string> words;
    std::string w;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!w.empty()) words.push_back(w);
            w.clear();
        } else {
            w += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    if (!w.empty()) words.push_back(w);
    std::map<std::string, int> seen;
    for (size_t i = 0; i + 2 < words.size(); ++i)
        if (++seen[words[i] + " " + words[i + 1] + " " + words[i + 2]] >= 3) return true;
    return false;
}

}  // namespace

std::vector<PreviewRun> FindPreviewRuns(const std::string& checkpoints_root) {
    std::vector<PreviewRun> runs;
    std::error_code ec;
    if (checkpoints_root.empty() || !fs::is_directory(checkpoints_root, ec)) return runs;
    for (fs::recursive_directory_iterator it(checkpoints_root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (!it->is_directory(ec) || it->path().filename() != "generation_previews") continue;
        it.disable_recursion_pending();
        PreviewRun run;
        run.folder = it->path().generic_string();
        run.run_id = it->path().parent_path().filename().string();
        run.graph = it->path().parent_path().parent_path().filename().string();
        for (const auto& f : fs::directory_iterator(it->path(), ec)) {
            const std::string name = f.path().filename().string();
            if (name.rfind("epoch_", 0) != 0 || f.path().extension() != ".json") continue;
            try {
                run.epochs.push_back(std::stoi(name.substr(6)));
            } catch (...) {
                continue;
            }
            run.time = std::max(run.time, f.last_write_time(ec));
        }
        if (run.epochs.empty()) continue;
        std::sort(run.epochs.begin(), run.epochs.end());
        runs.push_back(std::move(run));
    }
    std::sort(runs.begin(), runs.end(), [](const PreviewRun& a, const PreviewRun& b) { return a.time > b.time; });
    return runs;
}

PreviewSample PreviewSampleFrom(const std::string& prompt, const std::string& generated, const std::string& stop) {
    PreviewSample s;
    s.generated = generated;
    s.stop = stop;
    s.question = Tagged(prompt, "question");
    if (s.question.empty()) {
        const size_t user = prompt.rfind("User:");
        s.question = Trim(user == std::string::npos ? prompt : prompt.substr(user + 5));
    }
    s.evidence_ref = Tagged(prompt, "reference");
    Ref evidence, cited;
    s.has_ref = !s.evidence_ref.empty() && FirstRef(s.evidence_ref, evidence);
    if (FirstRef(generated, cited, &s.cited_ref) && s.has_ref) {
        s.book = cited.book == evidence.book;
        s.chapter = s.book && cited.chapter == evidence.chapter;
        s.verse = s.chapter && cited.verse == evidence.verse;
    }
    s.loops = Loops(generated);
    return s;
}

bool LoadPreviewEpoch(const fs::path& path, PreviewEpoch& epoch, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open " + path.string();
        return false;
    }
    try {
        const auto j = nlohmann::json::parse(in);
        epoch = {};
        epoch.run_id = j.value("run_id", std::string());
        epoch.epoch = j.value("completed_epoch", 0);
        epoch.sampling = j.value("sampling", std::string());
        epoch.max_new_tokens = j.value("max_new_tokens", 0);
        epoch.cancelled = j.value("cancelled", false);
        for (const auto& s : j.value("samples", nlohmann::json::array()))
            epoch.samples.push_back(PreviewSampleFrom(s.value("prompt", std::string()), s.value("generated_text", std::string()),
                                                      s.value("stop_reason", std::string())));
        return true;
    } catch (const std::exception& e) {
        error = std::string("cannot read the preview: ") + e.what();
        return false;
    }
}

std::string StopWords(const std::string& stop) {
    if (stop == "eos") return "end";
    if (stop == "max_tokens" || stop == "max_new_tokens") return "token limit";
    if (stop == "user_cancelled" || stop == "cancelled") return "cancelled";
    if (stop == "context_limit" || stop == "max_context") return "context limit";
    return stop.empty() ? "unknown" : stop;
}

PreviewSummary SummarizePreview(const PreviewEpoch& epoch) {
    PreviewSummary s;
    for (const auto& x : epoch.samples) {
        ++s.samples;
        if (x.has_ref) ++s.with_refs;
        if (x.book) ++s.right_book;
        if (x.chapter) ++s.right_chapter;
        if (x.verse) ++s.right_verse;
        if (StopWords(x.stop) == "end") ++s.stopped_itself;
        if (StopWords(x.stop) == "token limit") ++s.hit_limit;
        if (x.loops) ++s.loops;
    }
    return s;
}

std::vector<PreviewSlice> PreviewSlices(const PreviewEpoch& epoch) {
    std::vector<PreviewSlice> slices;
    std::map<std::string, size_t> index;
    for (const auto& x : epoch.samples) {
        const std::string name = "stop = " + StopWords(x.stop);
        if (!index.count(name)) {
            index[name] = slices.size();
            slices.push_back({name, 0, 0});
        }
        auto& slice = slices[index[name]];
        ++slice.rows;
        if (x.verse) ++slice.right_verse;
    }
    PreviewSlice loops{"loops", 0, 0};
    for (const auto& x : epoch.samples)
        if (x.loops) {
            ++loops.rows;
            if (x.verse) ++loops.right_verse;
        }
    if (loops.rows) slices.push_back(loops);
    return slices;
}

ClassifierResults ClassifierFromPredictions(const std::shared_ptr<arrow::Table>& table, size_t worst_rows) {
    ClassifierResults r;
    if (!table) {
        r.error = "no test predictions";
        return r;
    }
    const auto truth_col = table->GetColumnByName("true_class");
    const auto pred_col = table->GetColumnByName("predicted_class");
    const auto conf_col = table->GetColumnByName("confidence");
    if (!truth_col || !pred_col) {
        r.error = "the table has no true_class / predicted_class columns";
        return r;
    }
    std::vector<int64_t> truth, pred;
    std::vector<double> conf;
    const auto read_int = [](const std::shared_ptr<arrow::ChunkedArray>& col, std::vector<int64_t>& out) {
        for (const auto& chunk : col->chunks()) {
            const auto& a = static_cast<const arrow::Int64Array&>(*chunk);
            for (int64_t i = 0; i < a.length(); ++i) out.push_back(a.IsValid(i) ? a.Value(i) : -1);
        }
    };
    if (truth_col->type()->id() != arrow::Type::INT64 || pred_col->type()->id() != arrow::Type::INT64) {
        r.error = "true_class and predicted_class must be whole numbers";
        return r;
    }
    read_int(truth_col, truth);
    read_int(pred_col, pred);
    if (conf_col && conf_col->type()->id() == arrow::Type::DOUBLE) {
        for (const auto& chunk : conf_col->chunks()) {
            const auto& a = static_cast<const arrow::DoubleArray&>(*chunk);
            for (int64_t i = 0; i < a.length(); ++i) conf.push_back(a.IsValid(i) ? a.Value(i) : 0.0);
        }
    }
    conf.resize(truth.size(), 0.0);
    r.rows = static_cast<int64_t>(truth.size());

    // Class names from score_<name> columns, in column order; else indices.
    int64_t max_class = -1;
    for (size_t i = 0; i < truth.size(); ++i) max_class = std::max({max_class, truth[i], pred[i]});
    for (const auto& field : table->schema()->fields())
        if (field->name().rfind("score_", 0) == 0) r.classes.push_back(field->name().substr(6));
    if (static_cast<int64_t>(r.classes.size()) <= max_class) {
        r.classes.clear();
        for (int64_t c = 0; c <= max_class; ++c) r.classes.push_back(std::to_string(c));
    }
    const size_t k = r.classes.size();
    r.confusion.assign(k, std::vector<int64_t>(k, 0));
    int64_t right = 0;
    std::vector<WorstRow> wrong;
    for (size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] < 0 || pred[i] < 0) continue;
        ++r.confusion[static_cast<size_t>(truth[i])][static_cast<size_t>(pred[i])];
        if (truth[i] == pred[i]) ++right;
        else wrong.push_back({static_cast<int64_t>(i) + 1, r.classes[static_cast<size_t>(truth[i])],
                              r.classes[static_cast<size_t>(pred[i])], conf[i]});
    }
    r.accuracy = r.rows ? static_cast<double>(right) / static_cast<double>(r.rows) : 0.0;
    for (size_t c = 0; c < k; ++c) {
        ClassRow row;
        row.name = r.classes[c];
        int64_t predicted = 0;
        for (size_t t = 0; t < k; ++t) {
            row.support += r.confusion[c][t];
            predicted += r.confusion[t][c];
        }
        row.recall = row.support ? static_cast<double>(r.confusion[c][c]) / static_cast<double>(row.support) : 0.0;
        row.precision = predicted ? static_cast<double>(r.confusion[c][c]) / static_cast<double>(predicted) : 0.0;
        r.per_class.push_back(row);
    }
    std::stable_sort(wrong.begin(), wrong.end(), [](const WorstRow& a, const WorstRow& b) { return a.confidence > b.confidence; });
    if (wrong.size() > worst_rows) wrong.resize(worst_rows);
    r.worst = std::move(wrong);
    r.ok = true;
    return r;
}

}  // namespace cyxwiz::studio
