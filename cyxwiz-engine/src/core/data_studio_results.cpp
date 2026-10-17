#include "data_studio_results.h"

#include "arrow_number.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
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

namespace {

// The error columns of a regression table: "error", or "error_0", "error_1", ...
std::vector<std::string> ErrorColumns(const std::shared_ptr<arrow::Table>& table) {
    if (table->GetColumnByName("error")) return {"error"};
    std::vector<std::string> names;
    for (int t = 0; table->GetColumnByName("error_" + std::to_string(t)); ++t) names.push_back("error_" + std::to_string(t));
    return names;
}

std::string Number(double x) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", x);
    return buf;
}

}  // namespace

TestRowErrors RowErrorsFromPredictions(const std::shared_ptr<arrow::Table>& table) {
    TestRowErrors out;
    if (!table) {
        out.error = "no test predictions";
        return out;
    }
    if (const auto metadata = table->schema()->metadata()) {
        const int at = metadata->FindKey("source_dataset");
        if (at >= 0) out.source_dataset = metadata->value(at);
    }
    for (double id : ArrowNumbers(table->GetColumnByName("row_id")))
        out.row_ids.push_back(std::isnan(id) ? -1 : static_cast<int64_t>(id));
    if (out.row_ids.empty() || out.source_dataset.empty()) {
        out.error = "this test run did not record which rows of the dataset it tested";
        return out;
    }
    const size_t rows = static_cast<size_t>(table->num_rows());
    out.errors.assign(rows, 0.0);
    out.classifier = table->GetColumnByName("true_class") != nullptr;
    if (out.classifier) {
        const std::vector<double> truth = ArrowNumbers(table->GetColumnByName("true_class"));
        const std::vector<double> pred = ArrowNumbers(table->GetColumnByName("predicted_class"));
        if (truth.size() != rows || pred.size() != rows) {
            out.error = "the table has no true_class / predicted_class columns";
            return out;
        }
        for (size_t i = 0; i < rows; ++i) out.errors[i] = truth[i] == pred[i] ? 0.0 : 1.0;
    } else {
        const std::vector<std::string> names = ErrorColumns(table);
        for (const auto& name : names) {
            const std::vector<double> e = ArrowNumbers(table->GetColumnByName(name));
            if (e.size() != rows) {
                out.error = "'" + name + "' is not a number column";
                return out;
            }
            for (size_t i = 0; i < rows; ++i)
                if (!std::isnan(e[i])) out.errors[i] += std::abs(e[i]) / static_cast<double>(names.size());
        }
    }
    out.ok = true;
    return out;
}

// ---- ColumnGatherer ---------------------------------------------------------

ColumnGatherer::ColumnGatherer(const std::vector<int64_t>& row_ids) : row_ids_(row_ids), order_(row_ids.size()) {
    for (size_t i = 0; i < order_.size(); ++i) order_[i] = i;
    std::stable_sort(order_.begin(), order_.end(), [&](size_t a, size_t b) { return row_ids_[a] < row_ids_[b]; });
    column_.valid.assign(row_ids_.size(), 0);
}

bool ColumnGatherer::Add(const std::shared_ptr<arrow::ChunkedArray>& piece, std::string& error) {
    if (!piece) {
        error = "the column could not be read";
        return false;
    }
    if (!typed_) {
        const auto& type = piece->type();
        if (arrow::is_nested(type->id()) || type->id() == arrow::Type::BINARY || type->id() == arrow::Type::LARGE_BINARY) {
            error = "a " + type->ToString() + " column cannot be sliced by value";
            return false;
        }
        column_.numeric = IsArrowNumber(type->id());
        if (column_.numeric) column_.numbers.assign(row_ids_.size(), 0.0);
        else column_.labels.assign(row_ids_.size(), std::string());
        typed_ = true;
    }
    int64_t chunk_start = offset_;
    for (const auto& chunk : piece->chunks()) {
        const int64_t chunk_end = chunk_start + chunk->length();
        while (cursor_ < order_.size() && row_ids_[order_[cursor_]] < chunk_end) {
            const size_t at = order_[cursor_++];
            const int64_t row = row_ids_[at];
            if (row < chunk_start) {
                error = "row " + std::to_string(row) + " is not a row of the table";
                return false;
            }
            const int64_t local = row - chunk_start;
            if (chunk->IsNull(local)) continue;
            if (column_.numeric) {
                const double v = ArrowNumberAt(*chunk, local);
                if (std::isnan(v)) continue;
                column_.numbers[at] = v;
            } else {
                const auto scalar = chunk->GetScalar(local);
                if (!scalar.ok()) continue;
                column_.labels[at] = (*scalar)->ToString();
            }
            column_.valid[at] = 1;
        }
        chunk_start = chunk_end;
    }
    offset_ = chunk_start;
    return true;
}

bool ColumnGatherer::Finish(GatheredColumn& out, std::string& error) {
    if (cursor_ < order_.size()) {
        error = "the test used row " + std::to_string(row_ids_[order_[cursor_]] + 1) + ", but the table has " +
                std::to_string(offset_) + " rows now (it changed since the test)";
        return false;
    }
    out = std::move(column_);
    return true;
}

// ---- SliceErrors ------------------------------------------------------------

ErrorSlices SliceErrors(const GatheredColumn& column, const std::vector<double>& errors, size_t max_values, size_t bins) {
    ErrorSlices out;
    const size_t rows = std::min(errors.size(), column.valid.size());
    if (rows == 0) return out;
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) total += errors[i];
    out.overall = total / static_cast<double>(rows);

    struct Acc {
        int64_t rows = 0;
        double sum = 0.0;
    };
    Acc missing;
    std::vector<ValueSlice> slices;
    const auto add = [&](const std::string& value, const Acc& a) {
        if (a.rows) slices.push_back({value, a.rows, a.sum / static_cast<double>(a.rows)});
    };

    std::map<double, Acc> by_number;
    std::map<std::string, Acc> by_label;
    for (size_t i = 0; i < rows; ++i) {
        Acc& a = !column.valid[i] ? missing : column.numeric ? by_number[column.numbers[i]] : by_label[column.labels[i]];
        ++a.rows;
        a.sum += errors[i];
    }

    if (column.numeric && by_number.size() > max_values && bins > 1) {
        // Ranges of about rows / bins each; a value never splits across two.
        out.binned = true;
        int64_t valid = 0;
        for (const auto& [v, a] : by_number) valid += a.rows;
        const double per_bin = static_cast<double>(valid) / static_cast<double>(bins);
        Acc bin;
        double lo = by_number.begin()->first, hi = lo;
        int64_t seen = 0;
        size_t made = 0;
        for (auto it = by_number.begin(); it != by_number.end(); ++it) {
            if (bin.rows == 0) lo = it->first;
            hi = it->first;
            bin.rows += it->second.rows;
            bin.sum += it->second.sum;
            seen += it->second.rows;
            const bool last = std::next(it) == by_number.end();
            if (last || (made + 1 < bins && static_cast<double>(seen) >= per_bin * static_cast<double>(made + 1))) {
                add(Number(lo) + " to " + Number(hi), bin);
                bin = {};
                ++made;
            }
        }
    } else {
        std::vector<std::pair<std::string, Acc>> values;
        if (column.numeric)
            for (const auto& [v, a] : by_number) values.emplace_back(Number(v), a);
        else
            for (const auto& [v, a] : by_label) values.emplace_back(v.empty() ? "(empty)" : v, a);
        std::stable_sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.second.rows > b.second.rows; });
        Acc other;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i < max_values) add(values[i].first, values[i].second);
            else {
                other.rows += values[i].second.rows;
                other.sum += values[i].second.sum;
            }
        }
        add("(other values)", other);
    }
    add("(missing)", missing);
    std::stable_sort(slices.begin(), slices.end(), [](const ValueSlice& a, const ValueSlice& b) {
        return a.error > b.error || (a.error == b.error && a.rows > b.rows);
    });
    out.slices = std::move(slices);
    return out;
}

}  // namespace cyxwiz::studio
