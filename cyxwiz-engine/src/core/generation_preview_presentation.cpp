#include "generation_preview_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <tuple>

namespace cyxwiz {
namespace {

GenerationPreviewCheckKind CheckKindFromName(const std::string& name) {
    for (auto kind : {GenerationPreviewCheckKind::Contains, GenerationPreviewCheckKind::NotContains,
                      GenerationPreviewCheckKind::Regex, GenerationPreviewCheckKind::EndsWithStop})
        if (name == GenerationPreviewCheckKindName(kind)) return kind;
    return GenerationPreviewCheckKind::None;
}

std::vector<std::string> Words(const std::string& text) {
    std::vector<std::string> words;
    std::string word;
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '\'' || c >= 0x80) {
            word.push_back(static_cast<char>(c < 0x80 ? std::tolower(c) : c));
        } else if (!word.empty()) {
            words.push_back(std::move(word));
            word.clear();
        }
    }
    if (!word.empty()) words.push_back(std::move(word));
    return words;
}

std::string OneDecimal(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f", value);
    return buffer;
}

std::string Shown(const std::string& text) {
    std::string shown;
    for (char c : text) {
        if (c == '\n') shown += "\\n";
        else if (c == '\r') shown += "\\r";
        else if (c == '\t') shown += "\\t";
        else shown.push_back(c);
    }
    return shown;
}

std::string WithThousands(long long value) {
    std::string digits = std::to_string(value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

int PointOrder(const std::string& point) { return point == "baseline" ? 0 : 1; }

}  // namespace

std::optional<GenerationPreviewReport> LoadGenerationPreviewReport(const std::filesystem::path& path,
                                                                   std::string& error) {
    nlohmann::json document;
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            error = "cannot open " + path.string();
            return std::nullopt;
        }
        input >> document;
    } catch (const std::exception& exception) {
        error = path.filename().string() + " is not valid JSON (" + exception.what() + ")";
        return std::nullopt;
    }
    if (!document.is_object() || document.value("schema", 0) != kGenerationPreviewReportSchema) {
        error = path.filename().string() + " was written by an older Engine";
        return std::nullopt;
    }
    try {
        GenerationPreviewReport report;
        report.file = path;
        report.run_id = document.value("run_id", "");
        report.point = document.value("point", "epoch");
        report.epoch = document.value("epoch", 0);
        report.total_epochs = document.value("total_epochs", 0);
        report.step = document.value("step", 0LL);
        report.weights = document.value("weights", "current");
        report.mode = document.value("mode", "continuation");
        report.chat_template = document.value("template", "");
        report.decoding = document.value("decoding", "greedy");
        report.temperature = document.value("temperature", 0.0f);
        report.top_k = document.value("top_k", 0);
        report.top_p = document.value("top_p", 1.0f);
        report.seed = document.value("seed", 0u);
        report.repetition_penalty = document.value("repetition_penalty", 1.0f);
        report.max_new_tokens = document.value("max_new_tokens", 0);
        report.context = document.value("context", 0);
        if (document.contains("stop_texts") && document.at("stop_texts").is_array())
            for (const auto& stop : document.at("stop_texts"))
                if (stop.is_string()) report.stop_texts.push_back(stop.get<std::string>());
        report.elapsed_ms = document.value("elapsed_ms", 0.0);
        report.cancelled = document.value("cancelled", false);
        if (document.contains("validation_loss") && document.at("validation_loss").is_number())
            report.validation_loss = document.at("validation_loss").get<double>();
        for (const auto& item : document.value("samples", nlohmann::json::array())) {
            GenerationPreviewSample sample;
            sample.prompt_index = item.value("prompt_index", 0);
            sample.sample_index = item.value("sample_index", 0);
            sample.prompt = item.value("prompt", "");
            sample.rendered_prompt = item.value("rendered_prompt", sample.prompt);
            sample.text = item.value("text", "");
            sample.stop_reason = item.value("stop_reason", "");
            sample.stop_text = item.value("stop_text", "");
            if (item.contains("prompt_token_ids") && item.at("prompt_token_ids").is_array())
                sample.prompt_tokens = static_cast<int>(item.at("prompt_token_ids").size());
            if (item.contains("new_token_ids") && item.at("new_token_ids").is_array())
                sample.new_tokens = static_cast<int>(item.at("new_token_ids").size());
            if (item.contains("check") && item.at("check").is_object()) {
                sample.check.kind = CheckKindFromName(item.at("check").value("kind", ""));
                sample.check.value = item.at("check").value("value", "");
            }
            sample.check_result = EvaluateGenerationPreviewCheck(sample.check, sample.text, sample.stop_reason);
            sample.repeated_phrase = DetectRepeatedPhrase(sample.text);
            report.samples.push_back(std::move(sample));
        }
        return report;
    } catch (const std::exception& exception) {
        error = path.filename().string() + " has an unexpected shape (" + exception.what() + ")";
        return std::nullopt;
    }
}

std::vector<GenerationPreviewReport> LoadGenerationPreviewReports(const std::filesystem::path& directory,
                                                                  std::vector<std::string>* skipped) {
    std::vector<GenerationPreviewReport> reports;
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) return reports;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->path().extension() != ".json") continue;
        std::string error;
        if (auto report = LoadGenerationPreviewReport(it->path(), error)) {
            reports.push_back(std::move(*report));
        } else if (skipped) {
            skipped->push_back(error);
        }
    }
    std::sort(reports.begin(), reports.end(), [](const auto& a, const auto& b) {
        return std::make_tuple(PointOrder(a.point), a.epoch, a.step, a.file.filename().string()) <
               std::make_tuple(PointOrder(b.point), b.epoch, b.step, b.file.filename().string());
    });
    return reports;
}

std::vector<GenerationPreviewRun> FindGenerationPreviewRuns(const std::filesystem::path& checkpoints_root) {
    namespace fs = std::filesystem;
    std::vector<GenerationPreviewRun> runs;
    std::error_code ec;
    if (checkpoints_root.empty() || !fs::is_directory(checkpoints_root, ec)) return runs;
    for (fs::recursive_directory_iterator it(checkpoints_root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (!it->is_directory(ec) || it->path().filename() != "generation_previews") continue;
        it.disable_recursion_pending();
        GenerationPreviewRun run;
        run.folder = it->path();
        run.run_id = it->path().parent_path().filename().string();
        run.graph = it->path().parent_path().parent_path().filename().string();
        std::error_code inner;
        for (fs::directory_iterator f(it->path(), inner), stop; !inner && f != stop; f.increment(inner)) {
            if (f->path().extension() != ".json") continue;
            ++run.files;
            run.time = std::max(run.time, f->last_write_time(inner));
        }
        if (run.files > 0) runs.push_back(std::move(run));
    }
    std::sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) { return a.time > b.time; });
    return runs;
}

std::string GenerationPreviewFolderStamp(const std::filesystem::path& folder) {
    namespace fs = std::filesystem;
    std::error_code ec;
    size_t files = 0;
    fs::file_time_type newest{};
    uintmax_t bytes = 0;
    for (fs::directory_iterator f(folder, ec), stop; !ec && f != stop; f.increment(ec)) {
        if (f->path().extension() != ".json") continue;
        std::error_code inner;
        ++files;
        newest = std::max(newest, f->last_write_time(inner));
        bytes += f->file_size(inner);
    }
    return std::to_string(files) + ":" + std::to_string(bytes) + ":" +
           std::to_string(newest.time_since_epoch().count());
}

bool ExportGenerationPreviewSamples(const std::vector<GenerationPreviewReport>& reports,
                                    const std::filesystem::path& path, std::string& error) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "cannot write " + path.string();
        return false;
    }
    for (const auto& report : reports) {
        for (const auto& sample : report.samples) {
            nlohmann::json line = {{"run_id", report.run_id}, {"point", report.point}, {"epoch", report.epoch},
                                   {"step", report.step}, {"weights", report.weights},
                                   {"prompt_index", sample.prompt_index}, {"sample_index", sample.sample_index},
                                   {"prompt", sample.prompt}, {"rendered_prompt", sample.rendered_prompt},
                                   {"output", sample.text}, {"stop_reason", sample.stop_reason},
                                   {"new_tokens", sample.new_tokens}};
            if (!sample.stop_text.empty()) line["stop_text"] = sample.stop_text;
            if (report.validation_loss) line["validation_loss"] = *report.validation_loss;
            if (sample.check.kind != GenerationPreviewCheckKind::None) {
                line["check"] = {{"kind", GenerationPreviewCheckKindName(sample.check.kind)},
                                 {"value", sample.check.value},
                                 {"passed", sample.check_result == GenerationPreviewCheckResult::Passed}};
            }
            output << line.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        }
    }
    if (!output) {
        error = "writing " + path.string() + " failed";
        return false;
    }
    return true;
}

std::filesystem::path GenerationPreviewReportPath(const std::filesystem::path& directory, const std::string& point,
                                                  int epoch, long long step) {
    std::string stem = point == "baseline" ? "baseline"
                       : point == "step"   ? "step_" + std::to_string(step)
                                           : "epoch_" + std::to_string(epoch);
    auto path = directory / (stem + ".json");
    std::error_code ec;
    for (int n = 2; std::filesystem::exists(path, ec); ++n) path = directory / (stem + "_" + std::to_string(n) + ".json");
    return path;
}

GenerationPreviewCheckResult EvaluateGenerationPreviewCheck(const GenerationPreviewCheck& check,
                                                            const std::string& text,
                                                            const std::string& stop_reason) {
    using Result = GenerationPreviewCheckResult;
    switch (check.kind) {
        case GenerationPreviewCheckKind::None:
            return Result::NotSet;
        case GenerationPreviewCheckKind::Contains:
            return text.find(check.value) != std::string::npos ? Result::Passed : Result::Failed;
        case GenerationPreviewCheckKind::NotContains:
            return text.find(check.value) == std::string::npos ? Result::Passed : Result::Failed;
        case GenerationPreviewCheckKind::Regex:
            try {
                return std::regex_search(text, std::regex(check.value)) ? Result::Passed : Result::Failed;
            } catch (const std::regex_error&) {
                return Result::Failed;
            }
        case GenerationPreviewCheckKind::EndsWithStop:
            return stop_reason == "eos" || stop_reason == "stop_text" ? Result::Passed : Result::Failed;
    }
    return Result::NotSet;
}

std::string DetectRepeatedPhrase(const std::string& text) {
    const auto words = Words(text);
    for (size_t n = 8; n >= 3; --n) {
        if (words.size() < n * 2) continue;
        const int needed = n >= 4 ? 2 : 3;
        std::map<std::string, std::pair<int, size_t>> seen;  // phrase -> (count, next free start)
        std::string best;
        int best_count = 0;
        for (size_t start = 0; start + n <= words.size(); ++start) {
            std::string phrase = words[start];
            for (size_t k = 1; k < n; ++k) phrase += " " + words[start + k];
            auto& [count, next_free] = seen[phrase];
            if (count > 0 && start < next_free) continue;  // Overlapping repeat.
            ++count;
            next_free = start + n;
            if (count >= needed && count > best_count) {
                best = phrase;
                best_count = count;
            }
        }
        if (!best.empty()) return best;
    }
    return {};
}

GenerationPreviewSummary SummarizeGenerationPreview(const GenerationPreviewReport& report) {
    GenerationPreviewSummary summary;
    for (const auto& sample : report.samples) {
        ++summary.samples;
        if (sample.check_result != GenerationPreviewCheckResult::NotSet) {
            ++summary.checks;
            if (sample.check_result == GenerationPreviewCheckResult::Passed) ++summary.checks_passed;
        }
        if (sample.stop_reason == "eos" || sample.stop_reason == "stop_text") ++summary.stopped_by_itself;
        if (!sample.repeated_phrase.empty()) ++summary.repeating;
    }
    return summary;
}

std::string GenerationPreviewPointLabel(const GenerationPreviewReport& report) {
    if (report.point == "baseline") return "Before training";
    if (report.point == "step")
        return "Step " + WithThousands(report.step) + " (epoch " + std::to_string(report.epoch + 1) + ")";
    if (report.total_epochs > 0)
        return "Epoch " + std::to_string(report.epoch) + " of " + std::to_string(report.total_epochs);
    return "Epoch " + std::to_string(report.epoch);
}

std::string GenerationPreviewStopLabel(const GenerationPreviewSample& sample, int max_new_tokens) {
    const std::string tokens = std::to_string(sample.new_tokens) + " tokens";
    if (sample.stop_reason == "eos") return "Stopped by itself · " + tokens;
    if (sample.stop_reason == "stop_text") return "Stopped at \"" + Shown(sample.stop_text) + "\" · " + tokens;
    if (sample.stop_reason == "max_tokens") return "Hit the " + std::to_string(max_new_tokens) + "-token limit";
    if (sample.stop_reason == "user_cancel") return "Cancelled · " + tokens;
    return "Failed";
}

std::string GenerationPreviewCheckLabel(const GenerationPreviewSample& sample) {
    if (sample.check_result == GenerationPreviewCheckResult::NotSet) return {};
    if (sample.check_result == GenerationPreviewCheckResult::Passed) return "Check passed";
    switch (sample.check.kind) {
        case GenerationPreviewCheckKind::Contains: return "No \"" + sample.check.value + "\"";
        case GenerationPreviewCheckKind::NotContains: return "Has \"" + sample.check.value + "\"";
        case GenerationPreviewCheckKind::Regex: return "Pattern not found";
        case GenerationPreviewCheckKind::EndsWithStop: return "Did not stop by itself";
        default: return "Check failed";
    }
}

std::string GenerationPreviewHowLine(const GenerationPreviewReport& report) {
    std::string line = report.mode == "chat" ? "chat" : "continuation";
    if (report.mode == "chat" && !report.chat_template.empty()) {
        for (auto value : {GenerationPreviewTemplate::RoleLines, GenerationPreviewTemplate::ChatML,
                           GenerationPreviewTemplate::Llama3})
            if (report.chat_template == GenerationPreviewTemplateName(value)) {
                std::string label = GenerationPreviewTemplateLabel(value);
                label[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(label[0])));
                line += " (" + label + ")";
            }
    }
    if (report.decoding == "sampling") {
        std::ostringstream sampling;
        sampling << " · sampling t=" << report.temperature;
        if (report.top_k > 0) sampling << " k=" << report.top_k;
        if (report.top_p < 1.0f) sampling << " p=" << report.top_p;
        sampling << " seed " << report.seed;
        line += sampling.str();
    } else {
        line += " · greedy";
    }
    if (report.repetition_penalty > 1.0f) {
        std::ostringstream penalty;
        penalty << " · repetition penalty " << report.repetition_penalty;
        line += penalty.str();
    }
    line += " · up to " + std::to_string(report.max_new_tokens) + " new tokens · " +
            (report.weights == "best" ? "best weights" : "current weights");
    return line;
}

std::vector<std::string> GenerationPreviewLogLines(const GenerationPreviewReport& report) {
    std::vector<std::string> lines;
    std::string point = report.point == "baseline" ? "before training"
                        : report.point == "step"   ? "step " + WithThousands(report.step)
                                                   : "epoch " + std::to_string(report.epoch) +
                                                         (report.total_epochs > 0 ? "/" + std::to_string(report.total_epochs) : "");
    size_t prompts = 0;
    for (const auto& sample : report.samples) prompts = std::max(prompts, static_cast<size_t>(sample.prompt_index + 1));
    lines.push_back("Generation · " + point + " · " + GenerationPreviewHowLine(report) + " · " +
                    std::to_string(prompts) + (prompts == 1 ? " prompt" : " prompts") + " · " +
                    OneDecimal(report.elapsed_ms / 1000.0) + " s" + (report.cancelled ? " · cancelled" : ""));
    bool several = false;
    for (const auto& sample : report.samples) several = several || sample.sample_index > 0;
    for (const auto& sample : report.samples) {
        std::string line = "  " + std::to_string(sample.prompt_index + 1);
        if (several) line += "." + std::to_string(sample.sample_index + 1);
        if (sample.stop_reason == "max_tokens") line += "  limit " + std::to_string(report.max_new_tokens);
        else if (sample.stop_reason == "eos") line += "  stopped " + std::to_string(sample.new_tokens);
        else if (sample.stop_reason == "stop_text") line += "  stopped at \"" + Shown(sample.stop_text) + "\"";
        else if (sample.stop_reason == "user_cancel") line += "  cancelled";
        else if (sample.stop_reason == "error") line += "  failed";
        else line += "  " + sample.stop_reason;
        if (!sample.repeated_phrase.empty()) line += " · repeats";
        if (sample.check_result == GenerationPreviewCheckResult::Passed) line += " · check passed";
        if (sample.check_result == GenerationPreviewCheckResult::Failed) line += " · check failed";
        line += "  \"" + GenerationPreviewExcerpt(sample.text, 70) + "\"";
        lines.push_back(std::move(line));
    }
    const auto summary = SummarizeGenerationPreview(report);
    std::string tail = "  ";
    if (summary.checks > 0)
        tail += "checks " + std::to_string(summary.checks_passed) + "/" + std::to_string(summary.checks) + " · ";
    tail += "stopped by itself " + std::to_string(summary.stopped_by_itself) + "/" + std::to_string(summary.samples) +
            " · repeating " + std::to_string(summary.repeating) + "/" + std::to_string(summary.samples);
    if (!report.file.empty()) tail += " · saved " + report.file.parent_path().filename().string() + "/" +
                                      report.file.filename().string();
    lines.push_back(std::move(tail));
    return lines;
}

std::string GenerationPreviewExcerpt(const std::string& text, size_t max_chars) {
    std::string out;
    size_t chars = 0;
    bool cut = false;
    bool space = false;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        const size_t length = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            space = !out.empty();
            ++i;
            continue;
        }
        if (chars + (space ? 1 : 0) >= max_chars) {
            cut = true;
            break;
        }
        if (space) {
            out.push_back(' ');
            ++chars;
            space = false;
        }
        out.append(text, i, std::min(length, text.size() - i));
        ++chars;
        i += length;
    }
    if (cut) out += "...";  // ASCII: the Console font has no U+2026
    return out;
}

}  // namespace cyxwiz
