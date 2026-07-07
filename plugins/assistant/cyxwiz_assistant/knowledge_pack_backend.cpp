#include "knowledge_pack_backend.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <httplib.h>
#include <set>
#include <sstream>

namespace cyxwiz::plugin::assistant {

namespace {

constexpr const char* kManifestSchema = "cyxwiz.assistant.knowledge_pack.v1";
constexpr const char* kDefaultRuntimeEndpoint = "http://127.0.0.1:8768/completion";

const std::set<std::string> kStopwords = {
    "a", "an", "and", "are", "can", "does", "file", "for", "from", "in",
    "is", "me", "of", "show", "source", "the", "this", "to", "what",
    "where", "which", "who",
};

const std::set<std::string> kBroadHelpTerms = {
    "assist",
    "assistant",
    "capabilities",
    "capability",
    "help",
    "overview",
    "use",
};

int SourceTypeBoost(const std::string& source_type) {
    if (source_type == "source") return 40;
    if (source_type == "cyxgraph") return 30;
    if (source_type == "cyxgraph_node") return 25;
    if (source_type == "cyxgraph_links") return 10;
    return 0;
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string JoinTags(const std::vector<std::string>& tags) {
    std::string out;
    for (const auto& tag : tags) {
        if (!out.empty()) out += ' ';
        out += tag;
    }
    return out;
}

std::string BuildRetrievalQuery(const AssistantRequest& request) {
    std::ostringstream query;
    query << request.user_text;
    if (request.command_name == "explain_trace") {
        query << " DebugTraceRecord DebugTraceRole ValidationIssue StudioDebugger";
        query << " " << request.debugger_context_json;
    } else if (request.command_name == "explain_training") {
        query << " TrainingTraceEvent TrainingTraceSummary terminal_reason";
        query << " " << request.training_context_json;
    } else {
        query << " " << request.selected_node_id;
        query << " " << request.debugger_context_json;
    }
    return query.str();
}

bool ParseLocalRuntimeEndpoint(
    const std::string& endpoint,
    std::string& host,
    std::string& path,
    std::string& error) {
    const std::string value = endpoint.empty() ? kDefaultRuntimeEndpoint : endpoint;
    constexpr const char* http_prefix = "http://";
    if (value.rfind(http_prefix, 0) != 0) {
        error = "Runtime endpoint must use http:// and localhost.";
        return false;
    }

    const auto after_scheme = value.substr(std::string(http_prefix).size());
    const auto slash = after_scheme.find('/');
    const auto authority = slash == std::string::npos ? after_scheme : after_scheme.substr(0, slash);
    path = slash == std::string::npos ? "/" : after_scheme.substr(slash);

    const auto colon = authority.rfind(':');
    const auto hostname = colon == std::string::npos ? authority : authority.substr(0, colon);
    if (hostname != "127.0.0.1" && hostname != "localhost") {
        error = "Runtime endpoint must point to localhost or 127.0.0.1.";
        return false;
    }
    if (path.empty() || path[0] != '/') {
        error = "Runtime endpoint path is invalid.";
        return false;
    }

    host = std::string(http_prefix) + authority;
    return true;
}

} // namespace

KnowledgePackBackend::KnowledgePackBackend(std::filesystem::path pack_dir)
    : pack_dir_(std::move(pack_dir)) {
    Load();
}

AssistantResponse KnowledgePackBackend::Run(const AssistantRequest& request) {
    AssistantResponse response;
    if (!loaded_) {
        response.error_code = "knowledge_pack_invalid";
        response.error_message = status_;
        return response;
    }
    if (request.user_text.empty() &&
        request.command_name == "ask" &&
        request.selected_node_id.empty()) {
        response.error_code = "invalid_request";
        response.error_message = "Enter a question before asking the assistant.";
        return response;
    }

    const auto retrieval_query = BuildRetrievalQuery(request);
    const auto hits = Search(retrieval_query, std::max(1, request.top_k));
    response.retrieval_hits = hits;
    response.retrieval_ok = !hits.empty();
    response.success = response.retrieval_ok;
    response.runtime_ok = false;
    response.parsed = false;

    for (const auto& hit : hits) {
        response.citations.push_back(hit.citation);
    }

    if (hits.empty()) {
        response.error_code = "retrieval_no_hits";
        response.error_message = "No matching evidence was found in the knowledge pack.";
        response.unknowns = "No matching local evidence was found.";
        return response;
    }

    const auto& top = hits.front();
    response.evidence = MakeEvidenceText(hits);
    response.answer =
        "Retrieval-only result. Top evidence: " + top.citation.path + ":" +
        std::to_string(top.citation.line_start) + "-" + std::to_string(top.citation.line_end);

    if (request.retrieval_only) {
        response.unknowns = "No model inference was performed in retrieval-only mode.";
        response.unsupported_or_not_implemented =
            "Model runtime synthesis was skipped by request.";
        return response;
    }

    const auto prompt = BuildPrompt(request, hits);
    std::string runtime_output;
    std::string runtime_error;
    if (!CallRuntime(request, prompt, runtime_output, runtime_error)) {
        response.success = false;
        response.runtime_ok = false;
        response.error_code = "runtime_unavailable";
        response.error_message = runtime_error;
        response.unknowns =
            "Model runtime did not return an answer. Retrieval hits are still available.";
        response.unsupported_or_not_implemented =
            "Full answer synthesis requires the local runtime proxy at http://127.0.0.1:8768/completion.";
        return response;
    }

    response.runtime_ok = true;
    response.raw_output = runtime_output;
    if (!ParseSections(runtime_output, response)) {
        response.success = false;
        response.parsed = false;
        response.error_code = "runtime_parse_failed";
        response.error_message = "Runtime output did not include all required answer sections.";
        return response;
    }

    response.success = true;
    response.parsed = true;
    return response;
}

void KnowledgePackBackend::Load() {
    try {
        const auto manifest_path = pack_dir_ / "manifest.json";
        std::ifstream manifest_stream(manifest_path);
        if (!manifest_stream) {
            status_ = "Knowledge pack manifest not found: " + manifest_path.string();
            return;
        }
        manifest_stream >> manifest_;
        if (manifest_.value("schema", "") != kManifestSchema) {
            status_ = "Unsupported knowledge pack schema: " + manifest_.value("schema", "");
            return;
        }

        const auto assets = manifest_.value("assets", nlohmann::json::object());
        const auto chunks_path = pack_dir_ / assets.value("chunks", "chunks.jsonl");
        const auto postings_path = pack_dir_ / assets.value("postings", "postings.json");

        std::ifstream chunk_stream(chunks_path);
        if (!chunk_stream) {
            status_ = "Knowledge pack chunks not found: " + chunks_path.string();
            return;
        }

        std::string line;
        while (std::getline(chunk_stream, line)) {
            if (line.empty()) continue;
            const auto raw = nlohmann::json::parse(line);
            Chunk chunk;
            chunk.id = raw.value("id", "");
            chunk.source_type = raw.value("source_type", "");
            chunk.path = raw.value("path", "");
            chunk.line_start = raw.value("line_start", 0);
            chunk.line_end = raw.value("line_end", 0);
            chunk.title = raw.value("title", "");
            chunk.text = raw.value("text", "");
            if (raw.contains("tags") && raw["tags"].is_array()) {
                chunk.tags = raw["tags"].get<std::vector<std::string>>();
            }
            chunks_.push_back(std::move(chunk));
        }

        std::ifstream postings_stream(postings_path);
        if (!postings_stream) {
            status_ = "Knowledge pack postings not found: " + postings_path.string();
            return;
        }
        const auto raw_postings = nlohmann::json::parse(postings_stream);
        for (auto it = raw_postings.begin(); it != raw_postings.end(); ++it) {
            std::vector<Posting> postings;
            for (const auto& item : it.value()) {
                postings.push_back({item.value("chunk", 0), item.value("count", 0)});
            }
            postings_[it.key()] = std::move(postings);
        }

        const auto expected_chunks = manifest_.value("chunk_count", 0);
        if (expected_chunks != static_cast<int>(chunks_.size())) {
            status_ = "Knowledge pack chunk count mismatch.";
            return;
        }

        loaded_ = true;
        status_ = "loaded";
    } catch (const std::exception& exc) {
        loaded_ = false;
        status_ = std::string("Failed to load knowledge pack: ") + exc.what();
    }
}

std::vector<AssistantRetrievalHit> KnowledgePackBackend::Search(
    const std::string& query,
    int top_k) const {
    const auto query_terms = Tokenize(query);
    const bool broad_help_query =
        std::find(query_terms.begin(), query_terms.end(), "cyxwiz") != query_terms.end()
        && (query_terms.size() <= 4
            || std::any_of(query_terms.begin(), query_terms.end(), [](const auto& term) {
                return kBroadHelpTerms.count(term) != 0;
            }));
    std::set<int> candidate_indexes;

    for (const auto& token : query_terms) {
        auto found = postings_.find(token);
        if (found == postings_.end()) continue;
        for (const auto& posting : found->second) {
            candidate_indexes.insert(posting.chunk);
        }
    }

    if (candidate_indexes.empty()) {
        for (int i = 0; i < static_cast<int>(chunks_.size()); ++i) {
            candidate_indexes.insert(i);
        }
    }

    std::vector<AssistantRetrievalHit> hits;
    for (int index : candidate_indexes) {
        if (index < 0 || index >= static_cast<int>(chunks_.size())) continue;
        const auto& chunk = chunks_[index];
        const int score = ScoreChunk(chunk, query_terms, broad_help_query);
        if (score <= 0) continue;

        AssistantRetrievalHit hit;
        hit.rank = 0;
        hit.score = static_cast<double>(score);
        hit.citation = {
            chunk.path,
            chunk.line_start,
            chunk.line_end,
            chunk.title,
            chunk.source_type,
        };
        hit.snippet = Preview(chunk.text, 280);
        hits.push_back(std::move(hit));
    }

    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.citation.path != b.citation.path) return a.citation.path < b.citation.path;
        return a.citation.line_start < b.citation.line_start;
    });
    if (hits.size() > static_cast<std::size_t>(top_k)) {
        hits.resize(static_cast<std::size_t>(top_k));
    }
    for (std::size_t i = 0; i < hits.size(); ++i) {
        hits[i].rank = static_cast<int>(i + 1);
    }
    return hits;
}

int KnowledgePackBackend::ScoreChunk(
    const Chunk& chunk,
    const std::vector<std::string>& query_terms,
    bool broad_help_query) const {
    const auto text = Lower(chunk.text);
    const auto title = Lower(chunk.title);
    const auto path = Lower(chunk.path);
    const auto tags = Lower(JoinTags(chunk.tags));
    const auto text_norm = Normalize(text);
    const auto title_norm = Normalize(title);
    const auto path_norm = Normalize(path);
    const auto tags_norm = Normalize(tags);

    int score = SourceTypeBoost(chunk.source_type);
    int matched_terms = 0;

    if (broad_help_query) {
        if (chunk.source_type == "markdown" || chunk.source_type == "text") {
            score += 120;
        }
        if (path.find("knowledge_seed") != std::string::npos
            || path.find("overview") != std::string::npos
            || path.find("readme") != std::string::npos) {
            score += 100;
        }
        if (path.find("usage") != std::string::npos
            || path.find("assistant") != std::string::npos
            || path.find("capabilities") != std::string::npos) {
            score += 40;
        }
    }

    for (const auto& term : query_terms) {
        if (term.empty()) continue;
        const auto term_norm = Normalize(term);
        const int exact_weight = std::min(static_cast<int>(term.size()), 16);
        const int text_hits = std::min(std::max(CountContains(text, term, 3),
                                                CountContains(text_norm, term_norm, 3)), 3);
        const int title_hits = std::min(std::max(CountContains(title, term, 2),
                                                 CountContains(title_norm, term_norm, 2)), 2);
        const int path_hits = std::min(std::max(CountContains(path, term, 2),
                                                CountContains(path_norm, term_norm, 2)), 2);
        const int tag_hits = std::min(std::max(CountContains(tags, term, 2),
                                               CountContains(tags_norm, term_norm, 2)), 2);
        if (text_hits || title_hits || path_hits || tag_hits) {
            matched_terms += 1;
        }
        score += text_hits * std::max(1, exact_weight / 4);
        score += title_hits * (10 + exact_weight);
        score += path_hits * (6 + exact_weight);
        score += tag_hits * (4 + exact_weight);
    }

    if (matched_terms > 0) {
        score += matched_terms * matched_terms * 3;
        if (matched_terms == static_cast<int>(query_terms.size())) {
            score += 30;
        } else if (matched_terms >= std::max(3, static_cast<int>(query_terms.size()) - 1)) {
            score += 15;
        }
    }

    return matched_terms > 0 ? score : 0;
}

std::string KnowledgePackBackend::MakeEvidenceText(
    const std::vector<AssistantRetrievalHit>& hits) const {
    std::ostringstream out;
    for (const auto& hit : hits) {
        out << "[E" << hit.rank << "] " << hit.citation.path << ":"
            << hit.citation.line_start << "-" << hit.citation.line_end
            << " title=" << hit.citation.title << "\n";
    }
    return out.str();
}

std::string KnowledgePackBackend::BuildPrompt(
    const AssistantRequest& request,
    const std::vector<AssistantRetrievalHit>& hits) const {
    std::ostringstream evidence;
    for (const auto& hit : hits) {
        evidence << "[E" << hit.rank << "] " << hit.citation.path << ":"
                 << hit.citation.line_start << "-" << hit.citation.line_end << "\n"
                 << "title: " << hit.citation.title << "\n"
                 << "type: " << hit.citation.source_type << "\n"
                 << "text:\n";
        if (const auto* chunk = FindChunk(hit.citation)) {
            evidence << chunk->text << "\n\n";
        } else {
            evidence << hit.snippet << "\n\n";
        }
    }

    std::ostringstream prompt;
    prompt << "You are the local CyxWiz source-aware assistant.\n\n"
           << "Answer only from the cited evidence below.\n"
           << "Separate facts from inference.\n"
           << "If evidence is missing, say what is missing.\n"
           << "Do not claim unsupported CyxWiz behavior exists.\n"
           << "Do not suggest graph or source mutation unless explicitly approved.\n\n"
           << "Command:\n"
           << request.command_name << "\n\n"
           << "Question:\n"
           << (request.user_text.empty() ? "(use selected CyxWiz context)" : request.user_text) << "\n\n"
           << "Selected engine context:\n"
           << "engine_version=" << request.engine_version << "\n"
           << "build_id=" << request.build_id << "\n"
           << "active_graph_path=" << request.active_graph_path << "\n"
           << "selected_node_id=" << request.selected_node_id << "\n"
           << "selected_trace_id=" << request.selected_trace_id << "\n"
           << "debugger_context_json=" << request.debugger_context_json << "\n"
           << "training_context_json=" << request.training_context_json << "\n\n"
           << "Evidence:\n"
           << evidence.str()
           << "Missing evidence notes:\n"
           << "- none\n\n"
           << "Return this structure:\n"
           << "Answer:\n"
           << "Evidence:\n"
           << "Unknowns:\n"
           << "Unsupported or not implemented:\n";
    return prompt.str();
}

bool KnowledgePackBackend::CallRuntime(
    const AssistantRequest& request,
    const std::string& prompt,
     std::string& output,
     std::string& error) const {
    std::string host;
    std::string path;
    if (!ParseLocalRuntimeEndpoint(request.runtime_endpoint, host, path, error)) {
        return false;
    }

    httplib::Client client(host);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(std::max(5, request.timeout_seconds), 0);
    client.set_write_timeout(5, 0);

    nlohmann::json body = {
        {"prompt", prompt},
        {"n_predict", 384},
        {"stream", false},
    };

    auto result = client.Post(path, body.dump(), "application/json");
    if (!result) {
        error = "Runtime proxy request failed: " + httplib::to_string(result.error());
        return false;
    }
    if (result->status < 200 || result->status >= 300) {
        error = "Runtime proxy returned HTTP " + std::to_string(result->status) + ": " + result->body;
        return false;
    }

    try {
        const auto payload = nlohmann::json::parse(result->body);
        if (payload.contains("content") && payload["content"].is_string()) {
            output = payload["content"].get<std::string>();
            return !output.empty();
        }
        if (payload.contains("response") && payload["response"].is_string()) {
            output = payload["response"].get<std::string>();
            return !output.empty();
        }
        error = "Runtime proxy response did not include content text.";
        return false;
    } catch (const std::exception& exc) {
        error = std::string("Runtime proxy returned invalid JSON: ") + exc.what();
        return false;
    }
}

bool KnowledgePackBackend::ParseSections(const std::string& output, AssistantResponse& response) const {
    enum class Section {
        None,
        Answer,
        Evidence,
        Unknowns,
        Unsupported,
    };

    auto trim = [](std::string value) {
        auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) {
            return !is_space(static_cast<unsigned char>(ch));
        }));
        value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) {
            return !is_space(static_cast<unsigned char>(ch));
        }).base(), value.end());
        return value;
    };

    Section section = Section::None;
    std::ostringstream answer;
    std::ostringstream evidence;
    std::ostringstream unknowns;
    std::ostringstream unsupported;

    auto append = [&](const std::string& value) {
        if (section == Section::Answer) answer << value << "\n";
        else if (section == Section::Evidence) evidence << value << "\n";
        else if (section == Section::Unknowns) unknowns << value << "\n";
        else if (section == Section::Unsupported) unsupported << value << "\n";
    };

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("Answer:", 0) == 0) {
            section = Section::Answer;
            append(line.substr(std::string("Answer:").size()));
        } else if (line.rfind("Evidence:", 0) == 0) {
            section = Section::Evidence;
            append(line.substr(std::string("Evidence:").size()));
        } else if (line.rfind("Unknowns:", 0) == 0) {
            section = Section::Unknowns;
            append(line.substr(std::string("Unknowns:").size()));
        } else if (line.rfind("Unsupported or not implemented:", 0) == 0) {
            section = Section::Unsupported;
            append(line.substr(std::string("Unsupported or not implemented:").size()));
        } else {
            append(line);
        }
    }

    response.answer = trim(answer.str());
    response.evidence = trim(evidence.str());
    response.unknowns = trim(unknowns.str());
    response.unsupported_or_not_implemented = trim(unsupported.str());
    return !response.answer.empty()
        && !response.evidence.empty()
        && !response.unknowns.empty()
        && !response.unsupported_or_not_implemented.empty();
}

const KnowledgePackBackend::Chunk* KnowledgePackBackend::FindChunk(
    const AssistantCitation& citation) const {
    for (const auto& chunk : chunks_) {
        if (chunk.path == citation.path
            && chunk.line_start == citation.line_start
            && chunk.line_end == citation.line_end
            && chunk.title == citation.title) {
            return &chunk;
        }
    }
    return nullptr;
}

std::vector<std::string> KnowledgePackBackend::Tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::set<std::string> seen;
    std::string current;

    auto flush = [&]() {
        if (current.empty()) return;
        auto token = Lower(current);
        current.clear();
        std::vector<std::string> variants = {token, Normalize(token)};
        if (token.size() > 10 && token.ends_with("vectorizer")) {
            auto base = token.substr(0, token.size() - std::string("vectorizer").size());
            variants.push_back(base);
            variants.push_back(Normalize(base));
        }
        for (const auto& variant : variants) {
            if (variant.size() <= 1) continue;
            if (kStopwords.count(variant)) continue;
            if (seen.insert(variant).second) {
                tokens.push_back(variant);
            }
        }
    };

    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '_' || ch == '.' || ch == '/' || ch == ':' || ch == '-') {
            current.push_back(static_cast<char>(ch));
        } else {
            flush();
        }
    }
    flush();
    return tokens;
}

std::string KnowledgePackBackend::Normalize(const std::string& text) {
    std::string out;
    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '_') {
            out.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return out;
}

std::string KnowledgePackBackend::Preview(const std::string& text, std::size_t limit) {
    std::string compact;
    compact.reserve(std::min(text.size(), limit));
    bool last_space = false;
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!last_space) {
                compact.push_back(' ');
                last_space = true;
            }
            continue;
        }
        compact.push_back(static_cast<char>(ch));
        last_space = false;
    }
    if (compact.size() <= limit) return compact;
    return compact.substr(0, limit) + "...";
}

int KnowledgePackBackend::CountContains(
    const std::string& haystack,
    const std::string& needle,
    int max_count) {
    if (needle.empty()) return 0;
    int count = 0;
    std::size_t pos = 0;
    while (count < max_count) {
        pos = haystack.find(needle, pos);
        if (pos == std::string::npos) break;
        ++count;
        pos += needle.size();
    }
    return count;
}

} // namespace cyxwiz::plugin::assistant
