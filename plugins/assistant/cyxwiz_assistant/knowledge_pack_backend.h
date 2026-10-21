#pragma once

#include "assistant_backend_contract.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz::plugin::assistant {

class KnowledgePackBackend final : public IAssistantBackend {
public:
    explicit KnowledgePackBackend(std::filesystem::path pack_dir);

    AssistantResponse Run(const AssistantRequest& request) override;

    bool IsLoaded() const { return loaded_; }
    const std::string& Status() const { return status_; }

private:
    struct Chunk {
        std::string id;
        std::string source_type;
        std::string path;
        int line_start = 0;
        int line_end = 0;
        std::string title;
        std::string text;
        std::vector<std::string> tags;
    };

    struct Posting {
        int chunk = 0;
        int count = 0;
    };

    void Load();
    std::vector<AssistantRetrievalHit> Search(const std::string& query, int top_k) const;
    int ScoreChunk(const Chunk& chunk, const std::vector<std::string>& query_terms, bool broad_help_query) const;
    std::string MakeEvidenceText(const std::vector<AssistantRetrievalHit>& hits) const;
    std::string BuildPrompt(const AssistantRequest& request,
                            const std::vector<AssistantRetrievalHit>& hits) const;
    bool CallRuntime(const AssistantRequest& request,
                     const std::string& prompt,
                     std::string& output,
                     std::string& error) const;
    bool ParseSections(const std::string& output, AssistantResponse& response) const;
    const Chunk* FindChunk(const AssistantCitation& citation) const;

    static std::vector<std::string> Tokenize(const std::string& text);
    static std::string Normalize(const std::string& text);
    static std::string Preview(const std::string& text, std::size_t limit);
    static int CountContains(const std::string& haystack, const std::string& needle, int max_count);

    std::filesystem::path pack_dir_;
    bool loaded_ = false;
    std::string status_ = "not loaded";
    nlohmann::json manifest_;
    std::vector<Chunk> chunks_;
    std::unordered_map<std::string, std::vector<Posting>> postings_;
};

} // namespace cyxwiz::plugin::assistant
