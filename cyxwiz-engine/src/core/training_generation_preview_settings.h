#pragma once
#include <charconv>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {
struct TrainingGenerationPreviewSettings {
    bool enabled = false;
    int every_epochs = 20;
    int max_new_tokens = 32;
    std::string prompts; // See TrainingPreviewPrompts; no hidden chat template.
    // Resolved from the selected training dataset, never entered independently.
    std::string vocabulary_artifact;
    int tokenizer_type = -1;
    bool lowercase = false;
    int context = 0;
};

// One prompt per line. When some line is exactly "---", prompts are instead
// separated by such lines and keep their line breaks, for chat-template prompts
// (TOFIX143); an empty last line before "---" gives the prompt a trailing newline.
inline std::vector<std::string> TrainingPreviewPrompts(const std::string& text) {
    if (text.size() > 8192) throw std::invalid_argument("Generation preview prompts exceed 8192 bytes");
    std::vector<std::string> lines;
    {
        std::istringstream input(text);
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back()=='\r') line.pop_back();
            lines.push_back(line);
        }
    }
    std::vector<std::string> prompts;
    const auto add = [&prompts](const std::string& prompt) {
        if (prompt.find_first_not_of(" \t\n") == std::string::npos) return;
        if (prompt.size()>2048 || prompt.find('\0')!=std::string::npos)
            throw std::invalid_argument("Each generation preview prompt must be at most 2048 bytes without NUL");
        prompts.push_back(prompt);
    };
    bool separated = false;
    for (const auto& line : lines) separated = separated || line == "---";
    if (separated) {
        std::string prompt;
        bool first = true;
        for (const auto& line : lines) {
            if (line == "---") { add(prompt); prompt.clear(); first = true; continue; }
            if (!first) prompt += '\n';
            prompt += line;
            first = false;
        }
        add(prompt);
    } else {
        for (const auto& line : lines) add(line);
    }
    if (prompts.empty() || prompts.size()>8)
        throw std::invalid_argument("Generation previews require 1 to 8 nonempty prompts: one per line, or separated by --- lines");
    return prompts;
}

inline TrainingGenerationPreviewSettings ParseTrainingGenerationPreview(
    const std::map<std::string,std::string>& params) {
    TrainingGenerationPreviewSettings value;
    const auto enabled=params.find("generation_preview_enabled");
    if (enabled==params.end() || enabled->second=="false") return value;
    if (enabled->second!="true") throw std::invalid_argument("generation_preview_enabled must be true or false");
    value.enabled=true;
    auto integer=[&](const char* key,int fallback,int upper) {
        const auto it=params.find(key);
        if(it==params.end()) return fallback;
        int n=0;
        const auto r=std::from_chars(it->second.data(),it->second.data()+it->second.size(),n);
        if(r.ec!=std::errc{} || r.ptr!=it->second.data()+it->second.size() || n<1 || n>upper)
            throw std::invalid_argument(std::string(key)+" is outside its supported positive integer range");
        return n;
    };
    value.every_epochs=integer("generation_preview_every_epochs",20,10000);
    value.max_new_tokens=integer("generation_preview_max_new_tokens",32,128);
    if(auto it=params.find("generation_preview_prompts");it!=params.end()) value.prompts=it->second;
    (void)TrainingPreviewPrompts(value.prompts);
    return value;
}

inline bool ShouldRunTrainingPreview(const TrainingGenerationPreviewSettings& s,int completed_epoch) {
    return s.enabled && s.every_epochs>0 && completed_epoch>0 && completed_epoch%s.every_epochs==0;
}
} // namespace cyxwiz
