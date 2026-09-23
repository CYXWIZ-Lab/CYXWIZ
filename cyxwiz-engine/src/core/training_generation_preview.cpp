#include "training_generation_preview.h"
#include "language_model_generation.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <stdexcept>

namespace cyxwiz {
namespace {
// Restore individual module modes, including mixed frozen/eval modules.
class EvaluationScope {
public:
    explicit EvaluationScope(SequentialModel& model):model_(model) {
        for(size_t i=0;i<model_.Size();++i) modes_.push_back(model_.GetModule(i)->IsTraining());
        model_.SetTraining(false);
    }
    ~EvaluationScope() {
        for(size_t i=0;i<modes_.size();++i) model_.GetModule(i)->SetTraining(modes_[i]);
    }
private:
    SequentialModel& model_;
    std::vector<bool> modes_;
};
std::string Hash(const std::string& text) {
    const auto hash=Utilities::HashText(text,"sha256");
    if(!hash.success) throw std::runtime_error("Generation preview hash failed: "+hash.error_message);
    return hash.sha256_hash;
}
}

TrainingGenerationPreview::TrainingGenerationPreview(
    const TrainingGenerationPreviewSettings& s,size_t model_vocabulary)
    :settings_(s),tokenizer_(static_cast<TokenizerType>(s.tokenizer_type)) {
    if(!s.enabled || s.every_epochs<1 || s.every_epochs>10000 || s.max_new_tokens<1 || s.max_new_tokens>128 ||
       s.context<2 || s.context>65536 || s.tokenizer_type<0 || s.tokenizer_type>3)
        throw std::invalid_argument("Invalid resolved generation preview configuration");
    std::istringstream artifact(s.vocabulary_artifact);
    if(!tokenizer_.GetVocabulary().LoadFromStream(artifact))
        throw std::invalid_argument("Generation preview requires the training vocabulary artifact");
    tokenizer_.SetLowercase(s.lowercase);
    tokenizer_.SetPadding(false); tokenizer_.SetTruncation(false);
    tokenizer_.SetAddBos(false); tokenizer_.SetAddEos(false);
    tokenizer_.ValidateVocabulary();
    if(tokenizer_.GetVocabulary().Size()!=model_vocabulary)
        throw std::invalid_argument("Generation preview vocabulary differs from model output width");
    prompts_=TrainingPreviewPrompts(s.prompts);
    for(const auto& prompt:prompts_) {
        const auto encoded=tokenizer_.Encode(prompt);
        if(encoded.empty() || encoded.size()>=static_cast<size_t>(s.context))
            throw std::invalid_argument("Generation preview prompt must leave at least one token of context");
        for(int id:encoded) if(id==tokenizer_.GetVocabulary().PadIndex())
            throw std::invalid_argument("Generation preview prompt contains PAD control token");
        prompt_ids_.emplace_back(encoded.begin(),encoded.end());
    }
    vocabulary_hash_=Hash(s.vocabulary_artifact);
    prompt_hash_=Hash(nlohmann::json(prompts_).dump());
}

bool TrainingGenerationPreview::Run(SequentialModel& model,int epoch,const std::string& run_id,
    const std::string& output_directory,const std::function<bool()>& should_stop) {
    if(!ShouldRunTrainingPreview(settings_,epoch)) return true;
    namespace fs=std::filesystem;
    const fs::path directory(output_directory);
    fs::create_directories(directory);
    const auto path=directory/("epoch_"+std::to_string(epoch)+".json");
    if(fs::exists(path)) throw std::runtime_error("Generation preview artifact already exists: "+path.string());
    const auto start=std::chrono::steady_clock::now();
    nlohmann::json report={{"run_id",run_id},{"completed_epoch",epoch},{"weights","current_epoch"},
        {"tokenizer_sha256",vocabulary_hash_},{"prompt_set_sha256",prompt_hash_},
        {"sampling","greedy"},{"context",settings_.context},{"max_new_tokens",settings_.max_new_tokens},
        {"samples",nlohmann::json::array()}};
    spdlog::info("Generation preview started: run={} epoch={} weights=current_epoch prompts={}",run_id,epoch,prompts_.size());
    bool cancelled=false;
    {
        EvaluationScope modes(model);
        for(size_t i=0;i<prompts_.size();++i) {
            if(should_stop && should_stop()) { cancelled=true; break; }
            LanguageModelGenerationConfig config;
            config.max_new_tokens=static_cast<size_t>(settings_.max_new_tokens);
            config.max_context_tokens=static_cast<size_t>(settings_.context);
            config.eos_token_id=tokenizer_.GetVocabulary().EosIndex();
            config.should_cancel=should_stop;
            const auto result=GenerateTokenIdsWithReport(model,prompt_ids_[i],config);
            std::vector<int> new_ids;
            new_ids.reserve(result.new_token_ids.size());
            for (int64_t id : result.new_token_ids) {
                if (id < 0 || id >= static_cast<int64_t>(tokenizer_.GetVocabulary().Size()) ||
                    id > static_cast<int64_t>((std::numeric_limits<int>::max)()))
                    throw std::runtime_error("Generation preview returned an invalid token ID");
                new_ids.push_back(static_cast<int>(id));
            }
            auto sample=nlohmann::json{{"prompt",prompts_[i]},{"prompt_token_ids",prompt_ids_[i]},
                {"new_token_ids",result.new_token_ids},{"generated_text",tokenizer_.Decode(new_ids)},
                {"stop_reason",LanguageModelGenerationStopReasonName(result.stop_reason)}};
            spdlog::info("Generation preview sample: epoch={} {}",epoch,
                sample.dump(-1,' ',false,nlohmann::json::error_handler_t::replace));
            report["samples"].push_back(std::move(sample));
            if(result.stop_reason==LanguageModelGenerationStopReason::UserCancelled) {cancelled=true;break;}
        }
    }
    report["cancelled"]=cancelled;
    report["elapsed_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::ofstream output(path,std::ios::binary);
    output.exceptions(std::ios::failbit|std::ios::badbit);
    output<<report.dump(2,' ',false,nlohmann::json::error_handler_t::replace)<<'\n';
    output.close();
    spdlog::info("Generation preview {}: epoch={} elapsed_ms={} artifact={}",
        cancelled?"cancelled":"completed",epoch,report["elapsed_ms"].get<double>(),path.string());
    return !cancelled;
}
} // namespace cyxwiz
