#include "training_generation_preview.h"
#include "generation_preview_presentation.h"
#include "language_model_generation.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
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
    if(!s.enabled) throw std::invalid_argument("Generation previews are off");
    if(const auto error=ValidateTrainingGenerationPreview(s); !error.empty()) throw std::invalid_argument(error);
    if(s.context<2 || s.context>65536 || s.tokenizer_type<0 || s.tokenizer_type>3)
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
    for(size_t i=0;i<s.prompts.size();++i) {
        rendered_.push_back(RenderGenerationPreviewPrompt(s,s.prompts[i].text).text);
        const auto encoded=tokenizer_.Encode(rendered_.back());
        if(encoded.empty() || encoded.size()>=static_cast<size_t>(s.context))
            throw std::invalid_argument("Generation previews: prompt "+std::to_string(i+1)+" is "+
                std::to_string(encoded.size())+" tokens, it must leave room in the "+std::to_string(s.context)+
                "-token context");
        for(int id:encoded) if(id==tokenizer_.GetVocabulary().PadIndex())
            throw std::invalid_argument("Generation preview prompt contains PAD control token");
        prompt_ids_.emplace_back(encoded.begin(),encoded.end());
    }
    stop_texts_=GenerationPreviewStopTexts(s);
    vocabulary_hash_=Hash(s.vocabulary_artifact);
    nlohmann::json stored;
    for(const auto& [key,value]:SerializeTrainingGenerationPreview(s)) stored[key]=value;
    settings_hash_=Hash(stored.dump());
}

bool TrainingGenerationPreview::Run(SequentialModel& model,const GenerationPreviewPoint& point,
    const std::string& run_id,const std::filesystem::path& directory,const std::function<bool()>& should_stop) {
    namespace fs=std::filesystem;
    fs::create_directories(directory);
    const auto path=GenerationPreviewReportPath(directory,point.kind,point.epoch,point.step);
    const auto start=std::chrono::steady_clock::now();
    const auto& s=settings_;
    const bool sampling=s.decoding==GenerationPreviewDecoding::Sampling;
    nlohmann::json report={{"schema",kGenerationPreviewReportSchema},{"run_id",run_id},{"point",point.kind},
        {"epoch",point.epoch},{"total_epochs",point.total_epochs},{"step",point.step},{"weights",point.weights},
        {"mode",GenerationPreviewModeName(s.mode)},
        {"template",s.mode==GenerationPreviewMode::Chat?GenerationPreviewTemplateName(s.chat_template):""},
        {"decoding",GenerationPreviewDecodingName(s.decoding)},{"temperature",sampling?s.temperature:0.0f},
        {"top_k",sampling?s.top_k:0},{"top_p",sampling?s.top_p:1.0f},{"seed",s.seed},
        {"repetition_penalty",s.repetition_penalty},{"max_new_tokens",s.max_new_tokens},{"context",s.context},
        {"stop_texts",stop_texts_},{"tokenizer_sha256",vocabulary_hash_},{"settings_sha256",settings_hash_},
        {"samples",nlohmann::json::array()}};
    if(point.validation_loss) report["validation_loss"]=*point.validation_loss;
    spdlog::debug("Generation preview started: run={} point={} epoch={} step={} weights={} prompts={}",
                  run_id,point.kind,point.epoch,point.step,point.weights,prompt_ids_.size());
    const auto vocabulary_size=static_cast<int64_t>(tokenizer_.GetVocabulary().Size());
    const auto eos=tokenizer_.GetVocabulary().EosIndex();
    const auto decode=[&](const std::vector<int64_t>& ids) {
        std::vector<int> values;
        values.reserve(ids.size());
        for(int64_t id:ids) {
            if(id<0 || id>=vocabulary_size || id>static_cast<int64_t>((std::numeric_limits<int>::max)()))
                throw std::runtime_error("Generation preview returned an invalid token ID");
            if(id==eos) continue;
            values.push_back(static_cast<int>(id));
        }
        return tokenizer_.Decode(values);
    };
    bool cancelled=false;
    {
        EvaluationScope modes(model);
        for(size_t i=0;i<prompt_ids_.size() && !cancelled;++i) {
            for(int n=0;n<s.samples_per_prompt;++n) {
                if(should_stop && should_stop()) { cancelled=true; break; }
                LanguageModelGenerationConfig config;
                config.max_new_tokens=static_cast<size_t>(s.max_new_tokens);
                config.max_context_tokens=static_cast<size_t>(s.context);
                config.eos_token_id=eos;
                config.should_cancel=should_stop;
                config.repetition_penalty=s.repetition_penalty;
                if(sampling) {
                    config.sampling_mode=LanguageModelSamplingMode::Multinomial;
                    config.temperature=s.temperature;
                    config.top_k=static_cast<size_t>(s.top_k);
                    config.top_p=s.top_p;
                }
                std::string matched;
                if(!stop_texts_.empty()) {
                    config.stop_check=[&](const std::vector<int64_t>& ids) {
                        const auto text=decode(ids);
                        for(const auto& stop:stop_texts_)
                            if(text.find(stop)!=std::string::npos) { matched=stop; return true; }
                        return false;
                    };
                }
                const uint32_t seed=s.seed+static_cast<uint32_t>(i*1000+n);
                const auto result=GenerateTokenIdsWithReport(model,prompt_ids_[i],config,seed);
                auto text=decode(result.new_token_ids);
                if(result.stop_reason==LanguageModelGenerationStopReason::StopText && !matched.empty())
                    text=text.substr(0,text.find(matched));
                nlohmann::json sample={{"prompt_index",i},{"sample_index",n},{"prompt",s.prompts[i].text},
                    {"rendered_prompt",rendered_[i]},{"prompt_token_ids",prompt_ids_[i]},
                    {"new_token_ids",result.new_token_ids},{"text",text},
                    {"stop_reason",LanguageModelGenerationStopReasonName(result.stop_reason)},{"stop_text",matched}};
                if(s.prompts[i].check.kind!=GenerationPreviewCheckKind::None)
                    sample["check"]={{"kind",GenerationPreviewCheckKindName(s.prompts[i].check.kind)},
                                     {"value",s.prompts[i].check.value}};
                spdlog::debug("Generation sample: point={} epoch={} step={} prompt={} sample={} {}",point.kind,
                    point.epoch,point.step,i+1,n+1,sample.dump(-1,' ',false,nlohmann::json::error_handler_t::replace));
                report["samples"].push_back(std::move(sample));
                if(result.stop_reason==LanguageModelGenerationStopReason::UserCancelled) { cancelled=true; break; }
            }
        }
    }
    report["cancelled"]=cancelled;
    report["elapsed_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    {
        std::ofstream output(path,std::ios::binary);
        output.exceptions(std::ios::failbit|std::ios::badbit);
        output<<report.dump(2,' ',false,nlohmann::json::error_handler_t::replace)<<'\n';
    }
    std::string error;
    if(const auto written=LoadGenerationPreviewReport(path,error)) {
        std::string block;
        for(const auto& line:GenerationPreviewLogLines(*written)) block+=(block.empty()?"":"\n")+line;
        spdlog::info("{}",block);
    } else {
        spdlog::warn("Generation preview report {} could not be read back: {}",path.string(),error);
    }
    return !cancelled;
}
} // namespace cyxwiz
