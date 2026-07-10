#pragma once

#include "../panel.h"
#include "../../inference/language_model_inference_contract.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class Tokenizer;
class SequentialModel;

class LanguageModelGenerationPanel : public Panel {
public:
    LanguageModelGenerationPanel();
    ~LanguageModelGenerationPanel() override;

    void Render() override;
    const char* GetIcon() const override { return "LM"; }

private:
    void RenderPrompt();
    void RenderControls();
    void RenderResult();
    void RunGeneration();
    void CheckModelCompatibility();
    void LoadTokenizerFromCyxModel();
    void LoadModelAndTokenizerFromCyxModel();
    void RequireImportedModelPackageContract() const;
    SequentialModel* ActiveModel() const;
    std::vector<int64_t> ParsePromptIds() const;
    std::vector<int64_t> CurrentPromptIdsForProbe() const;
    std::unique_ptr<Tokenizer> BuildTokenizer() const;
    static std::string JoinTokenIds(const std::vector<int64_t>& ids);

private:
    bool use_text_prompt_ = false;
    char prompt_ids_[2048] = "1 2 3";
    char text_prompt_[4096] = "hello world";
    char vocab_file_[512] = "";
    char cyxmodel_path_[512] = "";
    int tokenizer_type_idx_ = 1;
    bool lowercase_ = true;
    int max_length_ = 512;
    bool add_bos_ = false;
    bool add_eos_ = false;
    bool use_packaged_tokenizer_ = false;
    bool use_imported_model_ = false;
    std::string packaged_tokenizer_config_json_;
    std::string packaged_tokenizer_vocab_text_;
    std::string packaged_tokenizer_summary_;
    std::unique_ptr<SequentialModel> imported_model_;
    std::string imported_model_source_;
    std::string imported_model_summary_;
    LanguageModelPackageContract imported_model_contract_;

    int max_new_tokens_ = 16;
    float temperature_ = 1.0f;
    int top_k_ = 0;
    float top_p_ = 1.0f;
    int eos_token_id_ = -1;
    int seed_ = 5489;
    bool multinomial_sampling_ = false;
    bool include_prompt_ = true;

    std::vector<int64_t> generated_ids_;
    std::string status_;
    std::string compatibility_status_;
    std::string active_model_status_;
    std::string generated_text_;
    bool has_result_ = false;
};

} // namespace cyxwiz
