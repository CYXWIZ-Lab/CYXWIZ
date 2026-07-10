#include "language_model_generation_panel.h"

#include "../../core/language_model_generation.h"
#include "../../core/training_manager.h"
#include "../../core/file_dialogs.h"
#include "../../core/formats/cyxmodel_format.h"
#include "../../core/model_importer.h"
#include "../../inference/text_inference_input.h"
#include "../../inference/language_model_inference_contract.h"
#include "../icons.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/tokenizer.h>

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace cyxwiz {

LanguageModelGenerationPanel::LanguageModelGenerationPanel()
    : Panel("Language Model Generation", false) {
    status_ =
        "Train a decoder model that returns Float32 [1, seq, vocab] logits, "
        "then enter prompt token IDs.";
}

LanguageModelGenerationPanel::~LanguageModelGenerationPanel() = default;

void LanguageModelGenerationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(760, 560), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Language Model Generation###LanguageModelGeneration",
                     &visible_)) {
        ImGui::Text("%s Language Model Generation", ICON_FA_WAND_MAGIC_SPARKLES);
        ImGui::TextWrapped(
            "Current contract: the model must return Float32 logits shaped "
            "[1, seq, vocab]. Text mode uses a CyxWiz vocabulary file; raw "
            "token-ID mode remains available for debugging.");
        ImGui::Separator();

        RenderPrompt();
        ImGui::Separator();
        RenderControls();
        ImGui::Separator();
        RenderResult();
    }
    ImGui::End();
}

void LanguageModelGenerationPanel::RenderPrompt() {
    ImGui::Text("%s Prompt", ICON_FA_KEYBOARD);
    if (ImGui::RadioButton("Text prompt", use_text_prompt_)) {
        use_text_prompt_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Raw token IDs", !use_text_prompt_)) {
        use_text_prompt_ = false;
    }

    if (use_text_prompt_) {
        ImGui::InputTextMultiline("##TextPrompt",
                                  text_prompt_,
                                  sizeof(text_prompt_),
                                  ImVec2(-1, 96));

        ImGui::InputText("Vocabulary file", vocab_file_, sizeof(vocab_file_));
        ImGui::SameLine();
        if (ImGui::Button("Browse##GenerationVocab")) {
            auto result = FileDialogs::OpenFile(
                "Select Vocabulary File",
                {{"Vocabulary", "txt,vocab"},
                 {"All Files", "*"}});
            if (result) {
                std::snprintf(vocab_file_,
                              sizeof(vocab_file_),
                              "%s",
                              result->c_str());
            }
        }

        ImGui::InputText("CyxModel package", cyxmodel_path_, sizeof(cyxmodel_path_));
        ImGui::SameLine();
        if (ImGui::Button("Browse##GenerationCyxModel")) {
            auto result = FileDialogs::OpenFile(
                "Select CyxModel Package",
                {{"CyxModel", "cyxmodel"},
                 {"All Files", "*"}});
            if (result) {
                std::snprintf(cyxmodel_path_,
                              sizeof(cyxmodel_path_),
                              "%s",
                              result->c_str());
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load tokenizer assets")) {
            LoadTokenizerFromCyxModel();
        }
        ImGui::SameLine();
        if (ImGui::Button("Load model + tokenizer")) {
            LoadModelAndTokenizerFromCyxModel();
        }

        ImGui::Checkbox("Use packaged tokenizer assets", &use_packaged_tokenizer_);
        if (!packaged_tokenizer_summary_.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", packaged_tokenizer_summary_.c_str());
        }
        if (imported_model_) {
            ImGui::Checkbox("Use imported .cyxmodel model", &use_imported_model_);
            ImGui::SameLine();
            ImGui::TextDisabled("%s", imported_model_source_.c_str());
            if (!imported_model_summary_.empty()) {
                ImGui::TextDisabled("%s", imported_model_summary_.c_str());
            }
        }

        const char* tokenizer_types[] = {"Whitespace", "Word", "Character"};
        if (use_packaged_tokenizer_) {
            ImGui::BeginDisabled();
        }
        ImGui::Combo("Tokenizer", &tokenizer_type_idx_, tokenizer_types, 3);
        ImGui::SameLine();
        ImGui::Checkbox("Lowercase", &lowercase_);
        ImGui::SameLine();
        ImGui::Checkbox("BOS", &add_bos_);
        ImGui::SameLine();
        ImGui::Checkbox("EOS", &add_eos_);
        ImGui::InputInt("Max prompt length", &max_length_);
        if (use_packaged_tokenizer_) {
            ImGui::EndDisabled();
        }
        ImGui::TextDisabled(
            "Manual mode uses the selected vocabulary file. Packaged mode uses "
            "tokenizer/config.json and tokenizer/vocab.txt from a .cyxmodel.");
    } else {
        ImGui::InputTextMultiline("##PromptTokenIds",
                                  prompt_ids_,
                                  sizeof(prompt_ids_),
                                  ImVec2(-1, 96));
        ImGui::TextDisabled(
            "Use spaces, commas, or new lines. Example: 1 42 17");
    }
}

void LanguageModelGenerationPanel::RenderControls() {
    ImGui::Text("%s Generation controls", ICON_FA_SLIDERS);
    ImGui::PushItemWidth(180);
    ImGui::InputInt("Max new tokens", &max_new_tokens_);
    ImGui::InputFloat("Temperature", &temperature_, 0.05f, 0.25f, "%.3f");
    ImGui::InputInt("Top-K (0 disables)", &top_k_);
    ImGui::InputFloat("Top-P", &top_p_, 0.05f, 0.25f, "%.3f");
    ImGui::InputInt("EOS token (-1 disables)", &eos_token_id_);
    ImGui::InputInt("Seed", &seed_);
    ImGui::PopItemWidth();

    ImGui::Checkbox("Multinomial sampling", &multinomial_sampling_);
    ImGui::SameLine();
    ImGui::Checkbox("Include prompt in output", &include_prompt_);

    auto& training = TrainingManager::Instance();
    const bool has_model = (use_imported_model_ && imported_model_) ||
                           training.HasTrainedModel();
    active_model_status_ = use_imported_model_ && imported_model_
        ? "Active model: imported .cyxmodel"
        : (training.HasTrainedModel()
            ? "Active model: last trained model"
            : "Active model: none");
    ImGui::TextDisabled("%s", active_model_status_.c_str());
    if (!has_model) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.15f, 1.0f),
                           "%s No trained or imported model available",
                           ICON_FA_TRIANGLE_EXCLAMATION);
    }

    if (!has_model) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_FA_STETHOSCOPE " Check compatibility", ImVec2(190, 0))) {
        CheckModelCompatibility();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_PLAY " Generate", ImVec2(160, 0))) {
        RunGeneration();
    }
    if (!has_model) {
        ImGui::EndDisabled();
    }

    if (!compatibility_status_.empty()) {
        ImGui::TextWrapped("%s", compatibility_status_.c_str());
    }
}

void LanguageModelGenerationPanel::RenderResult() {
    ImGui::Text("%s Result", ICON_FA_TERMINAL);
    if (!status_.empty()) {
        ImGui::TextWrapped("%s", status_.c_str());
    }

    if (!stop_reason_.empty()) {
        ImGui::Text("Stop reason: %s", stop_reason_.c_str());
    }
    if (!has_result_) {
        return;
    }

    ImGui::Text("Prompt length: %zu", last_prompt_length_);
    ImGui::SameLine();
    if (last_max_context_length_ > 0) {
        ImGui::Text("Max context: %zu", last_max_context_length_);
    } else {
        ImGui::Text("Max context: unbounded");
    }
    ImGui::Text("Remaining generation budget: %zu", last_remaining_budget_);
    if (!sampling_settings_.empty()) {
        ImGui::TextWrapped("Sampling settings: %s", sampling_settings_.c_str());
    }

    const std::string joined = JoinTokenIds(generated_ids_);
    ImGui::Text("Generated token IDs");
    ImGui::InputTextMultiline("##GeneratedTokenIds",
                              const_cast<char*>(joined.c_str()),
                              joined.size() + 1,
                              ImVec2(-1, 120),
                              ImGuiInputTextFlags_ReadOnly);
    if (!generated_text_.empty()) {
        ImGui::Text("Decoded text");
        ImGui::InputTextMultiline("##GeneratedText",
                                  const_cast<char*>(generated_text_.c_str()),
                                  generated_text_.size() + 1,
                                  ImVec2(-1, 120),
                                  ImGuiInputTextFlags_ReadOnly);
    }

    if (!last_candidates_.empty() &&
        ImGui::BeginTable("##GenerationCandidateTable",
                          2,
                          ImGuiTableFlags_Borders |
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Token ID");
        ImGui::TableSetupColumn("Probability");
        ImGui::TableHeadersRow();
        const size_t visible_candidates =
            std::min<size_t>(last_candidates_.size(), 8);
        for (size_t i = 0; i < visible_candidates; ++i) {
            const auto& candidate = last_candidates_[i];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%lld", static_cast<long long>(candidate.token_id));
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.6f", candidate.probability);
        }
        ImGui::EndTable();
    }
}

void LanguageModelGenerationPanel::RunGeneration() {
    try {
        if (use_imported_model_) {
            if (use_text_prompt_ && !use_packaged_tokenizer_) {
                throw std::runtime_error(
                    "Imported .cyxmodel generation requires packaged tokenizer assets");
            }
            RequireImportedModelPackageContract();
        }

        std::unique_ptr<Tokenizer> tokenizer;
        std::vector<int64_t> prompt;
        size_t tokenizer_vocab_size = 0;
        size_t max_sequence_length = 0;
        if (use_text_prompt_) {
            tokenizer = BuildTokenizer();
            prompt = EncodeTextTokenIdsForGeneration(*tokenizer, text_prompt_);
            tokenizer_vocab_size = tokenizer->GetVocabulary().Size();
            max_sequence_length = static_cast<size_t>(std::max(0, tokenizer->GetMaxLength()));
        } else {
            prompt = ParsePromptIds();
        }
        if (use_imported_model_) {
            tokenizer_vocab_size = imported_model_contract_.tokenizer_vocabulary_size;
            max_sequence_length = imported_model_contract_.max_sequence_length;
        }

        const auto prompt_contract = ValidateLanguageModelPromptIds(
            prompt,
            max_sequence_length);
        if (!prompt_contract.compatible) {
            throw std::runtime_error("Invalid prompt: " + prompt_contract.error);
        }

        LanguageModelGenerationConfig config;
        config.max_new_tokens =
            static_cast<size_t>(std::max(1, max_new_tokens_));
        config.temperature = temperature_;
        config.top_k = top_k_ > 0 ? static_cast<size_t>(top_k_) : 0;
        config.top_p = top_p_;
        config.eos_token_id = static_cast<int64_t>(eos_token_id_);
        config.include_prompt = include_prompt_;
        config.max_context_tokens = max_sequence_length;
        config.sampling_mode = multinomial_sampling_
            ? LanguageModelSamplingMode::Multinomial
            : LanguageModelSamplingMode::Greedy;
        RequireValidLanguageModelGenerationConfig(config);
        if (config.max_context_tokens > 0 &&
            prompt.size() >= config.max_context_tokens) {
            throw std::runtime_error(
                "Prompt length leaves no generation budget for max context");
        }

        auto* model = ActiveModel();
        if (model == nullptr) {
            throw std::runtime_error("No trained or imported model is available");
        }

        Tensor contract_input({1, prompt.size()}, prompt.data(), DataType::Int64);
        const Tensor contract_logits = model->Forward(contract_input);
        const auto runtime_contract = ValidateLanguageModelRuntimeOutput(
            contract_logits,
            prompt.size(),
            tokenizer_vocab_size);
        if (!runtime_contract.compatible) {
            throw std::runtime_error(
                "Model output contract failed: " + runtime_contract.error);
        }

        const auto report = GenerateTokenIdsWithReport(
            *model,
            prompt,
            config,
            static_cast<uint32_t>(std::max(0, seed_)));
        generated_ids_ = report.token_ids;
        stop_reason_ = LanguageModelGenerationStopReasonName(report.stop_reason);
        last_prompt_length_ = report.prompt_length;
        last_max_context_length_ = max_sequence_length;
        last_remaining_budget_ = report.remaining_budget;
        last_candidates_ = report.steps.empty()
            ? std::vector<NextTokenCandidate>{}
            : report.steps.back().candidates;

        std::ostringstream settings;
        settings << (config.sampling_mode == LanguageModelSamplingMode::Multinomial
                         ? "multinomial"
                         : "greedy")
                 << ", max_new_tokens=" << config.max_new_tokens
                 << ", temperature=" << config.temperature
                 << ", top_k=" << config.top_k
                 << ", top_p=" << config.top_p
                 << ", eos=" << config.eos_token_id
                 << ", seed=" << std::max(0, seed_)
                 << ", include_prompt=" << (config.include_prompt ? "true" : "false");
        sampling_settings_ = settings.str();

        generated_text_.clear();
        if (tokenizer) {
            generated_text_ = DecodeGeneratedTokenIds(*tokenizer, generated_ids_);
        }
        has_result_ = true;
        status_ = "Generation completed: " +
                  std::to_string(report.new_token_ids.size()) +
                  " new token IDs.";
    } catch (const std::exception& e) {
        has_result_ = false;
        generated_ids_.clear();
        generated_text_.clear();
        stop_reason_ = "error";
        sampling_settings_.clear();
        last_prompt_length_ = 0;
        last_max_context_length_ = 0;
        last_remaining_budget_ = 0;
        last_candidates_.clear();
        status_ = std::string("Generation failed: ") + e.what();
    }
}

void LanguageModelGenerationPanel::CheckModelCompatibility() {
    try {
        auto* model = ActiveModel();
        if (model == nullptr) {
            throw std::runtime_error("No trained or imported model is available");
        }
        if (use_imported_model_) {
            if (use_text_prompt_ && !use_packaged_tokenizer_) {
                throw std::runtime_error(
                    "Imported .cyxmodel compatibility checks require packaged tokenizer assets");
            }
            RequireImportedModelPackageContract();
        }

        std::unique_ptr<Tokenizer> tokenizer;
        std::vector<int64_t> prompt;
        size_t tokenizer_vocab_size = 0;
        size_t max_sequence_length = 0;
        if (use_text_prompt_) {
            tokenizer = BuildTokenizer();
            prompt = EncodeTextTokenIdsForGeneration(*tokenizer, text_prompt_);
            tokenizer_vocab_size = tokenizer->GetVocabulary().Size();
            max_sequence_length = static_cast<size_t>(std::max(0, tokenizer->GetMaxLength()));
        } else {
            prompt = ParsePromptIds();
        }
        if (use_imported_model_) {
            tokenizer_vocab_size = imported_model_contract_.tokenizer_vocabulary_size;
            max_sequence_length = imported_model_contract_.max_sequence_length;
        }

        const auto prompt_contract = ValidateLanguageModelPromptIds(
            prompt,
            max_sequence_length);
        if (!prompt_contract.compatible) {
            throw std::runtime_error("Invalid prompt: " + prompt_contract.error);
        }

        Tensor input({1, prompt.size()}, prompt.data(), DataType::Int64);
        const Tensor logits = model->Forward(input);
        const auto runtime_contract = ValidateLanguageModelRuntimeOutput(
            logits,
            prompt.size(),
            tokenizer_vocab_size);
        if (!runtime_contract.compatible) {
            throw std::runtime_error(runtime_contract.error);
        }

        compatibility_status_ =
            "Compatible: model returned Float32 [1, " +
            std::to_string(runtime_contract.sequence_length) + ", " +
            std::to_string(runtime_contract.vocab_size) + "] logits";
        if (tokenizer_vocab_size > 0) {
            compatibility_status_ +=
                "; tokenizer vocab=" + std::to_string(tokenizer_vocab_size);
            if (use_imported_model_) {
                compatibility_status_ +=
                    ", eos=" + std::to_string(imported_model_contract_.eos_token_id);
            } else if (tokenizer) {
                compatibility_status_ +=
                    ", eos=" + std::to_string(tokenizer->GetVocabulary().EosIndex());
            }
        }
        compatibility_status_ += ".";
    } catch (const std::exception& e) {
        compatibility_status_ =
            std::string("Not compatible for generation: ") + e.what();
    }
}

void LanguageModelGenerationPanel::RequireImportedModelPackageContract() const {
    if (!use_imported_model_) {
        return;
    }
    if (imported_model_contract_.package_path.empty()) {
        throw std::runtime_error(
            "Imported model package contract is not available");
    }
    if (!imported_model_contract_.compatible) {
        throw std::runtime_error(
            "Package contract failed: " + imported_model_contract_.error);
    }
}

void LanguageModelGenerationPanel::LoadTokenizerFromCyxModel() {
    try {
        if (cyxmodel_path_[0] == '\0') {
            throw std::invalid_argument("Choose a .cyxmodel package first");
        }

        formats::CyxModelFormat format;
        std::string config_json;
        std::string vocab_text;
        if (!format.ExtractTextTokenizerAssets(cyxmodel_path_,
                                               config_json,
                                               vocab_text)) {
            throw std::runtime_error(
                "No tokenizer assets found in package: " +
                format.GetLastError());
        }

        TextTokenizerPackage package;
        std::string error;
        if (!LoadTextTokenizerPackage(config_json, vocab_text, package, error)) {
            throw std::runtime_error(error);
        }
        if (!package.has_vocabulary) {
            throw std::runtime_error(
                "Package tokenizer assets do not include a vocabulary");
        }

        if (package.tokenizer) {
            eos_token_id_ = package.tokenizer->GetVocabulary().EosIndex();
            max_length_ = package.tokenizer->GetMaxLength();
            lowercase_ = package.tokenizer->GetLowercase();
            switch (package.tokenizer->GetType()) {
                case TokenizerType::Whitespace: tokenizer_type_idx_ = 0; break;
                case TokenizerType::Word: tokenizer_type_idx_ = 1; break;
                case TokenizerType::Character: tokenizer_type_idx_ = 2; break;
            }
            packaged_tokenizer_summary_ =
                "packaged tokenizer: vocab=" +
                std::to_string(package.tokenizer->GetVocabulary().Size()) +
                ", max_len=" + std::to_string(max_length_) +
                ", eos=" + std::to_string(eos_token_id_);
        }

        packaged_tokenizer_config_json_ = std::move(config_json);
        packaged_tokenizer_vocab_text_ = std::move(vocab_text);
        use_packaged_tokenizer_ = true;
        status_ = "Loaded packaged tokenizer assets from .cyxmodel.";
    } catch (const std::exception& e) {
        use_packaged_tokenizer_ = false;
        packaged_tokenizer_config_json_.clear();
        packaged_tokenizer_vocab_text_.clear();
        packaged_tokenizer_summary_.clear();
        status_ = std::string("Failed to load packaged tokenizer: ") + e.what();
    }
}

void LanguageModelGenerationPanel::LoadModelAndTokenizerFromCyxModel() {
    try {
        if (cyxmodel_path_[0] == '\0') {
            throw std::invalid_argument("Choose a .cyxmodel package first");
        }

        auto model = std::make_unique<SequentialModel>();
        ModelImporter importer;
        ImportOptions options;
        const auto result = importer.Import(cyxmodel_path_, *model, options);
        if (!result.success) {
            throw std::runtime_error(
                result.error_message.empty()
                    ? importer.GetLastError()
                    : result.error_message);
        }

        LoadTokenizerFromCyxModel();
        if (!use_packaged_tokenizer_) {
            throw std::runtime_error(status_);
        }

        TextTokenizerPackage contract_tokenizer_package;
        std::string contract_tokenizer_error;
        if (!LoadTextTokenizerPackage(packaged_tokenizer_config_json_,
                                      packaged_tokenizer_vocab_text_,
                                      contract_tokenizer_package,
                                      contract_tokenizer_error)) {
            throw std::runtime_error(contract_tokenizer_error);
        }

        imported_model_ = std::move(model);
        imported_model_source_ = cyxmodel_path_;
        imported_model_summary_.clear();
        const auto probe = importer.ProbeFile(cyxmodel_path_);
        const auto package_contract = ValidateLanguageModelPackageContract(
            probe,
            &contract_tokenizer_package,
            cyxmodel_path_);
        imported_model_contract_ = package_contract;
        imported_model_summary_ =
            "package: family=" +
            (package_contract.model_family.empty() ? std::string("unspecified")
                                                   : package_contract.model_family) +
            ", generation=" +
            (package_contract.supports_generation ? std::string("yes")
                                                  : std::string("no")) +
            ", contract=" +
            (package_contract.generation_output_contract.empty()
                 ? std::string("unspecified")
                 : package_contract.generation_output_contract) +
            ", tokenizer_vocab=" +
            std::to_string(package_contract.tokenizer_vocabulary_size) +
            ", max_len=" + std::to_string(package_contract.max_sequence_length) +
            ", eos=" + std::to_string(package_contract.eos_token_id);
        compatibility_status_ = package_contract.compatible
            ? "Package contract is compatible; use Check compatibility to "
              "validate the active runtime graph."
            : "Package contract failed: " + package_contract.error;
        use_imported_model_ = true;
        status_ = package_contract.compatible
            ? "Loaded model and tokenizer assets from .cyxmodel."
            : "Loaded model package, but generation contract failed: " +
                  package_contract.error;
    } catch (const std::exception& e) {
        imported_model_.reset();
        imported_model_source_.clear();
        imported_model_summary_.clear();
        imported_model_contract_ = {};
        use_imported_model_ = false;
        status_ = std::string("Failed to load model package: ") + e.what();
    }
}

SequentialModel* LanguageModelGenerationPanel::ActiveModel() const {
    if (use_imported_model_ && imported_model_) {
        return imported_model_.get();
    }
    return TrainingManager::Instance().GetLastTrainedModel();
}

std::vector<int64_t> LanguageModelGenerationPanel::ParsePromptIds() const {
    std::string normalized(prompt_ids_);
    for (char& c : normalized) {
        if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
            c = ' ';
        }
    }

    std::istringstream in(normalized);
    std::vector<int64_t> ids;
    int64_t id = 0;
    while (in >> id) {
        if (id < 0) {
            throw std::invalid_argument("Prompt token IDs must be non-negative");
        }
        ids.push_back(id);
    }
    if (ids.empty()) {
        throw std::invalid_argument("Prompt token IDs are required");
    }
    return ids;
}

std::vector<int64_t> LanguageModelGenerationPanel::CurrentPromptIdsForProbe() const {
    if (use_text_prompt_) {
        auto tokenizer = BuildTokenizer();
        return EncodeTextTokenIdsForGeneration(*tokenizer, text_prompt_);
    }
    return ParsePromptIds();
}

std::unique_ptr<Tokenizer> LanguageModelGenerationPanel::BuildTokenizer() const {
    if (use_packaged_tokenizer_) {
        if (packaged_tokenizer_config_json_.empty() ||
            packaged_tokenizer_vocab_text_.empty()) {
            throw std::invalid_argument(
                "Packaged tokenizer mode is enabled but no package assets are loaded");
        }
        TextTokenizerPackage package;
        std::string error;
        if (!LoadTextTokenizerPackage(packaged_tokenizer_config_json_,
                                      packaged_tokenizer_vocab_text_,
                                      package,
                                      error)) {
            throw std::runtime_error(error);
        }
        if (!package.has_vocabulary || !package.tokenizer) {
            throw std::runtime_error(
                "Packaged tokenizer does not contain a usable vocabulary");
        }
        return std::move(package.tokenizer);
    }

    TokenizerType type = TokenizerType::Word;
    if (tokenizer_type_idx_ == 0) {
        type = TokenizerType::Whitespace;
    } else if (tokenizer_type_idx_ == 2) {
        type = TokenizerType::Character;
    }

    auto tokenizer = std::make_unique<Tokenizer>(type);
    tokenizer->SetLowercase(lowercase_);
    tokenizer->SetMaxLength(std::max(1, max_length_));
    tokenizer->SetPadding(true);
    tokenizer->SetTruncation(true);
    tokenizer->SetAddBos(add_bos_);
    tokenizer->SetAddEos(add_eos_);

    if (vocab_file_[0] == '\0') {
        throw std::invalid_argument("Text mode requires a vocabulary file");
    }
    if (!tokenizer->GetVocabulary().LoadFromFile(vocab_file_)) {
        throw std::runtime_error(
            "Failed to load vocabulary file: " + std::string(vocab_file_));
    }
    return tokenizer;
}

std::string LanguageModelGenerationPanel::JoinTokenIds(
    const std::vector<int64_t>& ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            out << ' ';
        }
        out << ids[i];
    }
    return out.str();
}

} // namespace cyxwiz
