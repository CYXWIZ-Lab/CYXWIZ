#include "nas_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <cmath>

namespace cyxwiz {

NASPanel::NASPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));

    // Default search config
    search_config_.population_size = 20;
    search_config_.generations = 10;
    search_config_.elite_count = 2;
    search_config_.mutation_rate = 0.3;
    search_config_.crossover_rate = 0.5;
}

NASPanel::~NASPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        is_searching_ = false;
        compute_thread_->join();
    }
}

void NASPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 800), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_NETWORK_WIRED " Neural Architecture Search###NAS", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load() || is_searching_.load()) {
            if (is_searching_.load()) {
                RenderSearchProgress();
            } else {
                ImGui::Text("%s Computing...", ICON_FA_SPINNER);
            }
        } else {
            RenderInputConfig();
            ImGui::Spacing();

            if (ImGui::CollapsingHeader(ICON_FA_CHART_PIE " Current Architecture Score", ImGuiTreeNodeFlags_DefaultOpen)) {
                RenderCurrentScore();
            }

            ImGui::Spacing();

            if (ImGui::CollapsingHeader(ICON_FA_WAND_MAGIC_SPARKLES " Mutation Controls")) {
                RenderMutationControls();
            }

            ImGui::Spacing();

            if (ImGui::CollapsingHeader(ICON_FA_LIGHTBULB " Architecture Suggestions")) {
                RenderSuggestions();
            }

            ImGui::Spacing();

            if (ImGui::CollapsingHeader(ICON_FA_DNA " Evolutionary Search")) {
                RenderEvolutionarySearch();
            }

            if (has_search_result_) {
                ImGui::Spacing();
                RenderSearchResults();
            }
        }
    }
    ImGui::End();
}

void NASPanel::RenderToolbar() {
    if (!has_score_ && !has_search_result_) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportNAS");
    }

    if (!has_score_ && !has_search_result_) ImGui::EndDisabled();

    RenderExportOptions();
}

void NASPanel::RenderInputConfig() {
    ImGui::Text("%s Input Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    // Input type
    const char* input_types[] = {"Tabular Data", "Image Data"};
    ImGui::Text("Input Type:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##InputType", &input_type_, input_types, IM_ARRAYSIZE(input_types))) {
        has_score_ = false;
    }

    if (input_type_ == 0) {
        // Tabular
        ImGui::Text("Features:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##Features", &input_features_)) {
            input_features_ = std::max(1, input_features_);
            has_score_ = false;
        }
    } else {
        // Image
        ImGui::Text("Channels:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (ImGui::InputInt("##Channels", &input_channels_)) {
            input_channels_ = std::max(1, input_channels_);
            has_score_ = false;
        }

        ImGui::SameLine();
        ImGui::Text("Height:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (ImGui::InputInt("##Height", &input_height_)) {
            input_height_ = std::max(1, input_height_);
            has_score_ = false;
        }

        ImGui::SameLine();
        ImGui::Text("Width:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (ImGui::InputInt("##Width", &input_width_)) {
            input_width_ = std::max(1, input_width_);
            has_score_ = false;
        }
    }

    // Output size
    ImGui::Text("Output Size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##OutputSize", &output_size_)) {
        output_size_ = std::max(1, output_size_);
        has_score_ = false;
    }

    // Task type
    const char* task_types[] = {"Classification", "Regression", "Image Classification"};
    ImGui::Text("Task Type:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##TaskType", &task_type_, task_types, IM_ARRAYSIZE(task_types))) {
        has_score_ = false;
    }
}

void NASPanel::RenderCurrentScore() {
    // Get current architecture button
    if (ImGui::Button(ICON_FA_DOWNLOAD " Load from Node Editor")) {
        if (get_architecture_callback_) {
            auto [nodes, links] = get_architecture_callback_();
            current_nodes_ = nodes;
            current_links_ = links;
            ScoreCurrentArchitecture();
        } else {
            ImGui::TextDisabled("Node editor callback not set");
        }
    }

    if (!has_score_) {
        ImGui::TextDisabled("Click above to score the current architecture");
        return;
    }

    ImGui::Spacing();

    // Score display
    if (ImGui::BeginTable("ScoreTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 200);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto AddScoreRow = [](const char* name, double value, double max_good = 1.0) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();

            float normalized = static_cast<float>(value / max_good);
            ImVec4 color;
            if (normalized >= 0.7f) color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            else if (normalized >= 0.4f) color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
            else color = ImVec4(0.8f, 0.4f, 0.2f, 1.0f);

            ImGui::TextColored(color, "%.3f", value);
            ImGui::SameLine();
            ImGui::ProgressBar(static_cast<float>(value), ImVec2(100, 0));
        };

        AddScoreRow("Overall Score", current_score_.overall_score);
        AddScoreRow("Complexity Score", current_score_.complexity_score);
        AddScoreRow("Efficiency Score", current_score_.efficiency_score);
        AddScoreRow("Depth Score", current_score_.depth_score);
        AddScoreRow("Diversity Score", current_score_.diversity_score);
        AddScoreRow("Connectivity Score", current_score_.connectivity_score);

        ImGui::EndTable();
    }

    // Raw metrics
    ImGui::Spacing();
    ImGui::Text("Raw Metrics:");

    char buf[64];
    snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(current_score_.total_params));
    ImGui::BulletText("Total Parameters: %s", buf);

    snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(current_score_.trainable_params));
    ImGui::BulletText("Trainable Parameters: %s", buf);

    snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(current_score_.total_flops));
    ImGui::BulletText("Estimated FLOPs: %s", buf);

    ImGui::BulletText("Layer Count: %d", current_score_.layer_count);
    ImGui::BulletText("Trainable Layers: %d", current_score_.trainable_layer_count);

    // Architecture summary
    if (!current_score_.architecture_summary.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("Summary: %s", current_score_.architecture_summary.c_str());
    }
}

void NASPanel::RenderMutationControls() {
    if (current_nodes_.empty()) {
        ImGui::TextDisabled("Load an architecture first to apply mutations");
        return;
    }

    ImGui::Text("Apply a mutation to the current architecture:");
    ImGui::Spacing();

    // Mutation buttons in a grid
    if (ImGui::Button(ICON_FA_PLUS " Add Layer", ImVec2(140, 0))) {
        MutateArchitecture(MutationType::AddLayer);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MINUS " Remove Layer", ImVec2(140, 0))) {
        MutateArchitecture(MutationType::RemoveLayer);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Swap Layer", ImVec2(140, 0))) {
        MutateArchitecture(MutationType::SwapLayer);
    }

    if (ImGui::Button(ICON_FA_EXPAND " Change Units", ImVec2(140, 0))) {
        MutateArchitecture(MutationType::ChangeUnits);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_BOLT " Change Activation", ImVec2(140, 0))) {
        MutateArchitecture(MutationType::ChangeActivation);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_SHUFFLE " Random", ImVec2(140, 0))) {
        MutateArchitecture(MutationType::Random);
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Mutations are applied to the current architecture in memory.");
    ImGui::TextDisabled("Use 'Apply to Node Editor' to see the changes.");

    if (!current_nodes_.empty() && apply_architecture_callback_) {
        ImGui::Spacing();
        if (ImGui::Button(ICON_FA_UPLOAD " Apply to Node Editor", ImVec2(200, 0))) {
            apply_architecture_callback_(current_nodes_, current_links_);
        }
    }
}

void NASPanel::RenderSuggestions() {
    // Task type for suggestions
    const char* task_names[] = {"classification", "regression", "image_classification"};

    ImGui::Text("Generate architecture suggestions for your task:");
    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Generate Suggestions", ImVec2(200, 0))) {
        GenerateSuggestions();
    }

    if (suggestions_.empty()) {
        ImGui::TextDisabled("Click above to generate suggestions");
        return;
    }

    ImGui::Spacing();

    // List suggestions
    for (size_t i = 0; i < suggestions_.size(); i++) {
        char label[64];
        snprintf(label, sizeof(label), "Architecture %zu (Score: %.3f)", i + 1,
                suggestion_scores_[i].overall_score);

        bool is_selected = (selected_suggestion_ == static_cast<int>(i));

        if (ImGui::Selectable(label, is_selected)) {
            selected_suggestion_ = static_cast<int>(i);
        }

        if (is_selected) {
            // Show details
            ImGui::Indent();
            ImGui::Text("Layers: %d, Params: %lld",
                       suggestion_scores_[i].layer_count,
                       static_cast<long long>(suggestion_scores_[i].total_params));
            ImGui::TextWrapped("%s", suggestion_scores_[i].architecture_summary.c_str());
            ImGui::Unindent();
        }
    }

    if (selected_suggestion_ >= 0 && apply_architecture_callback_) {
        ImGui::Spacing();
        if (ImGui::Button(ICON_FA_CHECK " Apply Selected", ImVec2(150, 0))) {
            apply_architecture_callback_(
                suggestions_[selected_suggestion_].first,
                suggestions_[selected_suggestion_].second);
        }
    }
}

void NASPanel::RenderEvolutionarySearch() {
    ImGui::Text("Evolutionary Architecture Search");
    ImGui::TextDisabled("Automatically evolve architectures using genetic algorithms");
    ImGui::Spacing();

    // Config toggle
    if (ImGui::Button(show_search_config_ ? "Hide Config" : "Show Config")) {
        show_search_config_ = !show_search_config_;
    }

    if (show_search_config_) {
        ImGui::Indent();

        ImGui::Text("Population Size:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##PopSize", &search_config_.population_size);

        ImGui::Text("Generations:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##Generations", &search_config_.generations);

        ImGui::Text("Elite Count:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##EliteCount", &search_config_.elite_count);

        ImGui::Text("Mutation Rate:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        float mut_rate = static_cast<float>(search_config_.mutation_rate);
        if (ImGui::SliderFloat("##MutRate", &mut_rate, 0.1f, 0.9f)) {
            search_config_.mutation_rate = mut_rate;
        }

        ImGui::Text("Crossover Rate:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        float cross_rate = static_cast<float>(search_config_.crossover_rate);
        if (ImGui::SliderFloat("##CrossRate", &cross_rate, 0.1f, 0.9f)) {
            search_config_.crossover_rate = cross_rate;
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Start search button
    bool can_search = !current_nodes_.empty();
    if (!can_search) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Start Evolutionary Search", ImVec2(220, 0))) {
        RunEvolutionarySearch();
    }

    if (!can_search) ImGui::EndDisabled();

    if (!can_search) {
        ImGui::SameLine();
        ImGui::TextDisabled("Load an architecture first");
    }
}

void NASPanel::RenderSearchProgress() {
    ImGui::Text("%s Evolutionary Search in Progress...", ICON_FA_DNA);
    ImGui::Spacing();

    int gen = search_generation_.load();
    double best = search_best_score_.load();

    ImGui::Text("Generation: %d / %d", gen, search_config_.generations);
    ImGui::Text("Best Score: %.4f", best);

    float progress = static_cast<float>(gen) / search_config_.generations;
    ImGui::ProgressBar(progress, ImVec2(-1, 0));

    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_STOP " Stop Search")) {
        is_searching_ = false;
    }
}

void NASPanel::RenderSearchResults() {
    if (!search_result_.success) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "Search failed: %s",
                          search_result_.error_message.c_str());
        return;
    }

    ImGui::Text(ICON_FA_TROPHY " Search Results");
    ImGui::Separator();

    ImGui::Text("Generations: %d", search_result_.total_generations);
    ImGui::Text("Evaluations: %d", search_result_.total_evaluations);
    ImGui::Text("Best Score: %.4f", search_result_.best_score.overall_score);

    ImGui::Spacing();

    // Generation progress chart
    if (!search_result_.generation_best.empty()) {
        std::vector<double> gen_indices(search_result_.generation_best.size());
        std::vector<double> gen_scores(search_result_.generation_best.size());

        for (size_t i = 0; i < search_result_.generation_best.size(); i++) {
            gen_indices[i] = static_cast<double>(i + 1);
            gen_scores[i] = search_result_.generation_best[i].overall_score;
        }

        if (ImPlot::BeginPlot("Score Evolution", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Generation", "Best Score");
            ImPlot::PlotLine("Best", gen_indices.data(), gen_scores.data(),
                            static_cast<int>(gen_indices.size()));
            ImPlot::EndPlot();
        }
    }

    // Best architecture details
    ImGui::Text("Best Architecture:");
    ImGui::BulletText("Layers: %d", search_result_.best_score.layer_count);
    ImGui::BulletText("Parameters: %lld",
                     static_cast<long long>(search_result_.best_score.total_params));
    ImGui::TextWrapped("Summary: %s", search_result_.best_score.architecture_summary.c_str());

    // Apply best
    if (apply_architecture_callback_) {
        ImGui::Spacing();
        if (ImGui::Button(ICON_FA_CROWN " Apply Best Architecture", ImVec2(220, 0))) {
            apply_architecture_callback_(
                search_result_.best_architecture,
                search_result_.best_links);
        }
    }
}

void NASPanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportNAS")) {
        ImGui::Text("Export NAS Results");
        ImGui::Separator();

        ImGui::InputText("File Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Save CSV")) {
            std::lock_guard<std::mutex> lock(result_mutex_);

            std::ofstream file(export_path_);
            if (file) {
                file << "Metric,Value\n";

                if (has_score_) {
                    file << "Overall Score," << current_score_.overall_score << "\n";
                    file << "Complexity Score," << current_score_.complexity_score << "\n";
                    file << "Efficiency Score," << current_score_.efficiency_score << "\n";
                    file << "Depth Score," << current_score_.depth_score << "\n";
                    file << "Total Params," << current_score_.total_params << "\n";
                    file << "Total FLOPs," << current_score_.total_flops << "\n";
                    file << "Layer Count," << current_score_.layer_count << "\n";
                }

                if (has_search_result_) {
                    file << "\nEvolutionary Search Results\n";
                    file << "Generations," << search_result_.total_generations << "\n";
                    file << "Evaluations," << search_result_.total_evaluations << "\n";
                    file << "Best Score," << search_result_.best_score.overall_score << "\n";
                }

                spdlog::info("Exported NAS results to: {}", export_path_);
            }

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void NASPanel::SetGetArchitectureCallback(
    std::function<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>()> cb) {
    get_architecture_callback_ = cb;
}

void NASPanel::SetApplyArchitectureCallback(
    std::function<void(const std::vector<gui::MLNode>&, const std::vector<gui::NodeLink>&)> cb) {
    apply_architecture_callback_ = cb;
}

void NASPanel::ScoreCurrentArchitecture() {
    if (current_nodes_.empty()) return;

    std::vector<size_t> input_shape;
    if (input_type_ == 0) {
        input_shape = {1, static_cast<size_t>(input_features_)};
    } else {
        input_shape = {1, static_cast<size_t>(input_channels_),
                       static_cast<size_t>(input_height_),
                       static_cast<size_t>(input_width_)};
    }

    current_score_ = NASEvaluator::ScoreArchitecture(
        current_nodes_, current_links_, input_shape, search_config_);

    has_score_ = current_score_.success;

    if (has_score_) {
        spdlog::info("Architecture scored: overall = {:.3f}", current_score_.overall_score);
    }
}

void NASPanel::MutateArchitecture(MutationType type) {
    if (current_nodes_.empty()) return;

    auto [new_nodes, new_links] = NASEvaluator::MutateArchitecture(
        current_nodes_, current_links_, type);

    current_nodes_ = std::move(new_nodes);
    current_links_ = std::move(new_links);

    // Re-score
    ScoreCurrentArchitecture();

    spdlog::info("Applied mutation, new score: {:.3f}", current_score_.overall_score);
}

void NASPanel::GenerateSuggestions() {
    const char* task_names[] = {"classification", "regression", "image_classification"};
    std::string task = task_names[task_type_];

    std::vector<size_t> input_shape;
    if (input_type_ == 0) {
        input_shape = {1, static_cast<size_t>(input_features_)};
    } else {
        input_shape = {1, static_cast<size_t>(input_channels_),
                       static_cast<size_t>(input_height_),
                       static_cast<size_t>(input_width_)};
    }

    suggestions_ = NASEvaluator::SuggestArchitectures(task, input_shape, output_size_, 5);

    // Score each suggestion
    suggestion_scores_.clear();
    for (const auto& [nodes, links] : suggestions_) {
        auto score = NASEvaluator::ScoreArchitecture(nodes, links, input_shape, search_config_);
        suggestion_scores_.push_back(score);
    }

    selected_suggestion_ = -1;

    spdlog::info("Generated {} architecture suggestions", suggestions_.size());
}

void NASPanel::RunEvolutionarySearch() {
    if (is_searching_.load() || current_nodes_.empty()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_searching_ = true;
    search_generation_ = 0;
    search_best_score_ = 0.0;

    auto initial_nodes = current_nodes_;
    auto initial_links = current_links_;
    auto config = search_config_;

    std::vector<size_t> input_shape;
    if (input_type_ == 0) {
        input_shape = {1, static_cast<size_t>(input_features_)};
    } else {
        input_shape = {1, static_cast<size_t>(input_channels_),
                       static_cast<size_t>(input_height_),
                       static_cast<size_t>(input_width_)};
    }

    compute_thread_ = std::make_unique<std::thread>([this, initial_nodes, initial_links, input_shape, config]() {
        try {
            auto callback = [this](int generation, const ArchitectureScore& best) {
                search_generation_ = generation;
                search_best_score_ = best.overall_score;

                // Check if search should stop
                return is_searching_.load();
            };

            // Note: The callback signature doesn't match - we'd need to modify this
            // For now, use a simpler approach
            auto result = NASEvaluator::EvolveArchitecture(
                initial_nodes, initial_links, input_shape, config,
                [this](int gen, const ArchitectureScore& best) {
                    search_generation_ = gen;
                    search_best_score_ = best.overall_score;
                });

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                search_result_ = std::move(result);
                has_search_result_ = search_result_.success;
            }

            if (has_search_result_) {
                spdlog::info("Evolutionary search complete. Best score: {:.4f}",
                            search_result_.best_score.overall_score);
            }

        } catch (const std::exception& e) {
            spdlog::error("Evolutionary search error: {}", e.what());
        }

        is_searching_ = false;
    });
}

} // namespace cyxwiz
