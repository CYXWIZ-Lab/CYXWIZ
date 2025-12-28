#include "hyperparam_search_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

HyperparamSearchPanel::HyperparamSearchPanel()
    : Panel("Hyperparameter Search", false)
{
    search_engine_ = std::make_unique<HyperparamSearch>();

    // Set up callbacks
    search_engine_->SetOnTrialStart([this](int trial_id, const std::map<std::string, float>& params) {
        current_trial_ = trial_id + 1;
        spdlog::info("HPO: Trial {} started", trial_id + 1);
    });

    search_engine_->SetOnTrialProgress([this](int trial_id, int epoch, float loss, float acc) {
        // Could update real-time plots here
    });

    search_engine_->SetOnTrialComplete([this](const TrialResult& result) {
        cached_results_ = search_engine_->GetResults();

        if (result.completed) {
            if (result.val_loss < best_val_loss_) {
                best_trial_id_ = result.trial_id;
                best_val_loss_ = result.val_loss;
                best_val_accuracy_ = result.val_accuracy;
            }
        }
    });

    search_engine_->SetOnSearchComplete([this](const std::vector<TrialResult>& results) {
        is_running_ = false;
        cached_results_ = results;
        spdlog::info("HPO: Search complete with {} trials", results.size());
    });

    // Default search space with common hyperparameters
    config_.space.AddLogScale("learning_rate", 1e-5f, 1e-1f);
    config_.space.AddDiscrete("batch_size", 16, 128, 16);
    config_.space.AddCategorical("optimizer", {"SGD", "Adam", "AdamW"});
}

void HyperparamSearchPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_MAGNIFYING_GLASS_CHART " Hyperparameter Search###HyperparamSearch",
                     &visible_, ImGuiWindowFlags_MenuBar)) {

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Search")) {
                if (ImGui::MenuItem(ICON_FA_PLAY " Start", nullptr, false, !is_running_)) {
                    StartSearch();
                }
                if (ImGui::MenuItem(ICON_FA_STOP " Stop", nullptr, false, is_running_)) {
                    StopSearch();
                }
                ImGui::Separator();
                if (ImGui::MenuItem(ICON_FA_TRASH " Clear Results", nullptr, false, !is_running_)) {
                    cached_results_.clear();
                    best_trial_id_ = -1;
                    best_val_loss_ = std::numeric_limits<float>::max();
                    best_val_accuracy_ = 0.0f;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Update progress from search engine
        if (is_running_) {
            current_trial_ = search_engine_->GetCurrentTrial() + 1;
            total_trials_ = search_engine_->GetTotalTrials();
            current_progress_ = search_engine_->GetProgress();
            elapsed_seconds_ = search_engine_->GetElapsedSeconds();
        }

        // Tabs for different sections
        if (ImGui::BeginTabBar("HPOTabs")) {
            if (ImGui::BeginTabItem(ICON_FA_SLIDERS " Configuration")) {
                RenderSearchConfig();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem(ICON_FA_LIST " Parameters")) {
                RenderParameterSpace();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Progress")) {
                RenderProgress();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem(ICON_FA_TABLE " Results")) {
                RenderResults();
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    // Add parameter dialog
    if (show_add_param_dialog_) {
        RenderAddParameterDialog();
    }
}

void HyperparamSearchPanel::RenderSearchConfig() {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text(ICON_FA_GEAR " Search Configuration");
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    // Strategy selection
    ImGui::Text("Search Strategy:");
    int strategy = static_cast<int>(config_.strategy);
    if (ImGui::RadioButton("Grid Search", &strategy, 0)) {
        config_.strategy = SearchStrategy::Grid;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Random Search", &strategy, 1)) {
        config_.strategy = SearchStrategy::Random;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Bayesian (Beta)", &strategy, 2)) {
        config_.strategy = SearchStrategy::Bayesian;
    }

    ImGui::Spacing();

    // Trial settings
    ImGui::Text("Trial Settings:");
    ImGui::InputInt("Max Trials", &config_.max_trials);
    config_.max_trials = std::max(1, config_.max_trials);

    ImGui::InputInt("Epochs per Trial", &config_.epochs_per_trial);
    config_.epochs_per_trial = std::max(1, config_.epochs_per_trial);

    ImGui::InputInt("Early Stopping Patience", &config_.early_stopping_patience);
    config_.early_stopping_patience = std::max(0, config_.early_stopping_patience);

    ImGui::Spacing();

    // Objective
    ImGui::Text("Optimization Objective:");
    const char* objectives[] = {"Validation Loss", "Validation Accuracy"};
    int obj_idx = (config_.objective == "val_loss") ? 0 : 1;
    if (ImGui::Combo("Objective", &obj_idx, objectives, 2)) {
        config_.objective = (obj_idx == 0) ? "val_loss" : "val_accuracy";
        config_.minimize = (obj_idx == 0);
    }

    ImGui::Spacing();

    // Resource limits
    ImGui::Text("Resource Limits:");
    float max_hours = config_.max_time_seconds / 3600.0f;
    if (ImGui::SliderFloat("Max Time (hours)", &max_hours, 0.1f, 24.0f, "%.1f")) {
        config_.max_time_seconds = max_hours * 3600.0f;
    }

    ImGui::Spacing();

    // Dataset settings
    ImGui::Text("Dataset Settings:");
    ImGui::SliderFloat("Validation Split", &config_.validation_split, 0.1f, 0.5f, "%.0f%%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::InputInt("Default Batch Size", &config_.batch_size);
    config_.batch_size = std::max(1, config_.batch_size);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Start/Stop buttons
    if (!is_running_) {
        if (ImGui::Button(ICON_FA_PLAY " Start Search", ImVec2(150, 35))) {
            StartSearch();
        }
    } else {
        if (ImGui::Button(ICON_FA_STOP " Stop Search", ImVec2(150, 35))) {
            StopSearch();
        }
    }

    // Show estimated trials for grid search
    if (config_.strategy == SearchStrategy::Grid && !config_.space.parameters.empty()) {
        ImGui::SameLine();
        size_t total = config_.space.GetTotalCombinations();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "Grid search will run %zu trials", total);
    }
}

void HyperparamSearchPanel::RenderParameterSpace() {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text(ICON_FA_SLIDERS " Search Space");
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    // Add parameter button
    if (ImGui::Button(ICON_FA_PLUS " Add Parameter")) {
        show_add_param_dialog_ = true;
        memset(param_name_, 0, sizeof(param_name_));
        param_type_ = 0;
        param_min_ = 0.0f;
        param_max_ = 1.0f;
        param_step_ = 1;
        memset(param_options_, 0, sizeof(param_options_));
    }

    ImGui::Spacing();

    // Parameter list
    if (config_.space.parameters.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "No parameters defined. Click 'Add Parameter' to start.");
    } else {
        if (ImGui::BeginTable("ParameterTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < config_.space.parameters.size(); ++i) {
                const auto& param = config_.space.parameters[i];
                ImGui::TableNextRow();

                // Name
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", param.name.c_str());

                // Type
                ImGui::TableSetColumnIndex(1);
                switch (param.type) {
                    case ParamType::Continuous: ImGui::Text("Continuous"); break;
                    case ParamType::Discrete: ImGui::Text("Discrete"); break;
                    case ParamType::LogScale: ImGui::Text("Log Scale"); break;
                    case ParamType::Categorical: ImGui::Text("Categorical"); break;
                }

                // Min/Max or Options
                ImGui::TableSetColumnIndex(2);
                if (param.type == ParamType::Categorical) {
                    ImGui::Text("-");
                } else {
                    ImGui::Text("%.4g", param.min_value);
                }

                ImGui::TableSetColumnIndex(3);
                if (param.type == ParamType::Categorical) {
                    ImGui::Text("%zu opts", param.options.size());
                } else {
                    ImGui::Text("%.4g", param.max_value);
                }

                // Remove button
                ImGui::TableSetColumnIndex(4);
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::SmallButton(ICON_FA_TRASH)) {
                    RemoveParameter(i);
                }
                ImGui::PopID();
            }

            ImGui::EndTable();
        }
    }
}

void HyperparamSearchPanel::RenderProgress() {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text(ICON_FA_CHART_LINE " Search Progress");
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    // Status
    if (is_running_) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_CIRCLE " Running");
    } else if (search_engine_->IsComplete()) {
        ImGui::TextColored(ImVec4(0.3f, 0.5f, 0.8f, 1.0f), ICON_FA_CHECK " Complete");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_CIRCLE " Idle");
    }

    ImGui::Spacing();

    // Progress bar
    ImGui::Text("Trial Progress:");
    ImGui::ProgressBar(current_progress_, ImVec2(-1, 0),
        (total_trials_ > 0) ?
            (std::to_string(current_trial_) + " / " + std::to_string(total_trials_)).c_str() :
            "0 / 0");

    // Time elapsed
    int hours = static_cast<int>(elapsed_seconds_) / 3600;
    int mins = (static_cast<int>(elapsed_seconds_) % 3600) / 60;
    int secs = static_cast<int>(elapsed_seconds_) % 60;
    ImGui::Text("Elapsed Time: %02d:%02d:%02d", hours, mins, secs);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Best result so far
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text(ICON_FA_TROPHY " Best Result");
    ImGui::PopFont();

    if (best_trial_id_ >= 0) {
        ImGui::Text("Trial #%d", best_trial_id_ + 1);
        ImGui::Text("Validation Loss: %.4f", best_val_loss_);
        ImGui::Text("Validation Accuracy: %.2f%%", best_val_accuracy_ * 100.0f);

        // Show best parameters
        TrialResult best = search_engine_->GetBestResult();
        if (!best.parameters.empty() || !best.categorical_params.empty()) {
            ImGui::Spacing();
            ImGui::Text("Best Parameters:");
            for (const auto& [name, value] : best.parameters) {
                ImGui::BulletText("%s: %.6g", name.c_str(), value);
            }
            for (const auto& [name, value] : best.categorical_params) {
                ImGui::BulletText("%s: %s", name.c_str(), value.c_str());
            }
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No results yet");
    }

    ImGui::Spacing();

    // Loss over trials plot
    if (!cached_results_.empty() && ImPlot::BeginPlot("Loss Over Trials", ImVec2(-1, 200))) {
        std::vector<float> trial_ids;
        std::vector<float> val_losses;
        std::vector<float> best_losses;

        float running_best = std::numeric_limits<float>::max();
        for (const auto& result : cached_results_) {
            if (result.completed) {
                trial_ids.push_back(static_cast<float>(result.trial_id + 1));
                val_losses.push_back(result.val_loss);
                running_best = std::min(running_best, result.val_loss);
                best_losses.push_back(running_best);
            }
        }

        if (!trial_ids.empty()) {
            ImPlot::SetupAxes("Trial", "Loss");
            ImPlot::PlotScatter("Validation Loss", trial_ids.data(), val_losses.data(), static_cast<int>(trial_ids.size()));
            ImPlot::PlotLine("Best So Far", trial_ids.data(), best_losses.data(), static_cast<int>(trial_ids.size()));
        }

        ImPlot::EndPlot();
    }
}

void HyperparamSearchPanel::RenderResults() {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text(ICON_FA_TABLE " Trial Results");
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    if (cached_results_.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "No results yet. Start a search to see results here.");
        return;
    }

    // Sort results if needed
    std::vector<TrialResult> sorted_results = cached_results_;
    if (sort_column_ >= 0) {
        std::sort(sorted_results.begin(), sorted_results.end(),
            [this](const TrialResult& a, const TrialResult& b) {
                float val_a, val_b;
                switch (sort_column_) {
                    case 0: val_a = static_cast<float>(a.trial_id); val_b = static_cast<float>(b.trial_id); break;
                    case 1: val_a = a.val_loss; val_b = b.val_loss; break;
                    case 2: val_a = a.val_accuracy; val_b = b.val_accuracy; break;
                    case 3: val_a = a.train_loss; val_b = b.train_loss; break;
                    case 4: val_a = a.duration_seconds; val_b = b.duration_seconds; break;
                    default: return false;
                }
                return sort_ascending_ ? (val_a < val_b) : (val_a > val_b);
            });
    }

    // Results table
    if (ImGui::BeginTable("ResultsTable", 6,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Sortable |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
        ImVec2(0, 300))) {

        ImGui::TableSetupColumn("Trial", ImGuiTableColumnFlags_DefaultSort);
        ImGui::TableSetupColumn("Val Loss", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Val Acc", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Train Loss", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_NoSort);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        // Handle sorting
        ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs();
        if (sort_specs && sort_specs->SpecsDirty) {
            if (sort_specs->SpecsCount > 0) {
                sort_column_ = sort_specs->Specs[0].ColumnIndex;
                sort_ascending_ = (sort_specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending);
            }
            sort_specs->SpecsDirty = false;
        }

        for (const auto& result : sorted_results) {
            ImGui::TableNextRow();

            // Highlight best result
            if (result.trial_id == best_trial_id_) {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(50, 100, 50, 100));
            }

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%d", result.trial_id + 1);

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.4f", result.val_loss);

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.2f%%", result.val_accuracy * 100.0f);

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.4f", result.train_loss);

            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.1fs", result.duration_seconds);

            ImGui::TableSetColumnIndex(5);
            if (result.completed) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_CHECK);
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), ICON_FA_XMARK);
            }
        }

        ImGui::EndTable();
    }

    // Export button
    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export Results")) {
        // TODO: Export to CSV
        spdlog::info("Export results not yet implemented");
    }
}

void HyperparamSearchPanel::RenderAddParameterDialog() {
    ImGui::OpenPopup("Add Parameter");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Add Parameter", &show_add_param_dialog_,
        ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Text("Add a new parameter to the search space:");
        ImGui::Spacing();

        // Parameter name
        ImGui::Text("Name:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ParamName", param_name_, sizeof(param_name_));

        ImGui::Spacing();

        // Parameter type
        ImGui::Text("Type:");
        ImGui::RadioButton("Continuous", &param_type_, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Discrete", &param_type_, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Log Scale", &param_type_, 2);
        ImGui::SameLine();
        ImGui::RadioButton("Categorical", &param_type_, 3);

        ImGui::Spacing();

        // Type-specific options
        if (param_type_ == 3) {  // Categorical
            ImGui::Text("Options (comma-separated):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##Options", param_options_, sizeof(param_options_));
        } else {
            ImGui::Text("Range:");
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Min", &param_min_);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Max", &param_max_);

            if (param_type_ == 1) {  // Discrete
                ImGui::SetNextItemWidth(100);
                ImGui::InputInt("Step", &param_step_);
                param_step_ = std::max(1, param_step_);
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Action buttons
        if (ImGui::Button("Add", ImVec2(100, 0))) {
            AddParameter();
            show_add_param_dialog_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0))) {
            show_add_param_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void HyperparamSearchPanel::StartSearch() {
    if (is_running_) return;

    if (!config_.space.IsValid()) {
        spdlog::error("Invalid search space. Please add at least one parameter.");
        return;
    }

    search_engine_->SetConfig(config_);
    search_engine_->Start();
    is_running_ = true;
    current_trial_ = 0;
    total_trials_ = search_engine_->GetTotalTrials();
    current_progress_ = 0.0f;
    elapsed_seconds_ = 0.0f;

    spdlog::info("Starting hyperparameter search with {} trials", total_trials_);
}

void HyperparamSearchPanel::StopSearch() {
    if (!is_running_) return;

    search_engine_->Stop();
    is_running_ = false;
    spdlog::info("Hyperparameter search stopped");
}

void HyperparamSearchPanel::AddParameter() {
    if (strlen(param_name_) == 0) {
        spdlog::warn("Parameter name cannot be empty");
        return;
    }

    switch (param_type_) {
        case 0:  // Continuous
            config_.space.AddContinuous(param_name_, param_min_, param_max_);
            break;
        case 1:  // Discrete
            config_.space.AddDiscrete(param_name_, static_cast<int>(param_min_),
                                      static_cast<int>(param_max_), param_step_);
            break;
        case 2:  // Log Scale
            config_.space.AddLogScale(param_name_, param_min_, param_max_);
            break;
        case 3: {  // Categorical
            std::vector<std::string> options;
            std::string opts_str = param_options_;
            size_t pos = 0;
            while ((pos = opts_str.find(',')) != std::string::npos) {
                std::string opt = opts_str.substr(0, pos);
                // Trim whitespace
                size_t start = opt.find_first_not_of(' ');
                size_t end = opt.find_last_not_of(' ');
                if (start != std::string::npos && end != std::string::npos) {
                    options.push_back(opt.substr(start, end - start + 1));
                }
                opts_str.erase(0, pos + 1);
            }
            // Add last option
            size_t start = opts_str.find_first_not_of(' ');
            size_t end = opts_str.find_last_not_of(' ');
            if (start != std::string::npos && end != std::string::npos) {
                options.push_back(opts_str.substr(start, end - start + 1));
            }

            if (!options.empty()) {
                config_.space.AddCategorical(param_name_, options);
            } else {
                spdlog::warn("No valid options provided for categorical parameter");
            }
            break;
        }
    }

    spdlog::info("Added parameter: {}", param_name_);
}

void HyperparamSearchPanel::RemoveParameter(size_t index) {
    if (index < config_.space.parameters.size()) {
        std::string name = config_.space.parameters[index].name;
        config_.space.parameters.erase(config_.space.parameters.begin() + index);
        spdlog::info("Removed parameter: {}", name);
    }
}

} // namespace cyxwiz
