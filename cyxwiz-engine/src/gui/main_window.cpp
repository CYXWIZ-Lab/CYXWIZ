#include "../core/legacy_dataset_batchers.h"
#include "../core/session_query_service.h"
#include "../core/pipeline_type_names.h"
#include "../core/image_quality_analysis.h"
#include "../core/training_export_metadata.h"
#include "separate_windows.h"
// Windows header order fix - must come first to prevent winsock conflicts
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h> // Before Parquet/gRPC headers undefine Windows OPTIONAL.
#endif

#include "../core/legacy_dataset_batchers.h"
#include "../core/training_export_metadata.h"
#include "main_window.h"
#include "appearance_settings.h"
#include "panel_memory.h"
#include "../core/live_graph_compile.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "loaders/data_loader.h"
#include "node_config_dialog.h"
#include "../core/graph_topology_utils.h"
#include "graph_training_launcher.h"
#include "../core/training_input_preparation.h"
#include "engine_graph_training_dispatch.h"
#include "node_editor.h"
#include "console.h"
#include "viewport.h"
#include "properties.h"
#include "dock_style.h"
#include "icons.h"
#include "theme.h"
#include "../core/error_codes.h"
#include "../core/debug_graph_trace_executor.h"
#include "../core/debug_error_code_timeline.h"
#include "../core/debug_export_correlation_tracer.h"
#include "../core/debug_runtime_backend_classifier.h"
#include "../core/debug_slow_path_detector.h"
#include "../core/debug_trace_record.h"
#include "../core/debug_operator_trace_producer.h"
#include "../core/compute_runtime_config.h"
#include "../core/compute_runtime_paths.h"
#include "../core/execution_device_preferences.h"
#include "../core/keyboard_shortcuts.h"
#include "../extension/python/python_node_factory.h"
#include "../extension/python/python_node_provider.h"
#include "python_bundle_ui.h"
#include "frame_metrics.h"
#include "../core/sequence_arrow_batcher.h"
#include "panels/toolbar.h"
#include "panels/asset_browser.h"
#include "panels/training_dashboard.h"
#include "panels/training_plot_panel.h"
#include "panels/script_editor.h"
#include "panels/table_viewer.h"
#include "panels/annotation_editor_panel.h"
#include "panels/connection_dialog.h"
#include "panels/job_status_panel.h"
#include "panels/p2p_training_panel.h"
#include "panels/wallet_panel.h"
#include "panels/task_progress_panel.h"
#include "panels/pattern_browser.h"
#include "panels/query_console.h"
#include "panels/custom_node_editor.h"
#include "panels/theme_editor.h"
#include "panels/memory_panel.h"
#include "panels/memory_monitor.h"
#include "panels/variable_explorer.h"
#include "panels/plot_output_panel.h"
#include "panels/export_dialog.h"
#include "panels/import_dialog.h"
#include "panels/deployment_dialog.h"
#include "panels/model_summary_panel.h"
#include "panels/architecture_diagram.h"
#include "panels/lr_finder_panel.h"
#include "panels/studio_debugger_panel.h"
#include "../core/studio_debugger_presentation.h"
#include "panels/dim_reduction_panel.h"
#include "panels/gradcam_panel.h"
#include "panels/feature_importance_panel.h"
#include "panels/nas_panel.h"
#include "panels/hyperparam_search_panel.h"
#include "panels/serving_panel.h"
#include "panels/dnn_inference_panel.h"
#include "panels/kmeans_panel.h"
#include "panels/dbscan_panel.h"
#include "panels/hierarchical_panel.h"
#include "panels/gmm_panel.h"
#include "panels/cluster_eval_panel.h"
#include "panels/matrix_calculator_panel.h"
#include "panels/eigen_decomp_panel.h"
#include "panels/svd_panel.h"
#include "panels/qr_panel.h"
#include "panels/cholesky_panel.h"
#include "panels/fft_panel.h"
#include "panels/spectrogram_panel.h"
#include "panels/filter_designer_panel.h"
#include "panels/convolution_panel.h"
#include "panels/wavelet_panel.h"
#include "panels/gradient_descent_panel.h"
#include "panels/convexity_panel.h"
#include "panels/lp_panel.h"
#include "panels/qp_panel.h"
#include "panels/differentiation_panel.h"
#include "panels/integration_panel.h"
#include "panels/decomposition_panel.h"
#include "panels/acf_pacf_panel.h"
#include "panels/stationarity_panel.h"
#include "data_studio/folds_view.h"
#include "../core/node_executors/training_evaluation_operator.h"
#include "panels/seasonality_panel.h"
#include "panels/forecasting_panel.h"
#include "panels/tokenization_panel.h"
#include "panels/word_frequency_panel.h"
#include "panels/tfidf_panel.h"
#include "panels/embeddings_panel.h"
#include "panels/sentiment_panel.h"
#include "panels/language_model_generation_panel.h"
// Utilities panels (Phase 12)
#include "panels/calculator_panel.h"
#include "panels/unit_converter_panel.h"
#include "panels/random_generator_panel.h"
#include "panels/hash_generator_panel.h"
#include "panels/json_viewer_panel.h"
#include "panels/regex_tester_panel.h"
#include "panels/cloud_browser.h"
#include "panels/node_browser_panel.h"
#include "panels/node_info_panel.h"
#include "../core/node_metadata.h"
#include "panels/cloud_dataset_manager.h"
#include "panels/plugin_manager_panel.h"
#include "data_studio/data_studio_panel.h"
#include "dashboard/dashboard_links.h"
#include "../plugin/plugin_manager.h"
#include "../plugin/plugin_context.h"
#include "../plugin/registries/plugin_panel_registry.h"
#include "tutorial/tutorial_system.h"
#include "../core/async_task_manager.h"
#include "../core/dataset_catalog.h"
#include "../scripting/scripting_engine.h"
#include "../scripting/startup_script_manager.h"
#include "../network/job_manager.h"
#include "../network/grpc_client.h"
#include "../network/reservation_client.h"
#include "../network/p2p_client.h"
#include "../network/datastream_client.h"
#include "../auth/auth_client.h"
#include "../core/pipeline_execution_task.h"
#include "../core/project_manager.h"
#include "../core/engine_config.h"
#include "../core/cache_prune.h"
#include "../core/data_budget.h"
#include "../core/data_registry.h"
#include "../core/arrow_dataset.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/graph_compiler.h"
#include "../core/debug_executor.h"
#include "../core/debug_session_manager.h"
#include "../core/preflight_validator.h"
#include "../core/smoke_run_executor.h"
#include "../core/debug_recommendation_engine.h"
#include "../core/debug_gradient_health_analyzer.h"
#include "../core/debug_loss_metric_explainer.h"
#include "../core/debug_training_stall_detector.h"
#include "../core/debug_memory_ownership_tracer.h"
#include "../core/debug_node_inspector.h"
#include "../core/training_trace_collector.h"
#include "../core/debug_run_store.h"
#include "../core/runtime_log_store.h"
#include "../core/training_executor.h"
#include "../core/training_manager.h"
#include "../core/test_manager.h"
#include "../core/test_predictions_metrics.h"
#include "../core/test_predictions_table.h"
#include "../core/graph_compile_task.h"
#include "../core/test_dataset_selection.h"
#include "../core/split_partitioning.h"
#include "../core/model_converter.h"
#include "../core/model_importer.h"
#include "../core/node_metadata_registry.h"
#include "../serving/inference_server.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/word_embeddings.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <sstream>
#include <thread>

#include "../core/file_dialogs.h"

namespace gui {

namespace {

// FNV-1a 64-bit. Deterministic, dependency-free, and fine for detecting
// "did the graph change" - this is not a cryptographic use.
inline void HashFnvMix(uint64_t& h, const std::string& s) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
}
inline void HashFnvMix(uint64_t& h, int v) {
    HashFnvMix(h, std::to_string(v));
    HashFnvMix(h, std::string(1, '|'));
}

// Stable fingerprint of the graph's *trainable* structure: node ids, node
// types, node parameter maps, link endpoints + types. Positions and
// cosmetic fields are intentionally excluded so dragging a node around
// the canvas doesn't invalidate the staleness cache. Hash is stable
// across runs as long as the underlying identity of nodes / links is
// preserved.
uint64_t HashGraphStructure(const std::vector<MLNode>& nodes,
                            const std::vector<NodeLink>& links) {
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis

    // Sort-stable copies by id so reordering doesn't change the hash.
    std::vector<const MLNode*> sorted_nodes;
    sorted_nodes.reserve(nodes.size());
    for (const auto& n : nodes) sorted_nodes.push_back(&n);
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [](const MLNode* a, const MLNode* b) { return a->id < b->id; });

    for (const MLNode* np : sorted_nodes) {
        HashFnvMix(h, np->id);
        HashFnvMix(h, static_cast<int>(np->type));
        HashFnvMix(h, np->name);
        // Parameters: iterate in sorted key order (std::map already does).
        for (const auto& [k, v] : np->parameters) {
            HashFnvMix(h, k);
            HashFnvMix(h, v);
            HashFnvMix(h, std::string(1, ';'));
        }
        // Which extension, version and code: swapping one changes the model
        // (TOFIX125 P6). Built-in nodes hash as before.
        if (!np->extension_type_id.empty()) {
            HashFnvMix(h, np->extension_type_id);
            HashFnvMix(h, np->extension_version);
            HashFnvMix(h, np->extension_content_hash);
        }
        HashFnvMix(h, std::string(1, '#'));
    }

    std::vector<const NodeLink*> sorted_links;
    sorted_links.reserve(links.size());
    for (const auto& l : links) sorted_links.push_back(&l);
    std::sort(sorted_links.begin(), sorted_links.end(),
              [](const NodeLink* a, const NodeLink* b) {
                  if (a->from_node != b->from_node) return a->from_node < b->from_node;
                  if (a->from_pin  != b->from_pin)  return a->from_pin  < b->from_pin;
                  if (a->to_node   != b->to_node)   return a->to_node   < b->to_node;
                  return a->to_pin < b->to_pin;
              });
    for (const NodeLink* lp : sorted_links) {
        HashFnvMix(h, lp->from_node);
        HashFnvMix(h, lp->from_pin);
        HashFnvMix(h, lp->to_node);
        HashFnvMix(h, lp->to_pin);
        HashFnvMix(h, static_cast<int>(lp->type));
        HashFnvMix(h, std::string(1, '>'));
    }
    return h;
}

std::string FindTestingLabelColumn(
    const std::vector<MLNode>& nodes,
    const std::string& dataset_name,
    int data_source_node_id) {

    const MLNode* fallback_data_input = nullptr;
    for (const auto& node : nodes) {
        if (node.type != NodeType::DataInput) {
            continue;
        }

        if (node.id == data_source_node_id) {
            auto label_it = node.parameters.find("label_column");
            if (label_it != node.parameters.end() && !label_it->second.empty()) {
                return label_it->second;
            }
            return {};
        }

        if (!fallback_data_input) {
            fallback_data_input = &node;
        }

        auto dataset_it = node.parameters.find("dataset_name");
        if (dataset_it == node.parameters.end() || dataset_it->second.empty()) {
            dataset_it = node.parameters.find("dataset");
        }
        if (!dataset_name.empty() &&
            dataset_it != node.parameters.end() &&
            dataset_it->second == dataset_name) {
            auto label_it = node.parameters.find("label_column");
            if (label_it != node.parameters.end() && !label_it->second.empty()) {
                return label_it->second;
            }
            return {};
        }
    }

    if (fallback_data_input) {
        auto label_it = fallback_data_input->parameters.find("label_column");
        if (label_it != fallback_data_input->parameters.end() &&
            !label_it->second.empty()) {
            return label_it->second;
        }
    }
    return {};
}

std::string JsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    escaped += ' ';
                } else {
                    escaped += c;
                }
                break;
        }
    }
    return escaped;
}

std::string BuildAssistantGraphContextJson(NodeEditor& editor) {
    const auto& nodes = editor.GetNodes();
    const auto& links = editor.GetLinks();
    const int selected_node_id = editor.GetSelectedNodeId();

    std::ostringstream out;
    out << "{";
    out << "\"node_count\":" << nodes.size() << ",";
    out << "\"link_count\":" << links.size() << ",";
    out << "\"graph_hash\":\"" << HashGraphStructure(nodes, links) << "\",";
    out << "\"selected_node\":";

    const auto selected = std::find_if(nodes.begin(), nodes.end(),
        [selected_node_id](const MLNode& node) { return node.id == selected_node_id; });
    if (selected == nodes.end()) {
        out << "null";
    } else {
        out << "{";
        out << "\"id\":" << selected->id << ",";
        out << "\"name\":\"" << JsonEscape(selected->name) << "\",";
        out << "\"type\":\"" << JsonEscape(editor.GetNodeTypeDisplayName(selected->type)) << "\"";
        out << "}";
    }

    out << "}";
    return out.str();
}

cyxwiz::plugin::AssistantContextSnapshot BuildAssistantContextSnapshot(
    NodeEditor* node_editor,
    cyxwiz::StudioDebuggerPanel* studio_debugger_panel) {
    cyxwiz::plugin::AssistantContextSnapshot assistant_context;
    assistant_context.engine_version = "dev";
    assistant_context.build_id = "local";
    assistant_context.selected_panel = "MainWindow";

    auto& project_manager = cyxwiz::ProjectManager::Instance();
    if (project_manager.HasActiveProject()) {
        assistant_context.workspace_root = project_manager.GetProjectRoot();
    }
    if (node_editor) {
        assistant_context.active_graph_path = node_editor->GetCurrentFilePath();
        const int selected_node_id = node_editor->GetSelectedNodeId();
        if (selected_node_id >= 0) {
            assistant_context.selected_node_id = std::to_string(selected_node_id);
        }
        assistant_context.debugger_context_json =
            "{\"schema\":\"cyxwiz.assistant.context_bundle.v1\",\"graph\":" +
            BuildAssistantGraphContextJson(*node_editor) +
            ",\"debugger\":null}";
    }
    if (studio_debugger_panel) {
        assistant_context.selected_trace_id =
            studio_debugger_panel->GetSelectedTraceIdForAssistant();
        assistant_context.debugger_context_json =
            "{\"schema\":\"cyxwiz.assistant.context_bundle.v1\",\"graph\":" +
            (node_editor ? BuildAssistantGraphContextJson(*node_editor) : std::string("null")) +
            ",\"debugger\":" +
            studio_debugger_panel->BuildAssistantDebuggerContextJson() +
            "}";
        assistant_context.training_context_json =
            studio_debugger_panel->BuildAssistantTrainingContextJson();
    }
    return assistant_context;
}

std::string NowLocalTimestampForDebugStore() {
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now_time);
#else
    localtime_r(&now_time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

std::string MakeDebuggerRunId(uint64_t graph_hash) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::ostringstream out;
    out << "local-debug-" << std::hex << graph_hash << "-" << std::dec << ms;
    return out.str();
}

std::string MakeArtifactRunId(const std::string& action,
                              uint64_t graph_hash) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::ostringstream out;
    out << "artifact-" << (action.empty() ? "action" : action) << "-"
        << std::hex << graph_hash << "-" << std::dec << ms;
    return out.str();
}

bool DirectoryHasRegularFile(const std::filesystem::path& path) {
    if (!std::filesystem::is_directory(path)) {
        return false;
    }
    std::error_code error;
    for (std::filesystem::directory_iterator it(path, error), end;
         !error && it != end;
         it.increment(error)) {
        if (it->is_regular_file(error)) {
            return true;
        }
    }
    return false;
}

cyxwiz::DebugArtifactObservation ObserveModelArtifact(
    const std::string& path,
    const cyxwiz::ProbeResult& probe,
    bool inspection_supported) {
    cyxwiz::DebugArtifactObservation observed;
    observed.available = inspection_supported;
    observed.manifest_valid = inspection_supported && probe.valid;
    observed.format_version = probe.format_version;
    observed.model_family = probe.model_family;
    if (!inspection_supported) {
        observed.evidence_scope = "unobserved_probe_unsupported";
        return observed;
    }

    if (probe.valid) {
        observed.parameter_count =
            static_cast<size_t>(std::max(0, probe.num_parameters));
        observed.layer_count =
            static_cast<size_t>(std::max(0, probe.num_layers));
        observed.optimizer_state_present = probe.has_optimizer_state;
        observed.training_history_present = probe.has_training_history;
        observed.sequence_assets_present = probe.has_sequence;
        observed.tree_model_artifact_present =
            probe.has_tree_model_artifact;
        observed.input_contract = probe.supports_bert_encoder
            ? probe.bert_encoder_input_kind
            : std::string{};
        observed.output_contract = probe.supports_generation
            ? probe.generation_output_contract
            : (probe.supports_bert_encoder
                ? probe.bert_encoder_output_contract
                : std::string{});
    }

    const std::filesystem::path artifact_path(path);
    if (probe.format == cyxwiz::ModelFormat::CyxModel &&
        std::filesystem::is_directory(artifact_path)) {
        observed.evidence_scope = "cyxmodel_directory_manifest_and_inventory";
        observed.manifest_present =
            std::filesystem::is_regular_file(artifact_path / "manifest.json");
        observed.training_config_present =
            std::filesystem::is_regular_file(artifact_path / "config.json");
        observed.weights_manifest_present = std::filesystem::is_regular_file(
            artifact_path / "weights" / "manifest.json");
        observed.graph_present = std::filesystem::is_regular_file(
            artifact_path / "graph.cyxgraph");
        observed.tokenizer_config_present = std::filesystem::is_regular_file(
            artifact_path / "tokenizer" / "config.json");
        observed.tokenizer_vocabulary_present =
            std::filesystem::is_regular_file(
                artifact_path / "tokenizer" / "vocab.txt");
        observed.optimizer_state_present =
            DirectoryHasRegularFile(artifact_path / "optimizer");
        observed.training_history_present =
            std::filesystem::is_regular_file(artifact_path / "history.json");
        observed.sequence_assets_present =
            DirectoryHasRegularFile(artifact_path / "sequence");
        observed.tree_model_artifact_present =
            std::filesystem::is_regular_file(
                artifact_path / "tree" / "model.json");
    } else if (probe.format == cyxwiz::ModelFormat::CyxModel &&
               std::filesystem::is_regular_file(artifact_path)) {
        observed.evidence_scope = "cyxw_binary_header";
    } else if (probe.format == cyxwiz::ModelFormat::Safetensors) {
        observed.evidence_scope = "safetensors_header";
    } else {
        observed.evidence_scope = "manifest_probe";
        observed.graph_present = probe.has_graph;
        observed.tokenizer_config_present = probe.has_tokenizer;
        observed.tokenizer_vocabulary_present = probe.has_vocabulary;
    }
    return observed;
}

} // anonymous namespace

MainWindow::MainWindow()
    : first_time_layout_(true) {

    // Run Parquet cache hygiene once at startup. Drops anything older than
    // 30 days and trims the directory back under 10 GB if it grew over the
    // cap during the previous session. Filesystem ops on a small directory,
    // fast enough to do synchronously without measurable startup impact.
    cyxwiz::ParquetBackedDataset::PruneCache();

    // Merge persisted GPU placement evidence (runtime fallbacks, preflight
    // probes) into the process-global store so the first compile of this
    // session already routes around known-unsafe device/shape keys.

    // Original panels
    node_editor_ = std::make_unique<NodeEditor>();
    node_editor_->SetGraphPreparationBusyCallback([this] { return test_preparation_task_ != nullptr; });
    node_editor_->SetTrainingDataLoadingCallback([this] { return !data_input_loads_.empty(); });
    console_ = std::make_unique<Console>();
    viewport_ = std::make_unique<Viewport>();
    properties_ = std::make_unique<Properties>();

    // Initialize scripting engine (shared resource)
    scripting_engine_ = std::make_shared<scripting::ScriptingEngine>();
    console_->SetScriptingEngine(scripting_engine_);
    console_->SetAssistantCommandHandler(
        [](const cyxwiz::plugin::AssistantCommandRequest& request) {
            return cyxwiz::plugin::PluginManager::Instance()
                .RunAssistantCommand(request);
        });

    // New panel system
    toolbar_ = std::make_unique<cyxwiz::ToolbarPanel>();
    asset_browser_ = std::make_unique<cyxwiz::AssetBrowserPanel>();
    training_plot_panel_ = std::make_shared<cyxwiz::TrainingPlotPanel>();  // Now named "Training Dashboard"
    // Prepared-data cache actions from the dashboard's Data preparation card.
    training_plot_panel_->SetSavePathCallback([](const std::string& title, const std::string& default_name) {
        const std::string root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
        const std::string folder = root.empty() ? std::string() : (std::filesystem::path(root) / "exports").string();
        return cyxwiz::FileDialogs::SaveFile(title.c_str(), {{"CSV Files", "csv"}, {"All Files", "*"}},
                                             folder.empty() ? nullptr : folder.c_str(), default_name.c_str());
    });
    training_plot_panel_->SetMaterializationActionCallback(
        [weak_panel = std::weak_ptr<cyxwiz::TrainingPlotPanel>(training_plot_panel_)](
            const std::string& action) {
            const std::filesystem::path project_root =
                cyxwiz::ProjectManager::Instance().GetProjectRoot();
            const auto publish_usage = [weak_panel, project_root]() {
                const auto usage = MeasureGraphMaterializationCache(project_root);
                const auto config = GraphMaterializationCacheConfig(project_root);
                cyxwiz::AsyncTaskManager::Instance().PostToMainThread(
                    [weak_panel, usage,
                     directory = cyxwiz::MaterializationCacheDirectory(config).string(),
                     limit = config.max_total_bytes]() {
                        if (auto panel = weak_panel.lock()) {
                            panel->SetMaterializationCacheInfo(
                                directory, usage.entries, usage.total_bytes, limit);
                        }
                    });
            };
            if (action == "rebuild") {
                RequestGraphMaterializationRebuild(true);
                spdlog::info("Prepared data will be rebuilt on the next training run");
            } else if (action == "cancel_rebuild") {
                RequestGraphMaterializationRebuild(false);
            } else if (action == "refresh") {
                cyxwiz::AsyncTaskManager::Instance().RunAsync(
                    "Measure prepared-data cache",
                    [publish_usage](cyxwiz::LambdaTask&) { publish_usage(); });
            } else if (action == "clear") {
                cyxwiz::AsyncTaskManager::Instance().RunAsync(
                    "Clear prepared-data cache",
                    [project_root, publish_usage, weak_panel](cyxwiz::LambdaTask& task) {
                        task.ReportProgress(0.1f, "Removing prepared datasets...");
                        const auto cleared = ClearGraphMaterializationCache(project_root);
                        cyxwiz::AsyncTaskManager::Instance().PostToMainThread(
                            [weak_panel, cleared]() {
                                if (auto panel = weak_panel.lock()) {
                                    panel->SetMaterializationClearResult(
                                        cleared.removed_entries, cleared.freed_bytes,
                                        cleared.error);
                                }
                            });
                        publish_usage();
                    });
            }
        });

    script_editor_ = std::make_unique<cyxwiz::ScriptEditorPanel>();
    table_viewer_ = std::make_unique<cyxwiz::TableViewerPanel>();
    annotation_editor_panel_ = std::make_unique<cyxwiz::AnnotationEditorPanel>();
    job_status_panel_ = std::make_unique<cyxwiz::JobStatusPanel>();
    p2p_training_panel_ = std::make_unique<cyxwiz::P2PTrainingPanel>();
    wallet_panel_ = std::make_unique<gui::WalletPanel>();
    task_progress_panel_ = std::make_unique<cyxwiz::TaskProgressPanel>();
    pattern_browser_ = std::make_unique<cyxwiz::PatternBrowserPanel>();
    query_console_ = std::make_unique<cyxwiz::QueryConsolePanel>();
    custom_node_editor_ = std::make_unique<gui::CustomNodeEditorPanel>();
    theme_editor_ = std::make_unique<gui::ThemeEditorPanel>();
    memory_panel_ = std::make_unique<cyxwiz::MemoryPanel>();
    memory_monitor_ = std::make_unique<cyxwiz::MemoryMonitor>();
    variable_explorer_ = std::make_unique<cyxwiz::VariableExplorerPanel>();
    variable_explorer_->SetScriptingEngine(scripting_engine_);
    table_viewer_->SetScriptingEngine(scripting_engine_.get());  // Data Viewer reads again (TOFIX133 P5)

    // Plot output panel (like MATLAB's figure window)
    plot_output_panel_ = std::make_unique<cyxwiz::PlotOutputPanel>();
    plot_output_panel_->SetScriptingEngine(scripting_engine_);

    // Test results panel

    // Export/Import dialogs
    export_dialog_ = std::make_unique<cyxwiz::ExportDialog>();
    import_dialog_ = std::make_unique<cyxwiz::ImportDialog>();
    deployment_dialog_ = std::make_unique<cyxwiz::DeploymentDialog>();
    studio_debugger_panel_ = std::make_unique<cyxwiz::StudioDebuggerPanel>();

    // Model Analysis panels (Phase 2)
    model_summary_panel_ = std::make_unique<cyxwiz::ModelSummaryPanel>();
    model_summary_panel_->SetNodeEditor(node_editor_.get());

    architecture_diagram_ = std::make_unique<cyxwiz::ArchitectureDiagram>();
    architecture_diagram_->SetNodeEditor(node_editor_.get());

    lr_finder_panel_ = std::make_unique<cyxwiz::LRFinderPanel>();
    lr_finder_panel_->SetNodeEditor(node_editor_.get());

    // Statistics panels (Phase 4)

    // Advanced Tools panels (Phase 5)
    dim_reduction_panel_ = std::make_unique<cyxwiz::DimReductionPanel>();
    gradcam_panel_ = std::make_unique<cyxwiz::GradCAMPanel>();
    feature_importance_panel_ = std::make_unique<cyxwiz::FeatureImportancePanel>();
    feature_importance_panel_->SetLoadCheckpointCallback([this]() { LoadCheckpointForTesting(); });
    nas_panel_ = std::make_unique<cyxwiz::NASPanel>();
    hyperparam_search_panel_ = std::make_unique<cyxwiz::HyperparamSearchPanel>();
    serving_panel_ = std::make_unique<cyxwiz::ServingPanel>();
    dnn_inference_panel_ = std::make_unique<cyxwiz::DNNInferencePanel>();

    // Clustering panels (Phase 6A)
    kmeans_panel_ = std::make_unique<cyxwiz::KMeansPanel>();
    dbscan_panel_ = std::make_unique<cyxwiz::DBSCANPanel>();
    hierarchical_panel_ = std::make_unique<cyxwiz::HierarchicalPanel>();
    gmm_panel_ = std::make_unique<cyxwiz::GMMPanel>();
    cluster_eval_panel_ = std::make_unique<cyxwiz::ClusterEvalPanel>();

    // Model Evaluation panels (Phase 6B)

    // Linear Algebra panels (Phase 7)
    matrix_calculator_panel_ = std::make_unique<cyxwiz::MatrixCalculatorPanel>();
    eigen_decomp_panel_ = std::make_unique<cyxwiz::EigenDecompPanel>();
    svd_panel_ = std::make_unique<cyxwiz::SVDPanel>();
    qr_panel_ = std::make_unique<cyxwiz::QRPanel>();
    cholesky_panel_ = std::make_unique<cyxwiz::CholeskyPanel>();

    // Signal Processing panels (Phase 8)
    fft_panel_ = std::make_unique<cyxwiz::FFTPanel>();
    spectrogram_panel_ = std::make_unique<cyxwiz::SpectrogramPanel>();
    filter_designer_panel_ = std::make_unique<cyxwiz::FilterDesignerPanel>();
    convolution_panel_ = std::make_unique<cyxwiz::ConvolutionPanel>();
    wavelet_panel_ = std::make_unique<cyxwiz::WaveletPanel>();

    // Optimization & Calculus panels (Phase 9)
    gradient_descent_panel_ = std::make_unique<cyxwiz::GradientDescentPanel>();
    convexity_panel_ = std::make_unique<cyxwiz::ConvexityPanel>();
    lp_panel_ = std::make_unique<cyxwiz::LPPanel>();
    qp_panel_ = std::make_unique<cyxwiz::QPPanel>();
    differentiation_panel_ = std::make_unique<cyxwiz::DifferentiationPanel>();
    integration_panel_ = std::make_unique<cyxwiz::IntegrationPanel>();

    // Time Series Analysis panels (Phase 10)
    decomposition_panel_ = std::make_unique<cyxwiz::DecompositionPanel>();
    acf_pacf_panel_ = std::make_unique<cyxwiz::ACFPACFPanel>();
    stationarity_panel_ = std::make_unique<cyxwiz::StationarityPanel>();
    seasonality_panel_ = std::make_unique<cyxwiz::SeasonalityPanel>();
    forecasting_panel_ = std::make_unique<cyxwiz::ForecastingPanel>();

    // Text Processing panels (Phase 11)
    tokenization_panel_ = std::make_unique<cyxwiz::TokenizationPanel>();
    word_frequency_panel_ = std::make_unique<cyxwiz::WordFrequencyPanel>();
    tfidf_panel_ = std::make_unique<cyxwiz::TFIDFPanel>();
    embeddings_panel_ = std::make_unique<cyxwiz::EmbeddingsPanel>();
    sentiment_panel_ = std::make_unique<cyxwiz::SentimentPanel>();
    language_model_generation_panel_ =
        std::make_unique<cyxwiz::LanguageModelGenerationPanel>();

    // Utilities panels (Phase 12)
    calculator_panel_ = std::make_unique<cyxwiz::CalculatorPanel>();
    unit_converter_panel_ = std::make_unique<cyxwiz::UnitConverterPanel>();
    random_generator_panel_ = std::make_unique<cyxwiz::RandomGeneratorPanel>();
    hash_generator_panel_ = std::make_unique<cyxwiz::HashGeneratorPanel>();
    json_viewer_panel_ = std::make_unique<cyxwiz::JSONViewerPanel>();
    regex_tester_panel_ = std::make_unique<cyxwiz::RegexTesterPanel>();
    
    // Cloud panels (DataStream)
    cloud_browser_panel_ = std::make_unique<gui::CloudBrowserPanel>();
    node_browser_panel_ = std::make_unique<gui::NodeBrowserPanel>();
    node_browser_panel_->SetNodeEditor(node_editor_.get());
    node_info_panel_ = std::make_unique<cyxwiz::NodeInfoPanel>();
    // Connect Node Browser hover to Info Panel
    if (node_browser_panel_ && node_info_panel_) {
        // Hover previews a node; a click keeps it in the Info panel.
        node_browser_panel_->SetNodeHoverCallback([this](cyxwiz::NodeType type) {
            node_info_panel_->PreviewNode(type);
        });
        node_browser_panel_->SetNodeSelectCallback([this](cyxwiz::NodeType type) {
            node_info_panel_->SetSelectedNode(type);
        });
        // Extension nodes share one NodeType and are identified by type id.
        node_browser_panel_->SetExtensionHoverCallback([this](const std::string& type_id) {
            node_info_panel_->PreviewExtension(type_id);
        });
        node_browser_panel_->SetExtensionSelectCallback([this](const std::string& type_id) {
            node_info_panel_->SetSelectedExtension(type_id);
        });
    }
    // A node clicked on the canvas is kept in the Info panel too.
    if (node_editor_ && node_info_panel_) {
        node_editor_->SetNodeSelectedCallback([this](const gui::MLNode& node) {
            if (node.type == gui::NodeType::PluginCustom && !node.extension_type_id.empty())
                node_info_panel_->SetSelectedExtension(node.extension_type_id);
            else
                node_info_panel_->SetSelectedNode(node.type);
        });
    }
    cloud_dataset_manager_panel_ = std::make_unique<gui::CloudDatasetManagerPanel>();

    // Plugin Manager panel
    plugin_manager_panel_ = std::make_unique<cyxwiz::PluginManagerPanel>();
    // A missing extension node, or an empty Plugins section, opens it (TOFIX125).
    if (node_editor_) {
        node_editor_->SetOpenPluginManagerCallback([this]() {
            if (plugin_manager_panel_) plugin_manager_panel_->SetVisible(true);
        });
    }

    // Data Studio panel (Phase 1 Week 1)
    data_studio_panel_ = std::make_unique<cyxwiz::DataStudioPanel>();
    data_studio_panel_->SetNodeEditor(node_editor_.get());

    // Data Studio's Show graph, Open dialog and Add Data Input bring the
    // CyxWiz Studio tab to the front (a node is added when Studio draws).
    data_studio_panel_->SetOpenNodeEditorCallback([this]() {
        if (node_editor_) {
            node_editor_->Raise();
            spdlog::debug("Raised CyxWiz Studio from Data Studio");
        }
    });
    // A dashboard's "Edit roles in Data Studio" (TOFIX134 P3.8).
    if (node_editor_) {
        node_editor_->SetOpenDataStudioProfileCallback([this](const std::string& dataset) {
            if (data_studio_panel_) data_studio_panel_->ShowProfile(dataset);
        });
        node_editor_->SetOpenDataStudioQueryCallback([this](const std::string& dataset, const std::string& sql) {
            if (data_studio_panel_) data_studio_panel_->ShowQuery(dataset, sql);
        });
        // Add to Dashboard from Data Studio's Visualize and Query tabs (TOFIX134 P3.11).
        auto& links = cyxwiz::dashboard::DashboardLinks();
        links.list = [this](const std::string& dataset) { return node_editor_->DashboardTargets(dataset); };
        links.new_source = [this](const std::string& dataset) { return node_editor_->DashboardSourceFor(dataset); };
        links.add = [this](int node_id, const std::string& dataset, const cyxwiz::dashboard::WidgetSpec& widget) {
            node_editor_->QueueDashboardWidget(node_id, dataset, widget);
        };
    }
    // Query results open in the Table Viewer (TOFIX134 P3.3).
    data_studio_panel_->SetOpenTableCallback([this](std::shared_ptr<arrow::Table> table, const std::string& name) {
        if (table_viewer_ && table) {
            table_viewer_->OpenArrow(std::move(table), name);
            table_viewer_->Show();
            ::gui::BringToFront(table_viewer_->GetName());  // in front, as from the other routes
        }
    });

    // Set NAS panel callbacks for node editor integration
    nas_panel_->SetGetArchitectureCallback([this]() -> std::pair<std::vector<MLNode>, std::vector<NodeLink>> {
        if (node_editor_) {
            return {node_editor_->GetNodes(), node_editor_->GetLinks()};
        }
        return {{}, {}};
    });

    nas_panel_->SetApplyArchitectureCallback([this](const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
        if (node_editor_) {
            node_editor_->InsertPattern(nodes, links);
        }
    });

    // Set pattern browser callback to insert patterns into node editor
    pattern_browser_->SetInsertCallback([this](const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
        if (node_editor_) {
            node_editor_->InsertPattern(nodes, links);
        }
    });

    // Set scripting engine for the Script Editor. The Console-owned Python
    // REPL already received the same shared engine above.
    script_editor_->SetScriptingEngine(scripting_engine_);

    // Expose TrainingPlotPanel to Python scripts through the scripting engine
    // Registration is deferred - actual Python import happens on first script execution
    if (scripting_engine_) {
    }

    // Connect Viewport to TrainingPlotPanel for real-time metrics display
    viewport_->SetTrainingPanel(training_plot_panel_.get());

    // Word-vector runs on the Training Dashboard (TOFIX152): a Word
    // Embeddings node's training from the pipeline (its dialog's Train), and
    // what the dashboard asks back (its dialog, cancel, nearest words).
    {
        struct LoadedVectors {
            std::string path;
            std::filesystem::file_time_type stamp{};
            std::shared_ptr<cyxwiz::WordVectors> vectors;
        };
        auto loaded = std::make_shared<LoadedVectors>();
        // The saved file, read once per change.
        const auto vectors_of = [loaded](const std::string& path) -> std::shared_ptr<cyxwiz::WordVectors> {
            std::error_code error;
            const auto stamp = std::filesystem::last_write_time(path, error);
            if (error) return nullptr;
            if (loaded->path != path || loaded->stamp != stamp || !loaded->vectors) {
                loaded->path = path;
                loaded->stamp = stamp;
                loaded->vectors = std::make_shared<cyxwiz::WordVectors>(cyxwiz::WordEmbeddings::Load(path));
            }
            return loaded->vectors->success ? loaded->vectors : nullptr;
        };
        const auto resolve = [](const std::string& path) {
            std::filesystem::path p(path);
            if (p.is_relative()) p = std::filesystem::path(cyxwiz::ProjectManager::Instance().GetProjectRoot()) / p;
            std::error_code error;
            const auto canonical = std::filesystem::weakly_canonical(p, error);
            return error ? p.lexically_normal() : canonical;
        };
        cyxwiz::TrainingPlotPanel::WordVectorHooks hooks;
        hooks.open_node_dialog = [this](int node_id) {
            if (!node_editor_ || !properties_) return;
            if (MLNode* node = node_editor_->GetNodeForConfiguration(node_id)) properties_->ConfigureNode(node);
        };
        hooks.cancel = [this](int) {
            if (node_editor_ && cyxwiz::IsPipelineExecutionTaskActive(node_editor_->GetDataPipelineTaskId())) {
                cyxwiz::AsyncTaskManager::Instance().Cancel(node_editor_->GetDataPipelineTaskId());
            } else {
                gui::CancelGraphTrainingPreparation();
            }
        };
        hooks.nearest = [vectors_of](const std::string& saved, const std::string& word, std::string& note) {
            note.clear();
            const auto vectors = vectors_of(saved);
            if (!vectors) {
                note = "Cannot read " + saved;
                return std::vector<std::pair<std::string, double>>{};
            }
            std::string error;
            auto rows = cyxwiz::WordEmbeddings::Nearest(*vectors, word, 10, &error);
            if (!error.empty()) note = error;
            else if (rows.empty()) note = "No vector for \"" + word + "\".";
            return rows;
        };
        hooks.words = [vectors_of](const std::string& saved) {
            const auto vectors = vectors_of(saved);
            return vectors ? vectors->words : std::vector<std::string>{};
        };
        hooks.export_folder = []() { return cyxwiz::ProjectManager::Instance().GetExportsPath(); };
        hooks.used_by = [this, resolve](int, const std::string& saved) {
            if (!node_editor_ || saved.empty()) return std::string();
            const auto file = resolve(saved);
            std::string used_by;
            for (const auto& node : node_editor_->GetNodes()) {
                if (node.type != NodeType::Embedding) continue;
                const auto weights = node.parameters.find("weights_file");
                if (weights == node.parameters.end() || weights->second.empty() || resolve(weights->second) != file) {
                    continue;
                }
                const auto param = [&](const char* key) {
                    const auto it = node.parameters.find(key);
                    return it == node.parameters.end() ? std::string("?") : it->second;
                };
                used_by += (used_by.empty() ? "" : ", ") + node.name + " (" + param("num_embeddings") + " \xC3\x97 " +
                           param("embedding_dim") + ")";
            }
            return used_by.empty() ? used_by : used_by + " \xC2\xB7 loads the file when the graph trains";
        };
        training_plot_panel_->SetWordVectorHooks(std::move(hooks));

        NodeEditor::WordVectorFeed feed;
        const std::weak_ptr<cyxwiz::TrainingPlotPanel> panel = training_plot_panel_;
        feed.figures = [panel](const cyxwiz::OperatorFigures& figures) {
            if (auto p = panel.lock()) p->RecordWordVectorFigures(figures);
        };
        feed.progress = [panel](int node_id, float share, const std::string& message) {
            if (auto p = panel.lock()) p->RecordWordVectorProgress(node_id, share, message);
        };
        feed.ended = [panel](int node_id, bool cancelled, const std::string& error) {
            if (auto p = panel.lock()) p->EndWordVectorRuns(node_id, cancelled, error);
        };
        feed.show_run = [panel](int node_id) {
            if (auto p = panel.lock()) p->ShowWordVectorRun(node_id);
        };
        feed.has_run = [panel](int node_id) {
            const auto p = panel.lock();
            return p && p->HasWordVectorRun(node_id);
        };
        feed.run_line = [panel](int node_id) {
            const auto p = panel.lock();
            return p ? p->WordVectorRunLine(node_id) : std::string();
        };
        node_editor_->SetWordVectorFeed(std::move(feed));
    }

    script_editor_->SetScriptOutputSink(console_.get());

    // Connect Node Editor to Script Editor for code generation output
    node_editor_->SetScriptEditor(script_editor_.get());

    // Connect Node Editor to Properties panel for node selection display
    node_editor_->SetPropertiesPanel(properties_.get());

    // Connect Node Editor to ScriptingEngine for Python-based RL training
    node_editor_->SetScriptingEngine(scripting_engine_.get());

    // Connect Properties panel to Node Editor for shape inference
    properties_->SetNodeEditor(node_editor_.get());
    // A step's input or output opens in the Table Viewer (TOFIX146). The
    // viewer holds rows in memory, so a very large output opens its first rows.
    properties_->SetOpenTableCallback([this](const std::string& dataset_name, const std::string& title) {
        if (!table_viewer_) return;
        auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(dataset_name);
        auto arrow_table = dataset ? dataset->GetArrowTable() : nullptr;
        if (!arrow_table) {
            spdlog::warn("Open in Table Viewer: '{}' is not in memory; run the step again", dataset_name);
            return;
        }
        constexpr int64_t kMaxRows = 200000;
        // Converted in a task; past kMaxRows the tab says "first N of M rows".
        table_viewer_->OpenArrow(arrow_table, title + " (" + dataset_name + ")", kMaxRows);
        table_viewer_->SetVisible(true);
        ::gui::BringToFront(table_viewer_->GetName());
    });

    live_compile_ = std::make_unique<cyxwiz::LiveGraphCompile>();
    properties_->SetLiveCompile(live_compile_.get());
    node_editor_->SetLiveCompile(live_compile_.get());

    // Connect Pattern Browser to Node Editor for proper node creation
    pattern_browser_->SetNodeEditor(node_editor_.get());

    // Connect Query Console to Node Editor for graph queries
    query_console_->SetNodeEditor(node_editor_.get());

    // Set up training callback for Node Editor
    node_editor_->SetCompileCallback([this]() {
        this->CompileGraphAndReport();
    });

    // Local Debug (F6): a background Studio Debugger run (compile gate,
    // preflight, one synthetic batch through DebugExecutor).
    node_editor_->SetDebugCallback([this]() {
        this->LocalDebugGraphAndReport();
    });

    if (studio_debugger_panel_) {
        // One run path for the panel's Run, F6 and Explain Node: the graph is
        // frozen here on the UI thread and the run executes on a worker.
        studio_debugger_panel_->SetRunDebugCallback([this](cyxwiz::StudioDebuggerRunMode mode,
                                                           int sample_index,
                                                           int explain_node_id) {
            std::vector<MLNode> nodes;
            std::vector<NodeLink> links;
            cyxwiz::StudioDebuggerRunInputs inputs;
            inputs.project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
            inputs.rebuild_prepared_data =
                studio_debugger_panel_->ConsumeRebuildPreparedDataRequest();
            if (node_editor_) {
                nodes = node_editor_->GetNodes();
                links = node_editor_->GetLinks();
                for (const auto& [id, position] : node_editor_->GetCachedNodePositions()) {
                    inputs.node_positions[id] = {position.x, position.y};
                }
            }
            return [this, mode, sample_index, explain_node_id,
                    nodes = std::move(nodes),
                    links = std::move(links),
                    inputs = std::move(inputs)](
                       const cyxwiz::StudioDebuggerRunControl& control) mutable {
                cyxwiz::StudioDebuggerSnapshot session;
                this->BuildStudioDebuggerSessionFromSnapshot(
                    session, mode, sample_index, std::move(nodes), std::move(links),
                    explain_node_id, &control, &inputs);
                return session;
            };
        });
        studio_debugger_panel_->SetRunCompletedCallback(
            [this](const cyxwiz::StudioDebuggerSnapshot& session) {
                // Staleness cache for the Train gate: only a successful Local
                // Debug of this exact graph counts.
                if (session.has_debug_result && session.debug_result.success) {
                    last_debug_graph_hash_ = session.graph_hash;
                    have_last_debug_result_ = true;
                    spdlog::info("Studio Debugger: cached debug hash {:#018x}",
                                 last_debug_graph_hash_);
                }
            });
        studio_debugger_panel_->SetGraphDomainCallback([this]() -> std::string {
            if (!node_editor_) {
                return {};
            }
            try {
                cyxwiz::GraphCompiler compiler;
                const auto config = compiler.Compile(node_editor_->GetNodes(),
                                                     node_editor_->GetLinks());
                return cyxwiz::StudioDebuggerDomainLabel(config.preprocessing_domain);
            } catch (const std::exception&) {
                return {};
            }
        });
        studio_debugger_panel_->SetFocusNodeCallback([this](int node_id) {
            if (node_editor_) {
                node_editor_->Show();
                node_editor_->FocusNode(node_id);
            }
        });
    }

    node_editor_->SetExplainNodeCallback([this](int node_id) {
        if (!node_editor_ || !studio_debugger_panel_) {
            return;
        }
        studio_debugger_panel_->Show();
        if (!studio_debugger_panel_->StartRun(
                cyxwiz::StudioDebuggerRunMode::Preflight, 0, node_id)) {
            spdlog::info("Explain Node: a Studio Debugger run is already active");
        }
    });

    node_editor_->SetTrainCallback([this](const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
        this->StartTrainingFromGraph(nodes, links);
        // Show Training Dashboard when training starts
        if (training_plot_panel_) {
            training_plot_panel_->SetVisible(true);
        }
    });

    // Open the Custom Node Editor panel from the Studio toolbar
    node_editor_->SetOpenCustomNodeEditorCallback([this]() {
        if (custom_node_editor_) {
            custom_node_editor_->Show();
            spdlog::info("Opened Custom Node Editor panel from Studio toolbar");
        }
    });

    // Set up callbacks in the toolbar
    toolbar_->SetResetLayoutCallback([this]() {
        this->ResetDockLayout();
    });

    toolbar_->SetSaveLayoutCallback([this]() {
        this->SaveLayout();
    });

    toolbar_->SetSaveProjectSettingsCallback([this]() {
        this->SaveProjectSettings();
    });

    // View > Theme saves through gui::SetThemePreset (Engine-wide).
    toolbar_->SetAppThemeChangedCallback([](int theme_index) {
        spdlog::info("App theme changed: {}", theme_index);
    });

    toolbar_->SetLoadCheckpointCallback([this]() { LoadCheckpointForTesting(); });


    // Data enters only through Data Input nodes: both open a new one's dialog.
    toolbar_->SetImportDatasetCallback([this]() {
        if (node_editor_) node_editor_->AddDataInputAndConfigure();
    });

    toolbar_->SetCreateCustomDatasetCallback([this]() {
        if (node_editor_) node_editor_->AddDataInputAndConfigure();
    });

    // Tokenize - opens existing TokenizationPanel
    toolbar_->SetTokenizeDatasetCallback([this]() {
        if (tokenization_panel_) {
            tokenization_panel_->SetVisible(true);
            spdlog::info("Opened Tokenization panel");
        }
    });

    toolbar_->SetAugmentDatasetCallback([this]() {
        if (node_editor_) {
            node_editor_->Show();
            spdlog::info("Opened Node Editor - use Augmentation nodes for data augmentation");
        }
    });

    // Row counts, columns and value summaries: Data Studio's Profile tab.
    toolbar_->SetDatasetStatisticsCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->ShowProfile(data_studio_panel_->GetActiveDataset());
    });

    // Compile Graph - dry-run the graph compilation and show results in a popup
    toolbar_->SetCompileGraphCallback([this]() {
        CompileGraphAndReport();
    });

    // Local Debug - same action as F6 and the graph toolbar button.
    toolbar_->SetLocalDebugCallback([this]() {
        LocalDebugGraphAndReport();
    });

    // Start Training - compiles and runs training from node editor graph
    toolbar_->SetStartTrainingCallback([this]() {
        if (node_editor_) {
            auto nodes = node_editor_->GetNodes();
            auto links = node_editor_->GetLinks();
            StartTrainingFromGraph(nodes, links);
            if (training_plot_panel_) {
                training_plot_panel_->SetVisible(true);
            }
            spdlog::info("Started training from Train menu");
        }
    });

    // Pause Training
    toolbar_->SetPauseTrainingCallback([]() {
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (tm.IsTrainingActive()) {
            tm.PauseTraining();
            spdlog::info("Paused training");
        }
    });

    // Tools > Monitoring > Clear Cache: the prepared-data cache of this
    // project, after the dashboard's confirmation. Garbage Collection:
    // device buffers back to the driver and a Python gc pass (TOFIX129 G3).
    toolbar_->SetClearCacheCallback([this]() {
        if (training_plot_panel_) training_plot_panel_->RequestClearPreparedDataCache();
    });
    toolbar_->SetRunGCCallback([this]() {
        cyxwiz::ReleaseUnusedDeviceMemory();
        if (scripting_engine_ && !scripting_engine_->IsCommandRunning() && !scripting_engine_->IsScriptRunning()) {
            scripting_engine_->ExecuteCommandAsync("import gc; gc.collect()");
            spdlog::info("Garbage collection: device buffers released, Python gc.collect() requested");
        } else {
            spdlog::info("Garbage collection: device buffers released (Python busy, gc skipped)");
        }
    });

    // File > Save: settings and layout, every open script, and the graph
    // (asks where to put the graph the first time). Owner decision G4.
    toolbar_->SetSaveProjectCallback([this]() {
        SaveProjectSettings();
        if (script_editor_) script_editor_->SaveAllFiles();
        if (node_editor_ && !node_editor_->SaveCurrentGraph()) {
            spdlog::warn("File > Save: the graph was not saved");
        }
    });

    // Resume Training
    toolbar_->SetResumeTrainingCallback([]() {
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (tm.IsTrainingActive() && tm.IsPaused()) {
            tm.ResumeTraining();
            spdlog::info("Resumed training");
        }
    });

    // Stop Training
    toolbar_->SetStopTrainingCallback([]() {
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (tm.IsTrainingActive()) {
            tm.StopTraining();
            spdlog::info("Stopped training");
        }
    });

    // Training Settings - opens Training Dashboard
    toolbar_->SetTrainingSettingsCallback([this]() {
        if (training_plot_panel_) {
            training_plot_panel_->SetVisible(true);
            spdlog::info("Opened Training Dashboard");
        }
    });

    // Optimizer Settings: training takes its optimizer from the graph, so the
    // item selects the graph's optimizer node (its settings show in
    // Properties) instead of a separate window with its own copy.
    toolbar_->SetOptimizerSettingsCallback([this]() {
        const cyxwiz::TrainingConfiguration* config = live_compile_ ? live_compile_->Config() : nullptr;
        if (node_editor_ && config && config->optimizer_node_id >= 0) {
            node_editor_->FocusNode(config->optimizer_node_id);
            return;
        }
        ShowOperationError("No optimizer in this graph",
                           "Training takes its optimizer from the graph. Add one from the Nodes panel "
                           "(Training: Adam, AdamW, SGD, ...), connect it to the loss, and its settings show in "
                           "Properties.");
    });

    // Set up Run Test callback
    toolbar_->SetRunTestCallback([this]() {
        if (node_editor_) {
            auto nodes = node_editor_->GetNodes();
            auto links = node_editor_->GetLinks();
            StartTestingFromGraph(nodes, links);
        }
    });

    // Train > Test > Run Quick Test: the first 10 batches of the test split
    // (TOFIX129 G3); the result says it is partial.
    toolbar_->SetRunQuickTestCallback([this]() {
        if (node_editor_) {
            cyxwiz::TestManager::Instance().SetNextRunBatchLimit(10);
            StartTestingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
        }
    });
    // A test's results are Data Studio > Results > Test (TOFIX129 D4).
    if (cyxwiz::TestView* test = data_studio_panel_ ? data_studio_panel_->GetTestView() : nullptr) {
        test->run_test = [this]() {
            if (node_editor_) StartTestingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
        };
    }
    // Results > Cross-validation / Learning curve train the open graph again (TOFIX129 D4).
    // Their nodes' second pin reads the last run's table from the registry.
    cyxwiz::TrainingEvaluationOperator::SetResultsLookup([](const std::string& name) -> std::shared_ptr<arrow::Table> {
        const auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(name);
        return dataset ? dataset->GetArrowTable() : nullptr;
    });
    if (data_studio_panel_) {
        for (const auto view_kind : {cyxwiz::ResultsView::View::CrossValidation, cyxwiz::ResultsView::View::LearningCurve}) {
            const std::shared_ptr<cyxwiz::FoldsView> view = data_studio_panel_->GetFoldsView(view_kind);
            const std::weak_ptr<cyxwiz::FoldsView> weak = view;
            view->run = [this, weak](const cyxwiz::CrossValidationSettings& settings) {
                if (const auto v = weak.lock()) StartFoldsFromGraph(v, settings);
            };
            view->cancel = [this]() {
                if (folds_task_) cyxwiz::AsyncTaskManager::Instance().Cancel(folds_task_);
            };
            view->run_blocker = [this]() { return FoldsBlocker(); };
            const bool curve = view_kind == cyxwiz::ResultsView::View::LearningCurve;
            view->graph_node = [this, curve]() -> std::optional<cyxwiz::FoldsView::GraphNode> {
                const auto node = FindFoldsNode(curve);
                if (!node) return std::nullopt;
                return cyxwiz::FoldsView::GraphNode{node->name, node->settings, node->error};
            };
            view->add_to_graph = [this, curve](const cyxwiz::CrossValidationSettings& settings) {
                return AddFoldsNode(curve, settings);
            };
        }
    }
    // Train > Evaluate: Results' views, running.
    const auto run_folds = [this](cyxwiz::ResultsView::View kind) {
        if (!data_studio_panel_) return;
        data_studio_panel_->ShowResults(kind);
        if (const auto view = data_studio_panel_->GetFoldsView(kind)) view->RequestRun();
    };
    toolbar_->SetCrossValidateCallback([run_folds]() { run_folds(cyxwiz::ResultsView::View::CrossValidation); });
    toolbar_->SetLearningCurveCallback([run_folds]() { run_folds(cyxwiz::ResultsView::View::LearningCurve); });
    toolbar_->SetCompareTestResultsCallback([this]() {
        if (!data_studio_panel_) return;
        data_studio_panel_->GetTestView()->ShowComparison();
        data_studio_panel_->ShowResults(cyxwiz::ResultsView::View::Test);
    });
    toolbar_->SetExportTestReportCallback([this]() {
        if (!data_studio_panel_ || !data_studio_panel_->GetTestView()->ExportReport())
            ShowOperationError("No test results", "Run a test first, then export its report.");
    });
    toolbar_->SetViewTestResultsCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->ShowResults(cyxwiz::ResultsView::View::Test);
    });

    const std::weak_ptr<int> test_window_lifetime = test_callback_lifetime_;
    cyxwiz::TestManager::Instance().SetOnTestingEnd([this, test_window_lifetime](bool success, const cyxwiz::TestingMetrics& metrics) {
        if (test_window_lifetime.expired()) return;
        if (success && !metrics.causal_lm_mode) {
            // The run's per-sample results as the "test_predictions" table, so
            // Results and the ROC / PR nodes can read them like any dataset;
            // row_id joins them back to the tested table's columns.
            const std::shared_ptr<arrow::Table> table = cyxwiz::PredictionsTableFromMetrics(metrics);
            if (table) {
                cyxwiz::DataRegistry::Instance().RegisterArrowTable(table, cyxwiz::kTestPredictionsDataset);
                spdlog::info("Published {} test predictions as '{}'{}", table->num_rows(),
                             cyxwiz::kTestPredictionsDataset,
                             table->GetColumnByName("row_id") ? " (rows of '" + metrics.source_dataset + "')" : "");
            } else {
                // No per-row results kept: an older run's table must not stand in.
                cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(cyxwiz::kTestPredictionsDataset);
            }
        }
        if (data_studio_panel_) {
            data_studio_panel_->GetTestView()->SetResults(metrics);
            data_studio_panel_->ShowResults(cyxwiz::ResultsView::View::Test);
        }
        if (!success) {
            spdlog::error(
                "Testing failed and was surfaced in Data Studio > Results > Test: {}",
                metrics.status_message.empty()
                    ? "unknown testing failure"
                    : metrics.status_message);
        }
    });

    // Set up Export Model callback
    toolbar_->SetExportModelCallback([this](int format_index) { OpenModelExport(format_index); });

    export_dialog_->SetExportCompleteCallback(
        [this](const cyxwiz::ExportResult& result,
               const cyxwiz::ExportOptions& options,
               uint64_t graph_hash,
               const std::string& source_graph,
               bool optimizer_available,
               bool metrics_available) {
            cyxwiz::DebugArtifactConsistencyInput input;
            input.action = "export";
            input.artifact_kind = cyxwiz::GetFormatName(options.format);
            input.artifact_path = result.output_path;
            input.producer_name = "ModelExporter";
            input.graph_hash = graph_hash;
            input.operation_success = result.success;
            input.operation_status = result.success
                ? "Model export completed."
                : result.error_message;
            input.source_graph_content = source_graph;

            const bool is_cyxmodel =
                options.format == cyxwiz::ModelFormat::CyxModel;
            const bool inspection_supported = is_cyxmodel ||
                options.format == cyxwiz::ModelFormat::Safetensors;
            cyxwiz::ProbeResult probe;
            probe.format = options.format;
            if (result.success && inspection_supported) {
                cyxwiz::ModelImporter importer;
                probe = importer.ProbeFile(result.output_path);
                if (is_cyxmodel && probe.valid && probe.has_graph) {
                    const auto graph = importer.ExtractGraph(result.output_path);
                    if (graph) {
                        input.artifact_graph_content = *graph;
                    }
                }
            }
            input.observed = ObserveModelArtifact(
                result.output_path,
                probe,
                result.success && inspection_supported);

            const bool is_cyxmodel_directory =
                is_cyxmodel &&
                std::filesystem::is_directory(result.output_path);
            input.expected.manifest_expected = is_cyxmodel_directory;
            input.expected.training_config_expected = is_cyxmodel_directory;
            input.expected.weights_manifest_expected = is_cyxmodel_directory;
            input.expected.graph_expected =
                is_cyxmodel && options.include_graph &&
                !source_graph.empty();
            const auto family_it =
                options.custom_metadata.find("model_family");
            const std::string expected_family =
                family_it == options.custom_metadata.end()
                    ? std::string{}
                    : family_it->second;
            const bool tokenizer_required =
                probe.supports_generation || probe.supports_bert_encoder ||
                expected_family == "causal_lm" ||
                expected_family == "bert_encoder";
            input.expected.tokenizer_config_required = tokenizer_required;
            input.expected.tokenizer_vocabulary_required =
                tokenizer_required;
            input.expected.optimizer_state_expected =
                is_cyxmodel && options.include_optimizer_state &&
                optimizer_available;
            input.expected.training_history_expected =
                is_cyxmodel && options.include_training_history &&
                metrics_available;
            if (result.success) {
                input.expected.parameter_count = static_cast<size_t>(
                    std::max(0, result.num_parameters));
                if (is_cyxmodel) {
                    input.expected.layer_count = static_cast<size_t>(
                        std::max(0, result.num_layers));
                }
            }
            const auto expected_input_it =
                options.custom_metadata.find("bert_encoder_input_kind");
            if (expected_input_it != options.custom_metadata.end()) {
                input.expected.input_contract = expected_input_it->second;
            }
            const auto generation_output_it = options.custom_metadata.find(
                "generation_output_contract");
            const auto bert_output_it = options.custom_metadata.find(
                "bert_encoder_output_contract");
            if (generation_output_it != options.custom_metadata.end()) {
                input.expected.output_contract = generation_output_it->second;
            } else if (bert_output_it != options.custom_metadata.end()) {
                input.expected.output_contract = bert_output_it->second;
            }
            RecordArtifactConsistencyTrace(input);
        });

    // File > Open Model File...: reads a model file and opens its graph.
    toolbar_->SetImportModelCallback([this]() {
        if (import_dialog_) import_dialog_->Open();
    });

    import_dialog_->SetOpenGraphCallback(
        [this](const cyxwiz::ProbeResult& probe,
               const std::string& graph_json,
               const std::string& input_path) {
            bool graph_loaded = false;
            if (node_editor_ && node_editor_->LoadGraphFromString(graph_json)) {
                graph_loaded = true;
                node_editor_->Show();
                spdlog::info("Opened the graph of {} (weights not loaded)", input_path);
            } else {
                spdlog::error("Open Model File: Studio could not load the graph of {}", input_path);
            }

            uint64_t graph_hash = 0;
            if (graph_loaded) {
                graph_hash = HashGraphStructure(
                    node_editor_->GetNodes(), node_editor_->GetLinks());
            }
            cyxwiz::DebugArtifactConsistencyInput input;
            input.action = "import_inspection";
            input.artifact_kind = cyxwiz::GetFormatName(probe.format);
            input.artifact_path = input_path;
            input.producer_name = "ModelImporter";
            input.graph_hash = graph_hash;
            input.operation_success = graph_loaded;
            input.operation_status = graph_loaded
                ? "Opened the model file's graph; weights were not loaded."
                : "Studio could not load the model file's graph.";
            input.artifact_graph_content = graph_json;
            const bool inspection_supported =
                probe.format == cyxwiz::ModelFormat::CyxModel ||
                probe.format == cyxwiz::ModelFormat::Safetensors;
            input.observed = ObserveModelArtifact(
                input_path, probe, inspection_supported);
            const bool is_cyxmodel =
                probe.format == cyxwiz::ModelFormat::CyxModel;
            const bool is_cyxmodel_directory =
                is_cyxmodel &&
                std::filesystem::is_directory(input_path);
            input.expected.manifest_expected = is_cyxmodel_directory;
            input.expected.training_config_expected = is_cyxmodel_directory;
            input.expected.weights_manifest_expected = is_cyxmodel_directory;
            input.expected.graph_expected = is_cyxmodel && probe.has_graph;
            input.expected.tokenizer_config_required =
                probe.supports_generation || probe.supports_bert_encoder;
            input.expected.tokenizer_vocabulary_required =
                probe.supports_generation || probe.supports_bert_encoder;
            input.expected.optimizer_state_expected = false;
            input.expected.training_history_expected = false;
            RecordArtifactConsistencyTrace(input);
            return graph_loaded;
        });

    // Set up Model Conversion callbacks
    toolbar_->SetConvertBinaryToDirCallback([]() {
        spdlog::info("Binary to Directory conversion requested");

        // Step 1: Select input binary file
        auto input_result = cyxwiz::FileDialogs::OpenModel();
        if (!input_result) {
            spdlog::info("Binary to Directory conversion cancelled");
            return;
        }

        std::string input_path = *input_result;

        // Verify it's a binary file
        if (!cyxwiz::ModelConverter::IsBinaryFormat(input_path)) {
            spdlog::error("Selected file is not a binary .cyxmodel format");
            return;
        }

        // Step 2: Select output folder
        auto output_result = cyxwiz::FileDialogs::SelectOutputFolder();
        if (!output_result) {
            spdlog::info("Binary to Directory conversion cancelled (output selection)");
            return;
        }

        // Generate output path
        std::filesystem::path in_path(input_path);
        std::string output_path = *output_result + "/" + in_path.stem().string() + ".cyxmodel";

        spdlog::info("Converting binary '{}' to directory '{}'", input_path, output_path);

        if (cyxwiz::ModelConverter::BinaryToDirectory(input_path, output_path)) {
            spdlog::info("Successfully converted to directory format: {}", output_path);
        } else {
            spdlog::error("Conversion failed: {}", cyxwiz::ModelConverter::GetLastError());
        }
    });

    toolbar_->SetConvertDirToBinaryCallback([]() {
        spdlog::info("Directory to Binary conversion requested");

        // Step 1: Select input directory
        auto input_result = cyxwiz::FileDialogs::SelectFolder("Select .cyxmodel Directory");
        if (!input_result) {
            spdlog::info("Directory to Binary conversion cancelled");
            return;
        }

        std::string input_path = *input_result;

        // Verify it's a directory format
        if (!cyxwiz::ModelConverter::IsDirectoryFormat(input_path)) {
            spdlog::error("Selected path is not a valid .cyxmodel directory format");
            return;
        }

        // Step 2: Select output file
        auto output_result = cyxwiz::FileDialogs::SaveModel();
        if (!output_result) {
            spdlog::info("Directory to Binary conversion cancelled (output selection)");
            return;
        }

        std::string output_path = *output_result;

        spdlog::info("Converting directory '{}' to binary '{}'", input_path, output_path);

        if (cyxwiz::ModelConverter::DirectoryToBinary(input_path, output_path)) {
            spdlog::info("Successfully converted to binary format: {}", output_path);
        } else {
            spdlog::error("Conversion failed: {}", cyxwiz::ModelConverter::GetLastError());
        }
    });

    // Set up Deploy to Server callback
    toolbar_->SetDeployToServerCallback([this]() {
        if (deployment_dialog_) {
            deployment_dialog_->Open();
            spdlog::info("Opened Deployment dialog");
        }
    });

    // Set up Custom Node Editor callback
    toolbar_->SetOpenCustomNodeEditorCallback([this]() {
        if (custom_node_editor_) {
            custom_node_editor_->Show();
            spdlog::info("Opened Custom Node Editor panel");
        }
    });

    // Set up Icon Pack selection callbacks
    toolbar_->SetSetIconPackCallback([this](int pack) {
        if (node_editor_) {
            node_editor_->SetIconPack(static_cast<gui::IconPack>(pack));
            const char* pack_names[] = {"FontAwesome", "Tabler", "Remix", "Lucide", "Iconoir", "Phosphor"};
            spdlog::info("Changed icon pack to: {}", pack < 6 ? pack_names[pack] : "Unknown");
        }
    });
    toolbar_->SetGetIconPackCallback([this]() -> int {
        return node_editor_ ? static_cast<int>(node_editor_->GetIconPack()) : 0;
    });

    // Set up Node Editor operation callbacks (Nodes menu)
    toolbar_->SetAddDenseNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::Dense, "Dense (128)");
        }
    });
    toolbar_->SetAddConvNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::Conv2D, "Conv2D (32)");
        }
    });
    toolbar_->SetAddPoolingNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::MaxPool2D, "MaxPool2D");
        }
    });
    toolbar_->SetAddDropoutNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::Dropout, "Dropout (0.5)");
        }
    });
    toolbar_->SetAddBatchNormNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::BatchNorm, "BatchNorm");
        }
    });
    toolbar_->SetAddAttentionNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::MultiHeadAttention, "MultiHeadAttention");
        }
    });
    toolbar_->SetDuplicateNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->DuplicateSelectedNodes();
        }
    });
    toolbar_->SetDeleteNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->DeleteSelectedNodes();
        }
    });
    toolbar_->SetGroupNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->GroupSelectedNodes();
        }
    });
    toolbar_->SetUngroupNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->UngroupSelectedNodes();
        }
    });

    // Set up Theme Editor callback
    toolbar_->SetOpenThemeEditorCallback([this]() {
        if (theme_editor_) {
            theme_editor_->Show();
            spdlog::info("Opened Theme Editor panel");
        }
    });

    // Set up Profiler callback
    toolbar_->SetOpenProfilerCallback([this]() {
        if (studio_debugger_panel_) {
            studio_debugger_panel_->ShowRuntimeProfile();
            spdlog::info(
                "Opened canonical Studio Debugger runtime profile");
        }
    });

    // Set up Memory Monitor callback (System Monitor in Profile menu)
    toolbar_->SetOpenMemoryMonitorCallback([this]() {
        if (memory_monitor_) {
            memory_monitor_->Toggle();
            spdlog::info("Toggled Memory Monitor");
        }
    });

    // Set up Memory Panel callback (Memory Visualization in Profile menu)
    toolbar_->SetOpenMemoryPanelCallback([this]() {
        if (memory_panel_) {
            memory_panel_->Show();
            spdlog::info("Opened Memory Visualization panel");
        }
    });

    // Set up Compute Device selection callback (Preferences -> Device tab)
    toolbar_->SetComputeDeviceChangedCallback(
        [](cyxwiz::DeviceType type,
           int device_id,
           const std::string& physical_fingerprint,
           std::string& error) {
            const auto result = cyxwiz::CommitExecutionDeviceSelection(
                {type, device_id, physical_fingerprint},
                [](const cyxwiz::PendingExecutionDeviceSelection& selection) {
                    cyxwiz::PreferredComputeRoute preferred;
                    preferred.type = selection.type;
                    preferred.last_device_id = selection.device_id;
                    preferred.physical_fingerprint =
                        selection.physical_fingerprint;
                    std::string persistence_error;
                    if (!cyxwiz::UpdatePreferredComputeRouteAtomic(
                            cyxwiz::GetComputeRuntimeConfigPath(),
                            preferred,
                            persistence_error)) {
                        throw std::runtime_error(persistence_error);
                    }
                    cyxwiz::CommitExecutionDeviceSelectionState(selection);
                });
            if (!result.committed) {
                error = "stage=" + std::string(
                    cyxwiz::DeviceSelectionTransactionStageName(result.stage)) +
                    " reason=" + result.message;
                spdlog::warn(
                    "Compute device selection was not committed: {}", error);
                return false;
            }
            spdlog::info(
                "Committed compute device for next training run: backend={} device={}",
                cyxwiz::ExecutionDeviceSelectionBackendName(type),
                device_id);
            return true;
        });

    // Set up Verbose Python Logging pointer (View menu - Developer Tools)
    if (scripting_engine_) {
        toolbar_->SetVerbosePythonLogPtr(scripting_engine_->GetVerboseLoggingPtr());
        toolbar_->SetPythonDiagnosticsCallback([this]() -> std::string {
            return scripting_engine_->GetPythonRuntimeDiagnostics();
        });
    }

    // Set up Model Analysis callbacks (Tools menu - Phase 2)
    toolbar_->SetOpenModelSummaryCallback([this]() {
        if (model_summary_panel_) {
            model_summary_panel_->Toggle();
            if (model_summary_panel_->IsVisible()) {
                model_summary_panel_->RefreshAnalysis();
            }
            spdlog::info("Toggled Model Summary panel");
        }
    });

    toolbar_->SetOpenArchitectureDiagramCallback([this]() {
        if (architecture_diagram_) {
            architecture_diagram_->Toggle();
            if (architecture_diagram_->IsVisible()) {
                architecture_diagram_->RefreshDiagram();
            }
            spdlog::info("Toggled Architecture Diagram panel");
        }
    });

    toolbar_->SetOpenLRFinderCallback([this]() {
        if (lr_finder_panel_) {
            lr_finder_panel_->Toggle();
            spdlog::info("Toggled Learning Rate Finder panel");
        }
    });

    // Data > Explore and Tools > Statistics > Summary Statistics / Distribution
    // Fitter: Data Studio's Explore tabs (TOFIX129 D1; the old windows are gone).
    toolbar_->SetOpenCorrelationMatrixCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Correlation);
    });

    toolbar_->SetOpenMissingValuePanelCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Missing);
    });

    toolbar_->SetOpenOutlierDetectionCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Outliers);
    });

    // Set up Statistics callbacks (Tools menu - Phase 4)
    toolbar_->SetOpenDescriptiveStatsCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Distribution);
    });

    toolbar_->SetOpenHypothesisTestCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Tests);
    });

    toolbar_->SetOpenDistributionFitterCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Distribution);
    });

    toolbar_->SetOpenRegressionCallback([this]() {
        if (data_studio_panel_)
            data_studio_panel_->ShowExplore(data_studio_panel_->GetActiveDataset(), cyxwiz::DataStudioPanel::ExploreTab::Regression);
    });

    // Advanced Tools callbacks (Phase 5)
    toolbar_->SetOpenDimReductionCallback([this]() {
        if (dim_reduction_panel_) {
            dim_reduction_panel_->Toggle();
            spdlog::info("Toggled Dimensionality Reduction panel");
        }
    });

    toolbar_->SetOpenGradCAMCallback([this]() {
        if (gradcam_panel_) {
            gradcam_panel_->Toggle();
            spdlog::info("Toggled Grad-CAM panel");
        }
    });

    toolbar_->SetOpenFeatureImportanceCallback([this]() {
        if (feature_importance_panel_) {
            feature_importance_panel_->Toggle();
            spdlog::info("Toggled Feature Importance panel");
        }
    });

    toolbar_->SetOpenNASCallback([this]() {
        if (nas_panel_) {
            nas_panel_->Toggle();
            spdlog::info("Toggled Neural Architecture Search panel");
        }
    });

    toolbar_->SetOpenHyperparamSearchCallback([this]() {
        if (hyperparam_search_panel_) {
            hyperparam_search_panel_->Toggle();
            spdlog::info("Toggled Hyperparameter Search panel");
        }
    });

    toolbar_->SetShowPluginNodesCallback([this](const std::string& plugin_id) {
        if (node_browser_panel_) node_browser_panel_->ShowPluginNodes(plugin_id);
    });
    toolbar_->SetOpenServingCallback([this]() {
        if (serving_panel_) {
            serving_panel_->Toggle();
            spdlog::info("Toggled Model Serving panel");
        }
    });

    toolbar_->SetOpenDNNInferenceCallback([this]() {
        if (dnn_inference_panel_) {
            dnn_inference_panel_->Toggle();
            spdlog::info("Toggled DNN Inference panel");
        }
    });

    // Clustering callbacks (Phase 6A)
    toolbar_->SetOpenKMeansCallback([this]() {
        if (kmeans_panel_) {
            kmeans_panel_->Toggle();
            spdlog::info("Toggled K-Means Clustering panel");
        }
    });

    toolbar_->SetOpenDBSCANCallback([this]() {
        if (dbscan_panel_) {
            dbscan_panel_->Toggle();
            spdlog::info("Toggled DBSCAN panel");
        }
    });

    toolbar_->SetOpenHierarchicalCallback([this]() {
        if (hierarchical_panel_) {
            hierarchical_panel_->Toggle();
            spdlog::info("Toggled Hierarchical Clustering panel");
        }
    });

    toolbar_->SetOpenGMMCallback([this]() {
        if (gmm_panel_) {
            gmm_panel_->Toggle();
            spdlog::info("Toggled GMM panel");
        }
    });

    toolbar_->SetOpenClusterEvalCallback([this]() {
        if (cluster_eval_panel_) {
            cluster_eval_panel_->Toggle();
            spdlog::info("Toggled Cluster Evaluation panel");
        }
    });

    // Tools > Evaluation: Confusion Matrix, ROC / AUC and PR Curve are Data
    // Studio > Results > Test (TOFIX129 D4); with no test yet it opens on
    // A table..., which reads any table of predictions.
    const auto open_test_view = [this]() {
        if (!data_studio_panel_) return;
        if (!data_studio_panel_->GetTestView()->HasResults()) data_studio_panel_->GetTestView()->ShowTable();
        data_studio_panel_->ShowResults(cyxwiz::ResultsView::View::Test);
    };
    toolbar_->SetOpenConfusionMatrixCallback(open_test_view);
    toolbar_->SetOpenROCAUCCallback(open_test_view);
    toolbar_->SetOpenPRCurveCallback(open_test_view);

    toolbar_->SetOpenCrossValidationCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->ShowResults(cyxwiz::ResultsView::View::CrossValidation);
    });

    toolbar_->SetOpenLearningCurvesCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->ShowResults(cyxwiz::ResultsView::View::LearningCurve);
    });

    // Data > Transform: a step in Data Studio's Prepare (TOFIX129 D2; the old windows are gone).
    toolbar_->SetOpenNormalizationCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->AddTransformStep(gui::NodeType::MinMaxScaler);
    });
    toolbar_->SetOpenStandardizationCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->AddTransformStep(gui::NodeType::StandardScaler);
    });
    toolbar_->SetOpenLogTransformCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->AddTransformStep(gui::NodeType::LogTransform);
    });
    toolbar_->SetOpenBoxCoxCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->AddTransformStep(gui::NodeType::PowerTransform, {{"family", "box-cox"}});
    });
    toolbar_->SetOpenFeatureScalingCallback([this]() {
        if (data_studio_panel_) data_studio_panel_->AddTransformStep(gui::NodeType::StandardScaler, {}, true);
    });

    // Linear Algebra callbacks (Phase 7)
    toolbar_->SetOpenMatrixCalculatorCallback([this]() {
        if (matrix_calculator_panel_) {
            matrix_calculator_panel_->Toggle();
            spdlog::info("Toggled Matrix Calculator panel");
        }
    });

    toolbar_->SetOpenEigenDecompCallback([this]() {
        if (eigen_decomp_panel_) {
            eigen_decomp_panel_->Toggle();
            spdlog::info("Toggled Eigen Decomposition panel");
        }
    });

    toolbar_->SetOpenSVDCallback([this]() {
        if (svd_panel_) {
            svd_panel_->Toggle();
            spdlog::info("Toggled SVD panel");
        }
    });

    toolbar_->SetOpenQRCallback([this]() {
        if (qr_panel_) {
            qr_panel_->Toggle();
            spdlog::info("Toggled QR Decomposition panel");
        }
    });

    toolbar_->SetOpenCholeskyCallback([this]() {
        if (cholesky_panel_) {
            cholesky_panel_->Toggle();
            spdlog::info("Toggled Cholesky Decomposition panel");
        }
    });

    // Signal Processing callbacks (Phase 8)
    toolbar_->SetOpenFFTCallback([this]() {
        if (fft_panel_) {
            fft_panel_->Toggle();
            spdlog::info("Toggled FFT panel");
        }
    });
    toolbar_->SetOpenSpectrogramCallback([this]() {
        if (spectrogram_panel_) {
            spectrogram_panel_->Toggle();
            spdlog::info("Toggled Spectrogram panel");
        }
    });
    toolbar_->SetOpenFilterDesignerCallback([this]() {
        if (filter_designer_panel_) {
            filter_designer_panel_->Toggle();
            spdlog::info("Toggled Filter Designer panel");
        }
    });
    toolbar_->SetOpenConvolutionCallback([this]() {
        if (convolution_panel_) {
            convolution_panel_->Toggle();
            spdlog::info("Toggled Convolution panel");
        }
    });
    toolbar_->SetOpenWaveletCallback([this]() {
        if (wavelet_panel_) {
            wavelet_panel_->Toggle();
            spdlog::info("Toggled Wavelet Transform panel");
        }
    });

    // Optimization & Calculus callbacks (Phase 9)
    toolbar_->SetOpenGradientDescentCallback([this]() {
        if (gradient_descent_panel_) {
            gradient_descent_panel_->Toggle();
            spdlog::info("Toggled Gradient Descent panel");
        }
    });
    toolbar_->SetOpenConvexityCallback([this]() {
        if (convexity_panel_) {
            convexity_panel_->Toggle();
            spdlog::info("Toggled Convexity Analyzer panel");
        }
    });
    toolbar_->SetOpenLPCallback([this]() {
        if (lp_panel_) {
            lp_panel_->Toggle();
            spdlog::info("Toggled Linear Programming panel");
        }
    });
    toolbar_->SetOpenQPCallback([this]() {
        if (qp_panel_) {
            qp_panel_->Toggle();
            spdlog::info("Toggled Quadratic Programming panel");
        }
    });
    toolbar_->SetOpenDifferentiationCallback([this]() {
        if (differentiation_panel_) {
            differentiation_panel_->Toggle();
            spdlog::info("Toggled Numerical Differentiation panel");
        }
    });
    toolbar_->SetOpenIntegrationCallback([this]() {
        if (integration_panel_) {
            integration_panel_->Toggle();
            spdlog::info("Toggled Numerical Integration panel");
        }
    });

    // Time Series Analysis callbacks (Phase 10)
    toolbar_->SetOpenDecompositionCallback([this]() {
        if (decomposition_panel_) {
            decomposition_panel_->Toggle();
            spdlog::info("Toggled Time Series Decomposition panel");
        }
    });
    toolbar_->SetOpenACFPACFCallback([this]() {
        if (acf_pacf_panel_) {
            acf_pacf_panel_->Toggle();
            spdlog::info("Toggled ACF/PACF panel");
        }
    });
    toolbar_->SetOpenStationarityCallback([this]() {
        if (stationarity_panel_) {
            stationarity_panel_->Toggle();
            spdlog::info("Toggled Stationarity Testing panel");
        }
    });
    toolbar_->SetOpenSeasonalityCallback([this]() {
        if (seasonality_panel_) {
            seasonality_panel_->Toggle();
            spdlog::info("Toggled Seasonality Detection panel");
        }
    });
    toolbar_->SetOpenForecastingCallback([this]() {
        if (forecasting_panel_) {
            forecasting_panel_->Toggle();
            spdlog::info("Toggled Forecasting panel");
        }
    });

    // Text Processing callbacks (Phase 11)
    toolbar_->SetOpenTokenizationCallback([this]() {
        if (tokenization_panel_) {
            tokenization_panel_->SetVisible(true);
            spdlog::info("Opened Tokenization panel");
        }
    });
    toolbar_->SetOpenWordFrequencyCallback([this]() {
        if (word_frequency_panel_) {
            word_frequency_panel_->SetVisible(true);
            spdlog::info("Opened Word Frequency panel");
        }
    });
    toolbar_->SetOpenTFIDFCallback([this]() {
        if (tfidf_panel_) {
            tfidf_panel_->SetVisible(true);
            spdlog::info("Opened TF-IDF panel");
        }
    });
    toolbar_->SetOpenEmbeddingsCallback([this]() {
        if (embeddings_panel_) {
            embeddings_panel_->SetVisible(true);
            spdlog::info("Opened Embeddings panel");
        }
    });
    toolbar_->SetOpenSentimentCallback([this]() {
        if (sentiment_panel_) {
            sentiment_panel_->SetVisible(true);
            spdlog::info("Opened Sentiment Analysis panel");
        }
    });
    toolbar_->SetOpenLanguageModelGenerationCallback([this]() {
        if (language_model_generation_panel_) {
            language_model_generation_panel_->SetVisible(true);
            spdlog::info("Opened Language Model Generation panel");
        }
    });

    // Utilities panel callbacks (Phase 12)
    toolbar_->SetOpenCalculatorCallback([this]() {
        if (calculator_panel_) {
            calculator_panel_->SetVisible(true);
            spdlog::info("Opened Calculator panel");
        }
    });
    toolbar_->SetOpenUnitConverterCallback([this]() {
        if (unit_converter_panel_) {
            unit_converter_panel_->SetVisible(true);
            spdlog::info("Opened Unit Converter panel");
        }
    });
    toolbar_->SetOpenRandomGeneratorCallback([this]() {
        if (random_generator_panel_) {
            random_generator_panel_->SetVisible(true);
            spdlog::info("Opened Random Generator panel");
        }
    });
    toolbar_->SetOpenHashGeneratorCallback([this]() {
        if (hash_generator_panel_) {
            hash_generator_panel_->SetVisible(true);
            spdlog::info("Opened Hash Generator panel");
        }
    });
    toolbar_->SetOpenJSONViewerCallback([this]() {
        if (json_viewer_panel_) {
            json_viewer_panel_->SetVisible(true);
            spdlog::info("Opened JSON Viewer panel");
        }
    });
    toolbar_->SetOpenRegexTesterCallback([this]() {
        if (regex_tester_panel_) {
            regex_tester_panel_->SetVisible(true);
            spdlog::info("Opened Regex Tester panel");
        }
    });

    // Set minimap visibility pointers for View -> Minimaps menu
    toolbar_->SetNodeEditorMinimapPtr(node_editor_->GetShowMinimapPtr());
    toolbar_->SetScriptEditorMinimapPtr(script_editor_->GetShowMinimapPtr());

    // Register callbacks with ProjectManager for project lifecycle events.
    auto& lifecycle_project_manager = cyxwiz::ProjectManager::Instance();
    lifecycle_project_manager.SetOnProjectOpened([this](const std::string& project_root) {
        this->OnProjectOpened(project_root);
    });

    lifecycle_project_manager.SetOnProjectClosed([this](const std::string& project_root) {
        this->OnProjectClosed(project_root);
    });

    lifecycle_project_manager.SetOnProjectVenvReady([this](const std::string& project_root) {
        this->OnProjectVenvReady(project_root);
    });

    // The project stays open while its graph trains, tests or runs: closing it
    // cancelled the run's tasks and dropped its datasets without a word.
    lifecycle_project_manager.SetCloseGuard([this](const std::string& operation) {
        const std::string reason = node_editor_ ? node_editor_->GetGraphReplacementBlockReason() : std::string();
        if (reason.empty()) return true;
        ShowOperationError("Cannot " + operation,
                           "The project stays open while " + reason +
                               ". Stop it, or wait for it to finish, then try again.");
        return false;
    });

    // Start Page can open a project before MainWindow exists. Synchronize the
    // newly registered observers because the subsequent same-project open is
    // intentionally idempotent and does not emit another callback.
    if (lifecycle_project_manager.HasActiveProject()) {
        OnProjectOpened(lifecycle_project_manager.GetProjectRoot());
    }

    // Set up Open Script in Editor callback (called with file path to open)
    toolbar_->SetOpenScriptInEditorCallback([this](const std::string& file_path) { OpenScriptFile(file_path); });

    // Script > Run Script / Stop Script (the same actions as F5 / Shift+F5 in the editor)
    toolbar_->SetRunScriptCallback([this]() {
        if (script_editor_) {
            script_editor_->SetVisible(true);
            script_editor_->RunScript();
        }
    });
    toolbar_->SetStopScriptCallback([this]() {
        if (script_editor_) script_editor_->StopScript();
    });

    toolbar_->SetFrameOverlayPtr(FrameMetrics::Instance().enabled_ptr());

    // Menu bar state: focus window, selection, training and script state
    // (TOFIX129). Read by the menus, the palette and the shortcut handler.
    toolbar_->SetMenuStateProvider([this]() {
        cyxwiz::ToolbarPanel::MenuStateSnapshot s;
        switch (KeyboardShortcutManager::Instance().GetActiveContext()) {
            case KeyboardContext::NodeEditor: s.focus = cyxwiz::menu::Context::StudioCanvas; break;
            case KeyboardContext::ScriptEditor:
            case KeyboardContext::CompletionPopup: s.focus = cyxwiz::menu::Context::ScriptEditor; break;
            default: break;
        }
        if (node_editor_) s.selected_nodes = node_editor_->GetSelectedNodeCount();
        auto& tm = cyxwiz::TrainingManager::Instance();
        s.training = !tm.IsTrainingActive() ? cyxwiz::menu::TrainingState::Idle
                     : (tm.IsPaused() ? cyxwiz::menu::TrainingState::Paused : cyxwiz::menu::TrainingState::Running);
        if (script_editor_) {
            s.script_open = script_editor_->HasEditableScript();
            s.script_running = script_editor_->IsScriptRunning();
        }
        return s;
    });

    // Set up Open Python Console callback
    toolbar_->SetOpenPythonConsoleCallback([this]() {
        if (console_) {
            if (console_->ActivatePythonRepl()) {
                spdlog::info("Opened Python REPL in Console");
            } else {
                spdlog::warn("Python REPL requires an active project");
            }
        }
    });

    // Set up Save All callback
    toolbar_->SetSaveAllCallback([this]() {
        SaveAllFiles();
        spdlog::info("All files saved via toolbar");
    });

    // Set up Exit callback (called when user confirms exit)
    toolbar_->SetExitCallback([this]() {
        if (exit_request_callback_) {
            exit_request_callback_();
        }
    });


    // Edit > Find / Replace: the Script Editor's inline widget (TOFIX133 P2).
    toolbar_->SetOpenFindCallback([this](bool replace) {
        if (script_editor_) script_editor_->OpenFind(replace);
    });
    // Replace in Files: editor state (TOFIX129 G3)
    toolbar_->SetFileHasUnsavedChangesCallback([this](const std::string& path) {
        return script_editor_ && script_editor_->HasUnsavedChangesFor(path);
    });
    toolbar_->SetReloadOpenFileCallback([this](const std::string& path) {
        if (script_editor_) script_editor_->ReloadOpenFile(path);
    });

    // Set up edit operation callbacks
    toolbar_->SetUndoCallback([this]() {
        if (script_editor_) {
            script_editor_->Undo();
        }
    });

    toolbar_->SetRedoCallback([this]() {
        if (script_editor_) {
            script_editor_->Redo();
        }
    });

    toolbar_->SetCutCallback([this]() {
        if (script_editor_) {
            script_editor_->Cut();
        }
    });

    toolbar_->SetCopyCallback([this]() {
        if (script_editor_) {
            script_editor_->Copy();
        }
    });

    toolbar_->SetPasteCallback([this]() {
        if (script_editor_) {
            script_editor_->Paste();
        }
    });

    toolbar_->SetDeleteCallback([this]() {
        if (script_editor_) {
            script_editor_->Delete();
        }
    });

    toolbar_->SetSelectAllCallback([this]() {
        if (script_editor_) {
            script_editor_->SelectAll();
        }
    });

    // Set up comment toggle callbacks
    toolbar_->SetToggleLineCommentCallback([this]() {
        if (script_editor_) {
            script_editor_->ToggleLineComment();
        }
    });

    toolbar_->SetToggleBlockCommentCallback([this]() {
        if (script_editor_) {
            script_editor_->ToggleBlockComment();
        }
    });

    // Set up Go to Line callback
    if (script_editor_) script_editor_->SetGoToLineRequest([this]() { if (toolbar_) toolbar_->OpenGoToLineDialog(); });
    if (script_editor_)
        script_editor_->SetOpenImageCallback([this](const std::vector<unsigned char>& png, const std::string& title) {
            if (plot_output_panel_) plot_output_panel_->ShowImage(png, title);
        });
    if (variable_explorer_) {
        // Variable Explorer (TOFIX133 P5): the session or an open notebook;
        // View data opens the value in the Table Viewer.
        auto& view = variable_explorer_->View();
        view.scopes = [this]() {
            std::vector<cyxwiz::VariablesView::Scope> out{{"", "Python session", "scripts and Console"}};
            if (script_editor_)
                for (auto& s : script_editor_->NotebookScopes()) out.push_back(std::move(s));
            return out;
        };
        view.on_open_table = [this](const cyxwiz::VariablesView::OpenRequest& request,
                                    const scripting::VariablesService::Result& result) {
            if (!table_viewer_) return;
            table_viewer_->OpenVariable(request, result);
            ::gui::BringToFront(table_viewer_->GetName());
        };
        view.on_insert_name = [this](const std::string& name) {
            if (script_editor_) script_editor_->InsertTextAtCursor(name);
        };
    }
    if (script_editor_) {
        // Notebook table results open in the Table Viewer (TOFIX133 P4 board 5).
        script_editor_->SetOpenVariableCallback([this](const cyxwiz::VariablesView::OpenRequest& request,
                                                       const scripting::VariablesService::Result& result) {
            if (!table_viewer_) return;
            table_viewer_->OpenVariable(request, result);
            ::gui::BringToFront(table_viewer_->GetName());
        });
        script_editor_->SetOpenTableCallback([this](std::shared_ptr<cyxwiz::DataTable> table) {
            if (!table_viewer_) return;
            table_viewer_->SetTable(std::move(table));
            table_viewer_->SetVisible(true);
            ::gui::BringToFront(table_viewer_->GetName());
        });
    }
    toolbar_->SetGoToLineCallback([this](int line) {
        if (script_editor_) {
            script_editor_->GoToLine(line);
        }
    });

    // Set up line operation callbacks
    toolbar_->SetDuplicateLineCallback([this]() {
        if (script_editor_) {
            script_editor_->DuplicateLine();
        }
    });

    toolbar_->SetMoveLineUpCallback([this]() {
        if (script_editor_) {
            script_editor_->MoveLineUp();
        }
    });

    toolbar_->SetMoveLineDownCallback([this]() {
        if (script_editor_) {
            script_editor_->MoveLineDown();
        }
    });

    toolbar_->SetIndentCallback([this]() {
        if (script_editor_) {
            script_editor_->Indent();
        }
    });

    toolbar_->SetOutdentCallback([this]() {
        if (script_editor_) {
            script_editor_->Outdent();
        }
    });

    // Set up text transformation callbacks
    toolbar_->SetTransformUppercaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToUppercase();
        }
    });

    toolbar_->SetTransformLowercaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToLowercase();
        }
    });

    toolbar_->SetTransformTitleCaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToTitleCase();
        }
    });

    // Set up sort and join lines callbacks
    toolbar_->SetSortLinesAscCallback([this]() {
        if (script_editor_) {
            script_editor_->SortLinesAscending();
        }
    });

    toolbar_->SetSortLinesDescCallback([this]() {
        if (script_editor_) {
            script_editor_->SortLinesDescending();
        }
    });

    toolbar_->SetJoinLinesCallback([this]() {
        if (script_editor_) {
            script_editor_->JoinLines();
        }
    });

    // Set up editor settings callbacks (Preferences -> Script Editor synchronization)

    toolbar_->SetEditorTabSizeCallback([this](int tab_size) {
        if (script_editor_) {
            script_editor_->SetTabSize(tab_size);
            spdlog::info("Editor tab size changed to {}", tab_size);
        }
    });

    toolbar_->SetEditorFontScaleCallback([](float scale) {
        gui::SetCodeTextScale(scale);
        spdlog::info("Code text scale changed to {}", scale);
    });

    toolbar_->SetEditorShowWhitespaceCallback([this](bool show) {
        if (script_editor_) {
            script_editor_->SetShowWhitespace(show);
            spdlog::info("Editor show whitespace changed to {}", show);
        }
    });

    toolbar_->SetEditorWordWrapCallback([this](bool wrap) {
        if (script_editor_) {
            script_editor_->SetWordWrap(wrap);
            spdlog::info("Editor word wrap changed to {}", wrap);
        }
    });

    toolbar_->SetEditorAutoIndentCallback([this](bool indent) {
        if (script_editor_) {
            script_editor_->SetAutoIndent(indent);
            spdlog::info("Editor auto indent changed to {}", indent);
        }
    });

    // Initialize toolbar editor settings from script editor's current values
    if (script_editor_) {
        toolbar_->SetEditorTabSize(script_editor_->GetTabSize());
        toolbar_->SetEditorFontScale(script_editor_->GetFontScale());
        toolbar_->SetEditorShowWhitespace(script_editor_->GetShowWhitespace());
        toolbar_->SetEditorWordWrap(script_editor_->GetWordWrap());
        toolbar_->SetEditorAutoIndent(script_editor_->GetAutoIndent());

        // Set up callback for when settings change in Script Editor (View menu)
        // This syncs changes back to the Preferences dialog and saves to project
        script_editor_->SetOnSettingsChangedCallback([this]() {
            if (toolbar_ && script_editor_) {
                toolbar_->SetEditorTabSize(script_editor_->GetTabSize());
                toolbar_->SetEditorFontScale(script_editor_->GetFontScale());
                toolbar_->SetEditorShowWhitespace(script_editor_->GetShowWhitespace());
                toolbar_->SetEditorWordWrap(script_editor_->GetWordWrap());
                toolbar_->SetEditorAutoIndent(script_editor_->GetAutoIndent());
            }
            // Save settings to project file immediately
            SaveProjectSettings();
        });
    }

    // Set up asset browser double-click callback to open files in script editor
    asset_browser_->SetOnAssetDoubleClick([this](const cyxwiz::AssetBrowserPanel::AssetItem& item) {
        if (!item.is_directory && script_editor_) {
            std::string ext = std::filesystem::path(item.absolute_path).extension().string();
            for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

            // Python sources and notebooks (.py, .pyw, .cyx, .ipynb) and other
            // text files open in the Script Editor.
            if (cyxwiz::scriptfile::IsPythonSource(item.absolute_path) || ext == ".txt" ||
                ext == ".json" || ext == ".md" || ext == ".csv" || ext == ".tsv" ||
                ext == ".yaml" || ext == ".yml" || ext == ".toml" ||
                ext == ".ini" || ext == ".cfg" || ext == ".conf") {
                script_editor_->SetVisible(true);  // a closed Script Editor opens
                script_editor_->OpenFile(item.absolute_path);
                spdlog::info("Opened file in script editor: {}", item.name);
            }
        }
    });

    asset_browser_->SetOnCreateDataInput([this](const std::string& path) {
        if (node_editor_) node_editor_->AddDataInputFromAsset(path);
    });

    // A loaded dataset opens in Data Studio's Query tab (TOFIX140).
    asset_browser_->SetOnOpenInDataStudio([this](const std::string& dataset) {
        if (!data_studio_panel_) return;
        data_studio_panel_->ShowQuery(dataset, "SELECT * FROM \"" + dataset + "\" LIMIT 100");
    });

    // Set up asset browser callback for "Open in Node Editor" (.cyxgraph files)
    asset_browser_->SetOnOpenInNodeEditor([this](const std::string& path) {
        OpenGraphInNodeEditor(path);
    });

    // Initialize startup script manager
    startup_script_manager_ = std::make_unique<scripting::StartupScriptManager>(scripting_engine_);

    // Load and execute startup scripts
    if (startup_script_manager_->LoadConfig()) {
        spdlog::debug("Executing startup scripts...");
        startup_script_manager_->ExecuteAll(console_.get());
    } else {
        spdlog::debug("No startup scripts configured or startup_scripts.txt not found");
    }

    // Install custom dock node handler for Unreal-style tabs
    DockStyle::InstallCustomHandler();
    auto& dock_style = GetDockStyle();
    dock_style.SetSidebarPosition(gui::SidebarOnLeft() ? gui::SidebarPosition::Left
                                                        : gui::SidebarPosition::Right);
    dock_style.SetSidebarAutoHide(false);

    // Initialize Tutorial System
    auto& tutorial = cyxwiz::TutorialSystem::Instance();
    tutorial.SetWindowRectCallback([](const std::string& window_name) -> ImRect {
        ImGuiWindow* window = ImGui::FindWindowByName(window_name.c_str());
        if (window) {
            return ImRect(window->Pos, ImVec2(window->Pos.x + window->Size.x, window->Pos.y + window->Size.y));
        }
        return ImRect();
    });
    tutorial.LoadProgress();

    // Check for first-launch welcome
    if (tutorial.ShouldShowWelcome()) {
        tutorial.MarkWelcomeShown();
        tutorial.SaveProgress();
        // Optionally start the getting started tutorial automatically
        // tutorial.StartTutorial("getting_started");
    }

    // Register panels with sidebar for hide/unhide toggles. Their state
    // now is the built-in default that Reset to Default returns to.
    RegisterPanelsWithSidebar();
    for (const auto& panel : GetDockStyle().GetPanels())
        if (panel.visible_ptr) default_panel_visibility_[panel.name] = *panel.visible_ptr;

    // Set default panel visibility (hides tool panels on first launch)
    SetDefaultPanelVisibility();
    // Then the panels open in the last session (TOFIX129 0.6).
    ApplyRememberedPanels();

    spdlog::debug("MainWindow initialized with docking layout system");
}

void MainWindow::OpenScriptFile(const std::string& file_path) {
    if (!script_editor_) return;
    script_editor_->OpenFile(file_path);
    script_editor_->SetVisible(true);  // Show the Script Editor panel
    spdlog::info("Opened script in editor: {}", file_path);
}

MainWindow::~MainWindow() {
    spdlog::info("MainWindow destructor: starting cleanup");
    cyxwiz::ProjectManager::Instance().SetCloseGuard(nullptr);
    cyxwiz::python_ui::SetProvider(nullptr);
    cyxwiz::python_ui::SetOpenCodeHandler(nullptr);
    test_callback_lifetime_.reset();
    if (test_preparation_task_) test_preparation_task_->RequestCancel();
    auto& test_manager = cyxwiz::TestManager::Instance();
    test_manager.SetOnTestingEnd(nullptr);
    test_manager.StopTesting();
    test_manager.WaitForTestingStop();


    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    if (training_mgr.IsTrainingActive()) {
        spdlog::info("~MainWindow: stopping active training before panel cleanup");
        training_mgr.StopTraining();
        training_mgr.WaitForTrainingStop();
    }

    // Persist placement evidence gathered this session (covers compile-time
    // probe results even when no training ran).

    spdlog::info("~MainWindow: training_plot_panel_ (before scripting_engine)");
    training_plot_panel_.reset();

    // IMPORTANT: Destroy panels that hold shared_ptr<ScriptingEngine> BEFORE
    // the MainWindow's scripting_engine_, so all references are released
    // before we begin any Python cleanup
    spdlog::info("~MainWindow: console_ (holds scripting_engine)");
    console_.reset();
    spdlog::info("~MainWindow: script_editor_ (holds scripting_engine)");
    script_editor_.reset();
    spdlog::info("~MainWindow: variable_explorer_ (holds scripting_engine)");
    variable_explorer_.reset();
    spdlog::info("~MainWindow: plot_output_panel_ (holds scripting_engine)");
    plot_output_panel_.reset();
    spdlog::info("~MainWindow: startup_script_manager_ (holds scripting_engine)");
    startup_script_manager_.reset();

    // NOW reset MainWindow's scripting_engine_ - it should be the last reference
    spdlog::info("~MainWindow: scripting_engine_");
    scripting_engine_.reset();
    spdlog::info("~MainWindow: regex_tester_panel_");
    regex_tester_panel_.reset();
    spdlog::info("~MainWindow: json_viewer_panel_");
    json_viewer_panel_.reset();
    spdlog::info("~MainWindow: hash_generator_panel_");
    hash_generator_panel_.reset();
    spdlog::info("~MainWindow: random_generator_panel_");
    random_generator_panel_.reset();
    spdlog::info("~MainWindow: unit_converter_panel_");
    unit_converter_panel_.reset();
    spdlog::info("~MainWindow: calculator_panel_");
    calculator_panel_.reset();
    spdlog::info("~MainWindow: sentiment_panel_");
    sentiment_panel_.reset();
    spdlog::info("~MainWindow: embeddings_panel_");
    embeddings_panel_.reset();
    spdlog::info("~MainWindow: tfidf_panel_");
    tfidf_panel_.reset();
    spdlog::info("~MainWindow: word_frequency_panel_");
    word_frequency_panel_.reset();
    spdlog::info("~MainWindow: tokenization_panel_");
    tokenization_panel_.reset();
    spdlog::info("~MainWindow: forecasting_panel_");
    forecasting_panel_.reset();
    spdlog::info("~MainWindow: seasonality_panel_");
    seasonality_panel_.reset();
    spdlog::info("~MainWindow: stationarity_panel_");
    stationarity_panel_.reset();
    spdlog::info("~MainWindow: acf_pacf_panel_");
    acf_pacf_panel_.reset();
    spdlog::info("~MainWindow: decomposition_panel_");
    decomposition_panel_.reset();
    spdlog::info("~MainWindow: integration_panel_");
    integration_panel_.reset();
    spdlog::info("~MainWindow: differentiation_panel_");
    differentiation_panel_.reset();
    spdlog::info("~MainWindow: qp_panel_");
    qp_panel_.reset();
    spdlog::info("~MainWindow: lp_panel_");
    lp_panel_.reset();
    spdlog::info("~MainWindow: convexity_panel_");
    convexity_panel_.reset();
    spdlog::info("~MainWindow: gradient_descent_panel_");
    gradient_descent_panel_.reset();
    spdlog::info("~MainWindow: wavelet_panel_");
    wavelet_panel_.reset();
    spdlog::info("~MainWindow: convolution_panel_");
    convolution_panel_.reset();
    spdlog::info("~MainWindow: filter_designer_panel_");
    filter_designer_panel_.reset();
    spdlog::info("~MainWindow: spectrogram_panel_");
    spectrogram_panel_.reset();
    spdlog::info("~MainWindow: fft_panel_");
    fft_panel_.reset();
    spdlog::info("~MainWindow: cholesky_panel_");
    cholesky_panel_.reset();
    spdlog::info("~MainWindow: qr_panel_");
    qr_panel_.reset();
    spdlog::info("~MainWindow: svd_panel_");
    svd_panel_.reset();
    spdlog::info("~MainWindow: eigen_decomp_panel_");
    eigen_decomp_panel_.reset();
    spdlog::info("~MainWindow: matrix_calculator_panel_");
    matrix_calculator_panel_.reset();
    spdlog::info("~MainWindow: cluster_eval_panel_");
    cluster_eval_panel_.reset();
    spdlog::info("~MainWindow: gmm_panel_");
    gmm_panel_.reset();
    spdlog::info("~MainWindow: hierarchical_panel_");
    hierarchical_panel_.reset();
    spdlog::info("~MainWindow: dbscan_panel_");
    dbscan_panel_.reset();
    spdlog::info("~MainWindow: kmeans_panel_");
    kmeans_panel_.reset();
    spdlog::info("~MainWindow: serving_panel_");
    serving_panel_.reset();
    spdlog::info("~MainWindow: hyperparam_search_panel_");
    hyperparam_search_panel_.reset();
    spdlog::info("~MainWindow: nas_panel_");
    nas_panel_.reset();
    spdlog::info("~MainWindow: feature_importance_panel_");
    feature_importance_panel_.reset();
    spdlog::info("~MainWindow: gradcam_panel_");
    gradcam_panel_.reset();
    spdlog::info("~MainWindow: dim_reduction_panel_");
    dim_reduction_panel_.reset();
    spdlog::info("~MainWindow: lr_finder_panel_");
    lr_finder_panel_.reset();
    spdlog::info("~MainWindow: architecture_diagram_");
    architecture_diagram_.reset();
    spdlog::info("~MainWindow: model_summary_panel_");
    model_summary_panel_.reset();
    spdlog::info("~MainWindow: deployment_dialog_");
    deployment_dialog_.reset();
    spdlog::info("~MainWindow: import_dialog_");
    import_dialog_.reset();
    spdlog::info("~MainWindow: export_dialog_");
    export_dialog_.reset();
    // variable_explorer_ already reset at the beginning
    spdlog::info("~MainWindow: memory_monitor_");
    memory_monitor_.reset();
    spdlog::info("~MainWindow: memory_panel_");
    memory_panel_.reset();
    spdlog::info("~MainWindow: theme_editor_");
    theme_editor_.reset();
    spdlog::info("~MainWindow: custom_node_editor_");
    custom_node_editor_.reset();
    spdlog::info("~MainWindow: query_console_");
    query_console_.reset();
    spdlog::info("~MainWindow: pattern_browser_");
    pattern_browser_.reset();
    spdlog::info("~MainWindow: task_progress_panel_");
    task_progress_panel_.reset();
    spdlog::info("~MainWindow: wallet_panel_");
    wallet_panel_.reset();
    spdlog::info("~MainWindow: p2p_training_panel_");
    p2p_training_panel_.reset();
    spdlog::info("~MainWindow: job_status_panel_");
    job_status_panel_.reset();
    spdlog::info("~MainWindow: connection_dialog_");
    connection_dialog_.reset();
    spdlog::info("~MainWindow: table_viewer_");
    table_viewer_.reset();
    // script_editor_ and console_ already reset with the scripting-engine panels.
    // training_plot_panel_ already reset at the beginning
    spdlog::info("~MainWindow: asset_browser_");
    asset_browser_.reset();
    spdlog::info("~MainWindow: toolbar_");
    toolbar_.reset();
    spdlog::info("~MainWindow: properties_");
    properties_.reset();
    spdlog::info("~MainWindow: viewport_");
    viewport_.reset();
    spdlog::info("~MainWindow: node_editor_");
    node_editor_.reset();

    spdlog::info("~MainWindow: all panels destroyed");
}

void MainWindow::SetIdleLogPtr(bool* ptr) {
    if (toolbar_) {
        toolbar_->SetIdleLogPtr(ptr);
    }
}

bool MainWindow::OpenGraphInNodeEditor(const std::string& path) {
    if (!node_editor_) {
        return false;
    }

    if (!node_editor_->LoadGraph(path)) {
        spdlog::error("Failed to open graph file: {}", path);
        return false;
    }

    node_editor_->Show();
    spdlog::info("Opened graph in Node Editor: {}", std::filesystem::path(path).filename().string());
    return true;
}

void MainWindow::SetNetworkComponents(network::GRPCClient* client, network::JobManager* job_manager) {
    // Store job manager reference
    job_manager_ = job_manager;

    // Create connection dialog with network components
    connection_dialog_ = std::make_unique<cyxwiz::ConnectionDialog>(client, job_manager);

    // Connect NodeEditor so ConnectionDialog can access the model graph
    if (connection_dialog_ && node_editor_) {
        connection_dialog_->SetNodeEditor(node_editor_.get());
    }

    // Create and wire up ReservationClient for node reservation (async connect to avoid blocking startup)
    if (connection_dialog_ && client) {
        auto reservation_client = std::make_shared<network::ReservationClient>();
        // Set auth token from AuthClient if authenticated
        auto& auth = cyxwiz::auth::AuthClient::Instance();
        if (auth.IsAuthenticated()) {
            reservation_client->SetAuthToken(auth.GetJwtToken());
        }
        // Connect asynchronously - won't block the main thread
        reservation_client->ConnectAsync(cyxwiz::core::EngineConfig::Instance().GetCentralServerAddress());
        connection_dialog_->SetReservationClient(reservation_client);
        spdlog::debug("ReservationClient wired to ConnectionDialog (connecting in background)");
    }

    // Create a shared P2PClient for reservation-based connections
    if (connection_dialog_) {
        auto p2p_client = std::make_shared<network::P2PClient>();
        connection_dialog_->SetP2PClient(p2p_client);
        spdlog::debug("P2PClient created and wired to ConnectionDialog");
    }

    // Wire up P2PTrainingPanel to ConnectionDialog
    if (connection_dialog_ && p2p_training_panel_) {
        connection_dialog_->SetP2PTrainingPanel(p2p_training_panel_.get());
        spdlog::debug("P2PTrainingPanel wired to ConnectionDialog");
    }

    // Set JobManager for JobStatusPanel
    if (job_status_panel_) {
        job_status_panel_->SetJobManager(job_manager);
        job_manager->SetJobStatusPanel(job_status_panel_.get());
    }

    // Set up callback in toolbar to show connection dialog
    if (toolbar_) {
        toolbar_->SetConnectToServerCallback([this]() {
            if (connection_dialog_) {
                connection_dialog_->Show();
            }
        });

        // Set up auth callbacks to propagate JWT token to gRPC client
        toolbar_->SetOnLoginSuccessCallback([client](const std::string& jwt_token) {
            if (client) {
                client->SetAuthToken(jwt_token);
                spdlog::info("Auth token set on gRPC client after login");
            }
        });

        toolbar_->SetOnLogoutCallback([client]() {
            if (client) {
                client->ClearAuthToken();
                spdlog::info("Auth token cleared from gRPC client after logout");
            }
        });
    }

    spdlog::debug("Network components set in MainWindow");
}

void MainWindow::SetDataStreamClient(network::DataStreamClient* client) {
    datastream_client_ = client;

    // Pass DataStreamClient to cloud panels
    if (cloud_browser_panel_) {
        cloud_browser_panel_->SetDataStreamClient(client);
        spdlog::info("DataStreamClient set on CloudBrowserPanel");
    }
    if (cloud_dataset_manager_panel_) {
        cloud_dataset_manager_panel_->SetDataStreamClient(client);
        spdlog::info("DataStreamClient set on CloudDatasetManagerPanel");
    }

    spdlog::info("DataStreamClient set in MainWindow");
}

void MainWindow::StartJobMonitoring(const std::string& job_id) {
    spdlog::info("Starting P2P monitoring for job: {}", job_id);

    monitoring_job_id_ = job_id;

    // Note: The P2PClient won't be available immediately - it's created when
    // the job is assigned to a node. For now, we'll start monitoring with
    // the job_id, and the P2PTrainingPanel will connect when the client is ready.

    if (p2p_training_panel_) {
        // Start monitoring - this sets up the panel state
        // The P2PClient will be connected later when available
        p2p_training_panel_->StartMonitoring(job_id, "");
        p2p_training_panel_->Show();

        spdlog::info("P2PTrainingPanel now monitoring job: {}", job_id);
    }

    // Try to get the P2PClient if it's already available
    if (job_manager_) {
        auto p2p_client = job_manager_->GetP2PClient(job_id);
        if (p2p_client && p2p_training_panel_) {
            p2p_training_panel_->SetP2PClient(p2p_client);
            spdlog::info("P2PClient connected to monitoring panel");
        }
    }
}

bool MainWindow::IsScriptRunning() const {
    if (scripting_engine_) {
        return scripting_engine_->IsScriptRunning();
    }
    return false;
}

void MainWindow::StopRunningScript() {
    if (scripting_engine_) {
        scripting_engine_->StopScript();
    }
}

bool MainWindow::HasUnsavedFiles() const {
    if (script_editor_) {
        return script_editor_->HasUnsavedFiles();
    }
    return false;
}

std::vector<std::string> MainWindow::GetUnsavedFileNames() const {
    if (script_editor_) {
        return script_editor_->GetUnsavedFileNames();
    }
    return {};
}

void MainWindow::SaveAllFiles() {
    if (script_editor_) {
        script_editor_->SaveAllFiles();
    }
}

void MainWindow::PrepareForShutdown() {
    spdlog::info("MainWindow: Preparing for shutdown...");

    // Stop any running script first (this can take time to interrupt Python)
    if (scripting_engine_ && scripting_engine_->IsScriptRunning()) {
        spdlog::info("MainWindow: Stopping running script...");
        scripting_engine_->StopScript();
    }

    // Stop P2P training monitoring and cancel any model downloads
    if (p2p_training_panel_) {
        if (p2p_training_panel_->IsDownloading()) {
            spdlog::info("MainWindow: Cancelling model download...");
            p2p_training_panel_->CancelDownloadAndWait();
        }
        if (p2p_training_panel_->IsMonitoring()) {
            spdlog::info("MainWindow: Stopping P2P monitoring...");
            p2p_training_panel_->StopMonitoring();
        }
    }

    // Stop inference server if running
    if (serving_panel_ && serving_panel_->GetServer()) {
        spdlog::info("MainWindow: Stopping inference server...");
        serving_panel_->GetServer()->Stop();
    }

    spdlog::info("MainWindow: Shutdown preparation complete");
}

void MainWindow::ResetDockLayout() {
    // Rebuild the default docking layout on the next frame, even when a
    // layout already exists (first_time_layout_ alone keeps a split layout).
    reset_layout_requested_ = true;
    // The default panels open again, and only those.
    for (const auto& panel : GetDockStyle().GetPanels()) {
        const auto it = default_panel_visibility_.find(panel.name);
        if (panel.visible_ptr && it != default_panel_visibility_.end()) *panel.visible_ptr = it->second;
    }
    SetDefaultPanelVisibility(true);
    ForgetRememberedPanels();
    ::gui::MoveAllWindowsBack();
    spdlog::info("Dock layout reset requested");
}

void MainWindow::Render() {
    FrameMetrics::Instance().BeginFrame();
    TrackPanelChanges();
    // Handle global keyboard shortcuts
    HandleGlobalShortcuts();

    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    PollDataInputLoads();
    PollTrainingPipelineRun();
    // Dataset labels for the catalog: each Data Input node's name for its
    // dataset, so Data Studio and queries say "Spotify", not ds_datainput_1,
    // and each node's name for the table a pipeline run made from it.
    static double labels_at = -1.0;
    if (node_editor_ && ImGui::GetTime() - labels_at > 0.5) {
        labels_at = ImGui::GetTime();
        std::map<std::string, std::string> labels, targets, sources;
        // Data Inputs that share a name ("Data Input" x 3) are told apart by id.
        std::map<std::string, int> input_names;
        for (const auto& n : node_editor_->GetNodes())
            if (n.type == gui::NodeType::DataInput) ++input_names[n.name];
        for (const auto& n : node_editor_->GetNodes()) {
            if (n.type != gui::NodeType::DataInput) {
                // PipelineExecutor names a node's output ds_operator_<Type>_<id>.
                const std::string type = cyxwiz::PipelineTypeName(n.type);
                if (type != "Unknown" && !n.name.empty()) labels.emplace("ds_operator_" + type + "_" + std::to_string(n.id), n.name);
                continue;
            }
            std::vector<std::string> names{"ds_datainput_" + std::to_string(n.id)};  // read for a plot or a run (PipelineExecutor)
            auto it = n.parameters.find("dataset_name");
            if (it != n.parameters.end() && !it->second.empty()) names.push_back(it->second);
            auto label = n.parameters.find("label_column");
            auto file = n.parameters.find("file_path");
            const std::string shown = input_names[n.name] > 1 ? n.name + " #" + std::to_string(n.id) : n.name;
            for (const auto& dataset : names) {
                if (!n.name.empty()) labels.emplace(dataset, shown);
                if (label != n.parameters.end() && !label->second.empty()) targets.emplace(dataset, label->second);
                if (file != n.parameters.end() && !file->second.empty()) sources.emplace(dataset, file->second);
            }
        }
        // Every node a run reached, by the result name the run recorded (legacy
        // nodes name theirs ds_sql_<id>, ds_filter_<id>, ...; recipe steps are
        // not on the canvas): its node's name, " #id" when two share it.
        {
            std::map<std::string, int> step_names;
            std::vector<std::pair<std::string, const gui::MLNode*>> made;
            for (const auto& [id, facts] : node_editor_->GetNodeRunFacts()) {
                const gui::MLNode* step = node_editor_->FindStep(id);
                if (!step || step->type == gui::NodeType::DataInput || facts.output_dataset.empty() || step->name.empty()) continue;
                made.push_back({facts.output_dataset, step});
                ++step_names[step->name];
            }
            for (const auto& [dataset, step] : made)
                labels[dataset] = step_names[step->name] > 1 ? step->name + " #" + std::to_string(step->id) : step->name;
        }
        cyxwiz::DatasetCatalog::Instance().SetTargets(std::move(targets));
        cyxwiz::DatasetCatalog::Instance().SetSources(std::move(sources));
        static size_t labels_seen = 0;
        if (labels.size() != labels_seen) {
            labels_seen = labels.size();
            spdlog::info("Dataset catalog: {} dataset label(s) from the graph", labels_seen);
        }
        cyxwiz::DatasetCatalog::Instance().SetLabels(std::move(labels));
    }
    UpdatePythonNodes();
    UpdateLiveCompile();

    auto& plugin_manager = cyxwiz::plugin::PluginManager::Instance();
    if (plugin_manager.GetPluginCount() > 0) {
        plugin_manager.SetAssistantContextSnapshotForAll(
            BuildAssistantContextSnapshot(
                node_editor_.get(), studio_debugger_panel_.get()));
    }

    // Check if we need to connect P2PClient to monitoring panel
    if (!monitoring_job_id_.empty() && job_manager_ && p2p_training_panel_) {
        auto p2p_client = job_manager_->GetP2PClient(monitoring_job_id_);
        if (p2p_client) {
            // Check if P2PTrainingPanel doesn't have the client yet
            // (SetP2PClient is idempotent so we can call it multiple times safely)
            p2p_training_panel_->SetP2PClient(p2p_client);
        }
    }

    RenderDockSpace();

    // Render Unreal-style sidebar for panel toggles
    RenderSidebar();

    // Render new panel system - Toolbar (replaces old menu bar)
    TimedRender("toolbar", toolbar_);

    // Render new panels
    TimedRender("asset browser", asset_browser_);
    TimedRender("training plot", training_plot_panel_);  // Now "Training Dashboard"
    TimedRender("script editor", script_editor_);
    TimedRender("table viewer", table_viewer_);
    TimedRender("annotation editor", annotation_editor_panel_);
    TimedRender("connection dialog", connection_dialog_);
    TimedRender("job status", job_status_panel_);
    TimedRender("p2p training", p2p_training_panel_);
    TimedRender("wallet", wallet_panel_);
    TimedRender("task progress", task_progress_panel_);
    TimedRender("pattern browser", pattern_browser_);
    TimedRender("query console", query_console_);
    TimedRender("custom node editor", custom_node_editor_);
    TimedRender("theme editor", theme_editor_);
    TimedRender("memory", memory_panel_);
    TimedRender("memory monitor", memory_monitor_);
    cyxwiz::VariablesView::PollAll(scripting_engine_.get());  // results for every Variables view
    TimedRender("variable explorer", variable_explorer_);
    TimedRender("plot output", plot_output_panel_);
    TimedRender("export dialog", export_dialog_);
    TimedRender("import dialog", import_dialog_);
    TimedRender("deployment dialog", deployment_dialog_);
    TimedRender("studio debugger", studio_debugger_panel_);

    // Render Model Analysis panels (Phase 2)
    TimedRender("model summary", model_summary_panel_);
    TimedRender("architecture diagram", architecture_diagram_);
    TimedRender("lr finder", lr_finder_panel_);

    // Render Statistics panels (Phase 4)

    // Render Advanced Tools panels (Phase 5)
    TimedRender("dim reduction", dim_reduction_panel_);
    TimedRender("gradcam", gradcam_panel_);
    TimedRender("feature importance", feature_importance_panel_);
    TimedRender("nas", nas_panel_);
    TimedRender("hyperparam search", hyperparam_search_panel_);
    TimedRender("serving", serving_panel_);
    TimedRender("dnn inference", dnn_inference_panel_);

    // Render Clustering panels (Phase 6A)
    TimedRender("kmeans", kmeans_panel_);
    TimedRender("dbscan", dbscan_panel_);
    TimedRender("hierarchical", hierarchical_panel_);
    TimedRender("gmm", gmm_panel_);
    TimedRender("cluster eval", cluster_eval_panel_);

    // Render Model Evaluation panels (Phase 6B)

    // Linear Algebra panels (Phase 7)
    TimedRender("matrix calculator", matrix_calculator_panel_);
    TimedRender("eigen decomp", eigen_decomp_panel_);
    TimedRender("svd", svd_panel_);
    TimedRender("qr", qr_panel_);
    TimedRender("cholesky", cholesky_panel_);

    // Signal Processing panels (Phase 8)
    TimedRender("fft", fft_panel_);
    TimedRender("spectrogram", spectrogram_panel_);
    TimedRender("filter designer", filter_designer_panel_);
    TimedRender("convolution", convolution_panel_);
    TimedRender("wavelet", wavelet_panel_);

    // Optimization & Calculus panels (Phase 9)
    TimedRender("gradient descent", gradient_descent_panel_);
    TimedRender("convexity", convexity_panel_);
    TimedRender("lp", lp_panel_);
    TimedRender("qp", qp_panel_);
    TimedRender("differentiation", differentiation_panel_);
    TimedRender("integration", integration_panel_);

    // Time Series Analysis panels (Phase 10)
    TimedRender("decomposition", decomposition_panel_);
    TimedRender("acf pacf", acf_pacf_panel_);
    TimedRender("stationarity", stationarity_panel_);
    TimedRender("seasonality", seasonality_panel_);
    TimedRender("forecasting", forecasting_panel_);

    // Text Processing panels (Phase 11)
    TimedRender("tokenization", tokenization_panel_);
    TimedRender("word frequency", word_frequency_panel_);
    TimedRender("tfidf", tfidf_panel_);
    TimedRender("embeddings", embeddings_panel_);
    TimedRender("sentiment", sentiment_panel_);
    TimedRender("language model generation", language_model_generation_panel_);

    // Render Utilities panels (Phase 12)
    TimedRender("calculator", calculator_panel_);
    TimedRender("unit converter", unit_converter_panel_);
    TimedRender("random generator", random_generator_panel_);
    TimedRender("hash generator", hash_generator_panel_);
    TimedRender("json viewer", json_viewer_panel_);
    TimedRender("regex tester", regex_tester_panel_);
    
    // Cloud panels
    TimedRender("cloud browser", cloud_browser_panel_);
    TimedRender("node browser", node_browser_panel_);
    TimedRender("node info", node_info_panel_);
    TimedRender("cloud dataset manager", cloud_dataset_manager_panel_);

    // Plugin Manager panel
    TimedRender("plugin manager", plugin_manager_panel_);

    // Data Studio panel (Phase 1 Week 1)
    TimedRender("data studio", data_studio_panel_);

    // Render plugin-provided panels
    cyxwiz::plugin::PluginPanelRegistry::Instance().RenderAllVisible();

    // Render permission approval dialogs
    cyxwiz::plugin::PluginManager::Instance().RenderPermissionDialogs();
    cyxwiz::python_ui::RenderApprovalDialog();

    // Render original panels
    TimedRender("node editor", node_editor_);
    TimedRender("console", console_);
    TimedRender("viewport", viewport_);
    TimedRender("properties", properties_);

    // Render the Compile Graph result popup (triggered from Train -> Compile Graph menu)
    RenderCompileResultPopup();
    RenderOperationErrorPopup();
    RenderMaterializationMemoryConfirmationPopup();

    // Render tutorial overlay (on top of all panels)
    cyxwiz::TutorialSystem::Instance().Render();

    // Render status bar at the bottom of the screen
    FrameMetrics::Instance().Render();
    FrameMetrics::Instance().EndFrame();
    RenderStatusBar();
}

// Helper function to draw active tab indicator on a dock node
static void DrawDockNodeActiveTabIndicator(ImGuiDockNode* node) {
    if (!node) return;

    // Recursively process child nodes
    if (node->ChildNodes[0]) DrawDockNodeActiveTabIndicator(node->ChildNodes[0]);
    if (node->ChildNodes[1]) DrawDockNodeActiveTabIndicator(node->ChildNodes[1]);

    // Only process leaf nodes with tab bars
    if (!node->TabBar || node->Windows.Size == 0) return;

    ImGuiTabBar* tab_bar = node->TabBar;
    const DockTabStyle& style = GetDockStyle().GetStyle();

    // Only draw if we have an active tab and indicator is enabled
    if (!style.show_active_indicator || tab_bar->VisibleTabId == 0) return;

    ImGuiTabItem* active_tab = ImGui::TabBarFindTabByID(tab_bar, tab_bar->VisibleTabId);
    if (!active_tab) return;

    // Get the draw list for the host window
    ImGuiWindow* host_window = node->HostWindow;
    if (!host_window) return;

    ImDrawList* draw_list = host_window->DrawList;

    // Calculate tab position relative to the tab bar
    ImVec2 tab_bar_min = tab_bar->BarRect.Min;
    ImVec2 tab_min = ImVec2(tab_bar_min.x + active_tab->Offset, tab_bar_min.y);
    ImVec2 tab_max = ImVec2(tab_min.x + active_tab->Width, tab_bar_min.y + style.active_indicator_height);

    // Draw the indicator line at the TOP of the active tab
    ImU32 indicator_color = ImGui::ColorConvertFloat4ToU32(style.active_indicator_color);
    draw_list->AddRectFilled(tab_min, tab_max, indicator_color);
}

void MainWindow::UpdatePythonNodes() {
    // Bundles: every 2 s, cheap (file names, sizes and times; node.json is
    // read only when something changed).
    const double now = ImGui::GetTime();
    if (python_nodes_checked_at_ < 0.0 || now - python_nodes_checked_at_ >= 2.0) {
        python_nodes_checked_at_ = now;
        if (!python_node_provider_) {
            python_node_provider_ =
                std::make_unique<cyxwiz::PythonNodeProvider>(cyxwiz::PythonNodeApprovals::User());
            cyxwiz::python_ui::SetProvider(python_node_provider_.get());
            cyxwiz::python_ui::SetOpenCodeHandler([this](const std::filesystem::path& path) {
                OpenScriptFile(path.string());
            });
        }
        std::vector<std::filesystem::path> roots;
        auto& projects = cyxwiz::ProjectManager::Instance();
        if (projects.HasActiveProject()) {
            roots.push_back(cyxwiz::PythonNodeProvider::ProjectNodesFolder(projects.GetProjectRoot()));
        }
        roots.push_back(cyxwiz::UserPythonNodesFolder());
        for (const auto* plugin : cyxwiz::plugin::PluginManager::Instance().GetAllPlugins()) {
            if (plugin) roots.push_back(plugin->plugin_dir / "nodes");
        }
        python_node_provider_->Refresh(roots);
    }

    // Python itself: started here, on the UI thread, when a Python node
    // needed it (the factories never start it).
    auto& runtime = cyxwiz::PythonNodeRuntime::Instance();
    if (!runtime.Wanted() || !scripting_engine_) return;
    runtime.ClearWanted();
    if (scripting_engine_->IsInitialized()) return;
    std::string error;
    if (scripting_engine_->StartPython(&error)) {
        spdlog::info("Python started for Python nodes");
        python_start_error_logged_.clear();
        if (live_compile_) live_compile_->Invalidate();  // build and count the Python nodes now
    } else if (error != python_start_error_logged_) {
        python_start_error_logged_ = error;
        spdlog::warn("Python nodes need Python, which did not start: {}", error);
    }
}

void MainWindow::UpdateLiveCompile() {
    if (!live_compile_ || !node_editor_) return;
    // A Data Pipeline graph has no model: no training compile (it runs with
    // Execute Pipeline or a Plot node).
    if (node_editor_->GetExecutionMode() == gui::ExecutionMode::DuckDBPipeline) return;
    // Counting parameters builds the model once; not while training uses
    // the device.
    const bool training = cyxwiz::TrainingManager::Instance().IsTrainingActive();
    live_compile_->project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
    live_compile_->Update(node_editor_->GetNodes(), node_editor_->GetLinks(), !training);
    if (live_compile_->ResultSerial() == live_compile_serial_seen_ || !live_compile_->Config()) return;
    live_compile_serial_seen_ = live_compile_->ResultSerial();
    // Pins and links show this compile, as the Compile button does.
    node_editor_->SetAllNodesPinState(NodeEditor::NodePinState::CompilePassed);
    for (const auto& issue : live_compile_->Config()->issues) {
        if (issue.level == cyxwiz::IssueLevel::Error && issue.node_id >= 0) {
            node_editor_->SetNodePinState(issue.node_id, NodeEditor::NodePinState::CompileFailed);
        }
    }
}

void MainWindow::RenderDockSpace() {
    static bool opt_fullscreen = false;  // Set to false to show native title bar with window controls
    static bool opt_padding = false;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    // Status bar height constant - must match RenderStatusBar()
    const float status_bar_height = StatusBarHeight();

    // Always get viewport to fill the available space, minus status bar
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    // A pinned panel-toggle sidebar gets its own strip instead of being
    // drawn over the docked panels on that side.
    const DockStyle& dock_style = GetDockStyle();
    const float sidebar_reserve = dock_style.PinnedSidebarReserve();
    const bool sidebar_left = dock_style.GetSidebarPosition() == SidebarPosition::Left;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + (sidebar_left ? sidebar_reserve : 0.0f),
                                   viewport->WorkPos.y));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x - sidebar_reserve,
                                    viewport->WorkSize.y - status_bar_height));
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    if (opt_fullscreen) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    }

    if (!opt_padding)
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace", nullptr, window_flags);
    if (!opt_padding)
        ImGui::PopStyleVar();

    if (opt_fullscreen)
        ImGui::PopStyleVar(2);

    // DockSpace
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

        // After a fresh layout, bring the main tab of each area to the front
        // (otherwise the panel that opened last wins its slot). New dock tabs
        // select themselves in their first frames, and a dock node selects
        // only the tab focused while it updates, so this starts after a few
        // frames and focuses one window per frame, CyxWiz Studio last.
        if (front_tabs_countdown_ > 0) {
            static constexpr const char* kFrontTabs[] = {"Asset Browser", "Console", "Data Studio", "CyxWiz Studio"};
            constexpr int kCount = static_cast<int>(sizeof(kFrontTabs) / sizeof(kFrontTabs[0]));
            --front_tabs_countdown_;
            if (front_tabs_countdown_ < kCount) {
                const char* name = kFrontTabs[kCount - 1 - front_tabs_countdown_];
                if (ImGui::FindWindowByName(name)) ImGui::SetWindowFocus(name);
                // A file opened from the command line stays in front.
                if (front_tabs_countdown_ == 0 && script_editor_ && script_editor_->IsVisible() &&
                    script_editor_->HasOpenFiles() && ImGui::FindWindowByName(script_editor_->GetName()))
                    ImGui::SetWindowFocus(script_editor_->GetName());
            }
        }

        // View > Layout > Reset to Default: rebuild over the current layout
        if (reset_layout_requested_) {
            BuildInitialDockLayout();
            reset_layout_requested_ = false;
            first_time_layout_ = false;
        }

        // Build the initial layout ONLY if no saved layout exists
        if (first_time_layout_) {
            // Check if dockspace node already exists (loaded from imgui.ini)
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(dockspace_id);
            if (node == nullptr || !node->IsSplitNode()) {
                // No saved layout, build default
                BuildInitialDockLayout();
            }
            first_time_layout_ = false;
        }

        ::gui::SetMainDockSpace(dockspace_id);
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        // Draw Unreal-style active tab indicators on all dock nodes
        ImGuiDockNode* root_node = ImGui::DockBuilderGetNode(dockspace_id);
        if (root_node) {
            DrawDockNodeActiveTabIndicator(root_node);
        }
    }

    ImGui::End();
}

void MainWindow::BuildInitialDockLayout() {
    ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

    // Clear any existing layout
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    // Create the docking layout structure
    // We'll build this structure:
    // - Left column (Asset Browser) ~15%
    // - Center-Left (CyxWiz Studio) ~50% of center
    // - Center-Right (Data Studio) ~50% of center - side by side
    // - Right column (Properties) ~20%
    // - Bottom section split into:
    //   - Bottom-left (Console) ~70% of bottom
    //   - Bottom-right (Training Dashboard) ~30% of bottom
    //   - Bottom-bottom (Viewport)

    ImGuiID dock_id_left = 0;
    ImGuiID dock_id_right = 0;
    ImGuiID dock_id_center = dockspace_id;
    ImGuiID dock_id_center_left = 0;
    ImGuiID dock_id_center_right = 0;
    ImGuiID dock_id_bottom = 0;
    ImGuiID dock_id_bottom_left = 0;
    ImGuiID dock_id_bottom_right = 0;
    ImGuiID dock_id_bottom_bottom = 0;

    // Split left side for Asset Browser (15% width)
    dock_id_left = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Left, 0.15f, nullptr, &dock_id_center);

    // Split right side for Properties (20% width)
    dock_id_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.25f, nullptr, &dock_id_center);

    // Split bottom for Console area (25% height of remaining)
    dock_id_bottom = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Down, 0.30f, nullptr, &dock_id_center);

    // Split center horizontally: CyxWiz Studio (left) and Data Studio (right) side by side
    dock_id_center_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.5f, nullptr, &dock_id_center_left);

    // Split bottom section horizontally: left (Console) and right (Inspector)
    dock_id_bottom_right = ImGui::DockBuilderSplitNode(dock_id_bottom, ImGuiDir_Right, 0.25f, nullptr, &dock_id_bottom_left);

    // Split bottom-left further for Profiler/Viewport
    dock_id_bottom_bottom = ImGui::DockBuilderSplitNode(dock_id_bottom_left, ImGuiDir_Down, 0.40f, nullptr, &dock_id_bottom_left);

    // Dock windows to their designated areas
    // Window names must EXACTLY match the names in ImGui::Begin() calls in each panel
    ImGui::DockBuilderDockWindow("Asset Browser", dock_id_left);
    ImGui::DockBuilderDockWindow("CyxWiz Studio", dock_id_center_left);
    ImGui::DockBuilderDockWindow("Script Editor", dock_id_center_left); // Tabbed with CyxWiz Studio
    ImGui::DockBuilderDockWindow("Data Studio", dock_id_center_right); // Beside CyxWiz Studio
    ImGui::DockBuilderDockWindow("Properties", dock_id_right);
    ImGui::DockBuilderDockWindow("Console", dock_id_bottom_left);
    ImGui::DockBuilderDockWindow("Training Dashboard", dock_id_bottom_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_id_bottom_bottom);

    // Every other sidebar panel gets a home too, as a tab beside the panel
    // it belongs with, instead of opening as a small floating window
    // (TOFIX129 0.6). Names are the window ids the panels pass to Begin.
    const ImGuiID dock_id_right_bottom =
        ImGui::DockBuilderSplitNode(dock_id_right, ImGuiDir_Down, 0.45f, nullptr, &dock_id_right);
    for (const char* name : {"Nodes", "Pattern Browser"}) ImGui::DockBuilderDockWindow(name, dock_id_left);
    for (const char* name : {"Info", "Wallet"}) ImGui::DockBuilderDockWindow(name, dock_id_right_bottom);
    for (const char* name : {"Table Viewer", "Annotation Editor", "###Visualizer", "Query Console",
                             "Plot Output", "###CloudBrowser", "###DatasetManager", "Plugin Manager"})
        ImGui::DockBuilderDockWindow(name, dock_id_center_right);
    for (const char* name : {"###VariableExplorer", "Background Tasks", "Job Status & Orchestration",
                             "P2P Training Progress"})
        ImGui::DockBuilderDockWindow(name, dock_id_bottom_left);
    ImGui::DockBuilderDockWindow("###StudioDebuggerPanel", dock_id_bottom_right);

    // Finish the docking layout
    ImGui::DockBuilderFinish(dockspace_id);
    front_tabs_countdown_ = 8;

    spdlog::info("Initial dock layout built successfully");
    spdlog::info("  - Left: Asset Browser (15%)");
    spdlog::info("  - Center-Left: CyxWiz Studio (50%)");
    spdlog::info("  - Center-Right: Data Studio (50%)");
    spdlog::info("  - Right: Properties (20%)");
    spdlog::info("  - Bottom-Left: Console");
    spdlog::info("  - Bottom-Right: Training Dashboard");
    spdlog::info("  - Bottom-Bottom: Viewport");
}

void MainWindow::RegisterPanelsWithSidebar() {
    DockStyle& dock_style = GetDockStyle();

    // Clear any existing registrations
    dock_style.ClearPanels();

    // Sidebar and View > Panels share this list: grouped by area, one icon
    // per panel (TOFIX129). Order here is the order on screen.
    auto add = [&dock_style](const char* name, const char* icon, auto& panel, const char* group,
                             const char* shortcut = "") {
        if (panel) dock_style.RegisterPanel(name, icon, panel->GetVisiblePtr(), nullptr, group, shortcut);
    };

    // Workspace
    add("CyxWiz Studio", ICON_FA_DIAGRAM_PROJECT, node_editor_, "Workspace");
    add("Properties", ICON_FA_SLIDERS, properties_, "Workspace");
    add("Node Browser", ICON_FA_CUBES, node_browser_panel_, "Workspace");
    add("Node Info", ICON_FA_CIRCLE_INFO, node_info_panel_, "Workspace");
    add("Patterns", ICON_FA_LAYER_GROUP, pattern_browser_, "Workspace",
        cyxwiz::menu::ShortcutFor("view.patterns", cyxwiz::menu::Context::Any).c_str());
    add("Asset Browser", ICON_FA_IMAGES, asset_browser_, "Workspace");
    add("Script Editor", ICON_FA_CODE, script_editor_, "Workspace");
    add("Console", ICON_FA_TERMINAL, console_, "Workspace");
    add("Viewport", ICON_FA_CUBE, viewport_, "Workspace");

    // Data
    add("Data Studio", ICON_FA_WAND_MAGIC_SPARKLES, data_studio_panel_, "Data");
    add("Table Viewer", ICON_FA_TABLE, table_viewer_, "Data");
    add("Annotation Editor", ICON_FA_DRAW_POLYGON, annotation_editor_panel_, "Data");
    add("Query Console", ICON_FA_MAGNIFYING_GLASS_CHART, query_console_, "Data");
    add("Variable Explorer", ICON_FA_LIST_UL, variable_explorer_, "Data");
    add("Plot Output", ICON_FA_CHART_AREA, plot_output_panel_, "Data");

    // Training
    add("Training", ICON_FA_CHART_LINE, training_plot_panel_, "Training");
    add("Studio Debugger", ICON_FA_BUG, studio_debugger_panel_, "Training");
    add("Tasks", ICON_FA_HOURGLASS_HALF, task_progress_panel_, "Training");

    // Network
    add("Jobs", ICON_FA_LIST_CHECK, job_status_panel_, "Network");
    add("P2P Training", ICON_FA_NETWORK_WIRED, p2p_training_panel_, "Network");
    add("Cloud Browser", ICON_FA_CLOUD, cloud_browser_panel_, "Network");
    add("Cloud Manager", ICON_FA_CLOUD_ARROW_UP, cloud_dataset_manager_panel_, "Network");
    add("Wallet", ICON_FA_WALLET, wallet_panel_, "Network");

    // Extensions
    add("Plugin Manager", ICON_FA_PLUG, plugin_manager_panel_, "Extensions");

    // Command Palette: an action, pinned at the bottom of the sidebar
    if (toolbar_) {
        dock_style.RegisterPanel("Command Palette", ICON_FA_MAGNIFYING_GLASS, nullptr,
                                 [this]() { toolbar_->OpenCommandPalette(); }, "",
                                 cyxwiz::menu::ShortcutFor("view.command_palette", cyxwiz::menu::Context::Any));
    }

    spdlog::debug("Registered {} panels with sidebar", dock_style.GetPanels().size());
}

void MainWindow::SetDefaultPanelVisibility(bool force) {
    // Only set defaults on first launch (no imgui.ini exists), or on
    // View > Layout > Reset to Default.
    if (!first_time_layout_ && !force) {
        return;
    }

    spdlog::debug("Setting default panel visibility (first launch)");

    // === CORE PANELS ===
    // Original panels (node_editor_, console_, properties_, viewport_) don't have SetVisible
    // They use ImGui window visibility directly. We leave them alone.
    // New-style panels that should be visible:
    if (asset_browser_) asset_browser_->SetVisible(true);
    if (studio_debugger_panel_) studio_debugger_panel_->SetVisible(false);

    // === HIDE ALL OTHER PANELS ===
    // Users can enable these via View menu when needed

    // Main panels - hide by default
    if (training_plot_panel_) training_plot_panel_->SetVisible(false);
    if (script_editor_) script_editor_->SetVisible(false);
    if (table_viewer_) table_viewer_->SetVisible(false);
    if (job_status_panel_) job_status_panel_->SetVisible(false);
    if (p2p_training_panel_) p2p_training_panel_->SetVisible(false);
    if (wallet_panel_) wallet_panel_->SetVisible(false);
    if (task_progress_panel_) task_progress_panel_->SetVisible(false);
    if (pattern_browser_) pattern_browser_->SetVisible(false);
    if (query_console_) query_console_->SetVisible(false);
    if (custom_node_editor_) custom_node_editor_->SetVisible(false);
    if (theme_editor_) theme_editor_->SetVisible(false);
    if (memory_panel_) memory_panel_->SetVisible(false);
    if (memory_monitor_) memory_monitor_->SetVisible(false);
    if (variable_explorer_) variable_explorer_->SetVisible(false);
    if (plot_output_panel_) plot_output_panel_->SetVisible(false);

    // Model Analysis panels (Phase 2)
    if (model_summary_panel_) model_summary_panel_->SetVisible(false);
    if (architecture_diagram_) architecture_diagram_->SetVisible(false);
    if (lr_finder_panel_) lr_finder_panel_->SetVisible(false);

    // Statistics panels (Phase 4)

    // Advanced Tools panels (Phase 5)
    if (dim_reduction_panel_) dim_reduction_panel_->SetVisible(false);
    if (gradcam_panel_) gradcam_panel_->SetVisible(false);
    if (feature_importance_panel_) feature_importance_panel_->SetVisible(false);
    if (nas_panel_) nas_panel_->SetVisible(false);

    // Clustering panels (Phase 6A)
    if (kmeans_panel_) kmeans_panel_->SetVisible(false);
    if (dbscan_panel_) dbscan_panel_->SetVisible(false);
    if (hierarchical_panel_) hierarchical_panel_->SetVisible(false);
    if (gmm_panel_) gmm_panel_->SetVisible(false);
    if (cluster_eval_panel_) cluster_eval_panel_->SetVisible(false);

    // Model Evaluation panels (Phase 6B)

    // Linear Algebra panels (Phase 7)
    if (matrix_calculator_panel_) matrix_calculator_panel_->SetVisible(false);
    if (eigen_decomp_panel_) eigen_decomp_panel_->SetVisible(false);
    if (svd_panel_) svd_panel_->SetVisible(false);
    if (qr_panel_) qr_panel_->SetVisible(false);
    if (cholesky_panel_) cholesky_panel_->SetVisible(false);

    // Signal Processing panels (Phase 8)
    if (fft_panel_) fft_panel_->SetVisible(false);
    if (spectrogram_panel_) spectrogram_panel_->SetVisible(false);
    if (filter_designer_panel_) filter_designer_panel_->SetVisible(false);
    if (convolution_panel_) convolution_panel_->SetVisible(false);
    if (wavelet_panel_) wavelet_panel_->SetVisible(false);

    // Optimization & Calculus panels (Phase 9)
    if (gradient_descent_panel_) gradient_descent_panel_->SetVisible(false);
    if (convexity_panel_) convexity_panel_->SetVisible(false);
    if (lp_panel_) lp_panel_->SetVisible(false);
    if (qp_panel_) qp_panel_->SetVisible(false);
    if (differentiation_panel_) differentiation_panel_->SetVisible(false);
    if (integration_panel_) integration_panel_->SetVisible(false);

    // Time Series Analysis panels (Phase 10)
    if (decomposition_panel_) decomposition_panel_->SetVisible(false);
    if (acf_pacf_panel_) acf_pacf_panel_->SetVisible(false);
    if (stationarity_panel_) stationarity_panel_->SetVisible(false);
    if (seasonality_panel_) seasonality_panel_->SetVisible(false);
    if (forecasting_panel_) forecasting_panel_->SetVisible(false);

    // Text Processing panels (Phase 11)
    if (tokenization_panel_) tokenization_panel_->SetVisible(false);
    if (word_frequency_panel_) word_frequency_panel_->SetVisible(false);
    if (tfidf_panel_) tfidf_panel_->SetVisible(false);
    if (embeddings_panel_) embeddings_panel_->SetVisible(false);
    if (sentiment_panel_) sentiment_panel_->SetVisible(false);
    if (language_model_generation_panel_) language_model_generation_panel_->SetVisible(false);

    // Utilities panels (Phase 12)
    if (calculator_panel_) calculator_panel_->SetVisible(false);
    if (unit_converter_panel_) unit_converter_panel_->SetVisible(false);
    if (random_generator_panel_) random_generator_panel_->SetVisible(false);
    if (hash_generator_panel_) hash_generator_panel_->SetVisible(false);
    if (json_viewer_panel_) json_viewer_panel_->SetVisible(false);
    if (regex_tester_panel_) regex_tester_panel_->SetVisible(false);
    
    // Cloud panels (hidden by default - access via sidebar/view menu)
    if (cloud_browser_panel_) cloud_browser_panel_->SetVisible(false);
    if (cloud_dataset_manager_panel_) cloud_dataset_manager_panel_->SetVisible(false);

    // Plugin Manager (hidden by default)
    if (plugin_manager_panel_) plugin_manager_panel_->SetVisible(false);

    spdlog::debug("Default panel visibility set - showing only core panels");
}

bool MainWindow::StartDataInputLoads(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (!data_input_loads_.empty()) {
        spdlog::info("Train: the graph's data is still loading; training starts when it is loaded");
        return true;
    }
    if (!node_editor_) return false;
    const auto ids = cyxwiz::CollectDataInputsToLoad(nodes, links, [](const std::string& name) {
        return cyxwiz::loaders::GetByRegisteredDataset(name) != nullptr;
    });
    std::string names;
    for (const int id : ids) {
        MLNode* node = node_editor_->GetNodeForConfiguration(id);
        if (!node) continue;
        auto dialog = std::make_unique<DataInputDialog>(node);
        dialog->SetNodeEditor(node_editor_.get());
        dialog->Apply();
        spdlog::info("Train: loading '{}' first (its data is not loaded in this session)", node->name);
        names += (names.empty() ? "" : ", ") + node->name;
        data_input_loads_.push_back({id, node->name, std::move(dialog)});
    }
    if (data_input_loads_.empty()) return false;
    data_input_loads_project_ = cyxwiz::ProjectManager::Instance().GetProjectRoot();
    if (training_plot_panel_) {
        training_plot_panel_->SetPreparationState(true, "Loading " + names + " before training...", 0.01f);
        training_plot_panel_->SetVisible(true);
    }
    return true;
}

// Runs the graph's data steps through the Training Dataset input's feeder on
// the canvas executor (one run, one cache, the canvas shows it); training
// starts when it ends (PollTrainingPipelineRun).
bool MainWindow::StartTrainingPipelineRun(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (!node_editor_) return false;
    if (node_editor_->IsDataPipelineRunning()) {
        spdlog::warn("Train: the data pipeline is running; train when it ends");
        return false;
    }
    const int feeder = cyxwiz::TrainingInputFeeder(nodes, links);
    if (feeder < 0 || !node_editor_->ExecuteDataPipelineThrough(feeder, /*with_checks=*/false)) return false;
    training_pipeline_run_ = TrainingPipelineRun{node_editor_->GetDataPipelineTaskId(), feeder,
                                                 cyxwiz::ProjectManager::Instance().GetProjectRoot()};
    spdlog::info("Train: running the data steps before the Data Split first; training starts when they end");
    if (training_plot_panel_) {
        training_plot_panel_->SetPreparationState(true, "Running the data steps before training...", 0.01f);
        training_plot_panel_->SetVisible(true);
    }
    return true;
}

// The run Train started: when it ends, train on its result, or say why not.
void MainWindow::PollTrainingPipelineRun() {
    if (!training_pipeline_run_) return;
    if (!node_editor_) {
        training_pipeline_run_.reset();
        return;
    }
    const auto task = cyxwiz::AsyncTaskManager::Instance().GetTask(training_pipeline_run_->task_id);
    std::optional<cyxwiz::TaskInfo> info;
    if (task) info = task->GetInfo();
    if (info && (info->state == cyxwiz::TaskState::Pending || info->state == cyxwiz::TaskState::Running)) return;
    const TrainingPipelineRun run = *training_pipeline_run_;
    training_pipeline_run_.reset();
    if (cyxwiz::ProjectManager::Instance().GetProjectRoot() != run.project) {
        spdlog::info("Train: the project changed while the data steps ran; not training");
        EndFoldsLaunch("Not started: the project changed while the data steps ran.");
        return;
    }
    std::optional<std::string> result;
    if (info && info->state == cyxwiz::TaskState::Completed) result = node_editor_->LastRunResult(run.feeder);
    if (!result) {
        std::string why = "the run left no table";
        if (info && !info->error_message.empty()) why = info->error_message;
        else if (info && info->state == cyxwiz::TaskState::Cancelled) why = "the run was cancelled";
        spdlog::error("Train: the data steps before the Data Split failed: {}", why);
        compile_result_success_ = false;
        compile_result_issues_.clear();
        compile_result_issues_.push_back({cyxwiz::IssueLevel::Error, run.feeder, "",
                                          "The data steps before the Data Split failed: " + why,
                                          cyxwiz::errors::Training::InvalidTrainingSetup});
        ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the data steps failed");
        if (training_plot_panel_) training_plot_panel_->SetPreparationFailed("The data steps failed: " + why);
        EndFoldsLaunch("Not started: the data steps before the Data Split failed.");
        return;
    }
    spdlog::info("Train: the data steps ended; training reads '{}'", *result);
    training_pipeline_result_ = *result;
    StartTrainingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
}

// The loads Train started: when all have ended, train, or say which data
// could not be loaded and why.
void MainWindow::PollDataInputLoads() {
    if (data_input_loads_.empty()) return;
    bool busy = false;
    for (auto& load : data_input_loads_) {
        load.dialog->RebindNode(node_editor_ ? node_editor_->GetNodeForConfiguration(load.node_id) : nullptr);
        load.dialog->PollLoad();
        busy = busy || load.dialog->IsBusy();
    }
    if (busy) return;
    std::vector<cyxwiz::ValidationIssue> failures;
    std::string failed_names;
    for (const auto& load : data_input_loads_) {
        const MLNode* node = node_editor_ ? node_editor_->GetNodeForConfiguration(load.node_id) : nullptr;
        const auto name = node ? node->parameters.find("dataset_name") : std::map<std::string, std::string>::const_iterator{};
        if (node && name != node->parameters.end() && cyxwiz::loaders::GetByRegisteredDataset(name->second)) continue;
        const std::string why = load.dialog->LoadMessage().empty() ? "the node was removed" : load.dialog->LoadMessage();
        failures.push_back({cyxwiz::IssueLevel::Error, load.node_id, load.node_name,
                            "Could not load the data before training: " + why, ""});
        failed_names += (failed_names.empty() ? "" : ", ") + load.node_name;
    }
    data_input_loads_.clear();
    if (!node_editor_ || cyxwiz::ProjectManager::Instance().GetProjectRoot() != data_input_loads_project_) {
        spdlog::info("Train: the project changed while its data loaded; not training");
        EndFoldsLaunch("Not started: the project changed while its data loaded.");
        return;
    }
    const auto nodes = node_editor_->GetNodes();
    const auto links = node_editor_->GetLinks();
    if (!failures.empty()) {
        spdlog::error("Train: could not load {}", failed_names);
        BuildCompileResult(nodes, links);
        // The load's reason replaces the compiler's "open the node and click
        // Apply" for those nodes: Apply would fail the same way.
        std::erase_if(compile_result_issues_, [&](const cyxwiz::ValidationIssue& issue) {
            return issue.error_code == cyxwiz::errors::Compiler::DataNotLoaded &&
                   std::any_of(failures.begin(), failures.end(),
                               [&](const cyxwiz::ValidationIssue& failure) { return failure.node_id == issue.node_id; });
        });
        compile_result_issues_.insert(compile_result_issues_.begin(), failures.begin(), failures.end());
        compile_result_success_ = false;
        ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the data could not be loaded");
        if (training_plot_panel_) training_plot_panel_->SetPreparationFailed("Could not load " + failed_names);
        EndFoldsLaunch("Not started: could not load " + failed_names + " (the compile report says why).");
        return;
    }
    StartTrainingFromGraph(nodes, links);
}

void MainWindow::StartFoldsFromGraph(std::shared_ptr<cyxwiz::FoldsView> view, const cyxwiz::CrossValidationSettings& settings) {
    if (!node_editor_ || !view) return;
    if (const std::string why = FoldsBlocker(); !why.empty()) {
        cyxwiz::CrossValidationOutcome refused;
        refused.error = why;
        view->Finished(std::move(refused));
        return;
    }
    const auto node = FindFoldsNode(view->IsLearningCurve());
    const std::string results_table = node ? node->results_table
                                           : cyxwiz::TrainingEvaluationOperator::DefaultResultsTable(view->IsLearningCurve());
    folds_launch_ = FoldsLaunch{view, settings, results_table};
    view->Started();
    StartTrainingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
}

std::optional<MainWindow::FoldsNode> MainWindow::FindFoldsNode(bool learning_curve) const {
    if (!node_editor_) return std::nullopt;
    const NodeType type = learning_curve ? NodeType::LearningCurvesNode : NodeType::CrossValidationNode;
    for (const auto& n : node_editor_->GetNodes()) {
        if (n.type != type) continue;
        FoldsNode found;
        found.id = n.id;
        found.name = n.name;
        cyxwiz::TrainingEvaluationOperator::SettingsFromParameters(learning_curve, n.parameters, found.settings, found.results_table,
                                                                  found.error);
        return found;
    }
    return std::nullopt;
}

std::string MainWindow::AddFoldsNode(bool learning_curve, const cyxwiz::CrossValidationSettings& settings) {
    if (!node_editor_) return "No graph is open.";
    std::vector<const MLNode*> inputs;
    for (const auto& n : node_editor_->GetNodes())
        if (n.type == NodeType::DataInput) inputs.push_back(&n);
    if (inputs.size() != 1)
        return inputs.empty() ? "The graph has no Data Input to put it after."
                              : "The graph has several Data Inputs: add the node from the palette after the one it trains from.";
    const char* name = learning_curve ? "Learning Curves" : "Cross-Validation";
    node_editor_->RequestStepAfter(inputs.front()->id, learning_curve ? NodeType::LearningCurvesNode : NodeType::CrossValidationNode,
                                   name,
                                   cyxwiz::TrainingEvaluationOperator::ParametersFromSettings(
                                       learning_curve, settings, cyxwiz::TrainingEvaluationOperator::DefaultResultsTable(learning_curve)));
    return std::string("Added a ") + name + " node after \"" + inputs.front()->name +
           "\": the graph keeps these settings, and Run uses them from now on.";
}

void MainWindow::EndFoldsLaunch(const std::string& why) {
    if (!folds_launch_) return;
    const auto view = folds_launch_->view;
    folds_launch_.reset();
    cyxwiz::CrossValidationOutcome stopped;
    stopped.error = why;
    if (view) view->Finished(std::move(stopped));
}

std::string MainWindow::FoldsBlocker() const {
    if (!node_editor_) return "No graph is open";
    if (folds_task_ || folds_launch_) return "A cross-validation or learning curve is running";
    if (cyxwiz::TrainingManager::Instance().IsTrainingActive()) return "Train is running; run this when it has finished";
    if (HasActiveGraphTrainingPreparation()) return "A training is being prepared";
    if (test_preparation_task_ || cyxwiz::TestManager::Instance().IsTestingActive()) return "A test is running";
    return "";
}

// The last step of a marked launch: the prepared table to the runner, on a
// task that reports into the view.
::gui::GraphTrainingDispatch MainWindow::MakeFoldsDispatch(const FoldsLaunch& launch) {
    const std::weak_ptr<int> lifetime = test_callback_lifetime_;
    return [this, launch, lifetime](cyxwiz::TrainingConfiguration config, const std::string& dataset_name,
                                    const std::string& label_column, int epochs, int batch_size,
                                    std::weak_ptr<cyxwiz::TrainingPlotPanel>, std::function<void(bool)>) -> bool {
        if (cyxwiz::TrainingManager::Instance().IsTrainingActive())
            throw std::runtime_error("Train is running; run this again when it has finished.");
        const auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(dataset_name);
        if (!dataset)
            throw std::runtime_error("Cross-validation reads the prepared table from memory, and '" + dataset_name +
                                     "' is kept on disk. Raise the data budget (Preferences) or use fewer rows.");
        const std::shared_ptr<cyxwiz::FoldsView> view = launch.view;
        const cyxwiz::CrossValidationSettings settings = launch.settings;
        const std::string results_table = launch.results_table;
        const std::string name = settings.train_shares.empty() ? "Cross-validation" : "Learning curve";
        folds_task_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
            name,
            [view, settings, config, dataset, label_column, epochs, batch_size, name, results_table](cyxwiz::LambdaTask& task) {
                cyxwiz::CrossValidationCallbacks callbacks;
                callbacks.should_cancel = [&task] { return task.ShouldStop(); };
                callbacks.on_progress = [&task, view, name](const cyxwiz::CrossValidationProgress& p) {
                    view->Progress(p);
                    const float done = (static_cast<float>(p.run - 1) +
                                        (p.epochs > 0 ? static_cast<float>(p.epoch) / static_cast<float>(p.epochs) : 0.0f)) /
                                       static_cast<float>(std::max(1, p.runs));
                    task.ReportProgress(done, name + ": fold " + std::to_string(p.fold) + " of " + std::to_string(p.folds));
                };
                callbacks.on_run = [view](const cyxwiz::CrossValidationRun& run) { view->FoldDone(run); };
                cyxwiz::CrossValidationOutcome outcome =
                    cyxwiz::RunCrossValidation(config, dataset, label_column, epochs, batch_size, settings, callbacks);
                if (!outcome.ok && !outcome.cancelled) task.MarkFailed(outcome.error);
                // The run's table, for the node's second pin and any reader (as test_predictions is).
                if (outcome.ok) {
                    if (auto table = cyxwiz::CrossValidationTable(outcome)) {
                        cyxwiz::AsyncTaskManager::Instance().PostToMainThread([table, results_table]() {
                            cyxwiz::DataRegistry::Instance().RegisterArrowTable(table, results_table);
                        });
                    }
                }
                view->Finished(std::move(outcome));
            },
            nullptr,
            [this, lifetime](bool, const std::string&) {
                if (!lifetime.expired()) folds_task_ = 0;
            },
            lifetime);
        return true;
    };
}

namespace {

// The data the canvas runs already prepared for training (TOFIX178 S0-d): the
// feeder's result (a table, or sparse features, S0-f) when it and every step
// before it are unchanged since. Empty when the data must be prepared: no such
// result or a requested rebuild.
std::string CanvasPreparedTrainingInput(NodeEditor* editor, const std::vector<MLNode>& nodes,
                                        const std::vector<NodeLink>& links) {
    if (!editor || IsGraphMaterializationRebuildRequested()) return {};
    const int feeder = cyxwiz::TrainingInputFeeder(nodes, links);
    if (feeder < 0) {
        spdlog::debug("StartTrainingFromGraph: no step feeds the Training Dataset input from its main output");
        return {};
    }
    std::string result = editor->CurrentRunResult(feeder).value_or(std::string());
    spdlog::debug("StartTrainingFromGraph: canvas result of node {}: '{}'", feeder, result);
    return result;
}

}  // namespace

void MainWindow::StartTrainingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    // A launch Results > Cross-validation / Learning curve marked: Train's own
    // checks and preparation, then its own dispatch. A stop on the way ends it
    // with a word in the view, except while the data loads or the memory
    // question is open (both come back here).
    const std::optional<FoldsLaunch> folds = folds_launch_;
    bool folds_waiting = false;
    struct FoldsLaunchEnd {
        MainWindow* window;
        const bool& waiting;
        ~FoldsLaunchEnd() {
            if (!waiting) window->EndFoldsLaunch("Not started: Train's checks stopped it (the compile report says why).");
        }
    } folds_end{this, folds_waiting};
    if (!folds && (folds_task_ || folds_launch_)) {
        spdlog::warn("Train: a cross-validation or learning curve is running; cancel it in Data Studio > Results first");
        return;
    }
    if (test_preparation_task_ || cyxwiz::TestManager::Instance().IsTestingActive()) {
        spdlog::warn("Train: wait for Run Test preparation/evaluation or cancel it first");
        return;
    }
    // A reopened project keeps each Data Input's settings, not its loaded
    // data: load what training needs first, then train.
    if (StartDataInputLoads(nodes, links)) {
        folds_waiting = true;
        return;
    }

    const bool memory_confirmation_accepted =
        materialization_memory_confirmation_accepted_once_;
    const int accepted_memory_node_id =
        accepted_materialization_memory_node_id_;
    const uint64_t accepted_memory_estimated_bytes =
        accepted_materialization_memory_estimated_bytes_;
    const std::string accepted_memory_risk =
        std::move(accepted_materialization_memory_risk_);
    materialization_memory_confirmation_accepted_once_ = false;
    accepted_materialization_memory_node_id_ = -1;
    accepted_materialization_memory_estimated_bytes_ = 0;
    try {
        spdlog::info("StartTrainingFromGraph: Compiling {} nodes, {} links", nodes.size(), links.size());

        // === Pre-flight compile gate ===
        // Run the same Compile pass the explicit Compile button uses, and
        // populate compile_result_* so the popup can render structured
        // findings. If the graph has any Error-level issues, refuse to
        // start training and trigger the popup with a "blocked" header
        // instead of silently logging.
        BuildCompileResult(nodes, links);
        if (!compile_result_success_) {
            ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked);
            spdlog::error("StartTrainingFromGraph: blocked - graph has errors");
            return;
        }

        // === Local Debug staleness gate ===
        // After the compile gate passes, check whether the user has run
        // a successful Local Debug on the current graph. Three states:
        //   - no prior successful debug      => warn (or error under strict mode)
        //   - prior debug on DIFFERENT graph => warn (or error under strict mode)
        //   - prior debug on same graph      => silent
        // Under EngineConfig::RequireDebugBeforeTrain() the warning becomes
        // an Error-level issue and training refuses to start.
        const uint64_t current_graph_hash = HashGraphStructure(nodes, links);
        const bool same_as_last_debug =
            have_last_debug_result_ &&
            current_graph_hash == last_debug_graph_hash_;
        if (!same_as_last_debug) {
            const bool strict = cyxwiz::core::EngineConfig::Instance()
                                    .RequireDebugBeforeTrain();
            const std::string msg = have_last_debug_result_
                ? "Graph changed since last successful Local Debug run. "
                  "Consider F6 before F5."
                : "No Local Debug (F6) run on this graph yet. "
                  "Consider running one before Train.";
            const cyxwiz::IssueLevel lvl = strict
                ? cyxwiz::IssueLevel::Error
                : cyxwiz::IssueLevel::Warning;
            compile_result_issues_.push_back(
                {lvl, -1, "", msg, cyxwiz::errors::Ui::InvalidWorkflowState});
            if (strict) {
                compile_result_success_ = false;
                ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked,
                                  "this project requires a Local Debug (F6) run first");
                spdlog::error("StartTrainingFromGraph: blocked by "
                              "require_debug_before_train (hash mismatch)");
                return;
            }
            spdlog::warn("StartTrainingFromGraph: {}", msg);
        }

        // Compile a fresh config for the actual run. BuildCompileResult
        // already validated the graph; this Compile call will succeed
        // identically and produce the same TrainingConfiguration.
        cyxwiz::GraphCompiler compiler;
        cyxwiz::TrainingConfiguration config = compiler.Compile(nodes, links);

        if (!config.is_valid) {
            // Defensive - should be unreachable because BuildCompileResult
            // already validated. Keep as a safety net in case of races.
            spdlog::error("Graph compilation failed (post-gate): {}", config.error_message);
            return;
        }

        spdlog::info("Graph compiled successfully: {} layers, input={}, output={}",
                     config.layers.size(), config.input_size, config.output_size);

        auto& project = cyxwiz::ProjectManager::Instance();
        if (config.checkpoint_dir.empty() && project.HasActiveProject()) {
            config.checkpoint_dir = project.GetCheckpointsPath();
            spdlog::info(
                "StartTrainingFromGraph: using project checkpoint directory '{}'",
                config.checkpoint_dir);
        }
        // Embedding weights, the warm-start checkpoint and the Embedding
        // Output file are project-relative, like Data Input paths.
        if (project.HasActiveProject()) cyxwiz::ResolveProjectRelativePaths(config, project.GetProjectRoot());

        auto& registry = cyxwiz::DataRegistry::Instance();
        auto& tm = cyxwiz::TrainingManager::Instance();

        cyxwiz::MaterializationMemoryPolicy materialization_memory_policy;
        materialization_memory_policy.hard_limit_bytes = cyxwiz::DataBudgetBytes();

        // The canvas already prepared this data: training reads that table and
        // prepares nothing, so it needs no memory plan for a preparation either.
        // A Train that just ran the steps itself reads that run's result.
        std::string canvas_dataset = std::exchange(training_pipeline_result_, std::string());
        if (canvas_dataset.empty()) canvas_dataset = CanvasPreparedTrainingInput(node_editor_.get(), nodes, links);
        // The same table kept on disk by an earlier session (TOFIX178 S0-e2):
        // training loads it and nothing runs.
        const std::string project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
        cyxwiz::PreparedDataKey prepared_key;
        if (const int feeder = cyxwiz::TrainingInputFeeder(nodes, links); feeder >= 0) {
            prepared_key = cyxwiz::ComputePreparedDataKey(nodes, links, feeder, registry, project_root);
            spdlog::debug("StartTrainingFromGraph: prepared data key {}", prepared_key.cacheable
                              ? prepared_key.key.substr(0, 16)
                              : "none (" + prepared_key.reason + ")");
        }
        const bool prepared_on_disk = canvas_dataset.empty() && !IsGraphMaterializationRebuildRequested() &&
            cyxwiz::ProbePreparedData(GraphMaterializationCacheConfig(project_root), prepared_key);
        if (prepared_on_disk) spdlog::info("StartTrainingFromGraph: the prepared data is on disk from an earlier session");
        cyxwiz::MaterializationMemoryContext preflight_memory_context;
        preflight_memory_context.policy = materialization_memory_policy;
        GraphMaterializationPreflightResult memory_preflight;
        if (canvas_dataset.empty() && !prepared_on_disk) {
            spdlog::info("StartTrainingFromGraph: preparing materialization memory plan");
            memory_preflight = PreflightGraphMaterialization(
                nodes, links, config, registry,
                std::move(preflight_memory_context),
                cyxwiz::ProjectManager::Instance().GetProjectRoot());
        } else if (!canvas_dataset.empty()) {
            spdlog::info("StartTrainingFromGraph: the canvas run already prepared '{}'", canvas_dataset);
        }
        if (memory_preflight.blocked) {
            compile_result_success_ = false;
            compile_result_issues_.clear();
            compile_result_issues_.push_back({
                cyxwiz::IssueLevel::Error,
                memory_preflight.evidence.node_id,
                memory_preflight.evidence.node_name,
                memory_preflight.status_detail,
                cyxwiz::errors::Training::InvalidTrainingSetup});
            ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the prepared data would not fit in memory");
            spdlog::error(
                "StartTrainingFromGraph: pre-start materialization memory "
                "check blocked training: {}",
                memory_preflight.status_detail);
            return;
        }
        const auto memory_risk_rank = [](const std::string& risk) {
            if (risk == "risky") return 2;
            if (risk == "warning") return 1;
            return 0;
        };
        const bool accepted_current_estimate =
            memory_confirmation_accepted &&
            accepted_memory_node_id == memory_preflight.evidence.node_id &&
            accepted_memory_estimated_bytes ==
                memory_preflight.evidence.estimated_memory_bytes &&
            memory_risk_rank(memory_preflight.evidence.memory_risk_level) <=
                memory_risk_rank(accepted_memory_risk);
        if (memory_preflight.requires_confirmation &&
            !accepted_current_estimate) {
            pending_memory_confirmation_nodes_ = nodes;
            pending_memory_confirmation_links_ = links;
            materialization_memory_confirmation_node_id_ =
                memory_preflight.evidence.node_id;
            materialization_memory_confirmation_estimated_bytes_ =
                memory_preflight.evidence.estimated_memory_bytes;
            materialization_memory_confirmation_available_bytes_ =
                memory_preflight.evidence.available_memory_bytes;
            materialization_memory_confirmation_budget_bytes_ =
                memory_preflight.evidence.safe_memory_budget_bytes;
            materialization_memory_confirmation_detail_ =
                memory_preflight.status_detail;
            materialization_memory_confirmation_risk_ =
                memory_preflight.evidence.memory_risk_level;
            materialization_memory_confirmation_node_ =
                memory_preflight.evidence.node_name;
            show_materialization_memory_confirmation_popup_ = true;
            folds_waiting = true;
            spdlog::warn(
                "StartTrainingFromGraph: awaiting main-thread confirmation "
                "for {} materialization estimate at node '{}'",
                materialization_memory_confirmation_risk_,
                materialization_memory_confirmation_node_);
            return;
        }
        if (memory_preflight.requires_confirmation) {
            spdlog::warn(
                "StartTrainingFromGraph: user accepted {} materialization "
                "estimate at node '{}'",
                memory_preflight.evidence.memory_risk_level,
                memory_preflight.evidence.node_name);
        }

        // After the memory estimate: the data steps before the Split run once, in the pipeline run, the
        // canvas's own engine (TOFIX178 S0-e): run it through the Split's
        // input, then train on its result (a table or sparse features).
        // Non-table paths (images) are still prepared by training itself.
        const bool run_steps = config.training_data_needs_pipeline_run ||
                               (prepared_key.table_path && prepared_key.steps > 0);
        if (canvas_dataset.empty() && !prepared_on_disk && run_steps) {
            if (StartTrainingPipelineRun(nodes, links)) {
                folds_waiting = true;
                return;
            }
            compile_result_success_ = false;
            compile_result_issues_.clear();
            compile_result_issues_.push_back({cyxwiz::IssueLevel::Error, -1, "",
                                              "The data steps before the Data Split could not be started "
                                              "(a data pipeline run is in progress, or nothing feeds the Split).",
                                              cyxwiz::errors::Training::InvalidTrainingSetup});
            ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the data steps could not run");
            return;
        }

        // Set up node editor callback to update training animation and pin state.
        // When training ends, flip all nodes' pin state to Trained so pins go
        // solid green - the user's at-a-glance signal that the graph has been run.
        // (Not for a cross-validation: it leaves the trained state and the
        // Training Dashboard alone.)
        std::function<void(bool)> node_editor_callback;
        if (!folds) {
            node_editor_callback = [this](bool is_training) {
                if (!node_editor_) return;
                node_editor_->SetTrainingActive(is_training);
                if (!is_training) {
                    node_editor_->SetAllNodesPinState(NodeEditor::NodePinState::Trained);
                }
            };
        }

        auto dispatch_training = folds ? MakeFoldsDispatch(*folds) : MakeEngineGraphTrainingDispatch(registry, tm);
        const std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel =
            folds ? std::weak_ptr<cyxwiz::TrainingPlotPanel>() : std::weak_ptr<cyxwiz::TrainingPlotPanel>(training_plot_panel_);

        auto launch_result = StartGraphTrainingFromCompiledConfig(
            nodes, links, std::move(config), registry, plot_panel,
            node_editor_callback, dispatch_training,
            materialization_memory_policy,
            memory_preflight.estimate_available
                ? std::make_optional(memory_preflight.evidence)
                : std::nullopt,
            cyxwiz::ProjectManager::Instance().GetProjectRoot(),
            canvas_dataset,
            prepared_key);

        if (launch_result.started && folds) {
            folds_launch_.reset();  // the preparation runs; its task reports into the view
            spdlog::info("{} started", folds->settings.train_shares.empty() ? "Cross-validation" : "Learning curve");
        } else if (launch_result.started) {
            spdlog::info("Training started successfully");
            if (training_plot_panel_) {
                training_plot_panel_->SetVisible(true);
            }
        } else {
            compile_result_success_ = false;
            compile_result_issues_.clear();

            cyxwiz::ValidationIssue issue;
            issue.level = cyxwiz::IssueLevel::Error;
            const std::string detail = launch_result.status_detail.empty()
                ? launch_result.error_message
                : launch_result.status_detail;
            issue.message = launch_result.status_title.empty() ? detail
                                                               : launch_result.status_title + ": " + detail;
            issue.error_code = cyxwiz::errors::Training::InvalidTrainingSetup;
            compile_result_issues_.push_back(std::move(issue));
            ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the training launch was refused");
        }

    } catch (const std::exception& e) {
        spdlog::error("StartTrainingFromGraph exception: {}", e.what());
    } catch (...) {
        spdlog::error("StartTrainingFromGraph unknown exception");
    }
}

// ---------------------------------------------------------------------------
// Compile Graph (dry-run) - validates and compiles the node graph without
// starting training, then shows the result in a modal popup. Lets users catch
// config errors before committing to a full training run.
// ---------------------------------------------------------------------------

void MainWindow::CompileGraphAndReport() {
    if (!node_editor_) {
        compile_result_success_ = false;
        compile_result_has_config_ = false;
        compile_result_backend_placements_.clear();
        compile_result_issues_ = {{cyxwiz::IssueLevel::Error, -1, "", "The node editor is not available.",
                                   cyxwiz::errors::Ui::InvalidWorkflowState}};
        ShowCompileReport(cyxwiz::compile_report::Mode::Compile);
        return;
    }

    BuildCompileResult(node_editor_->GetNodes(), node_editor_->GetLinks());
    ShowCompileReport(cyxwiz::compile_report::Mode::Compile);
}

void MainWindow::LocalDebugGraphAndReport() {
    if (!node_editor_ || !studio_debugger_panel_) {
        compile_result_success_ = false;
        compile_result_has_config_ = false;
        compile_result_backend_placements_.clear();
        compile_result_issues_ = {{cyxwiz::IssueLevel::Error, -1, "",
                                   "The node editor or the Studio Debugger is not available.",
                                   cyxwiz::errors::Ui::InvalidWorkflowState}};
        ShowCompileReport(cyxwiz::compile_report::Mode::DebugBlocked);
        return;
    }

    // F6 runs Local Debug through the Studio Debugger's background run path
    // (compile gate, preflight, one synthetic batch), with progress and Stop.
    // The staleness cache is updated by the run-completed callback.
    studio_debugger_panel_->Show();
    if (!studio_debugger_panel_->StartRun(cyxwiz::StudioDebuggerRunMode::LocalDebug, 0)) {
        spdlog::info("LocalDebugGraphAndReport: a Studio Debugger run is already active");
    }
}

void MainWindow::RecordArtifactConsistencyTrace(
    const cyxwiz::DebugArtifactConsistencyInput& input) {
    if (!studio_debugger_panel_) {
        return;
    }

    cyxwiz::DebugExportCorrelationTracer tracer;
    const std::string run_id = MakeArtifactRunId(
        input.action, input.graph_hash);
    cyxwiz::DebugTraceRecord trace = tracer.BuildConsistencyTrace(
        run_id, input);

    cyxwiz::StudioDebuggerSnapshot session;
    session.run_id = run_id;
    session.graph_hash = input.graph_hash;
    session.success = trace.payload.value("success", false);
    const std::string outcome = trace.payload.value(
        "consistency_outcome", std::string{"unobserved"});
    session.failure_summary = session.success
        ? std::string{}
        : std::string("Artifact consistency outcome: ") + outcome;
    session.graph_summary =
        std::string("Artifact consistency outcome: ") + outcome;
    session.sample_summary = input.action + ": " + input.artifact_path;
    session.issues = trace.issues;
    session.traces.push_back(trace);
    session.studio_events.push_back({
        run_id,
        NowLocalTimestampForDebugStore(),
        input.graph_hash,
        -1,
        input.action.rfind("import", 0) == 0
            ? "Model.ImportConsistency"
            : "Model.ExportConsistency",
        session.success ? "completed" : "failed",
        input.operation_status
    });

    cyxwiz::DebugRecommendationEngine recommendation_engine;
    session.recommendations = recommendation_engine.Build(
        session.traces,
        session.issues,
        session.smoke_result,
        session.last_run,
        session.training_trace);

    cyxwiz::DebugRunStoreRecord record;
    record.summary.run_id = run_id;
    record.summary.timestamp = NowLocalTimestampForDebugStore();
    record.summary.graph_hash = input.graph_hash;
    record.summary.success = session.success;
    record.summary.summary = session.graph_summary;
    record.issues = session.issues;
    record.traces = session.traces;
    record.studio_events = session.studio_events;
    record.recommendations = session.recommendations;
    if (!cyxwiz::DebugRunStore::Save(record)) {
        spdlog::warn(
            "Failed to persist artifact consistency run {}", run_id);
    }
    session.run_history = cyxwiz::DebugRunStore::ListRecent(8);
    studio_debugger_panel_->SetSession(std::move(session));
    studio_debugger_panel_->Show();
}

bool MainWindow::BuildStudioDebuggerSession(cyxwiz::StudioDebuggerSnapshot& session,
                                            cyxwiz::StudioDebuggerRunMode mode,
                                            int sample_index) {
    if (!node_editor_) {
        session = cyxwiz::StudioDebuggerSnapshot{};
        session.failure_summary = "Node editor is not available.";
        return false;
    }

    auto nodes = node_editor_->GetNodes();
    auto links = node_editor_->GetLinks();
    return BuildStudioDebuggerSessionFromSnapshot(
        session, mode, sample_index, std::move(nodes), std::move(links));
}

bool MainWindow::BuildStudioDebuggerSessionFromSnapshot(
    cyxwiz::StudioDebuggerSnapshot& session,
    cyxwiz::StudioDebuggerRunMode mode,
    int sample_index,
    std::vector<MLNode> nodes,
    std::vector<NodeLink> links,
    int explain_node_id,
    const cyxwiz::StudioDebuggerRunControl* control,
    const cyxwiz::StudioDebuggerRunInputs* inputs) {
    const auto* node_positions = inputs ? &inputs->node_positions : nullptr;
    session = cyxwiz::StudioDebuggerSnapshot{};
    const auto run_started = std::chrono::steady_clock::now();
    session.mode = mode;
    // Planned as if Smoke were available; the real capability is known
    // once Compile reports the graph's data domain.
    cyxwiz::SmokeCapability planned_smoke;
    planned_smoke.supported = true;
    session.steps = cyxwiz::PlanStudioDebuggerSteps(mode, planned_smoke);
    bool stopped = false;
    auto step_started = std::chrono::steady_clock::now();
    auto find_step = [&session](const char* id) -> cyxwiz::StudioDebuggerStep* {
        for (auto& step : session.steps) {
            if (step.id == id) return &step;
        }
        return nullptr;
    };
    auto report_progress = [&session, control](const std::string& running) {
        if (!control || !control->on_progress) return;
        size_t done = 0;
        for (const auto& step : session.steps) {
            if (step.state != cyxwiz::StudioDebuggerStepState::Pending &&
                step.state != cyxwiz::StudioDebuggerStepState::Running) {
                ++done;
            }
        }
        const float progress = session.steps.empty()
            ? 0.0f : static_cast<float>(done) / static_cast<float>(session.steps.size());
        control->on_progress(session.steps, running, progress);
    };
    auto begin_step = [&](const char* id) {
        step_started = std::chrono::steady_clock::now();
        if (auto* step = find_step(id)) {
            step->state = cyxwiz::StudioDebuggerStepState::Running;
            report_progress(step->name);
        }
    };
    auto end_step = [&](const char* id, cyxwiz::StudioDebuggerStepState state,
                        const std::string& detail) {
        if (auto* step = find_step(id)) {
            step->state = state;
            if (!detail.empty()) step->detail = detail;
            step->seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_started).count();
            report_progress("");
        }
    };
    auto stop_requested = [&]() {
        if (!stopped && control && control->should_stop && control->should_stop()) {
            stopped = true;
        }
        return stopped;
    };
    // Outcome from the steps, never from the last stage that ran.
    auto finalize_outcome = [&]() {
        for (auto& step : session.steps) {
            if (step.state == cyxwiz::StudioDebuggerStepState::Running) {
                step.state = stopped ? cyxwiz::StudioDebuggerStepState::Stopped
                                     : cyxwiz::StudioDebuggerStepState::Failed;
            } else if (step.state == cyxwiz::StudioDebuggerStepState::Pending) {
                step.state = cyxwiz::StudioDebuggerStepState::Skipped;
            }
        }
        const bool has_warnings = std::any_of(
            session.issues.begin(), session.issues.end(),
            [](const cyxwiz::ValidationIssue& issue) {
                return issue.level != cyxwiz::IssueLevel::Info;
            });
        session.outcome = cyxwiz::AggregateStudioDebuggerOutcome(
            session.steps, stopped, has_warnings);
        session.success = session.outcome == cyxwiz::StudioDebuggerOutcome::Passed ||
            session.outcome == cyxwiz::StudioDebuggerOutcome::NeedsAttention;
        if (session.success) {
            session.failure_summary.clear();
        } else if (session.failure_summary.empty()) {
            if (stopped) {
                session.failure_summary = "Stopped before all steps finished.";
            }
            for (const auto& step : session.steps) {
                if (!session.failure_summary.empty()) break;
                if ((step.state == cyxwiz::StudioDebuggerStepState::Failed ||
                     step.state == cyxwiz::StudioDebuggerStepState::Unsupported) &&
                    step.required) {
                    session.failure_summary = step.detail.empty()
                        ? step.name + " did not pass." : step.detail;
                }
            }
        }
        session.duration_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - run_started).count();
    };

    session.node_count = nodes.size();
    session.link_count = links.size();
    session.graph_hash = HashGraphStructure(nodes, links);
    session.run_id = MakeDebuggerRunId(session.graph_hash);
    const bool run_full = mode == cyxwiz::StudioDebuggerRunMode::FullWorkflow;
    const bool run_local_debug = run_full || mode == cyxwiz::StudioDebuggerRunMode::LocalDebug;
    const bool run_smoke = run_full || mode == cyxwiz::StudioDebuggerRunMode::SmokeRun;
    const bool run_runtime = run_full || mode == cyxwiz::StudioDebuggerRunMode::RuntimeTrace;

    const char* mode_name = "FullWorkflow";
    switch (mode) {
        case cyxwiz::StudioDebuggerRunMode::FullWorkflow: mode_name = "FullWorkflow"; break;
        case cyxwiz::StudioDebuggerRunMode::Preflight: mode_name = "Preflight"; break;
        case cyxwiz::StudioDebuggerRunMode::LocalDebug: mode_name = "LocalDebug"; break;
        case cyxwiz::StudioDebuggerRunMode::SmokeRun: mode_name = "SmokeRun"; break;
        case cyxwiz::StudioDebuggerRunMode::RuntimeTrace: mode_name = "RuntimeTrace"; break;
    }
    const size_t selected_sample_index = static_cast<size_t>(std::max(0, sample_index));
    cyxwiz::DebugSession debug_session = cyxwiz::DebugSessionManager::StartSession(
        session.run_id, mode_name, session.graph_hash, nodes, links, selected_sample_index);
    session.traces = debug_session.traces;
    session.studio_events = debug_session.studio_events;
    // Canvas positions let the debugger lay the graph out like the canvas.
    if (node_positions && !node_positions->empty()) {
        for (auto& trace : session.traces) {
            if (trace.phase != "GraphSnapshot" || !trace.payload.contains("nodes") ||
                !trace.payload["nodes"].is_array()) {
                continue;
            }
            for (auto& node : trace.payload["nodes"]) {
                const auto position = node_positions->find(node.value("id", -1));
                if (position != node_positions->end()) {
                    node["x"] = position->second.first;
                    node["y"] = position->second.second;
                }
            }
        }
    }
    session.sample_summary = std::string("Studio Debugger mode: ") + mode_name +
        " | sample " + std::to_string(selected_sample_index);
    // tofix96: training evidence is this run's execution truth only while
    // that training is live. A finished run is shown as historical and does
    // not feed this run's execution summary, analyzers or recommendations
    // (Runtime Trace inspects it on purpose, so it may analyze it).
    const auto training_trace =
        cyxwiz::TrainingTraceCollector::LatestTrace();
    if (training_trace.available) {
        session.training_trace = training_trace;
        const bool live = cyxwiz::IsTrainingTraceLive(
            training_trace,
            cyxwiz::TrainingManager::Instance().IsTrainingActive());
        session.training_trace_historical = !live;
        if (live) {
            session.execution =
                cyxwiz::MakeDebugRunExecutionSummary(
                    session.training_trace,
                    mode == cyxwiz::StudioDebuggerRunMode::RuntimeTrace);
        }
    }
    const cyxwiz::TrainingTraceSummary no_training_evidence;
    const bool use_training_evidence = !session.training_trace_historical ||
        mode == cyxwiz::StudioDebuggerRunMode::RuntimeTrace;
    auto training_evidence = [&]() -> const cyxwiz::TrainingTraceSummary& {
        return use_training_evidence ? session.training_trace : no_training_evidence;
    };
    const cyxwiz::CrashRunSummary no_crash_evidence;
    auto crash_evidence = [&]() -> const cyxwiz::CrashRunSummary& {
        return use_training_evidence ? session.last_run : no_crash_evidence;
    };
    if (auto last_run = cyxwiz::CrashRunRecorder::LoadLastRun()) {
        session.last_run = *last_run;
    }
    const std::string& run_id = session.run_id;
    cyxwiz::DebugRunReplayCapsule replay_capsule =
        cyxwiz::MakeDebugRunReplayCapsule(
            debug_session, nullptr, session.execution);
    auto collect_runtime_diagnostics = [&session, &run_id, use_training_evidence]() {
        cyxwiz::RuntimeLogSnapshotRequest request;
        request.limit = cyxwiz::RuntimeLogStore::kDefaultCapacity;
        const auto snapshot =
            cyxwiz::RuntimeLogStore::Instance().Snapshot(request);
        std::vector<cyxwiz::RuntimeLogEvent> events;
        for (const auto& event : snapshot.events) {
            const bool same_debug_run = event.run_id == run_id;
            const bool linked_training_run = use_training_evidence &&
                !session.training_trace.run_id.empty() &&
                event.run_id == session.training_trace.run_id;
            if ((same_debug_run || linked_training_run) &&
                (!event.primary_error_code.empty() ||
                 !event.issue_codes.empty())) {
                events.push_back(event);
            }
        }
        return events;
    };
    auto append_error_timeline =
        [&session, &run_id, &collect_runtime_diagnostics, &crash_evidence]() {
            const bool already_present = std::any_of(
                session.traces.begin(), session.traces.end(),
                [](const cyxwiz::DebugTraceRecord& trace) {
                    return trace.phase == "ErrorCodeTimeline";
                });
            if (already_present) {
                return;
            }
            cyxwiz::DebugErrorCodeTimeline timeline;
            session.traces.push_back(timeline.BuildTrace(
                run_id,
                session.traces,
                collect_runtime_diagnostics(),
                crash_evidence()));
        };
    auto append_training_stall_analysis =
        [&session, &run_id, &training_evidence](
            const cyxwiz::TrainingConfiguration* training_config) {
            const bool already_present = std::any_of(
                session.traces.begin(), session.traces.end(),
                [](const cyxwiz::DebugTraceRecord& trace) {
                    return trace.phase == "TrainingStallAnalysis";
                });
            if (already_present) {
                return;
            }
            cyxwiz::DebugTrainingStallConfigEvidence evidence;
            if (training_config) {
                evidence.learning_rate_available = true;
                evidence.learning_rate = training_config->learning_rate;
            }
            cyxwiz::DebugTrainingStallDetector detector;
            session.traces.push_back(detector.BuildTrace(
                run_id, session.traces, training_evidence(), evidence));
        };
    const cyxwiz::TrainingConfiguration* loss_metric_config = nullptr;
    auto save_session = [&session, &replay_capsule, &loss_metric_config,
                         &nodes, &finalize_outcome]() {
        finalize_outcome();
        const bool loss_metric_present = std::any_of(
            session.traces.begin(), session.traces.end(),
            [](const cyxwiz::DebugTraceRecord& trace) {
                return trace.phase == "LossMetricExplanation";
            });
        if (loss_metric_config && !loss_metric_present) {
            cyxwiz::DebugLossMetricExplainer explainer;
            session.traces.push_back(explainer.BuildTrace(
                session.run_id, *loss_metric_config, nodes, session.traces));
        }
        const bool has_backend_placements = std::any_of(
            session.traces.begin(), session.traces.end(),
            [](const cyxwiz::DebugTraceRecord& trace) {
                return trace.phase == "BackendPlacement";
            });
        const bool backend_audit_present = std::any_of(
            session.traces.begin(), session.traces.end(),
            [](const cyxwiz::DebugTraceRecord& trace) {
                return trace.phase == "BackendDecisionAudit";
            });
        if (has_backend_placements && !backend_audit_present) {
            cyxwiz::DebugRuntimeBackendClassifier classifier;
            session.traces.push_back(classifier.BuildAuditTrace(
                session.run_id, session.traces, session.execution));
        }
        const bool gradient_health_present = std::any_of(
            session.traces.begin(), session.traces.end(),
            [](const cyxwiz::DebugTraceRecord& trace) {
                return trace.phase == "GradientHealth";
            });
        if (!gradient_health_present) {
            cyxwiz::DebugGradientHealthAnalyzer analyzer;
            session.traces.push_back(analyzer.BuildTrace(
                session.run_id, session.traces));
        }
        const bool lifecycle_present = std::any_of(
            session.traces.begin(), session.traces.end(),
            [](const cyxwiz::DebugTraceRecord& trace) {
                return trace.phase == "TensorLifecycle";
            });
        if (!lifecycle_present) {
            cyxwiz::DebugMemoryOwnershipTracer memory_tracer;
            session.traces.push_back(
                memory_tracer.BuildTensorLifecycleTrace(
                    session.run_id,
                    session.traces,
                    session.execution));
        }
        session.traces.push_back(
            cyxwiz::BuildStudioDebuggerRunSummaryTrace(session));
        cyxwiz::DebugRunStoreRecord record;
        record.summary.run_id = session.run_id;
        record.summary.timestamp = NowLocalTimestampForDebugStore();
        record.summary.graph_hash = session.graph_hash;
        record.summary.success = session.success;
        record.summary.summary = session.failure_summary.empty()
            ? std::string(cyxwiz::StudioDebuggerRunModeLabel(session.mode)) + ": " +
                  cyxwiz::StudioDebuggerOutcomeLabel(session.outcome)
            : session.failure_summary;
        record.summary.execution = session.execution;
        record.replay_capsule = replay_capsule;
        record.issues = session.issues;
        record.traces = session.traces;
        record.studio_events = session.studio_events;
        record.recommendations = session.recommendations;
        cyxwiz::DebugRunStore::Save(record);
        session.run_history = cyxwiz::DebugRunStore::ListRecent(8);
    };

    session.studio_events.push_back({
        run_id,
        NowLocalTimestampForDebugStore(),
        session.graph_hash,
        -1,
        "StudioDebugger.Run",
        "started",
        std::string("Studio Debugger started mode: ") + mode_name
    });

    if (mode == cyxwiz::StudioDebuggerRunMode::RuntimeTrace) {
        begin_step("runtime");
        cyxwiz::DebugRecommendationEngine recommendation_engine;
        cyxwiz::DebugSlowPathDetector slow_path_detector;
        session.traces.push_back(slow_path_detector.BuildTrace(
            run_id, session.traces, training_evidence()));
        append_training_stall_analysis(nullptr);
        append_error_timeline();
        session.recommendations = recommendation_engine.Build(
            session.traces, session.issues, session.smoke_result,
            crash_evidence(), training_evidence());
        const bool has_runtime_evidence =
            session.training_trace.available || !session.last_run.run_id.empty();
        end_step("runtime",
                 has_runtime_evidence ? cyxwiz::StudioDebuggerStepState::Passed
                                      : cyxwiz::StudioDebuggerStepState::Warning,
                 has_runtime_evidence
                     ? (session.training_trace_historical
                            ? "Loaded historical training evidence."
                            : "Loaded runtime evidence.")
                     : "No training trace or crash heartbeat was found.");
        session.graph_summary =
            "Runtime Trace loaded crash heartbeat and training trace without executing graph debug.";
        save_session();
        return true;
    }

    cyxwiz::TrainingConfiguration config;
    bool compile_success = false;
    bool compile_completed = false;
    std::string compile_summary;
    begin_step("compile");
    try {
        cyxwiz::GraphCompiler compiler;
        config = compiler.Compile(nodes, links);
        compile_completed = true;
        session.graph_domain = cyxwiz::StudioDebuggerDomainLabel(config.preprocessing_domain);
        session.issues = config.issues;
        compile_success = config.is_valid;
        if (compile_success) {
            loss_metric_config = &config;
        }

        std::ostringstream out;
        out << "Layers:          " << config.layers.size() << "\n";
        out << "Input size:      " << config.input_size << "\n";
        out << "Output size:     " << config.output_size << "\n";
        out << "Batch size:      " << config.batch_size << "\n";
        out << "Dataset:         " << (config.dataset_name.empty() ? "(none)" : config.dataset_name) << "\n";
        out << "Issues:          " << config.issues.size() << "\n";
        compile_summary = out.str();
        session.graph_summary = compile_summary;
    } catch (const std::exception& e) {
        session.failure_summary = std::string("Compile threw: ") + e.what();
        session.success = false;
        session.issues.push_back({
            cyxwiz::IssueLevel::Error,
            -1,
            "",
            session.failure_summary,
            cyxwiz::errors::Compiler::InvariantViolation
        });
        compile_summary = session.failure_summary;
    }

    replay_capsule = cyxwiz::MakeDebugRunReplayCapsule(
        debug_session,
        &config,
        session.execution,
        run_smoke ? 100 : 0);

    cyxwiz::DebugTraceRecord compile_trace = cyxwiz::DebugNodeTraceContract::Make(
        run_id,
        -1,
        "GraphCompiler",
        "Compile",
        "Compile",
        cyxwiz::DebugTraceRole::CompileArtifact,
        {},
        {},
        "graph",
        "GraphCompiler",
        compile_success ? "passed" : "failed");
    compile_trace.issues = session.issues;
    compile_trace.payload["success"] = compile_success;
    compile_trace.payload["node_count"] = nodes.size();
    compile_trace.payload["link_count"] = links.size();
    cyxwiz::DebugNodeTraceContract::AttachIssueSummary(compile_trace, session.issues);
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        compile_trace,
        "compile",
        "GraphCompiler",
        "cyxwiz-engine/src/core/graph_compiler.cpp",
        "cyxwiz::GraphCompiler::Compile");
    compile_trace.payload["summary"] = compile_summary;
    session.traces.push_back(std::move(compile_trace));

    if (!config.backend_placements.empty()) {
        cyxwiz::DebugRuntimeBackendClassifier backend_classifier;
        for (const auto& placement : config.backend_placements) {
            session.traces.push_back(backend_classifier.BuildPlacementTrace(
                run_id, session.graph_hash, placement));
        }
        session.studio_events.push_back({
            run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
            "BackendPlacementAudit", "captured",
            "Captured " + std::to_string(config.backend_placements.size()) +
                " compiler placement decision(s); actual runtime backend "
                "remains unobserved without same-run execution evidence."
        });
    }

    auto append_node_explanation =
        [&session, &run_id, &nodes, &links, explain_node_id](
            const cyxwiz::TrainingConfiguration* explanation_config) {
            if (explain_node_id < 0) {
                return;
            }
            const auto node_it = std::find_if(
                nodes.begin(), nodes.end(),
                [explain_node_id](const MLNode& node) {
                    return node.id == explain_node_id;
                });
            if (node_it == nodes.end()) {
                return;
            }
            std::string node_type = node_it->name;
            if (const auto* metadata =
                    cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(
                        node_it->type)) {
                node_type = metadata->name;
            }
            cyxwiz::DebugNodeInspector inspector;
            session.traces.push_back(inspector.BuildExplanationTrace(
                run_id,
                session.graph_hash,
                *node_it,
                node_type,
                explanation_config,
                nodes,
                links,
                session.traces,
                session.issues,
                session.recommendations));
            session.studio_events.push_back({
                run_id,
                NowLocalTimestampForDebugStore(),
                session.graph_hash,
                explain_node_id,
                "StudioDebugger.ExplainNode",
                "captured",
                "Captured node explanation from the frozen debug snapshot."
            });
        };

    if (!compile_success) {
        session.success = false;
        if (session.failure_summary.empty()) {
            session.failure_summary = "Compile gate failed.";
        }
        end_step("compile", cyxwiz::StudioDebuggerStepState::Failed,
                 session.failure_summary);
        session.studio_events.push_back({
            run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
            "Compile", "failed", session.failure_summary
        });
        if (explain_node_id >= 0) {
            cyxwiz::DebugRecommendationEngine recommendation_engine;
            session.recommendations = recommendation_engine.Build(
                session.traces, session.issues, session.smoke_result,
                crash_evidence(), training_evidence());
        }
        append_error_timeline();
        append_node_explanation(compile_completed ? &config : nullptr);
        save_session();
        return false;
    }
    session.studio_events.push_back({
        run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
        "Compile", "passed", "Compile gate passed before Local Debug."
    });
    end_step("compile", cyxwiz::StudioDebuggerStepState::Passed,
             std::to_string(config.layers.size()) + " layers, " +
                 session.graph_domain + " data");
    // Smoke capability from the compiled domain, before Smoke is dispatched.
    const cyxwiz::SmokeCapability smoke_capability =
        cyxwiz::EvaluateSmokeCapability(session.graph_domain);
    if (auto* smoke_step = find_step("smoke"); smoke_step && !smoke_capability.supported) {
        smoke_step->state = cyxwiz::StudioDebuggerStepState::Unsupported;
        smoke_step->detail = smoke_capability.reason + " " + smoke_capability.alternative;
        if (run_full) {
            smoke_step->required = false;
        }
    }
    auto finish_stopped = [&]() {
        session.studio_events.push_back({
            run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
            "StudioDebugger.Run", "stopped", "Stopped by the user."
        });
        append_error_timeline();
        save_session();
        return false;
    };
    if (stop_requested()) {
        return finish_stopped();
    }
    begin_step("preflight");

    cyxwiz::PreflightValidator preflight_validator;
    session.preflight = preflight_validator.Validate(config, nodes, links, session.graph_hash);
    session.preflight_summary = session.preflight.summary;
    cyxwiz::DebugTraceRecord preflight_trace = cyxwiz::DebugNodeTraceContract::Make(
        run_id,
        -1,
        "PreflightValidator",
        "Preflight",
        "Preflight",
        session.preflight.ready
            ? cyxwiz::DebugTraceRole::CompileArtifact
            : cyxwiz::DebugTraceRole::Warning,
        {},
        {},
        "graph",
        "PreflightValidator",
        session.preflight.ready ? "ready" : "blocked");
    preflight_trace.issues = session.preflight.issues;
    preflight_trace.payload["ready"] = session.preflight.ready;
    preflight_trace.payload["success"] = session.preflight.ready;
    cyxwiz::DebugNodeTraceContract::AttachIssueSummary(preflight_trace, session.preflight.issues);
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        preflight_trace,
        "preflight",
        "PreflightValidator",
        "cyxwiz-engine/src/core/preflight_validator.cpp",
        "cyxwiz::PreflightValidator::Validate");
    preflight_trace.payload["summary"] = session.preflight.summary;
    session.traces.push_back(std::move(preflight_trace));
    session.studio_events.push_back({
        run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
        "Preflight", session.preflight.ready ? "ready" : "blocked",
        session.preflight.ready ? "Preflight checks passed." : "Preflight reported blocking issues."
    });
    end_step("preflight",
             session.preflight.ready ? cyxwiz::StudioDebuggerStepState::Passed
                                     : cyxwiz::StudioDebuggerStepState::Failed,
             session.preflight.ready ? std::string{}
                                     : "Preflight reported blocking issues.");

    if (mode == cyxwiz::StudioDebuggerRunMode::Preflight) {
        session.failure_summary = session.preflight.ready ? "" : "Preflight reported issues.";
        append_training_stall_analysis(&config);
        cyxwiz::DebugRecommendationEngine recommendation_engine;
        session.recommendations = recommendation_engine.Build(
            session.traces, session.issues, session.smoke_result,
            crash_evidence(), training_evidence());
        append_error_timeline();
        append_node_explanation(&config);
        save_session();
        return session.success;
    }

    if (stop_requested()) {
        return finish_stopped();
    }
    if ((run_smoke || run_local_debug) && !config.dataset_name.empty()) {
        auto arrow_dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(
            config.dataset_name);
        if (arrow_dataset) {
            cyxwiz::DebugOperatorTraceProducer operator_trace_producer;
            auto operator_traces = operator_trace_producer.TracePreprocessingGraph(
                run_id,
                nodes,
                links,
                arrow_dataset->GetArrowTable(),
                config.dataset_name,
                selected_sample_index);
            const bool has_operator_traces = !operator_traces.empty();
            session.traces.insert(
                session.traces.end(),
                std::make_move_iterator(operator_traces.begin()),
                std::make_move_iterator(operator_traces.end()));
            if (has_operator_traces) {
                session.studio_events.push_back({
                    run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
                    "OperatorPreprocessingTrace", "captured",
                    "Captured operator-backed Arrow preprocessing trace."
                });
            }
        }
    }
    if (run_smoke && !smoke_capability.supported) {
        session.studio_events.push_back({
            run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
            "SmokeRun", "unsupported", smoke_capability.reason
        });
    } else if (run_smoke) {
        begin_step("smoke");
        cyxwiz::SmokeRunExecutor smoke_executor;
        cyxwiz::SmokeRunOptions smoke_options;
        smoke_options.cache_config = GraphMaterializationCacheConfig(
            inputs ? inputs->project_root : std::filesystem::path{});
        {
            auto& project = cyxwiz::ProjectManager::Instance();
            smoke_options.roots = {inputs ? inputs->project_root.string() : project.GetProjectRoot(),
                                   project.GetArtifactsPath(), project.GetExportsPath(),
                                   project.GetIngestionCachePath()};
        }
        if (inputs && inputs->rebuild_prepared_data) {
            smoke_options.cache_config.mode = cyxwiz::MaterializationCacheMode::Rebuild;
        }
        smoke_options.should_stop = [&stop_requested]() { return stop_requested(); };
        try {
            session.smoke_result = smoke_executor.RunTextSmoke(
                config, nodes, links, run_id, 100, smoke_options);
        } catch (const std::exception& e) {
            // A Smoke failure is this step's result; the run still finishes
            // and is saved.
            session.smoke_result = {};
            session.smoke_result.supported = true;
            session.smoke_result.summary = std::string("Smoke Run threw: ") + e.what();
            session.smoke_result.issues.push_back({
                cyxwiz::IssueLevel::Error, -1, "SmokeRun", session.smoke_result.summary,
                cyxwiz::errors::Training::TrainingExecutionFailed});
        }
        session.traces.insert(session.traces.end(),
                              std::make_move_iterator(session.smoke_result.traces.begin()),
                              std::make_move_iterator(session.smoke_result.traces.end()));
        if (session.smoke_result.stopped) {
            session.studio_events.push_back({
                run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
                "SmokeRun", "stopped", session.smoke_result.summary
            });
        } else if (session.smoke_result.supported) {
            session.studio_events.push_back({
                run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
                "SmokeRun", session.smoke_result.success ? "passed" : "failed",
                session.smoke_result.summary
            });
        } else {
            session.studio_events.push_back({
                run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
                "SmokeRun", "unsupported", session.smoke_result.summary
            });
        }
        if (!session.smoke_result.issues.empty()) {
            session.issues.insert(session.issues.end(),
                                  session.smoke_result.issues.begin(),
                                  session.smoke_result.issues.end());
        }
        end_step("smoke",
                 session.smoke_result.stopped
                     ? cyxwiz::StudioDebuggerStepState::Stopped
                 : !session.smoke_result.supported
                     ? cyxwiz::StudioDebuggerStepState::Unsupported
                     : (session.smoke_result.success
                            ? cyxwiz::StudioDebuggerStepState::Passed
                            : cyxwiz::StudioDebuggerStepState::Failed),
                 session.smoke_result.summary);
    }
    if (run_local_debug && stop_requested()) {
        return finish_stopped();
    }

    if (run_local_debug) begin_step("local_debug");
    if (run_local_debug) try {
        // The debugger continues to use the compiled configuration for
        // explanations, stall analysis, replay metadata, and persistence
        // after Local Debug completes. Preserve that session source of truth;
        // this once-per-debug-run copy is outside training hot loops.
        cyxwiz::DebugExecutor exe(config);
        session.debug_result = exe.Run();
        session.has_debug_result = true;
        session.success = session.debug_result.success;
        session.failure_summary = session.debug_result.failure_summary;

        if (!session.debug_result.issues.empty()) {
            session.issues.insert(
                session.issues.end(),
                session.debug_result.issues.begin(),
                session.debug_result.issues.end());
        }

        if (!session.debug_result.model_build_traces.empty()) {
            cyxwiz::DebugGraphTraceExecutor graph_trace_executor;
            auto model_build_traces = graph_trace_executor.TraceSteps(
                run_id, session.debug_result.model_build_traces);
            session.traces.insert(
                session.traces.end(),
                std::make_move_iterator(model_build_traces.begin()),
                std::make_move_iterator(model_build_traces.end()));
        }

        const bool local_debug_build_failed =
            session.debug_result.reached == cyxwiz::DebugStage::BuildModel &&
            !session.debug_result.success &&
            !session.debug_result.failure_summary.empty();
        if (local_debug_build_failed) {
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                -1,
                "LocalDebugBuildModel",
                "BuildModel",
                "BuildModel",
                cyxwiz::DebugTraceRole::Error,
                {},
                {},
                "model",
                "LocalDebug",
                "failed");
            record.payload["build_model_reached"] = true;
            record.payload["success"] = false;
            cyxwiz::DebugNodeTraceContract::AddError(
                record,
                session.debug_result.failure_summary,
                cyxwiz::errors::Training::ModelBuildFailed);
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_build_model",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }

        for (const auto& trace : session.debug_result.layer_traces) {
            const bool has_shape_mismatch = trace.has_shape_mismatch();
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                trace.node_id,
                trace.name,
                [&trace]() -> std::string {
                    if (const auto* metadata =
                            cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(trace.type)) {
                        return metadata->name;
                    }
                    return "Node type " + std::to_string(static_cast<int>(trace.type));
                }(),
                "Forward",
                cyxwiz::DebugTraceRole::Activation,
                trace.actual_input_shape,
                trace.actual_shape,
                "float32",
                "LocalDebug",
                (trace.has_nan || trace.has_inf) ? "warning" :
                    (has_shape_mismatch ? "shape_mismatch" : "ok"));
            record.duration_ms = trace.forward_ms;
            record.payload["module_index"] = trace.module_index;
            record.payload["compiled_layer_index"] =
                trace.compiled_layer_index;
            record.payload["module_name"] = trace.module_name;
            record.payload["predicted_input_shape"] =
                trace.predicted_input_shape;
            record.payload["actual_input_shape"] =
                trace.actual_input_shape;
            record.payload["input_shape_matches"] =
                trace.input_shape_matches;
            record.payload["predicted_shape"] = trace.predicted_shape;
            record.payload["actual_shape"] = trace.actual_shape;
            record.payload["output_shape_matches"] = trace.shape_matches;
            record.payload["shape_matches"] = trace.shape_matches;
            record.payload["all_shapes_match"] = !has_shape_mismatch;
            record.payload["shape_mismatch_kind"] =
                cyxwiz::ShapeMismatchKindName(trace.shape_mismatch);
            record.payload["is_first_shape_mismatch"] =
                trace.is_first_shape_mismatch;
            record.payload["upstream_node_id"] = trace.upstream_node_id;
            record.payload["upstream_node_name"] =
                trace.upstream_node_name;
            record.payload["has_nan"] = trace.has_nan;
            record.payload["has_inf"] = trace.has_inf;
            cyxwiz::AttachDebugNumericsPayload(
                record, trace.numerics, "activation");
            record.payload["success"] = !has_shape_mismatch &&
                !trace.has_nan && !trace.has_inf;
            if (has_shape_mismatch) {
                record.issues.push_back({
                    cyxwiz::IssueLevel::Warning,
                    trace.node_id,
                    trace.name,
                    std::string(
                        "Local Debug shape differed from compiler prediction") +
                        (trace.is_first_shape_mismatch
                            ? " at the first observed divergence."
                            : "."),
                    cyxwiz::errors::Compiler::TensorShapeMismatch
                });
                record.payload["suggested_fix"] = trace.upstream_node_id >= 0
                    ? "Inspect this node's configured shapes and the recorded "
                      "upstream graph node."
                    : "Inspect this node's configured shapes and the Local "
                      "Debug model input.";
            }
            if (trace.has_nan || trace.has_inf) {
                record.issues.push_back({
                    cyxwiz::IssueLevel::Warning,
                    trace.node_id,
                    trace.name,
                    "Local Debug forward output contained non-finite values.",
                    cyxwiz::errors::Training::TrainingExecutionFailed
                });
            }
            cyxwiz::DebugNodeTraceContract::AttachIssueSummary(record, record.issues);
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_forward",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }

        const bool local_debug_forward_failed =
            session.debug_result.reached == cyxwiz::DebugStage::Forward &&
            !session.debug_result.success &&
            !session.debug_result.failure_summary.empty();
        if (local_debug_forward_failed) {
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                -1,
                "LocalDebugForward",
                "Forward",
                "Forward",
                cyxwiz::DebugTraceRole::Activation,
                {},
                {},
                "float32",
                "LocalDebug",
                "failed");
            record.payload["forward_reached"] = true;
            record.payload["success"] = false;
            cyxwiz::DebugNodeTraceContract::AddError(
                record,
                session.debug_result.failure_summary,
                cyxwiz::errors::Training::TrainingExecutionFailed);
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_forward",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }

        const bool has_local_debug_loss =
            session.debug_result.reached == cyxwiz::DebugStage::Loss ||
            session.debug_result.reached == cyxwiz::DebugStage::Backward ||
            session.debug_result.reached == cyxwiz::DebugStage::OptimizerStep ||
            session.debug_result.reached == cyxwiz::DebugStage::Complete;
        if (has_local_debug_loss) {
            const bool loss_stage_failed =
                session.debug_result.reached == cyxwiz::DebugStage::Loss &&
                !session.debug_result.success &&
                !session.debug_result.failure_summary.empty();
            const auto loss_node = std::find_if(
                nodes.begin(), nodes.end(),
                [&config](const MLNode& node) {
                    return node.id == config.loss_node_id;
                });
            const std::string loss_node_name = loss_node == nodes.end()
                ? config.GetLossName()
                : loss_node->name;
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                config.loss_node_id,
                loss_node_name,
                config.GetLossName(),
                "Loss",
                cyxwiz::DebugTraceRole::Loss,
                {},
                {},
                "float32",
                "LocalDebug",
                session.debug_result.loss_finite ? "ok" : "failed");
            record.payload["loss"] = session.debug_result.loss_value;
            record.payload["loss_finite"] = session.debug_result.loss_finite;
            record.payload["loss_stage_failed"] = loss_stage_failed;
            record.payload["prediction_shape"] =
                session.debug_result.loss_prediction_shape;
            record.payload["target_shape"] =
                session.debug_result.loss_target_shape;
            record.payload["prediction_target_shape_check_available"] =
                session.debug_result.loss_target_compatibility.available;
            record.payload["prediction_target_shapes_compatible"] =
                session.debug_result.loss_target_compatibility.compatible;
            record.payload["prediction_target_shape_reason"] =
                session.debug_result.loss_target_compatibility.reason;
            record.payload["success"] = session.debug_result.loss_finite && !loss_stage_failed;
            if (!session.debug_result.loss_finite) {
                cyxwiz::DebugNodeTraceContract::AddError(
                    record,
                    loss_stage_failed
                        ? session.debug_result.failure_summary
                        : "Local Debug loss was not finite.",
                    cyxwiz::errors::Training::TrainingExecutionFailed);
            }
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_loss",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }

        const bool local_debug_backward_failed =
            session.debug_result.reached == cyxwiz::DebugStage::Backward &&
            !session.debug_result.success &&
            !session.debug_result.failure_summary.empty();
        if (local_debug_backward_failed) {
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                -1,
                "LocalDebugBackward",
                "Backward",
                "Backward",
                cyxwiz::DebugTraceRole::Error,
                {},
                {},
                "float32",
                "LocalDebug",
                "failed");
            record.payload["backward_reached"] = true;
            record.payload["success"] = false;
            cyxwiz::DebugNodeTraceContract::AddError(
                record,
                session.debug_result.failure_summary,
                cyxwiz::errors::Training::TrainingExecutionFailed);
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_backward",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }

        for (const auto& grad : session.debug_result.grad_norms) {
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                grad.node_id,
                grad.node_name.empty() ? grad.param_name : grad.node_name,
                "Parameter",
                "Backward",
                cyxwiz::DebugTraceRole::Gradient,
                {},
                grad.gradient_shape,
                "float32",
                "LocalDebug",
                !grad.has_gradient ? "missing_gradient" :
                    (grad.is_nan ? "nan" : (grad.is_zero ? "zero" : "ok")));
            record.payload["l2_norm"] = grad.l2_norm;
            record.payload["gradient_l2_norm"] = grad.l2_norm;
            record.payload["parameter_l2_norm"] =
                grad.parameter_l2_norm;
            record.payload["grad_parameter_ratio"] =
                grad.grad_parameter_ratio;
            record.payload["update_l2_norm"] = grad.update_l2_norm;
            record.payload["update_parameter_ratio"] =
                grad.update_parameter_ratio;
            record.payload["relative_norm_denominator_floor"] =
                cyxwiz::kDebugNormDenominatorFloor;
            record.payload["update_observed"] = grad.update_observed;
            record.payload["parameter_name"] = grad.param_name;
            record.payload["parameter_shape"] = grad.parameter_shape;
            record.payload["gradient_shape"] = grad.gradient_shape;
            record.payload["parameter_gradient_shapes_match"] =
                grad.has_gradient &&
                grad.parameter_shape == grad.gradient_shape;
            record.payload["module_index"] = grad.layer_index;
            record.payload["compiled_layer_index"] =
                grad.compiled_layer_index;
            record.payload["has_gradient"] = grad.has_gradient;
            record.payload["is_nan"] = grad.is_nan;
            record.payload["is_inf"] = grad.is_inf;
            record.payload["is_zero"] = grad.is_zero;
            cyxwiz::AttachDebugNumericsPayload(
                record, grad.numerics, "gradient");
            record.payload["missing_gradient_reason"] =
                grad.missing_gradient_reason;
            record.payload["success"] = grad.has_gradient &&
                !grad.is_nan && !grad.is_zero;
            if (!grad.has_gradient) {
                record.issues.push_back({
                    cyxwiz::IssueLevel::Warning,
                    grad.node_id,
                    grad.node_name.empty() ? grad.param_name : grad.node_name,
                    grad.missing_gradient_reason,
                    cyxwiz::errors::Training::TrainingExecutionFailed
                });
            } else if (grad.is_nan) {
                record.issues.push_back({
                    cyxwiz::IssueLevel::Warning,
                    grad.node_id,
                    grad.node_name.empty() ? grad.param_name : grad.node_name,
                    "Local Debug gradient norm was NaN.",
                    cyxwiz::errors::Training::TrainingExecutionFailed
                });
            } else if (grad.is_zero) {
                record.issues.push_back({
                    cyxwiz::IssueLevel::Warning,
                    grad.node_id,
                    grad.node_name.empty() ? grad.param_name : grad.node_name,
                    "Local Debug gradient norm was zero.",
                    cyxwiz::errors::Training::TrainingExecutionFailed
                });
            }
            cyxwiz::DebugNodeTraceContract::AttachIssueSummary(record, record.issues);
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_backward",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }
        const bool reached_optimizer_step =
            session.debug_result.reached == cyxwiz::DebugStage::OptimizerStep ||
            session.debug_result.reached == cyxwiz::DebugStage::Complete;
        if (reached_optimizer_step) {
            const bool optimizer_failed =
                session.debug_result.reached == cyxwiz::DebugStage::OptimizerStep &&
                !session.debug_result.success &&
                !session.debug_result.failure_summary.empty();
            cyxwiz::DebugTraceRecord record = cyxwiz::DebugNodeTraceContract::Make(
                run_id,
                -1,
                "LocalDebugOptimizerStep",
                "OptimizerStep",
                "OptimizerStep",
                cyxwiz::DebugTraceRole::OptimizerStep,
                {},
                {},
                "optimizer",
                "LocalDebug",
                optimizer_failed ? "failed" : "ok");
            record.payload["optimizer_step_reached"] = reached_optimizer_step;
            record.payload["success"] = !optimizer_failed;
            if (optimizer_failed) {
                cyxwiz::DebugNodeTraceContract::AddError(
                    record,
                    session.debug_result.failure_summary,
                    cyxwiz::errors::Training::TrainingExecutionFailed);
            }
            cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
                record,
                "local_debug_optimizer",
                "DebugExecutor",
                "cyxwiz-engine/src/core/debug_executor.cpp",
                "cyxwiz::DebugExecutor::Run");
            session.traces.push_back(std::move(record));
        }
        session.studio_events.push_back({
            run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
            "LocalDebug", session.debug_result.success ? "passed" : "failed",
            session.debug_result.success ? "Local Debug completed." : session.debug_result.failure_summary
        });

        std::ostringstream out;
        out << session.graph_summary;
        out << "\nDebugger:\n";
        const char* stage_name = "Unknown";
        switch (session.debug_result.reached) {
            case cyxwiz::DebugStage::NotRun:        stage_name = "NotRun"; break;
            case cyxwiz::DebugStage::BuildModel:    stage_name = "BuildModel"; break;
            case cyxwiz::DebugStage::Forward:       stage_name = "Forward"; break;
            case cyxwiz::DebugStage::Loss:          stage_name = "Loss"; break;
            case cyxwiz::DebugStage::Backward:      stage_name = "Backward"; break;
            case cyxwiz::DebugStage::OptimizerStep: stage_name = "OptimizerStep"; break;
            case cyxwiz::DebugStage::Complete:      stage_name = "Complete"; break;
        }
        out << "  Stage: " << stage_name << "\n";
        out << "  Forward time: " << session.debug_result.forward_total_ms << " ms\n";
        out << "  Backward time: " << session.debug_result.backward_total_ms << " ms\n";
        out << "  Optimizer time: " << session.debug_result.optimizer_step_ms << " ms\n";
        out << "  Loss: " << session.debug_result.loss_value
            << (session.debug_result.loss_finite ? "" : " (NON-FINITE)") << "\n";
        out << "  Params with grad: " << session.debug_result.params_with_grad << "\n";
        out << "  Params missing: " << session.debug_result.params_missing_grad << "\n";
        session.graph_summary = out.str();
        end_step("local_debug",
                 session.debug_result.success ? cyxwiz::StudioDebuggerStepState::Passed
                                              : cyxwiz::StudioDebuggerStepState::Failed,
                 session.debug_result.success ? std::string("Stage reached: ") + stage_name
                                              : session.debug_result.failure_summary);
    } catch (const std::exception& e) {
        session.failure_summary = std::string("Debug run threw: ") + e.what();
        end_step("local_debug", cyxwiz::StudioDebuggerStepState::Failed,
                 session.failure_summary);
        session.success = false;
        session.issues.push_back({
            cyxwiz::IssueLevel::Error,
            -1,
            "",
            session.failure_summary,
            cyxwiz::errors::Training::TrainingExecutionFailed
        });
        session.studio_events.push_back({
            run_id, NowLocalTimestampForDebugStore(), session.graph_hash, -1,
            "LocalDebug", "failed", session.failure_summary
        });
        append_error_timeline();
        save_session();
        return false;
    }

    // tofix96: the outcome comes from the per-step requirements in
    // finalize_outcome(); a later passing stage never erases an earlier
    // required failure.
    if (run_runtime && stop_requested()) {
        return finish_stopped();
    }

    if (run_runtime) {
        begin_step("runtime");
        const bool live_evidence =
            session.training_trace.available && !session.training_trace_historical;
        end_step("runtime",
                 live_evidence ? cyxwiz::StudioDebuggerStepState::Passed
                               : cyxwiz::StudioDebuggerStepState::Skipped,
                 live_evidence
                     ? "Live training " + session.training_trace.run_id
                     : (session.training_trace.available
                            ? "Only historical training evidence (" +
                                  session.training_trace.run_id + "); shown in Runtime."
                            : "No training is running."));
    }
    cyxwiz::DebugSlowPathLocalTimings local_timings;
    if (session.has_debug_result) {
        local_timings.forward_available =
            session.debug_result.forward_total_ms > 0.0f;
        local_timings.forward_ms = session.debug_result.forward_total_ms;
        local_timings.backward_available =
            session.debug_result.backward_total_ms > 0.0f;
        local_timings.backward_ms = session.debug_result.backward_total_ms;
        local_timings.optimizer_available =
            session.debug_result.optimizer_step_ms > 0.0f;
        local_timings.optimizer_ms = session.debug_result.optimizer_step_ms;
    }
    cyxwiz::DebugSlowPathDetector slow_path_detector;
    session.traces.push_back(slow_path_detector.BuildTrace(
        run_id, session.traces, training_evidence(), local_timings));
    append_training_stall_analysis(&config);
    append_error_timeline();
    cyxwiz::DebugRecommendationEngine recommendation_engine;
    session.recommendations = recommendation_engine.Build(
        session.traces, session.issues, session.smoke_result,
        crash_evidence(), training_evidence());

    append_node_explanation(&config);

    save_session();
    return session.success;
}

void MainWindow::BuildCompileResult(const std::vector<MLNode>& nodes,
                                     const std::vector<NodeLink>& links) {
    spdlog::info("BuildCompileResult: compiling {} nodes, {} links", nodes.size(), links.size());

    compile_result_issues_.clear();
    compile_result_has_config_ = false;
    compile_result_backend_placements_.clear();
    compile_result_success_ = false;
    if (properties_) {
        properties_->ClearBackendPlacementFacts();
    }

    try {
        cyxwiz::GraphCompiler compiler;
        cyxwiz::TrainingConfiguration config = compiler.Compile(nodes, links);

        compile_result_issues_ = config.issues;
        compile_result_backend_placements_ = config.backend_placements;
        compile_result_success_ = config.is_valid;
        // Launch readiness (data columns, device qualification, row limits):
        // reported by Compile, and therefore also before Train starts.
        if (config.is_valid) {
            const auto route = cyxwiz::PreviewRequestedRouteTrainingReadiness();
            for (auto& issue : gui::CheckGraphLaunchReadiness(
                     nodes, config, cyxwiz::DataRegistry::Instance(), &route)) {
                if (issue.level == cyxwiz::IssueLevel::Error) compile_result_success_ = false;
                compile_result_issues_.push_back(std::move(issue));
            }
        }
        if (properties_) {
            std::vector<properties_truth::BackendPlacementTruthFact> facts;
            facts.reserve(compile_result_backend_placements_.size());
            for (const auto& placement : compile_result_backend_placements_) {
                properties_truth::BackendPlacementTruthFact fact;
                fact.node_id = placement.node_id;
                fact.node_type = placement.node_type;
                fact.expected_backend = placement.expected_backend;
                fact.fallback_backend = placement.fallback_backend;
                fact.status = placement.status;
                fact.reason_code = placement.reason_code;
                fact.explanation = placement.explanation;
                fact.suggested_action = placement.suggested_action;
                facts.push_back(std::move(fact));
            }
            properties_->SetBackendPlacementFacts(std::move(facts));
        }

        // The configuration facts (the report's Configuration tab). Even when
        // the graph has errors, the partial facts show what the compiler
        // extracted before the issues.
        auto& facts = compile_result_config_;
        facts = {};
        compile_result_has_config_ = true;
        facts.layers = config.layers.size();
        facts.input_size = config.input_size;
        facts.input_shape = config.input_shape;
        facts.output_size = config.output_size;
        std::string dataset = config.dataset_name;
        std::string input_name;
        for (const auto& node : nodes) {
            if (node.type != NodeType::DataInput) continue;
            input_name = node.name;
            if (dataset.empty()) {
                auto it = node.parameters.find("dataset_name");
                if (it != node.parameters.end()) dataset = it->second;
            }
            break;
        }
        auto& reg = cyxwiz::DataRegistry::Instance();
        facts.dataset = dataset.empty() ? input_name : dataset;
        const bool registered = !dataset.empty() &&
            (reg.IsArrowDataset(dataset) || reg.IsParquetBackedDataset(dataset) || reg.IsTextDataset(dataset) ||
             reg.IsImageDataset(dataset) || reg.IsAudioDataset(dataset) || reg.IsSparseFeatureDataset(dataset) ||
             reg.HasDataset(dataset));
        if (!registered) {
            facts.backing = cyxwiz::compile_report::Backing::NotLoaded;
        } else if (reg.IsArrowDataset(dataset)) {
            if (auto arrow_ds = reg.GetArrowDataset(dataset)) {
                facts.backing = cyxwiz::compile_report::Backing::InMemoryArrow;
                facts.backing_megabytes = arrow_ds->GetMemoryUsage() / (1024.0 * 1024.0);
                facts.rows = static_cast<long long>(arrow_ds->GetNumRows());
            }
        } else if (reg.IsParquetBackedDataset(dataset)) {
            if (auto pq_ds = reg.GetParquetBackedDataset(dataset)) {
                facts.backing = cyxwiz::compile_report::Backing::DiskParquet;
                facts.backing_megabytes = pq_ds->GetFileSizeBytes() / (1024.0 * 1024.0);
                facts.rows = static_cast<long long>(pq_ds->GetNumRows());
                facts.row_groups = static_cast<long long>(pq_ds->GetNumRowGroups());
            }
        } else {
            facts.backing = cyxwiz::compile_report::Backing::OtherLoaded;
            if (const auto* text = reg.GetTextDatasetEntry(dataset)) {
                facts.backing_label = "text, loaded lazily";
                facts.rows = static_cast<long long>(text->num_samples);
            } else if (const auto* image = reg.GetImageDatasetEntry(dataset)) {
                facts.backing_label = "image folder, loaded lazily";
                facts.rows = static_cast<long long>(image->num_images);
            } else if (reg.IsAudioDataset(dataset)) {
                facts.backing_label = "audio folder, loaded lazily";
            } else {
                facts.backing_label = "loaded";
            }
        }
        for (const auto& node : nodes) {
            if (node.type == NodeType::DataSplit && facts.split_node.empty()) facts.split_node = node.name;
            if (node.type == NodeType::DataLoader && facts.loader_node.empty()) facts.loader_node = node.name;
        }
        facts.has_split = config.has_data_split;
        facts.train_ratio = config.train_ratio;
        facts.val_ratio = config.val_ratio;
        facts.test_ratio = config.test_ratio;
        facts.has_loader = config.has_data_loader;
        facts.batch_size = static_cast<int>(config.batch_size);
        facts.epochs = static_cast<int>(config.epochs);
        facts.shuffle = config.shuffle;
        facts.drop_last = config.drop_last;
        facts.num_workers = static_cast<int>(config.num_workers);
        facts.prefetch_factor = static_cast<int>(config.prefetch_factor);
        facts.log_interval = static_cast<int>(config.log_interval);
        facts.validation_freq = static_cast<int>(config.validation_freq);
        facts.seed = static_cast<unsigned long long>(config.dataloader_seed);
        facts.grad_accum_steps = static_cast<int>(config.grad_accum_steps);
        facts.loss = config.GetLossName();
        facts.optimizer = config.GetOptimizerName();
        facts.learning_rate = config.learning_rate;
        facts.normalize = config.preprocessing.has_normalization;
        facts.norm_mean = config.preprocessing.norm_mean;
        facts.norm_std = config.preprocessing.norm_std;
        facts.one_hot = config.preprocessing.has_onehot;
        facts.num_classes = static_cast<int>(config.preprocessing.num_classes);

        spdlog::info("BuildCompileResult: {} - {} layers, {} errors / {} warnings",
                     config.is_valid ? "valid" : "INVALID",
                     config.layers.size(),
                     config.CountIssues(cyxwiz::IssueLevel::Error),
                     config.CountIssues(cyxwiz::IssueLevel::Warning));

        // Populate pin state on the node editor so pins turn green
        // (compile passed) or red filled (compile failed) after compile.
        if (node_editor_) {
            node_editor_->SetAllNodesPinState(NodeEditor::NodePinState::CompilePassed);
            for (const auto& issue : config.issues) {
                if (issue.level == cyxwiz::IssueLevel::Error && issue.node_id >= 0) {
                    node_editor_->SetNodePinState(
                        issue.node_id, NodeEditor::NodePinState::CompileFailed);
                }
            }
        }
    } catch (const std::exception& e) {
        compile_result_success_ = false;
        cyxwiz::ValidationIssue exc_issue;
        exc_issue.level = cyxwiz::IssueLevel::Error;
        exc_issue.message = std::string("Compilation threw an exception: ") + e.what();
        exc_issue.error_code = cyxwiz::errors::Compiler::InvariantViolation;
        compile_result_issues_.push_back(std::move(exc_issue));
        spdlog::error("BuildCompileResult exception: {}", e.what());
    } catch (...) {
        compile_result_success_ = false;
        cyxwiz::ValidationIssue exc_issue;
        exc_issue.level = cyxwiz::IssueLevel::Error;
        exc_issue.message = "Compilation threw an unknown exception";
        exc_issue.error_code = cyxwiz::errors::Compiler::InvariantViolation;
        compile_result_issues_.push_back(std::move(exc_issue));
        spdlog::error("BuildCompileResult unknown exception");
    }
}

void MainWindow::StartTestingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    auto& training = cyxwiz::TrainingManager::Instance();
    if (test_preparation_task_ || training.IsTrainingActive() ||
        cyxwiz::TestManager::Instance().IsTestingActive()) {
        ShowOperationError("Cannot run test",
            "Another test preparation, training run or test is active. Wait for it to finish.");
        return;
    }
    auto model = training.GetActiveModel();
    if (!model) {
        ShowOperationError("Cannot run test",
            "No model is active. Train the graph, or use Train > Test > Load Checkpoint "
            "for Testing with the graph the checkpoint was trained with open.");
        return;
    }
    const auto fingerprint = HashGraphStructure(nodes, links);
    const std::weak_ptr<int> lifetime = test_callback_lifetime_;
    test_preparation_task_ = cyxwiz::SubmitGraphCompileTask(nodes, links,
        [this, lifetime, nodes, links, model, fingerprint](bool success,
            const std::string& error, cyxwiz::TrainingConfiguration config) {
            if (lifetime.expired()) return;
            test_preparation_task_.reset();
            if (!success) { ShowOperationError("Cannot run test", error); return; }
            auto& training = cyxwiz::TrainingManager::Instance();
            if (!node_editor_ || training.IsTrainingActive() ||
                cyxwiz::TestManager::Instance().IsTestingActive() ||
                training.GetActiveModel() != model ||
                HashGraphStructure(node_editor_->GetNodes(), node_editor_->GetLinks()) != fingerprint) {
                ShowOperationError("Cannot run test",
                    "The graph or model changed while the test was being prepared. Run Test again.");
                return;
            }
            try { StartTestingWithConfig(nodes, links, std::move(config)); }
            catch (const std::exception& error) { spdlog::error("Run Test launch failed: {}", error.what()); }
        });
}

void MainWindow::StartTestingWithConfig(const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links, cyxwiz::TrainingConfiguration config) {
    if (!config.is_valid) {
        spdlog::error("Graph compilation failed: {}", config.error_message);
        return;
    }

    spdlog::info("Graph compiled successfully: {} layers, input={}, output={}",
                 config.layers.size(), config.input_size, config.output_size);

    if (config.batch_size <= 0) {
        config.batch_size = 32;
    }

    auto& tm = cyxwiz::TrainingManager::Instance();
    if (!tm.HasTrainedModel()) {
        ShowOperationError("Cannot run test",
            "No model is active. Train a model, or use Train > Test > Load "
            "Checkpoint for Testing before running Test.");
        return;
    }

    const auto active_model_info = tm.GetActiveModelInfo();
    if (active_model_info.origin ==
            cyxwiz::ActiveModelOrigin::LoadedCheckpoint &&
        !active_model_info.graph_fingerprint.empty()) {
        std::ostringstream fingerprint;
        fingerprint << std::hex << HashGraphStructure(nodes, links);
        if (fingerprint.str() != active_model_info.graph_fingerprint) {
            ShowOperationError("Cannot run test",
                "The graph changed after the checkpoint was loaded (any edit, including "
                "a Data Input file, counts). Load the checkpoint again with this graph open, "
                "then run Test.");
            return;
        }
    }

    if (active_model_info.evaluation_config &&
        active_model_info.evaluation_config->sequence_batch.enabled) {
        config = *active_model_info.evaluation_config;
        spdlog::info("StartTestingFromGraph: using the active model sequence preparation contract");
    }
    auto test_selection =
        cyxwiz::ResolveGraphTestDataset(config);
    // A supplied Test input the pipeline run joined to the training rows
    // (TOFIX178 S0-g4): its rows, transformed by the steps before the Data
    // Split, are the model's prepared table's Test partition.
    if (test_selection.scope == cyxwiz::TestDatasetScope::EntireProvidedDataset &&
        !active_model_info.effective_dataset_name.empty()) {
        if (const auto prepared = cyxwiz::DataRegistry::Instance().GetArrowDataset(
                active_model_info.effective_dataset_name)) {
            const auto parts = cyxwiz::PartitionValues(prepared->GetArrowTable());
            if (std::find(parts.begin(), parts.end(), int8_t{2}) != parts.end()) {
                test_selection.scope = cyxwiz::TestDatasetScope::ConfiguredTestSplit;
                spdlog::info("StartTestingFromGraph: the supplied Test rows are in '{}', prepared with the "
                             "training rows",
                             active_model_info.effective_dataset_name);
            }
        }
    }
    std::string dataset_name = test_selection.dataset_name;
    const int test_source_node_id = test_selection.source_node_id >= 0
        ? test_selection.source_node_id
        : config.data_source_node_id;
    std::string label_column = test_selection.label_column;
    // Reuse the model's own prepared dataset only when the graph supplies no
    // Test input; a supplied Test dataset is exactly what should be scored.
    if (!active_model_info.effective_dataset_name.empty() &&
        test_selection.scope != cyxwiz::TestDatasetScope::EntireProvidedDataset) {
        dataset_name = active_model_info.effective_dataset_name;
        if (!active_model_info.effective_label_column.empty()) {
            label_column = active_model_info.effective_label_column;
        }
        config.dataset_name = dataset_name;
        config.dataset_roles.train.dataset_name = dataset_name;
        config.dataset_roles.train.label_column = label_column;
        spdlog::info(
            "StartTestingFromGraph: reusing active model's effective "
            "dataset '{}' (label='{}')",
            dataset_name,
            label_column);
    }
    if (label_column.empty()) {
        label_column = FindTestingLabelColumn(
            nodes, dataset_name, test_source_node_id);
    }
    if (dataset_name.empty()) {
        spdlog::error("StartTestingFromGraph: no dataset loaded. Configure the Data Input node first.");
        return;
    }
    config.dataset_name = dataset_name;
    config.data_source_node_id = test_source_node_id;
    spdlog::info(
        "StartTestingFromGraph: selected dataset '{}' from source node {} "
        "with scope={}",
        dataset_name,
        test_source_node_id,
        test_selection.scope ==
                cyxwiz::TestDatasetScope::EntireProvidedDataset
            ? "entire_provided_dataset"
            : "configured_test_split");

    auto model = tm.GetActiveModel();
    if (!model) {
        spdlog::error(
            "StartTestingFromGraph: active model became unavailable before testing.");
        return;
    }

    if (!active_model_info.model_device) {
        spdlog::error("StartTestingFromGraph: active model has no device ownership. "
                      "Reload it through Train > Test > Load Checkpoint for Testing.");
        return;
    }

    const int test_batch_size = config.batch_size;
    auto& registry = cyxwiz::DataRegistry::Instance();
    bool started = false;
    if (auto arrow_dataset = registry.GetArrowDataset(dataset_name)) {
        started = cyxwiz::TestManager::Instance().StartTestingArrow(
            std::move(config), std::move(arrow_dataset), label_column,
            test_selection.scope, test_batch_size, model, nullptr, active_model_info.model_device);
    } else if (auto parquet_dataset = registry.GetParquetBackedDataset(dataset_name)) {
        started = cyxwiz::TestManager::Instance().StartTestingParquet(
            std::move(config), std::move(parquet_dataset), label_column,
            test_selection.scope, test_batch_size, model, nullptr, active_model_info.model_device);
    } else {
        auto dataset = registry.GetDataset(dataset_name);
        if (!dataset) {
            spdlog::error("StartTestingFromGraph: dataset '{}' is not available in the legacy dataset registry.", dataset_name);
            spdlog::info("Testing currently uses the legacy DatasetHandle path for non-text datasets.");
            return;
        }

        started = cyxwiz::TestManager::Instance().StartTesting(
            std::move(config), cyxwiz::LegacyTestSource(dataset), test_batch_size, model, nullptr, active_model_info.model_device);
    }

    if (started) {
        spdlog::info("StartTestingFromGraph: testing started successfully");
    } else {
        spdlog::error("StartTestingFromGraph: failed to start testing");
    }
}

void MainWindow::RenderSidebar() {
    // Render the Unreal-style sidebar (auto-hides, appears on hover)
    GetDockStyle().RenderSidebarToggles(StatusBarHeight());
}


void MainWindow::DetectKeyboardContext() {
    auto& kb = KeyboardShortcutManager::Instance();

    // Check for modal dialogs first (highest priority)
    if (toolbar_) {
        if (toolbar_->IsFindInFilesDialogOpen() || toolbar_->IsReplaceInFilesDialogOpen() ||
            toolbar_->IsCommandPaletteOpen()) {
            kb.SetActiveContext(KeyboardContext::ModalDialog);
            return;
        }
    }

    // Check for completion popup in script editor
    if (script_editor_ && script_editor_->IsCompletionPopupOpen()) {
        kb.SetActiveContext(KeyboardContext::CompletionPopup);
        return;
    }

    // Check which panel is focused using panel-level focus tracking
    // This is more reliable than window name matching since panels track child window focus

    // Script Editor - uses IsFocused() which tracks child windows (TextEditor)
    if (script_editor_ && script_editor_->IsFocused()) {
        kb.SetActiveContext(KeyboardContext::ScriptEditor);
        return;
    }

    // For other panels, fall back to window name matching
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
        // Get the focused window name
        ImGuiWindow* focused = ImGui::GetCurrentContext()->NavWindow;
        if (focused) {
            std::string window_name = focused->Name ? focused->Name : "";

            // CyxWiz Studio detection
            if (window_name.find("CyxWiz Studio") != std::string::npos ||
                window_name.find("##NodeEditor") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::NodeEditor);
                return;
            }

            // Console detection
            if (window_name.find("Console") != std::string::npos ||
                window_name.find("Python Console") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::Console);
                return;
            }

            // Asset Browser detection
            if (window_name.find("Asset") != std::string::npos ||
                window_name.find("Project") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::AssetBrowser);
                return;
            }

            // Table Viewer detection
            if (window_name.find("Table") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::TableViewer);
                return;
            }
        }
    }

    // Default to main window context
    kb.SetActiveContext(KeyboardContext::MainWindow);
}

// ============================================================================
// Project Settings Persistence
// ============================================================================

void MainWindow::SaveLayout() {
    // Save to the default imgui.ini in the executable directory
    // This ensures consistent layout across all projects
    // IniFilename is absolute (Application): a relative path would follow
    // the working directory, which script runs move to the project root.
    if (const char* ini = ImGui::GetIO().IniFilename)
        ImGui::SaveIniSettingsToDisk(ini);
    spdlog::info("Saved layout to imgui.ini");
}

void MainWindow::SaveProjectSettings() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::debug("Editor settings not saved: no project open (they apply to this session)");
        return;
    }

    // Get current editor settings from script editor
    cyxwiz::EditorSettings& settings = pm.GetConfig().editor_settings;
    if (script_editor_) {
        settings.font_scale = script_editor_->GetFontScale();
        settings.tab_size = script_editor_->GetTabSize();
        settings.show_whitespace = script_editor_->GetShowWhitespace();
        settings.word_wrap = script_editor_->GetWordWrap();
        settings.auto_indent = script_editor_->GetAutoIndent();
        settings.syntax_highlighting = script_editor_->GetSyntaxHighlighting();

        // Persist open scripts for project restore
        pm.GetConfig().open_scripts.clear();
        auto open_paths = script_editor_->GetOpenFilePaths();
        for (const auto& path : open_paths) {
            pm.GetConfig().open_scripts.push_back(path);
        }
        pm.GetConfig().active_script_index = script_editor_->GetActiveTabIndex();
    }

    // Save application-wide settings
    settings.app_theme = static_cast<int>(GetTheme().GetCurrentPreset());
    settings.ui_scale = ImGui::GetIO().FontGlobalScale;

    // Save layout file
    SaveLayout();

    // Save project file (includes editor settings)
    pm.SaveProject();
    spdlog::info("Saved project settings");
}

void MainWindow::LoadProjectSettings() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        return;
    }

    const cyxwiz::EditorSettings& settings = pm.GetConfig().editor_settings;

    // Apply editor settings to script editor
    if (script_editor_) {
        script_editor_->SetTabSize(settings.tab_size);
        script_editor_->SetShowWhitespace(settings.show_whitespace);
        script_editor_->SetWordWrap(settings.word_wrap);
        script_editor_->SetAutoIndent(settings.auto_indent);
        script_editor_->SetSyntaxHighlighting(settings.syntax_highlighting);
    }

    // Sync settings to toolbar/preferences
    if (toolbar_) {
        toolbar_->SetEditorTabSize(settings.tab_size);
        toolbar_->SetEditorFontScale(gui::CodeTextScale());
        toolbar_->SetEditorShowWhitespace(settings.show_whitespace);
        toolbar_->SetEditorWordWrap(settings.word_wrap);
        toolbar_->SetEditorAutoIndent(settings.auto_indent);
    }

    // Theme and text sizes are Engine-wide (Preferences > Appearance); the
    // project's older app_theme / ui_scale fields are no longer applied.
    ImGui::GetIO().FontGlobalScale = 1.0f;

    // Restore open scripts
    const auto& open_scripts = pm.GetConfig().open_scripts;
    for (const auto& script_path : open_scripts) {
        std::string resolved_path = script_path;
        std::filesystem::path fs_path(script_path);
        if (fs_path.is_relative()) {
            resolved_path = pm.ResolveAssetPath(script_path);
        }
        if (script_editor_ && std::filesystem::exists(resolved_path)) {
            script_editor_->OpenFile(resolved_path);
        }
    }
    if (script_editor_ && !open_scripts.empty()) {
        script_editor_->SetActiveTabIndex(pm.GetConfig().active_script_index);
    }

    spdlog::debug("Loaded project settings (font_scale={:.1f}, tab_size={})", settings.font_scale, settings.tab_size);
}

void MainWindow::OnProjectOpened(const std::string& project_root) {
    spdlog::debug("Project opened: {}", project_root);

    if (console_) console_->SetProjectRoot(project_root);
    // Saved query results (dashboards) live in the project.
    cyxwiz::SessionQueryService::Instance().SetProjectRoot(project_root);
    cyxwiz::SetImageQualityProjectRoot(project_root);
    // DuckDB spills sorts, joins and group-bys into the project (TOFIX154).
    cyxwiz::SetDataSpillDirectory((std::filesystem::path(project_root) / "cache" / "duckdb_tmp").string());
    // The project's prepared-data caches: older than 30 days, then the oldest
    // past 10 GB, go (TOFIX160 P7; the temp cache is pruned at startup). A
    // file a loaded dataset reads stays.
    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Prune the project's prepared-data caches", [project_root](cyxwiz::LambdaTask& task) {
            const auto in_use = cyxwiz::DataRegistry::Instance().FileInUse();
            uint64_t freed = 0;
            for (const auto& cache : cyxwiz::PreparedDataCacheFolders(project_root, in_use)) {
                if (cache.label.find("this project") == std::string::npos) continue;
                freed += cyxwiz::PruneCacheFolder(cache.folder).freed_bytes;
            }
            task.MarkCompleted(freed ? cyxwiz::FormatDataBytes(freed) + " freed" : "nothing to remove");
        });

    // Load project settings and layout
    LoadProjectSettings();

    // If Python is already initialized, check for interpreter mismatch
    if (scripting_engine_ && scripting_engine_->IsInitialized()) {
        if (!scripting_engine_->ReloadPythonForProject()) {
            spdlog::warn("Python interpreter mismatch after project open (project: {})", project_root);
        }
    }

    // Show the project's files (SetProjectRoot scans when the root is new).
    if (asset_browser_) {
        asset_browser_->SetProjectRoot(project_root);
    }
}

void MainWindow::OnProjectClosed(const std::string& project_root) {
    spdlog::info("Project closed: {}", project_root);
    cyxwiz::SessionQueryService::Instance().SetProjectRoot("");
    cyxwiz::SetImageQualityProjectRoot("");
    cyxwiz::SetDataSpillDirectory("");

    // A pipeline run writes into the closed project's artifact/export paths;
    // it stops with the project (TOFIX101 section 7).
    if (node_editor_) node_editor_->CancelOwnedBackgroundWork();

    if (console_) console_->CloseProject(project_root);

    // Note: Settings should be saved before CloseProject() is called
    // (the toolbar handles this in its Close Project menu action)

    // If Python is already initialized, check for interpreter mismatch after closing
    if (scripting_engine_ && scripting_engine_->IsInitialized()) {
        if (!scripting_engine_->ReloadPythonForProject()) {
            spdlog::warn("Python interpreter mismatch after project close");
        }
    }

    // Clear the asset browser
    if (asset_browser_) {
        asset_browser_->Clear();
    }

    // Drop every tabular dataset (Arrow + Parquet-backed) so the next
    // project starts with an empty registry. Without this, datasets from
    // the previous project linger and can collide with same-named entries
    // (auto-generated from file stem) in the new project, leading to
    // stale dispatch in StartTrainingFromGraph.
    cyxwiz::DataRegistry::Instance().ClearAllTabularDatasets();

    // Reset the dock layout to default when project is closed
    // This gives a clean slate for the next project
    first_time_layout_ = true;  // This will trigger BuildInitialDockLayout on next render
}

void MainWindow::OnProjectVenvReady(const std::string& project_root) {
    spdlog::info("Project venv ready: {}", project_root);

    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::info("Skipping Python reload: no active project");
        return;
    }
    if (pm.GetProjectRoot() != project_root) {
        spdlog::info("Skipping Python reload: active project root differs");
        return;
    }

    if (scripting_engine_ && scripting_engine_->IsInitialized()) {
        if (!scripting_engine_->ReloadPythonForProject()) {
            spdlog::warn("Python interpreter mismatch after venv creation");
        }
    }
}

void MainWindow::RenderPythonEnvSetupStatus(const cyxwiz::ProjectManager& pm) {
    using SetupState = cyxwiz::ProjectManager::PythonEnvSetupState;
    const auto status = pm.GetPythonEnvSetupStatus();
    if (status.state == SetupState::None || status.project_root != pm.GetProjectRoot()) {
        return;
    }
    const auto age = std::chrono::steady_clock::now() - status.changed;
    ImGui::SameLine(0.0f, 16.0f);
    switch (status.state) {
        case SetupState::Pending:
            ImGui::TextColored(cyxwiz::ui::CurrentTokens().running, "%s %s", ICON_FA_SPINNER,
                               status.message.c_str());
            break;
        case SetupState::Ready:
            // Confirm briefly, then get out of the way.
            if (age < std::chrono::seconds(8)) {
                ImGui::TextColored(cyxwiz::ui::CurrentTokens().success, "%s %s",
                                   ICON_FA_CIRCLE_CHECK, status.message.c_str());
            }
            break;
        case SetupState::Failed:
            ImGui::TextColored(cyxwiz::ui::CurrentTokens().error, "%s Python environment setup failed",
                               ICON_FA_TRIANGLE_EXCLAMATION);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", status.message.c_str());
            }
            break;
        case SetupState::None:
            break;
    }
}

template <class PanelPtr>
void MainWindow::TimedRender(const char* name, PanelPtr& panel) {
    if (!panel) return;
    PanelTimer timer(name);
    panel->Render();
}

float MainWindow::StatusBarHeight() {
    return ImGui::GetTextLineHeight() + 8.0f;
}

void MainWindow::RenderStatusBar() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float status_bar_height = StatusBarHeight();

    // Position status bar at the bottom of the screen
    ::gui::NextWindowStaysInMain();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - status_bar_height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, status_bar_height));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::GetStyleColorVec4(ImGuiCol_MenuBarBg));

    if (ImGui::Begin("##StatusBar", nullptr, flags)) {
        // Left side: Project info
        auto& pm = cyxwiz::ProjectManager::Instance();
        if (pm.HasActiveProject()) {
            ImGui::Text("%s %s", ICON_FA_FOLDER_OPEN, pm.GetProjectName().c_str());
            RenderPythonEnvSetupStatus(pm);
        } else {
            ImGui::TextDisabled("%s No Project", ICON_FA_FOLDER);
        }

        // Centre: one sentence about the hovered menu item (TOFIX129).
        ImGui::SameLine();
        if (toolbar_ && !toolbar_->StatusHint().empty()) {
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            ImGui::TextUnformatted(toolbar_->StatusHint().c_str());
        }

        // Right side: Task indicator (if tasks are running)
        if (task_progress_panel_ && task_progress_panel_->HasActiveTasks()) {
            // Calculate position for right-aligned content
            std::string status = task_progress_panel_->GetStatusSummary();
            float text_width = ImGui::CalcTextSize(status.c_str()).x + 30; // + icon width

            ImGui::SameLine(ImGui::GetWindowWidth() - text_width - 16);

            // Show task indicator that opens the task panel when clicked
            if (cyxwiz::TaskStatusIndicator::Render()) {
                task_progress_panel_->Show();
            }

            ImGui::SameLine();
            ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_CheckMark), "%s", status.c_str());
        } else {
            // Show ready status on the right
            float text_width = ImGui::CalcTextSize("Ready").x + 20;
            ImGui::SameLine(ImGui::GetWindowWidth() - text_width - 16);
            ImGui::TextDisabled("%s Ready", ICON_FA_CIRCLE_CHECK);
        }
    }
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// Train > Load Checkpoint for Testing, also offered by Feature Importance:
// compiles the graph, prepares its data and makes the checkpoint the active model.
void MainWindow::LoadCheckpointForTesting() {
    auto& training = cyxwiz::TrainingManager::Instance();
    auto& testing = cyxwiz::TestManager::Instance();
    if (training.IsTrainingActive() || testing.IsTestingActive()) {
        ShowOperationError("Cannot load checkpoint",
            "Stop the active training or testing task first.");
        return;
    }
    if (!node_editor_) {
        ShowOperationError("Cannot load checkpoint", "The node editor is unavailable.");
        return;
    }
    if (node_editor_->GetNodes().empty()) {
        // The checkpoint's weights are loaded into the model built from the
        // graph on the canvas; without that graph there is nothing to load into.
        ShowOperationError("Cannot load checkpoint",
            "The canvas is empty. Open the graph this checkpoint was trained with, "
            "Apply its Data Inputs, then load the checkpoint.");
        return;
    }

    auto& project = cyxwiz::ProjectManager::Instance();
    const std::string default_path = project.HasActiveProject()
        ? project.GetCheckpointsPath()
        : std::string();
    auto selected = cyxwiz::FileDialogs::SelectFolder(
        "Select Checkpoint or Training Run",
        default_path.empty() ? nullptr : default_path.c_str());
    if (!selected) {
        return;
    }

    const auto nodes = node_editor_->GetNodes();
    const auto links = node_editor_->GetLinks();
    BuildCompileResult(nodes, links);
    if (!compile_result_success_) {
        ShowCompileReport(cyxwiz::compile_report::Mode::CheckpointBlocked);
        spdlog::error(
            "LoadCheckpoint: active graph did not pass the compile gate.");
        return;
    }

    cyxwiz::TrainingConfiguration config;
    try {
        cyxwiz::GraphCompiler compiler;
        config = compiler.Compile(nodes, links);
    } catch (const std::exception& exception) {
        ShowOperationError("Cannot load checkpoint",
            std::string("Graph compilation failed: ") + exception.what());
        return;
    }

    std::ostringstream fingerprint;
    fingerprint << std::hex << HashGraphStructure(nodes, links);
    const std::string graph_fingerprint = fingerprint.str();
    const std::string selected_path = *selected;
    const cyxwiz::StepRunRoots step_roots{project.GetProjectRoot(), project.GetArtifactsPath(),
                                          project.GetExportsPath(), project.GetIngestionCachePath()};
    auto load_result =
        std::make_shared<cyxwiz::CheckpointEvaluationLoadResult>();
    std::weak_ptr<cyxwiz::TrainingPlotPanel> dashboard =
        training_plot_panel_;

    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Load checkpoint for testing",
        [config = std::move(config), nodes, links, selected_path,
         graph_fingerprint, load_result,
         step_roots](cyxwiz::LambdaTask& task) {
            auto evaluation_config = config;
            task.ReportProgress(0.05f,
                                "Preparing evaluation dataset...");
            std::string effective_dataset_name =
                evaluation_config.dataset_name;
            if (effective_dataset_name.empty()) {
                effective_dataset_name =
                    evaluation_config.dataset_roles.train.dataset_name;
            }
            if (effective_dataset_name.empty()) {
                throw std::runtime_error(
                    "Checkpoint testing cannot resolve the graph's "
                    "source dataset. Apply the Data Input node first.");
            }

            // The data steps before the Data Split, as training ran them:
            // the table kept on disk, else a run of those steps (TOFIX178 S0-f2).
            if (const int feeder = cyxwiz::TrainingInputFeeder(nodes, links); feeder >= 0) {
                const auto prepared = cyxwiz::PrepareStepResult(
                    nodes, links, feeder, cyxwiz::DataRegistry::Instance(), step_roots,
                    GraphMaterializationCacheConfig(step_roots.project),
                    [&task]() { return task.ShouldStop(); },
                    [&task](float progress, const std::string& message) {
                        task.ReportProgress(0.05f + 0.55f * std::clamp(progress, 0.0f, 1.0f), message);
                    });
                if (prepared.cancelled) return;
                if (!prepared.ok)
                    throw std::runtime_error("Checkpoint evaluation preprocessing failed: " + prepared.error);
                effective_dataset_name = prepared.dataset;
            }
            std::string effective_label =
                evaluation_config.dataset_roles.train.label_column.empty()
                    ? evaluation_config.target.primary_column
                    : evaluation_config.dataset_roles.train.label_column;
            evaluation_config.dataset_name = effective_dataset_name;
            evaluation_config.dataset_roles.train.dataset_name =
                effective_dataset_name;
            evaluation_config.dataset_roles.train.label_column =
                effective_label;
            task.ReportProgress(0.65f, "Inspecting checkpoint...");
            *load_result = cyxwiz::TrainingManager::Instance()
                .LoadCheckpointForEvaluation(
                    evaluation_config,
                    cyxwiz::DataRegistry::Instance().GetArrowDataset(
                        evaluation_config.dataset_name),
                    selected_path, graph_fingerprint,
                    [&task]() { return task.IsCancelRequested(); });
            if (task.IsCancelRequested()) {
                return;
            }
            if (!load_result->success) {
                throw std::runtime_error(load_result->error_message);
            }
            task.ReportProgress(1.0f, "Checkpoint loaded for testing");
        },
        nullptr,
        [this, lifetime = std::weak_ptr<int>(test_callback_lifetime_), dashboard, load_result](
            bool success, const std::string& error) {
            if (!success || !load_result->success) {
                const std::string reason = error.empty() ? load_result->error_message : error;
                if (!lifetime.expired()) {
                    ShowOperationError("Cannot load checkpoint", reason);
                } else {
                    spdlog::error("LoadCheckpoint: {}", reason);
                }
                return;
            }

            const bool has_validation =
                !load_result->metadata.val_loss_history.empty() ||
                !load_result->metadata.val_accuracy_history.empty() ||
                load_result->metadata.val_loss != 0.0f ||
                load_result->metadata.val_accuracy != 0.0f;
            if (auto panel = dashboard.lock()) {
                panel->SetActiveCheckpointLoaded(
                    load_result->resolved_checkpoint_path,
                    load_result->metadata.epoch,
                    load_result->metadata.val_loss,
                    load_result->metadata.val_accuracy,
                    has_validation);
                panel->SetVisible(true);
            }
            spdlog::info(
                "LoadCheckpoint: active model is ready for Tools > Test");
        });
}

void MainWindow::OpenModelExport(int format_index) {
    if (export_dialog_) {
        cyxwiz::ModelFormat requested_format = cyxwiz::ModelFormat::CyxModel;
        if (format_index >= 0 && format_index <= 3) {
            requested_format = static_cast<cyxwiz::ModelFormat>(format_index);
        }

        // Get trained model from TrainingManager
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (tm.HasTrainedModel()) {
            auto model = tm.GetActiveModel();
            auto* optimizer = tm.GetLastOptimizer();
            auto& metrics = tm.GetLastMetrics();

            // Get current graph JSON from node editor
            std::string graph_json;
            if (node_editor_) {
                graph_json = node_editor_->GetGraphJson();
            }

            const auto trained_info = tm.GetActiveModelInfo();
            const auto nodes = node_editor_
                ? node_editor_->GetNodes()
                : std::vector<MLNode>{};
            const auto links = node_editor_
                ? node_editor_->GetLinks()
                : std::vector<NodeLink>{};
            export_dialog_->SetModelData(
                model,
                optimizer,
                &metrics,
                graph_json,
                node_editor_
                    ? HashGraphStructure(nodes, links)
                    : 0,
                trained_info.evaluation_config
                    ? cyxwiz::TrainingExportMetadata(*trained_info.evaluation_config)
                    : cyxwiz::ExportOptions{},
                trained_info.model_device);
            spdlog::info("Loaded trained model into Export dialog");
        } else {
            spdlog::warn("No trained model available for export");
        }

        if (node_editor_) export_dialog_->SetSourceGraph(node_editor_->GetCurrentFilePath());
        export_dialog_->Open(requested_format);
        spdlog::info("Opened Export Model dialog (format index: {})", format_index);
    }
}

} // namespace gui
