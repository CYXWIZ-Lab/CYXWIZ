#include "image_recipe_view.h"

#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/live_graph_compile.h"
#include "../../core/node_metadata_registry.h"
#include "../../core/texture_manager.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>

namespace cyxwiz {

namespace {

namespace ir = image_recipe;
using ::gui::NodeType;

constexpr float kThumb = 64.0f;
constexpr int kDraws = 4;
const char* kRowPayload = "CYXWIZ_RECIPE_ROW";

// The image steps Prepare adds: done once (device and OpenCV), and the
// random ones drawn per batch.
const NodeType kOnceSteps[] = {NodeType::CenterCrop,   NodeType::Grayscale,     NodeType::ImageGaussianBlur,
                               NodeType::MorphologyTransform, NodeType::ImageContrast, NodeType::ImageDenoise,
                               NodeType::ImageSharpen, NodeType::ImageEdges,    NodeType::ImageSmooth,
                               NodeType::ImageWarp,    NodeType::ImagePyramid};
const NodeType kOpenCvSteps[] = {NodeType::ImageContrast, NodeType::ImageDenoise, NodeType::ImageSharpen,
                                 NodeType::ImageEdges,    NodeType::ImageSmooth,  NodeType::ImageWarp,
                                 NodeType::ImagePyramid};
const NodeType kPerBatchSteps[] = {NodeType::RandomCrop,        NodeType::RandomResizedCrop, NodeType::HorizontalFlip,
                                   NodeType::VerticalFlip,      NodeType::ImageRotate,       NodeType::RandomAffine,
                                   NodeType::RandomPerspective, NodeType::ColorJitter,       NodeType::RandomGrayscale,
                                   NodeType::GaussianNoise,     NodeType::AdvancedAugment,   NodeType::AugmentationPreset};

uint32_t Upload(const PreviewPicture& p) {
    return p.rgba.empty() ? 0u : TextureManager::Instance().CreateTextureFromUint8Data(p.rgba.data(), p.width, p.height, 4);
}

const char* GroupTitle(ir::Group g) {
    return g == ir::Group::Once ? "Once, before training" : g == ir::Group::PerBatch ? "Per batch, training images only" : "";
}

const char* GroupNote(ir::Group g) {
    return g == ir::Group::Once ? "done when the images are prepared (kept in cache/images) \xC2\xB7 every image of every part"
                                : "drawn again for every batch, on the GPU \xC2\xB7 validation and test images skip them";
}

}  // namespace

// The sample's pictures: after each step done once, and each per-batch row's
// draws (the per-batch steps up to it).
struct ImageRecipeView::Previews {
    ImagePipelinePreview full;
    std::vector<std::pair<int, ImagePipelinePreview>> per_row;
};

ImageRecipeView::ImageRecipeView() = default;

ImageRecipeView::~ImageRecipeView() {
    if (task_) AsyncTaskManager::Instance().Cancel(task_);
    ReleaseTextures();
}

void ImageRecipeView::ReleaseTextures() {
    auto& textures = TextureManager::Instance();
    for (uint32_t id : textures_)
        if (id) textures.DeleteTexture(id);
    textures_.clear();
    row_texture_.clear();
    row_draws_.clear();
    sample_texture_ = 0;
}

bool ImageRecipeView::IsImageGraph(const ::gui::NodeEditor* editor) {
    const auto* live = editor ? editor->GetLiveCompile() : nullptr;
    const TrainingConfiguration* config = live ? live->Config() : nullptr;
    if (config && config->image_preprocessing.resize_mode != ResizeMode::None) return true;
    if (!editor) return false;
    // Before the first compile: a Resize on the canvas.
    return std::any_of(editor->GetNodes().begin(), editor->GetNodes().end(),
                       [](const ::gui::MLNode& n) { return n.type == NodeType::Resize; });
}

std::string ImageRecipeView::SamplePath(const std::string& dataset, size_t* count) const {
    *count = 0;
    const auto index = DataRegistry::Instance().GetArrowDataset(DataRegistry::ImageIndexName(dataset));
    const auto table = index ? index->GetArrowTable() : nullptr;
    const auto column = table ? table->GetColumnByName("path") : nullptr;
    if (!column || column->length() == 0) return {};
    *count = static_cast<size_t>(column->length());
    int64_t row = static_cast<int64_t>(sample_ % *count);
    for (int c = 0; c < column->num_chunks(); ++c) {
        const auto chunk = std::static_pointer_cast<arrow::StringArray>(column->chunk(c));
        if (row < chunk->length()) return chunk->GetString(row);
        row -= chunk->length();
    }
    return {};
}

void ImageRecipeView::UpdatePreviews(const TrainingConfiguration& config, const ir::View& view) {
    size_t count = 0;
    const std::string path = SamplePath(config.dataset_name, &count);
    if (path.empty()) {
        // The images were unloaded: no pictures of a sample that is gone.
        if (!task_ && !preview_key_.empty()) {
            ReleaseTextures();
            preview_key_.clear();
        }
        return;
    }
    const PreparedImageSpec spec = PreparedImageSpecFrom(config.image_preprocessing);
    const std::string key = path + "|" + ImageStepsKey(spec, config.image_augmentation, config.image_cv_steps);
    if (key == preview_key_ || task_) return;
    preview_key_ = key;
    // Each per-batch row's draws come from the device steps up to it.
    std::vector<std::pair<int, image::ImageAugmentation>> cuts;
    for (const auto& row : view.rows) {
        if (row.group != ir::Group::PerBatch || row.node_id < 0) continue;
        image::ImageAugmentation up_to = config.image_augmentation;
        up_to.ops.resize(std::min(up_to.ops.size(), row.ops_through));
        if (row.node_id != config.image_mix_node_id) up_to.mix = image::BatchMix::None;
        cuts.emplace_back(row.node_id, up_to);
    }
    auto result = std::make_shared<Previews>();
    const image::ImageAugmentation steps = config.image_augmentation;
    const std::vector<CvImageStep> cv_steps = config.image_cv_steps;
    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(path));
    std::weak_ptr<int> alive = alive_;
    task_ = AsyncTaskManager::Instance().RunAsync(
        "Image recipe pictures",
        [path, spec, steps, cv_steps, cuts, seed, result](LambdaTask&) {
            result->full = PreviewImagePipeline(path, spec, steps, cv_steps, 0, seed);
            for (const auto& [node, up_to] : cuts)
                result->per_row.emplace_back(node, PreviewImagePipeline(path, spec, up_to, cv_steps, kDraws, seed));
        },
        nullptr,
        [this, alive, result, view](bool, const std::string&) {
            if (alive.expired()) return;
            task_ = 0;
            ReleaseTextures();
            previews_ = std::make_unique<Previews>(std::move(*result));
            const auto keep = [&](uint32_t id) {
                if (id) textures_.push_back(id);
                return id;
            };
            sample_texture_ = keep(Upload(previews_->full.decoded));
            // Pictures done once: one per merged step, in order; a row takes
            // the last of its steps (a preset's ops share a row).
            size_t picture = 0, ops_seen = 0;
            for (const auto& row : view.rows) {
                if (row.group == ir::Group::Resize) {
                    if (row.node_id >= 0) row_texture_[row.node_id] = sample_texture_;
                    continue;
                }
                if (row.group != ir::Group::Once) continue;
                const size_t steps_in_row = row.opencv ? 1 : row.ops_through - ops_seen;
                if (!row.opencv) ops_seen = row.ops_through;
                picture += steps_in_row;
                if (picture >= 1 && picture <= previews_->full.fixed.size() && row.node_id >= 0)
                    row_texture_[row.node_id] = keep(Upload(previews_->full.fixed[picture - 1]));
            }
            uint32_t model_input = sample_texture_;
            for (const auto& row : view.rows)
                if (row.group == ir::Group::Once || row.group == ir::Group::Resize)
                    if (const auto at = row_texture_.find(row.node_id); at != row_texture_.end()) model_input = at->second;
            for (const auto& [node, preview] : previews_->per_row) {
                auto& draws = row_draws_[node];
                for (const auto& p : preview.draws) draws.push_back(keep(Upload(p)));
                if (!draws.empty() && draws.front()) model_input = draws.front();
            }
            // Normalize: the picture the model gets (it scales values, not pixels).
            for (const auto& row : view.rows)
                if (row.group == ir::Group::Normalize && row.node_id >= 0) row_texture_[row.node_id] = model_input;
        },
        alive_);
}

void ImageRecipeView::RenderAddMenu(::gui::NodeEditor* editor, int after_node, bool per_batch) {
    const auto& registry = NodeMetadataRegistry::Instance();
    const auto item = [&](NodeType type) {
        const auto* meta = registry.GetMetadata(type);
        if (!meta || meta->status != NodeImplementationStatus::Implemented) return;
        if (ImGui::MenuItem((meta->icon + "  " + meta->name).c_str())) {
            editor->RequestStepAfter(after_node, type);
            message_ = "Added " + meta->name + " (Ctrl+Z on the canvas undoes it).";
            message_error_ = false;
        }
        if (ImGui::IsItemHovered() && !meta->brief_description.empty()) ImGui::SetTooltip("%s", meta->brief_description.c_str());
    };
    if (per_batch) {
        for (const NodeType type : kPerBatchSteps) item(type);
        ImGui::Separator();
        // A step done the same way every time also runs per batch here; an
        // OpenCV step cannot follow a random one, so it is not offered.
        if (ImGui::BeginMenu("The same for every image")) {
            for (const NodeType type : kOnceSteps)
                if (std::find(std::begin(kOpenCvSteps), std::end(kOpenCvSteps), type) == std::end(kOpenCvSteps)) item(type);
            ImGui::EndMenu();
        }
    } else {
        for (const NodeType type : kOnceSteps) item(type);
    }
}

void ImageRecipeView::DropTarget(::gui::NodeEditor* editor, const ir::View& view, size_t index) {
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    if (!ImGui::BeginDragDropTarget()) return;
    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(
            kRowPayload, ImGuiDragDropFlags_AcceptBeforeDelivery | ImGuiDragDropFlags_AcceptNoDrawDefaultRect)) {
        const int node = *static_cast<const int*>(payload->Data);
        const auto dragged = std::find_if(view.rows.begin(), view.rows.end(), [node](const ir::Row& r) { return r.node_id == node; });
        if (dragged == view.rows.end()) {
            ImGui::EndDragDropTarget();
            return;
        }
        const size_t from = static_cast<size_t>(dragged - view.rows.begin());
        // The upper half puts it before this row (after the one above), the
        // lower half after it; Resize takes only after, the mix and Normalize
        // only before.
        const ir::Row& target = view.rows[index];
        const bool before = target.group == ir::Group::Normalize || target.mix ||
                            (target.group != ir::Group::Resize && ImGui::GetMousePos().y < (lo.y + hi.y) * 0.5f);
        const long long after_index = static_cast<long long>(index) - (before ? 1 : 0);
        const bool same = ir::MoveChangesNothing(view, from, after_index);
        const std::string refusal = same ? std::string() : ir::MoveRefusal(view, from, after_index);
        if (!refusal.empty()) {
            ImGui::SetTooltip("%s", refusal.c_str());
        } else if (!same) {
            // Where it lands: a wash on this row and a mark on the side it goes.
            const ui::Tokens& t = ui::CurrentTokens();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(lo, hi, ui::ToU32(t.hover), 4.0f);
            const float y = before ? lo.y - 3.0f : hi.y + 1.0f;
            dl->AddRectFilled(ImVec2(lo.x, y), ImVec2(hi.x, y + 2.0f), ui::ToU32(t.accent), 1.0f);
        }
        if (payload->IsDelivery() && !same) {
            if (refusal.empty() && after_index >= 0) {
                editor->RequestMoveStep(view.rows[from].node_id, view.rows[static_cast<size_t>(after_index)].node_id);
                message_ = "Moved " + view.rows[from].name + " (Ctrl+Z on the canvas undoes it).";
                message_error_ = false;
            } else if (!refusal.empty()) {
                message_ = refusal;
                message_error_ = true;
            }
        }
    }
    ImGui::EndDragDropTarget();
}

void ImageRecipeView::RenderRow(::gui::NodeEditor* editor, const ir::View& view, size_t index) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ir::Row& row = view.rows[index];
    ImGui::PushID(row.node_id >= 0 ? row.node_id : static_cast<int>(index) + 100000);
    const bool movable = ir::CanMove(row);
    const bool removable = row.group == ir::Group::Once || row.group == ir::Group::PerBatch;
    ImGui::BeginGroup();
    // The handle: drag the row to move its node.
    if (movable) {
        ui::GhostButton(ICON_FA_GRIP_VERTICAL "##drag");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drag to move this step; its node moves on the canvas");
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload(kRowPayload, &row.node_id, sizeof(row.node_id));
            ImGui::TextUnformatted(row.name.c_str());
            ImGui::EndDragDropSource();
        }
    } else {
        ImGui::Dummy(ImVec2(ui::ButtonWidth(ICON_FA_GRIP_VERTICAL, ui::ButtonSize::Small), ImGui::GetFrameHeight()));
    }
    ImGui::SameLine();
    const auto picture = row_texture_.find(row.node_id);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(kThumb, kThumb));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + kThumb, at.y + kThumb), ui::ToU32(t.bg_input), 4.0f);
    if (picture != row_texture_.end() && picture->second)
        dl->AddImageRounded(static_cast<ImTextureID>(picture->second), at, ImVec2(at.x + kThumb, at.y + kThumb),
                            ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 4.0f);
    const auto draws = row_draws_.find(row.node_id);
    if (draws != row_draws_.end() && !draws->second.empty() && draws->second.front())
        dl->AddImageRounded(static_cast<ImTextureID>(draws->second.front()), at, ImVec2(at.x + kThumb, at.y + kThumb),
                            ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 4.0f);
    ImGui::SameLine(0.0f, 12.0f);
    // Window-local x of the panel's right edge: nothing runs past it.
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::BeginGroup();
    const float text_x = ImGui::GetCursorPosX();
    ImGui::PushTextWrapPos(std::max(right, text_x + 1.0f));
    ImGui::TextUnformatted(row.name.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PushTextWrapPos(std::max(std::min(text_x + 320.0f, right), text_x + 1.0f));
    ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s", row.what.c_str(), row.output_text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    // A per-batch row's other draws on the sample: beside it as far as they
    // fit, else on a line of their own under the text.
    if (draws != row_draws_.end() && draws->second.size() > 1) {
        const float gap = 6.0f;
        const size_t extra = draws->second.size() - 1;
        const float beside = right - (ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + ImGui::GetScrollX()) - gap;
        size_t fit = beside > 0.0f ? static_cast<size_t>((beside + gap) / (kThumb + gap)) : 0;
        const bool own_line = fit == 0;
        if (own_line) {
            ImGui::Indent(ui::ButtonWidth(ICON_FA_GRIP_VERTICAL, ui::ButtonSize::Small) + kThumb + 20.0f);
            fit = std::max<size_t>(1, static_cast<size_t>((ImGui::GetContentRegionAvail().x + gap) / (kThumb + gap)));
        }
        for (size_t k = 1; k <= std::min(extra, fit); ++k) {
            if (!own_line || k > 1) ImGui::SameLine(0.0f, gap);
            if (draws->second[k])
                ImGui::Image(static_cast<ImTextureID>(draws->second[k]), ImVec2(kThumb, kThumb));
            else
                ImGui::Dummy(ImVec2(kThumb, kThumb));
        }
        if (own_line) ImGui::Unindent(ui::ButtonWidth(ICON_FA_GRIP_VERTICAL, ui::ButtonSize::Small) + kThumb + 20.0f);
    }
    // Actions, right aligned (on the next row when there is no room).
    const char* settings_label = open_settings_ == row.node_id ? "Hide settings" : "Settings";
    const float actions_w = ui::ButtonWidth(settings_label, ui::ButtonSize::Small) +
                            ui::ButtonWidth("Open dialog", ui::ButtonSize::Small) +
                            ImGui::CalcTextSize("Show on canvas").x + (removable ? ui::ButtonWidth("Remove", ui::ButtonSize::Small) : 0.0f) +
                            4.0f * ImGui::GetStyle().ItemSpacing.x + 16.0f;
    ui::SameLineRight(actions_w);
    if (row.node_id >= 0) {
        if (ui::LinkButton("Show on canvas") && show_on_canvas) show_on_canvas({row.node_id});
        ImGui::SameLine();
        if (ui::SecondaryButton(settings_label)) open_settings_ = open_settings_ == row.node_id ? -1 : row.node_id;
        ImGui::SameLine();
        if (ui::SecondaryButton("Open dialog")) editor->RequestOpenNode(row.node_id);
        if (removable) {
            ImGui::SameLine();
            if (ui::DangerButton("Remove")) {
                editor->RequestRemoveStep(row.node_id);
                message_ = "Removed " + row.name + " (Ctrl+Z on the canvas undoes it).";
                message_error_ = false;
                if (open_settings_ == row.node_id) open_settings_ = -1;
            }
        }
    }
    ImGui::EndGroup();
    // Dropping a row here puts it before or after this one.
    DropTarget(editor, view, index);

    // Its settings, with Properties' own editor.
    if (open_settings_ == row.node_id && row.node_id >= 0) {
        ::gui::MLNode* node = editor->GetNodeForConfiguration(row.node_id);
        ::gui::Properties* properties = editor->GetPropertiesPanel();
        if (node && properties) {
            std::string key = std::to_string(node->id);
            for (const auto& [k, v] : node->parameters) key += "|" + k + "=" + v;
            if (key != settings_key_) {
                properties->BuildSettingsRows(*node, settings_);
                settings_key_ = key;
            }
            ImGui::Indent(ui::ButtonWidth(ICON_FA_GRIP_VERTICAL, ui::ButtonSize::Small) + kThumb + 24.0f);
            properties->RenderSettingsRows(*node, settings_);
            ImGui::Unindent(ui::ButtonWidth(ICON_FA_GRIP_VERTICAL, ui::ButtonSize::Small) + kThumb + 24.0f);
        } else {
            ImGui::TextColored(t.text_dim, "Open the node's recipe on the canvas to change its settings.");
        }
    }
    ImGui::PopID();
}

void ImageRecipeView::Render(::gui::NodeEditor* editor) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!editor) return;
    const auto* live = editor->GetLiveCompile();
    const TrainingConfiguration* config = live ? live->Config() : nullptr;
    ir::Input in;
    in.config = config;
    for (const auto& node : editor->GetNodes())
        if (node.type == NodeType::Resize) in.resize_node_id = node.id;
    in.name_of = [editor](int id) {
        const ::gui::MLNode* node = editor->FindStep(id);
        return node ? node->name : std::string();
    };
    if (config) {
        const auto* entry = DataRegistry::Instance().GetImageDatasetEntry(config->dataset_name);
        in.dataset = entry ? config->dataset_name : std::string();
    }
    const ir::View view = ir::Build(in);

    // The header: what the recipe makes, and the sample its pictures use.
    ImGui::TextColored(t.text_bright, "Image recipe");
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", view.known ? view.subtitle.c_str() : view.reason.c_str());
    ImGui::PopTextWrapPos();
    if (!view.known) return;
    UpdatePreviews(*config, view);
    size_t count = 0;
    const std::string sample = SamplePath(config->dataset_name, &count);
    if (count > 0) {
        const std::string file = sample.substr(sample.find_last_of("/\\") + 1);
        // While the pictures are drawn (the first draw of a step builds its GPU code).
        const std::string label = (task_ ? "Drawing the pictures of " : "Sample ") + file;
        const float line_w = ImGui::CalcTextSize(label.c_str()).x + ui::ButtonWidth("Next sample", ui::ButtonSize::Small) + 24.0f;
        if (line_w <= ImGui::GetContentRegionAvail().x) {
            ui::SameLineRight(line_w);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.text_dim, "%s", label.c_str());
            ImGui::SameLine();
        } else {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", label.c_str());
            ImGui::PopTextWrapPos();
        }
        if (ui::SecondaryButton("Next sample")) ++sample_;
    } else {
        ImGui::TextColored(t.text_faint, "The pictures show once the images are loaded and indexed.");
    }
    if (message_graph_ != editor->GraphGeneration()) {
        message_.clear();
        open_settings_ = -1;
        settings_key_.clear();
        sample_ = 0;
        message_graph_ = editor->GraphGeneration();
    }
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.caution : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();

    // The groups, in training's order.
    const auto last_of = [&](ir::Group g) {
        int id = -1;
        for (const auto& row : view.rows)
            if (row.group == g && !row.mix) id = row.node_id;
        return id;
    };
    const int resize_id = in.resize_node_id;
    for (const ir::Group group : {ir::Group::Resize, ir::Group::Once, ir::Group::PerBatch, ir::Group::Normalize}) {
        const bool has_rows = std::any_of(view.rows.begin(), view.rows.end(), [&](const ir::Row& r) { return r.group == group; });
        if (group == ir::Group::Resize || group == ir::Group::Normalize) {
            for (size_t i = 0; i < view.rows.size(); ++i)
                if (view.rows[i].group == group) {
                    ui::BeginCard(group == ir::Group::Resize ? "recipe_resize" : "recipe_normalize");
                    ImGui::TextColored(t.text_dim, group == ir::Group::Resize ? "First" : "Last \xC2\xB7 on the GPU with each batch");
                    RenderRow(editor, view, i);
                    ui::EndCard();
                }
            continue;
        }
        ui::BeginCard(group == ir::Group::Once ? "recipe_once" : "recipe_per_batch");
        ImGui::TextColored(t.text_bright, "%s", GroupTitle(group));
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", GroupNote(group));
        ImGui::PopTextWrapPos();
        if (!has_rows)
            ImGui::TextColored(t.text_dim, group == ir::Group::Once ? "None: the images go to the model as resized."
                                                                    : "None in this graph: every epoch sees the same images.");
        for (size_t i = 0; i < view.rows.size(); ++i)
            if (view.rows[i].group == group) RenderRow(editor, view, i);
        // "+ Add": after the group's last row (an empty group: after the one before it).
        int after = last_of(group);
        if (after < 0) after = group == ir::Group::PerBatch ? (last_of(ir::Group::Once) >= 0 ? last_of(ir::Group::Once) : resize_id) : resize_id;
        const char* add = group == ir::Group::Once ? "+ Add a step##add_once" : "+ Add an augmentation##add_batch";
        if (after >= 0) {
            if (ui::SecondaryButton(add)) ImGui::OpenPopup(group == ir::Group::Once ? "##add_once_menu" : "##add_batch_menu");
            if (ImGui::BeginPopup(group == ir::Group::Once ? "##add_once_menu" : "##add_batch_menu")) {
                RenderAddMenu(editor, after, group == ir::Group::PerBatch);
                ImGui::EndPopup();
            }
        }
        ui::EndCard();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Drag a row by its handle to move its step; the canvas follows. Settings change the node itself.");
    ImGui::PopTextWrapPos();
}

}  // namespace cyxwiz
