#pragma once

#include "../core/training_generation_preview_settings.h"

#include <imgui.h>

#include <string>
#include <utility>
#include <vector>

namespace gui {

// Generation settings shared by the Data Loader's Generation tab and the LM
// Generation panel (TOFIX159 decision 7): one set of widgets, one wording.
// Each Render* draws its title line and returns true when a setting changed.

// Muted note wrapped at the current width.
void GenerationHint(const char* text);
// Puts the next item, `width` wide, on this line when it fits, else on the next.
void GenerationFlowSameLine(float width, float gap = -1.0f);
// Newlines shown as \n (stop texts, labels).
std::string GenerationShown(const std::string& text);
// Text segments in their own colours, wrapped at the current width.
void DrawGenerationSegments(const std::vector<std::pair<std::string, ImVec4>>& segments);

// Template choice, the system prompt (`system_note` after its label) and,
// for Role lines, the three labels.
bool RenderGenerationTemplateSettings(cyxwiz::TrainingGenerationPreviewSettings& settings, const char* title,
                                      const char* system_note);
// End-of-sequence and template stops, the user's stop texts and max new tokens.
bool RenderGenerationStopSettings(cyxwiz::TrainingGenerationPreviewSettings& settings, const char* title,
                                  std::string& new_stop);
// Greedy, sampling (temperature, top-k, top-p, min-p, seed and, with
// `samples_each`, samples per prompt) or beam search, and the penalties.
// `random_seed` (optional) adds "New seed for each reply" under the seed.
bool RenderGenerationDecodingSettings(cyxwiz::TrainingGenerationPreviewSettings& settings, const char* title,
                                      bool samples_each, bool* random_seed = nullptr);

}  // namespace gui
