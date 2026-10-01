// Script Editor debugger UI and keyboard handling.

#include "script_editor.h"
#include "../icons.h"
#include "../../scripting/debugger.h"
#include "../../scripting/scripting_engine.h"

#include <algorithm>
#include <memory>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {
// ============================================================================
// Debugger UI Functions
// ============================================================================

void ScriptEditorPanel::RenderDebugToolbar() {
    if (!debugger_) return;

    auto state = debugger_->GetState();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.2f, 1.0f));
    ImGui::BeginChild("##debug_toolbar", ImVec2(0, 35), ImGuiChildFlags_Border);

    // Debug status indicator
    if (state == scripting::DebugState::Paused) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_FA_PAUSE " Paused at line %d", debug_current_line_);
    } else if (state == scripting::DebugState::Running) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_FA_PLAY " Running...");
    } else if (state == scripting::DebugState::Stepping) {
        ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), ICON_FA_FORWARD_STEP " Stepping...");
    } else {
        ImGui::TextDisabled(ICON_FA_BUG " Debugger Disconnected");
    }

    ImGui::SameLine(ImGui::GetWindowWidth() - 280);

    // Continue/Pause button
    if (state == scripting::DebugState::Paused) {
        if (ImGui::Button(ICON_FA_PLAY " Continue")) {
            debugger_->Continue();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Continue execution (F5)");
        }
    } else if (state == scripting::DebugState::Running || state == scripting::DebugState::Stepping) {
        if (ImGui::Button(ICON_FA_PAUSE " Pause")) {
            debugger_->Pause();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Pause execution (F5)");
        }
    }

    ImGui::SameLine();

    // Step controls (only enabled when paused)
    ImGui::BeginDisabled(state != scripting::DebugState::Paused);

    if (ImGui::Button(ICON_FA_ARROW_DOWN " Over")) {
        debugger_->StepOver();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Step Over (F10)");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROW_RIGHT " Into")) {
        debugger_->StepInto();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Step Into (F11)");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROW_UP " Out")) {
        debugger_->StepOut();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Step Out (Shift+F11)");
    }

    ImGui::EndDisabled();

    ImGui::SameLine();

    // Stop button
    if (ImGui::Button(ICON_FA_STOP " Stop")) {
        debugger_->Stop();
        debug_mode_active_ = false;
        debug_current_line_ = -1;
        debug_current_cell_.clear();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Stop debugging (Shift+F5)");
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// F9 while debugging (TOFIX133 P0 item 7: dispatched once, from
// HandleKeyboardShortcuts, through core/script_keys).
void ScriptEditorPanel::ToggleBreakpointAtCursor() {
    if (tabs_.empty() || active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];

    if (tab->cell_mode) {
        // Cell mode - toggle breakpoint in selected cell
        if (tab->selected_cell >= 0 &&
            tab->selected_cell < static_cast<int>(tab->cell_manager.GetCellCount())) {
            Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
            if (cell.type == CellType::Code) {
                // Get current cursor line from editor
                int line = cell.editor.Doc().Primary().head.line + 1;  // 1-based

                // Toggle breakpoint
                auto it = std::find(cell.breakpoints.begin(), cell.breakpoints.end(), line);
                if (it != cell.breakpoints.end()) {
                    cell.breakpoints.erase(it);
                    if (debugger_) {
                        auto breakpoints = debugger_->GetBreakpointsForCell(cell.id);
                        for (const auto& bp : breakpoints) {
                            if (bp.line == line) {
                                debugger_->RemoveBreakpoint(bp.id);
                                break;
                            }
                        }
                    }
                } else {
                    cell.breakpoints.push_back(line);
                    if (debugger_) {
                        debugger_->AddBreakpoint(cell.id, line);
                    }
                }
            }
        }
    } else {
        // Traditional mode - toggle breakpoint in script
        int line = tab->editor.Doc().Primary().head.line + 1;  // 1-based

        std::string file_id = tab->filepath.empty() ? tab->filename : tab->filepath;

        // Toggle breakpoint
        auto it = std::find(tab->breakpoints.begin(), tab->breakpoints.end(), line);
        if (it != tab->breakpoints.end()) {
            tab->breakpoints.erase(it);
            if (debugger_) {
                auto breakpoints = debugger_->GetBreakpointsForCell(file_id);
                for (const auto& bp : breakpoints) {
                    if (bp.line == line) {
                        debugger_->RemoveBreakpoint(bp.id);
                        break;
                    }
                }
            }
        } else {
            tab->breakpoints.push_back(line);
            if (debugger_) {
                debugger_->AddBreakpoint(file_id, line);
            }
        }
    }
}

void ScriptEditorPanel::Debug() {
    if (!IsActiveTabEditable()) {
        spdlog::warn("No script to debug");
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Initialize debugger if not already done
    if (!debugger_) {
        debugger_ = std::make_unique<scripting::DebuggerManager>();
        if (scripting_engine_ && scripting_engine_->IsInitialized()) {
            // Get the raw ScriptingEngine pointer from the shared_ptr
            debugger_->Initialize(scripting_engine_.get());

            // Set up callbacks
            debugger_->SetBreakpointHitCallback([this](const std::string& cell_id, int line) {
                debug_mode_active_ = true;
                debug_current_cell_ = cell_id;
                debug_current_line_ = line;
                spdlog::info("Breakpoint hit at {}:{}", cell_id, line);
            });

            debugger_->SetStateChangedCallback([this](scripting::DebugState state) {
                if (state == scripting::DebugState::Disconnected) {
                    debug_mode_active_ = false;
                    debug_current_line_ = -1;
                    debug_current_cell_.clear();
                } else if (state == scripting::DebugState::Running) {
                    debug_mode_active_ = true;
                }
            });

            spdlog::info("Debugger initialized");
        } else {
            spdlog::error("Cannot initialize debugger: scripting engine not ready");
            debugger_.reset();
            return;
        }
    }

    // Get current cell content to debug
    if (tab->cell_mode && tab->selected_cell >= 0 &&
        tab->selected_cell < static_cast<int>(tab->cell_manager.GetCellCount())) {
        Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
        if (cell.type == CellType::Code) {
            cell.SyncSourceFromEditor();

            // Execute with debugging enabled
            debugger_->ExecuteWithDebug(cell.source, cell.id);
            debug_mode_active_ = true;
            spdlog::info("Started debugging cell {}", cell.id);
        }
    } else {
        // Debug whole script (traditional mode)
        std::string script = tab->editor.GetText();
        debugger_->ExecuteWithDebug(script, tab->filepath.empty() ? tab->filename : tab->filepath);
        debug_mode_active_ = true;
        spdlog::info("Started debugging script");
    }
}

} // namespace cyxwiz
