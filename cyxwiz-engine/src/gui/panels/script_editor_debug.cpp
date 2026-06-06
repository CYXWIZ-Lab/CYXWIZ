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

void ScriptEditorPanel::RenderBreakpointGutter(Cell& cell, int /*cell_index*/) {
    if (cell.type != CellType::Code) return;

    int line_count = cell.editor.GetTotalLines();
    if (line_count == 0) line_count = 1;

    float line_height = ImGui::GetTextLineHeightWithSpacing();
    float gutter_width = 20.0f;

    ImGui::BeginChild("##bp_gutter", ImVec2(gutter_width, line_count * line_height), ImGuiChildFlags_None);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cursor_start = ImGui::GetCursorScreenPos();

    for (int line = 0; line < line_count; ++line) {
        ImVec2 line_pos = ImVec2(cursor_start.x, cursor_start.y + line * line_height);
        ImVec2 center = ImVec2(line_pos.x + gutter_width * 0.5f, line_pos.y + line_height * 0.5f);
        float radius = 5.0f;

        // Check if this line has a breakpoint
        bool has_breakpoint = std::find(cell.breakpoints.begin(), cell.breakpoints.end(), line + 1) != cell.breakpoints.end();

        // Check if this is the current debug line
        bool is_current_line = debug_mode_active_ &&
                               debug_current_cell_ == cell.id &&
                               debug_current_line_ == line + 1;

        // Draw hover indicator
        ImGui::SetCursorScreenPos(line_pos);
        ImGui::InvisibleButton(("##bp_line_" + std::to_string(line)).c_str(), ImVec2(gutter_width, line_height));

        if (ImGui::IsItemHovered()) {
            draw_list->AddCircle(center, radius, IM_COL32(150, 150, 150, 150), 12, 1.0f);

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                // Toggle breakpoint
                if (has_breakpoint) {
                    cell.breakpoints.erase(
                        std::remove(cell.breakpoints.begin(), cell.breakpoints.end(), line + 1),
                        cell.breakpoints.end()
                    );

                    // Notify debugger
                    if (debugger_) {
                        auto breakpoints = debugger_->GetBreakpointsForCell(cell.id);
                        for (const auto& bp : breakpoints) {
                            if (bp.line == line + 1) {
                                debugger_->RemoveBreakpoint(bp.id);
                                break;
                            }
                        }
                    }
                } else {
                    cell.breakpoints.push_back(line + 1);

                    // Notify debugger
                    if (debugger_) {
                        debugger_->AddBreakpoint(cell.id, line + 1);
                    }
                }
            }
        }

        // Draw breakpoint indicator
        if (has_breakpoint) {
            draw_list->AddCircleFilled(center, radius, IM_COL32(200, 50, 50, 255), 12);
        }

        // Draw current line indicator (arrow)
        if (is_current_line) {
            ImVec2 arrow_points[3] = {
                ImVec2(center.x - 4, center.y - 4),
                ImVec2(center.x + 4, center.y),
                ImVec2(center.x - 4, center.y + 4)
            };
            draw_list->AddTriangleFilled(arrow_points[0], arrow_points[1], arrow_points[2],
                                         IM_COL32(255, 255, 0, 255));
        }
    }

    ImGui::EndChild();
}

void ScriptEditorPanel::RenderScriptBreakpointGutter(float height) {
    if (tabs_.empty() || active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];
    int line_count = tab->editor.GetTotalLines();
    if (line_count == 0) line_count = 1;

    float line_height = ImGui::GetTextLineHeightWithSpacing();
    float gutter_width = 20.0f;

    // Get editor scroll position to sync gutter scrolling
    // Note: TextEditor doesn't expose scroll position directly, so we estimate
    auto coords = tab->editor.GetCursorPosition();

    ImGui::BeginChild("##script_bp_gutter", ImVec2(gutter_width, height), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cursor_start = ImGui::GetCursorScreenPos();

    // Calculate visible lines based on height
    int visible_lines = static_cast<int>(height / line_height) + 1;
    int start_line = 0;  // Would need TextEditor scroll position for accurate sync

    for (int line = start_line; line < std::min(start_line + visible_lines + 5, line_count); ++line) {
        ImVec2 line_pos = ImVec2(cursor_start.x, cursor_start.y + (line - start_line) * line_height);
        ImVec2 center = ImVec2(line_pos.x + gutter_width * 0.5f, line_pos.y + line_height * 0.5f);
        float radius = 5.0f;

        int line_number = line + 1;  // 1-based line numbers

        // Check if this line has a breakpoint
        bool has_breakpoint = std::find(tab->breakpoints.begin(), tab->breakpoints.end(), line_number) != tab->breakpoints.end();

        // Check if this is the current debug line
        std::string file_id = tab->filepath.empty() ? tab->filename : tab->filepath;
        bool is_current_line = debug_mode_active_ &&
                               debug_current_cell_ == file_id &&
                               debug_current_line_ == line_number;

        // Draw hover indicator and handle clicks
        ImGui::SetCursorScreenPos(line_pos);
        ImGui::InvisibleButton(("##script_bp_line_" + std::to_string(line)).c_str(), ImVec2(gutter_width, line_height));

        if (ImGui::IsItemHovered()) {
            draw_list->AddCircle(center, radius, IM_COL32(150, 150, 150, 150), 12, 1.0f);

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                // Toggle breakpoint
                if (has_breakpoint) {
                    tab->breakpoints.erase(
                        std::remove(tab->breakpoints.begin(), tab->breakpoints.end(), line_number),
                        tab->breakpoints.end()
                    );

                    // Notify debugger
                    if (debugger_) {
                        auto breakpoints = debugger_->GetBreakpointsForCell(file_id);
                        for (const auto& bp : breakpoints) {
                            if (bp.line == line_number) {
                                debugger_->RemoveBreakpoint(bp.id);
                                break;
                            }
                        }
                    }
                } else {
                    tab->breakpoints.push_back(line_number);

                    // Notify debugger
                    if (debugger_) {
                        debugger_->AddBreakpoint(file_id, line_number);
                    }
                }
            }
        }

        // Draw breakpoint indicator
        if (has_breakpoint) {
            draw_list->AddCircleFilled(center, radius, IM_COL32(200, 50, 50, 255), 12);
        }

        // Draw current line indicator (arrow)
        if (is_current_line) {
            ImVec2 arrow_points[3] = {
                ImVec2(center.x - 4, center.y - 4),
                ImVec2(center.x + 4, center.y),
                ImVec2(center.x - 4, center.y + 4)
            };
            draw_list->AddTriangleFilled(arrow_points[0], arrow_points[1], arrow_points[2],
                                         IM_COL32(255, 255, 0, 255));
        }
    }

    ImGui::EndChild();
}

void ScriptEditorPanel::HandleDebugKeyboardShortcuts() {
    if (!debugger_) return;

    auto state = debugger_->GetState();

    // F5 - Continue / Start Debug
    if (ImGui::IsKeyPressed(ImGuiKey_F5)) {
        if (ImGui::GetIO().KeyShift) {
            // Shift+F5 - Stop debugging
            debugger_->Stop();
            debug_mode_active_ = false;
            debug_current_line_ = -1;
            debug_current_cell_.clear();
        } else if (state == scripting::DebugState::Paused) {
            debugger_->Continue();
        } else if (state == scripting::DebugState::Disconnected) {
            Debug(); // Start debugging
        }
    }

    // F10 - Step Over
    if (ImGui::IsKeyPressed(ImGuiKey_F10)) {
        if (state == scripting::DebugState::Paused) {
            debugger_->StepOver();
        }
    }

    // F11 - Step Into / Shift+F11 - Step Out
    if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
        if (state == scripting::DebugState::Paused) {
            if (ImGui::GetIO().KeyShift) {
                debugger_->StepOut();
            } else {
                debugger_->StepInto();
            }
        }
    }

    // F9 - Toggle breakpoint at current line
    if (ImGui::IsKeyPressed(ImGuiKey_F9)) {
        if (tabs_.empty() || active_tab_index_ < 0) return;

        auto& tab = tabs_[active_tab_index_];

        if (tab->cell_mode) {
            // Cell mode - toggle breakpoint in selected cell
            if (tab->selected_cell >= 0 &&
                tab->selected_cell < static_cast<int>(tab->cell_manager.GetCellCount())) {
                Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
                if (cell.type == CellType::Code) {
                    // Get current cursor line from editor
                    auto coords = cell.editor.GetCursorPosition();
                    int line = coords.mLine + 1; // 1-based

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
            auto coords = tab->editor.GetCursorPosition();
            int line = coords.mLine + 1; // 1-based

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
}

void ScriptEditorPanel::Debug() {
    if (tabs_.empty() || active_tab_index_ < 0) {
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
