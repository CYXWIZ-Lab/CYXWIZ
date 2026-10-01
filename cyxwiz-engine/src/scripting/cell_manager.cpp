#include "cell_manager.h"
#include "../core/cyx_format.h"
#include "scripting_engine.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>
#include <regex>

namespace cyxwiz {

CellManager::CellManager() {
    // Start with one empty code cell
    AddCell(CellType::Code);
}

CellManager::~CellManager() {
    Clear();
}

// ========== Cell Access ==========

Cell& CellManager::GetCell(int index) {
    if (!IsValidIndex(index)) {
        throw std::out_of_range("Cell index out of range");
    }
    return cells_[index];
}

const Cell& CellManager::GetCell(int index) const {
    if (!IsValidIndex(index)) {
        throw std::out_of_range("Cell index out of range");
    }
    return cells_[index];
}

Cell* CellManager::GetCellById(const std::string& id) {
    for (auto& cell : cells_) {
        if (cell.id == id) {
            return &cell;
        }
    }
    return nullptr;
}

// ========== Cell Operations ==========

int CellManager::AddCell(CellType type, int position) {
    std::lock_guard<std::mutex> lock(mutex_);

    Cell cell(type);

    if (position < 0 || position >= static_cast<int>(cells_.size())) {
        cells_.push_back(std::move(cell));
        return static_cast<int>(cells_.size()) - 1;
    } else {
        cells_.insert(cells_.begin() + position, std::move(cell));
        return position;
    }
}

bool CellManager::DeleteCell(int index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!IsValidIndex(index)) {
        return false;
    }

    // Don't delete the last cell
    if (cells_.size() <= 1) {
        spdlog::warn("Cannot delete the last cell");
        return false;
    }

    // Clean up outputs
    cells_[index].ClearOutputs();
    cells_.erase(cells_.begin() + index);

    spdlog::debug("Deleted cell at index {}", index);
    return true;
}

bool CellManager::MoveCell(int from, int to) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!IsValidIndex(from) || to < 0 || to > static_cast<int>(cells_.size())) {
        return false;
    }

    if (from == to) {
        return true;
    }

    Cell cell = std::move(cells_[from]);
    cells_.erase(cells_.begin() + from);

    if (to > from) {
        to--;
    }

    cells_.insert(cells_.begin() + to, std::move(cell));

    spdlog::debug("Moved cell from {} to {}", from, to);
    return true;
}

int CellManager::DuplicateCell(int index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!IsValidIndex(index)) {
        return -1;
    }

    const Cell& original = cells_[index];
    Cell copy(original.type, original.source);

    // Insert after the original
    int new_index = index + 1;
    cells_.insert(cells_.begin() + new_index, std::move(copy));

    spdlog::debug("Duplicated cell {} to {}", index, new_index);
    return new_index;
}

bool CellManager::MergeCells(int first, int second) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!IsValidIndex(first) || !IsValidIndex(second)) {
        return false;
    }

    if (second != first + 1) {
        spdlog::warn("Can only merge adjacent cells");
        return false;
    }

    // Merge source content
    Cell& cell1 = cells_[first];
    Cell& cell2 = cells_[second];

    cell1.source += "\n" + cell2.source;
    cell1.SyncEditorFromSource();

    // Remove second cell
    cell2.ClearOutputs();
    cells_.erase(cells_.begin() + second);

    spdlog::debug("Merged cells {} and {}", first, second);
    return true;
}

int CellManager::SplitCell(int index, int line) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!IsValidIndex(index)) {
        return -1;
    }

    Cell& cell = cells_[index];
    cell.SyncSourceFromEditor();

    // Split source by lines
    std::istringstream stream(cell.source);
    std::string first_part, second_part;
    std::string current_line;
    int line_num = 0;

    while (std::getline(stream, current_line)) {
        if (line_num < line) {
            if (!first_part.empty()) first_part += "\n";
            first_part += current_line;
        } else {
            if (!second_part.empty()) second_part += "\n";
            second_part += current_line;
        }
        line_num++;
    }

    if (second_part.empty()) {
        spdlog::warn("Nothing to split at line {}", line);
        return -1;
    }

    // Update first cell
    cell.source = first_part;
    cell.SyncEditorFromSource();

    // Create second cell
    Cell new_cell(cell.type, second_part);
    int new_index = index + 1;
    cells_.insert(cells_.begin() + new_index, std::move(new_cell));

    spdlog::debug("Split cell {} at line {}", index, line);
    return new_index;
}

bool CellManager::ChangeCellType(int index, CellType type) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!IsValidIndex(index)) {
        return false;
    }

    Cell& cell = cells_[index];
    if (cell.type == type) {
        return true;
    }

    // Sync source before type change
    cell.SyncSourceFromEditor();

    cell.type = type;
    cell.ClearOutputs();

    // Setup editor if changing to code
    if (type == CellType::Code) {
        cell.SetupCodeEditor();
    }

    spdlog::debug("Changed cell {} type to {}", index, static_cast<int>(type));
    return true;
}

void CellManager::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        cell.ClearOutputs();
    }
    cells_.clear();
    execution_counter_ = 0;
}

// ========== Output Management ==========

void CellManager::ClearCellOutput(int index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (IsValidIndex(index)) {
        cells_[index].ClearOutputs();
    }
}

void CellManager::ClearAllOutputs() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        cell.ClearOutputs();
    }
    execution_counter_ = 0;
}

void CellManager::AddCellOutput(int index, const CellOutput& output) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (IsValidIndex(index)) {
        cells_[index].AddOutput(output);

        if (on_cell_output_added_) {
            on_cell_output_added_(index, output);
        }
    }
}

// ========== Execution ==========

void CellManager::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void CellManager::RunCell(int index) {
    if (!IsValidIndex(index)) {
        return;
    }

    Cell& cell = cells_[index];

    // Only run code cells
    if (cell.type != CellType::Code) {
        spdlog::debug("Skipping non-code cell {}", index);
        return;
    }

    // Sync source from editor
    cell.SyncSourceFromEditor();

    if (cell.source.empty()) {
        spdlog::debug("Skipping empty cell {}", index);
        return;
    }

    ExecuteCellInternal(index);
}

void CellManager::RunAllCells() {
    for (int i = 0; i < static_cast<int>(cells_.size()); i++) {
        if (cells_[i].type == CellType::Code) {
            execution_queue_.push_back(i);
        }
    }

    if (!execution_queue_.empty()) {
        RunCell(execution_queue_.front());
        execution_queue_.erase(execution_queue_.begin());
    }
}

void CellManager::RunCellsAbove(int index) {
    for (int i = 0; i <= index && i < static_cast<int>(cells_.size()); i++) {
        if (cells_[i].type == CellType::Code) {
            execution_queue_.push_back(i);
        }
    }

    if (!execution_queue_.empty()) {
        RunCell(execution_queue_.front());
        execution_queue_.erase(execution_queue_.begin());
    }
}

void CellManager::RunCellsBelow(int index) {
    for (int i = index; i < static_cast<int>(cells_.size()); i++) {
        if (cells_[i].type == CellType::Code) {
            execution_queue_.push_back(i);
        }
    }

    if (!execution_queue_.empty()) {
        RunCell(execution_queue_.front());
        execution_queue_.erase(execution_queue_.begin());
    }
}

void CellManager::InterruptExecution() {
    if (scripting_engine_) {
        scripting_engine_->StopScript();
    }
    execution_queue_.clear();
    is_running_ = false;

    if (running_cell_index_ >= 0 && IsValidIndex(running_cell_index_)) {
        cells_[running_cell_index_].state = CellState::Error;
        cells_[running_cell_index_].AddOutput(CellOutput::Error("Execution interrupted"));
    }
    running_cell_index_ = -1;
}

void CellManager::ExecuteCellInternal(int index) {
    if (!scripting_engine_) {
        spdlog::error("No scripting engine set");
        return;
    }

    Cell& cell = cells_[index];

    // Clear previous outputs
    cell.ClearOutputs();

    // Update state
    cell.state = CellState::Running;
    running_cell_index_ = index;
    is_running_ = true;
    execution_counter_++;
    cell.execution_count = execution_counter_;

    if (on_cell_state_changed_) {
        on_cell_state_changed_(index);
    }

    spdlog::info("Executing cell {} [{}]", index, cell.execution_count);

    // Set up output callback for real-time output
    scripting_engine_->SetOutputCallback([this, index](const std::string& output) {
        OnExecutionOutput(index, output, false);
    });

    // Set up completion callback
    scripting_engine_->SetCompletionCallback([this, index](const scripting::ExecutionResult& result) {
        // Handle captured plots
        for (const auto& plot : result.plots) {
            CellOutput plot_output;
            plot_output.type = OutputType::Plot;
            plot_output.name = plot.label;
            plot_output.image_data = plot.png_data;
            plot_output.width = plot.width;
            plot_output.height = plot.height;
            plot_output.mime_type = "image/png";
            AddCellOutput(index, plot_output);
        }

        OnExecutionComplete(index, result.success, result.error_message);
    });

    // Execute asynchronously
    scripting_engine_->ExecuteScriptAsync(cell.source);
}

void CellManager::OnExecutionOutput(int cell_index, const std::string& output, bool is_error) {
    if (!IsValidIndex(cell_index)) return;

    CellOutput cell_output = is_error
        ? CellOutput::Error(output)
        : CellOutput::Text(output);

    AddCellOutput(cell_index, cell_output);
}

void CellManager::OnExecutionComplete(int cell_index, bool success, const std::string& error) {
    if (!IsValidIndex(cell_index)) return;

    Cell& cell = cells_[cell_index];

    if (success) {
        cell.state = CellState::Success;
    } else {
        cell.state = CellState::Error;
        if (!error.empty()) {
            AddCellOutput(cell_index, CellOutput::Error(error));
        }
    }

    running_cell_index_ = -1;
    is_running_ = false;

    if (on_cell_state_changed_) {
        on_cell_state_changed_(cell_index);
    }

    if (on_execution_complete_) {
        on_execution_complete_(cell_index);
    }

    // Run next cell in queue if any
    if (!execution_queue_.empty()) {
        int next_index = execution_queue_.front();
        execution_queue_.erase(execution_queue_.begin());
        RunCell(next_index);
    }

    spdlog::info("Cell {} execution complete. Success: {}", cell_index, success);
}

// ========== Serialization ==========

bool CellManager::HasCellMarkers(const std::string& content) {
    return cyx::HasCellMarkers(content);
}

namespace {
CellType ToCellType(cyx::CellKind kind) {
    switch (kind) {
        case cyx::CellKind::Markdown: return CellType::Markdown;
        case cyx::CellKind::Raw: return CellType::Raw;
        case cyx::CellKind::Code: break;
    }
    return CellType::Code;
}

cyx::CellKind ToCellKind(CellType type) {
    switch (type) {
        case CellType::Markdown: return cyx::CellKind::Markdown;
        case CellType::Raw: return cyx::CellKind::Raw;
        case CellType::Code: break;
    }
    return cyx::CellKind::Code;
}
}  // namespace

// The format lives in core/cyx_format (TOFIX133 P0 item 15): bare %% markers,
// comments before the first marker kept, empty cells kept, and a stable
// save/load round trip.
bool CellManager::ParseFromCyx(const std::string& content) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& cell : cells_) {
        cell.ClearOutputs();
    }
    cells_.clear();
    for (const auto& parsed : cyx::Parse(content)) {
        cells_.emplace_back(ToCellType(parsed.kind), parsed.source);
    }
    if (cells_.empty()) {
        cells_.emplace_back(CellType::Code);
    }
    spdlog::info("Parsed .cyx file with {} cells", cells_.size());
    return true;
}

std::string CellManager::SerializeToCyx() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<cyx::Cell> cells;
    cells.reserve(cells_.size());
    for (const auto& cell : cells_) {
        cells.push_back({ToCellKind(cell.type), cell.source});
    }
    return cyx::Serialize(cells);
}

// ========== Editor Theme ==========

void CellManager::ApplyEditorPalette(const TextEditor::Palette& palette) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        if (cell.type == CellType::Code) {
            cell.editor.SetPalette(palette);
        }
    }
}

void CellManager::ApplyTabSize(int size) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        if (cell.type == CellType::Code) {
            cell.editor.SetTabSize(size);
        }
    }
}

void CellManager::ApplyShowWhitespace(bool show) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        if (cell.type == CellType::Code) {
            cell.editor.SetShowWhitespaces(show);
        }
    }
}
void CellManager::ApplySyntaxHighlighting(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        if (cell.type == CellType::Code) {
            cell.editor.SetColorizerEnable(enabled);
        }
    }
}

} // namespace cyxwiz
