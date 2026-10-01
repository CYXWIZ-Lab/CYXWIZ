#include "cell_manager.h"
#include "../core/cyx_format.h"
#include "../core/notebook_format.h"
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
    }
}

// ========== Execution ==========
//
// TOFIX133 P0 items 8-10. The worker thread never touches cells_: its
// callbacks only post events to a mailbox (shared, so a closed notebook
// leaves nothing dangling), and Pump() applies them on the UI thread, then
// starts the next queued cell there. Before, the completion callback started
// the next cell on the worker itself (Run All stopped: the engine joined its
// own thread) and changed cells_ while the UI drew them. The queue holds cell
// ids, so adding or deleting cells during a run cannot shift it.

void CellManager::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

int CellManager::IndexOfId(const std::string& id) const {
    if (id.empty()) return -1;
    for (int i = 0; i < static_cast<int>(cells_.size()); ++i)
        if (cells_[i].id == id) return i;
    return -1;
}

int CellManager::GetRunningCellIndex() const {
    return IndexOfId(running_cell_id_);
}

void CellManager::Enqueue(int from, int to) {
    for (int i = std::max(0, from); i <= to && i < static_cast<int>(cells_.size()); ++i) {
        Cell& cell = cells_[i];
        if (cell.type != CellType::Code) continue;
        cell.SyncSourceFromEditor();
        if (cell.source.find_first_not_of(" \t\r\n") == std::string::npos) continue;  // nothing to run
        if (std::find(execution_queue_.begin(), execution_queue_.end(), cell.id) != execution_queue_.end()) continue;
        execution_queue_.push_back(cell.id);
        if (cell.id != running_cell_id_) cell.state = CellState::Queued;
    }
    StartNext();
}

void CellManager::RunCell(int index) {
    if (IsValidIndex(index)) Enqueue(index, index);
}

void CellManager::RunAllCells() {
    Enqueue(0, static_cast<int>(cells_.size()) - 1);
}

void CellManager::RunCellsAbove(int index) {
    Enqueue(0, index);
}

void CellManager::RunCellsBelow(int index) {
    Enqueue(index, static_cast<int>(cells_.size()) - 1);
}

void CellManager::InterruptExecution() {
    for (const auto& id : execution_queue_) {
        const int i = IndexOfId(id);
        if (i >= 0 && cells_[i].state == CellState::Queued) cells_[i].state = CellState::Idle;
    }
    execution_queue_.clear();
    if (scripting_engine_ && is_running_) {
        scripting_engine_->StopScript();  // the Done event marks the cell
    }
}

bool CellManager::StartNext() {
    if (is_running_ || !scripting_engine_) return false;
    // Another script (the editor, the console) may hold the engine: wait.
    if (scripting_engine_->IsScriptRunning()) return false;
    while (!execution_queue_.empty()) {
        const std::string id = execution_queue_.front();
        execution_queue_.erase(execution_queue_.begin());
        const int index = IndexOfId(id);
        if (index < 0) continue;  // deleted while queued
        ExecuteCellInternal(index);
        return true;
    }
    return false;
}

void CellManager::ExecuteCellInternal(int index) {
    Cell& cell = cells_[index];
    cell.ClearOutputs();
    cell.state = CellState::Running;
    running_cell_id_ = cell.id;
    is_running_ = true;
    cell.execution_count = ++execution_counter_;
    const std::uint64_t run = ++run_counter_;
    spdlog::info("Executing cell {} [{}]", index, cell.execution_count);

    std::weak_ptr<Mailbox> weak = mailbox_;
    scripting::ScriptingEngine::RunCallbacks callbacks;
    callbacks.on_output = [weak, run](const std::string& text) {
        if (auto box = weak.lock()) {
            std::lock_guard<std::mutex> lock(box->mutex);
            box->events.push_back({RunEvent::Kind::Output, run, text, {}, false, false, {}});
        }
    };
    callbacks.on_complete = [weak, run](const scripting::ExecutionResult& result) {
        auto box = weak.lock();
        if (!box) return;
        std::lock_guard<std::mutex> lock(box->mutex);
        for (const auto& plot : result.plots) {
            RunEvent e{RunEvent::Kind::Plot, run, {}, {}, false, false, {}};
            e.output.type = OutputType::Plot;
            e.output.name = plot.label;
            e.output.image_data = plot.png_data;
            e.output.width = plot.width;
            e.output.height = plot.height;
            e.output.mime_type = "image/png";
            box->events.push_back(std::move(e));
        }
        box->events.push_back(
            {RunEvent::Kind::Done, run, {}, {}, result.success, result.was_cancelled, result.error_message});
    };
    if (!scripting_engine_->ExecuteScriptAsync(cell.source, std::move(callbacks))) {
        // The engine was taken between the check and the start: queue again.
        cell.state = CellState::Queued;
        execution_queue_.insert(execution_queue_.begin(), cell.id);
        running_cell_id_.clear();
        is_running_ = false;
    }
}

void CellManager::Pump() {
    std::vector<RunEvent> events;
    {
        std::lock_guard<std::mutex> lock(mailbox_->mutex);
        events.swap(mailbox_->events);
    }
    for (auto& e : events) {
        if (e.run != run_counter_) continue;  // a run this notebook no longer tracks
        const int index = IndexOfId(running_cell_id_);
        switch (e.kind) {
            case RunEvent::Kind::Output:
                if (index >= 0) cells_[index].AddOutput(CellOutput::Text(e.text));
                break;
            case RunEvent::Kind::Plot:
                if (index >= 0) cells_[index].AddOutput(e.output);
                break;
            case RunEvent::Kind::Done:
                if (index >= 0) {
                    Cell& cell = cells_[index];
                    if (e.cancelled) {
                        cell.state = CellState::Error;
                        cell.AddOutput(CellOutput::Error("Execution interrupted"));
                    } else if (e.success) {
                        cell.state = CellState::Success;
                    } else {
                        cell.state = CellState::Error;
                        if (!e.error.empty()) cell.AddOutput(CellOutput::Error(e.error));
                    }
                    spdlog::info("Cell {} execution complete. Success: {}", index, e.success);
                }
                running_cell_id_.clear();
                is_running_ = false;
                if (e.cancelled || !e.success) {
                    // Stop the rest of Run All after an error, as notebooks do.
                    for (const auto& id : execution_queue_) {
                        const int q = IndexOfId(id);
                        if (q >= 0 && cells_[q].state == CellState::Queued) cells_[q].state = CellState::Idle;
                    }
                    execution_queue_.clear();
                }
                break;
        }
    }
    StartNext();
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

namespace {
CellType ToCellType(nb::CellKind kind) {
    switch (kind) {
        case nb::CellKind::Markdown: return CellType::Markdown;
        case nb::CellKind::Raw: return CellType::Raw;
        case nb::CellKind::Code: break;
    }
    return CellType::Code;
}

CellOutput FromNotebookOutput(const nb::Output& o) {
    CellOutput out;
    switch (o.kind) {
        case nb::Output::Kind::Stream:
            out = CellOutput::Stream(o.text, o.stream_name);
            break;
        case nb::Output::Kind::Error: {
            std::string text;
            for (const auto& line : o.traceback) text += nb::StripAnsi(line) + "\n";
            if (text.empty()) text = o.ename + ": " + o.evalue;
            out = CellOutput::Error(text);
            break;
        }
        case nb::Output::Kind::Result:
        case nb::Output::Kind::Display:
            if (!o.png_base64.empty()) {
                out.type = OutputType::Plot;
                out.mime_type = "image/png";
                out.image_data = nb::DecodeBase64(o.png_base64);
            } else {
                out = CellOutput::Text(o.text);
            }
            break;
    }
    out.ipynb_raw = o.raw;
    return out;
}

nb::Output ToNotebookOutput(const CellOutput& out, int execution_count) {
    nb::Output o;
    o.raw = out.ipynb_raw;
    switch (out.type) {
        case OutputType::Error: {
            const std::string raw = o.raw;
            o = nb::ErrorFromText(out.data);
            o.raw = raw;
            break;
        }
        case OutputType::Image:
        case OutputType::Plot:
            o.kind = nb::Output::Kind::Display;
            o.png_base64 = out.image_data.empty() ? out.data : nb::EncodeBase64(out.image_data);
            o.text = out.name.empty() ? "<Figure>" : out.name;
            break;
        case OutputType::Html:
            o.kind = nb::Output::Kind::Display;
            o.html = out.data;
            o.text = out.data;
            break;
        case OutputType::Text:
        case OutputType::Stream:
        case OutputType::Table:
        case OutputType::Markdown:
            o.kind = nb::Output::Kind::Stream;
            o.stream_name = out.name.empty() ? "stdout" : out.name;
            o.text = out.data;
            break;
    }
    o.execution_count = execution_count;
    return o;
}
}  // namespace

bool CellManager::ParseFromIpynb(const std::string& content, std::string* error) {
    nb::Notebook notebook;
    if (!nb::ParseIpynb(content, notebook, error)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& cell : cells_) {
        cell.ClearOutputs();
    }
    cells_.clear();
    for (const auto& parsed : notebook.cells) {
        Cell& cell = cells_.emplace_back(ToCellType(parsed.kind), parsed.source);
        cell.execution_count = parsed.execution_count;
        cell.ipynb_extra = parsed.extra;
        for (const auto& o : parsed.outputs) cell.outputs.push_back(FromNotebookOutput(o));
    }
    if (cells_.empty()) {
        cells_.emplace_back(CellType::Code);
    }
    ipynb_metadata_ = notebook.metadata;
    ipynb_minor_ = notebook.nbformat_minor;
    spdlog::info("Parsed .ipynb file with {} cells", cells_.size());
    return true;
}

std::string CellManager::SerializeToIpynb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nb::Notebook notebook;
    notebook.metadata = ipynb_metadata_.empty() ? nb::DefaultMetadata("") : ipynb_metadata_;
    notebook.nbformat_minor = ipynb_minor_;
    for (const auto& cell : cells_) {
        nb::Cell out;
        out.kind = cell.type == CellType::Markdown ? nb::CellKind::Markdown
                   : cell.type == CellType::Raw    ? nb::CellKind::Raw
                                                   : nb::CellKind::Code;
        out.source = cell.source;
        if (cell.type == CellType::Code) {
            out.execution_count = cell.execution_count;
            for (const auto& o : cell.outputs) out.outputs.push_back(ToNotebookOutput(o, cell.execution_count));
        }
        // nbformat 4.5 wants an id per cell.
        out.extra = cell.ipynb_extra.empty() ? "{\"id\":\"" + cell.id.substr(cell.id.rfind('-') + 1) + "\"}" : cell.ipynb_extra;
        notebook.cells.push_back(std::move(out));
    }
    return nb::SerializeIpynb(notebook);
}

// ========== Editor Theme ==========

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
            cell.editor.SetShowWhitespace(show);
        }
    }
}
void CellManager::ApplySyntaxHighlighting(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& cell : cells_) {
        if (cell.type == CellType::Code) {
            cell.editor.SetColorize(enabled);
        }
    }
}

} // namespace cyxwiz
