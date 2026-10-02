#pragma once

#include "cell.h"
#include "debug_types.h"
#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <chrono>
#include <stdexcept>

namespace scripting {
    class ScriptingEngine;
}

namespace cyxwiz {

/**
 * Manages cells for notebook-style script editing
 * Handles cell CRUD operations, execution, and serialization
 */
class CellManager {
public:
    CellManager();
    ~CellManager();

    // ========== Cell Access ==========

    /**
     * Get all cells
     */
    std::vector<Cell>& GetCells() { return cells_; }
    const std::vector<Cell>& GetCells() const { return cells_; }

    /**
     * Get cell count
     */
    int GetCellCount() const {
        if (cells_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("Cell count exceeds the editor index range");
        }
        return static_cast<int>(cells_.size());
    }

    /**
     * Get cell at index
     */
    Cell& GetCell(int index);
    const Cell& GetCell(int index) const;

    /**
     * Get cell by ID
     */
    Cell* GetCellById(const std::string& id);

    /**
     * Check if index is valid
     */
    bool IsValidIndex(int index) const {
        return index >= 0 && index < static_cast<int>(cells_.size());
    }

    // ========== Cell Operations ==========

    /**
     * Add a new cell
     * @param type Cell type (Code, Markdown, Raw)
     * @param position Insert position (-1 = end)
     * @return Index of new cell
     */
    int AddCell(CellType type, int position = -1);

    /**
     * Delete a cell
     * @param index Cell index to delete
     * @return True if deleted
     */
    bool DeleteCell(int index);

    /**
     * Move a cell
     * @param from Source index
     * @param to Destination index
     * @return True if moved
     */
    bool MoveCell(int from, int to);

    /**
     * Duplicate a cell
     * @param index Cell to duplicate
     * @return Index of new cell (-1 on failure)
     */
    int DuplicateCell(int index);

    /**
     * Merge two adjacent cells
     * @param first First cell index
     * @param second Second cell index (must be first + 1)
     * @return True if merged
     */
    bool MergeCells(int first, int second);

    /**
     * Split a cell at a line
     * @param index Cell to split
     * @param line Line number to split at (0-based)
     * @return Index of new cell (-1 on failure)
     */
    int SplitCell(int index, int line);

    /**
     * Change cell type
     * @param index Cell index
     * @param type New cell type
     * @return True if changed
     */
    bool ChangeCellType(int index, CellType type);

    /**
     * Clear all cells
     */
    void Clear();

    // ========== Output Management ==========

    /**
     * Clear outputs for a specific cell
     */
    void ClearCellOutput(int index);

    /**
     * Clear all cell outputs
     */
    void ClearAllOutputs();

    /**
     * Add output to a cell
     */
    void AddCellOutput(int index, const CellOutput& output);

    // ========== Execution ==========

    /**
     * Set scripting engine for execution
     */
    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);

    /**
     * Run a specific cell
     * @param index Cell index to run
     */
    void RunCell(int index);

    /**
     * Run one cell under the debugger (TOFIX133 P6) with these breakpoints
     * (its lines, 1-based).
     */
    void DebugCell(int index, std::vector<scripting::DebugBreakpoint> breakpoints, bool stop_on_error);

    /**
     * Run all cells in order
     */
    void RunAllCells();

    /**
     * Run all cells above (and including) index
     */
    void RunCellsAbove(int index);

    /**
     * Run all cells below (and including) index
     */
    void RunCellsBelow(int index);

    /**
     * Interrupt current execution
     */
    void InterruptExecution();

    /**
     * Restart (TOFIX133 P4, D4): interrupts, then forgets this notebook's
     * namespace and restarts the [n] count. Outputs stay. Cells run after
     * it wait until the restart is done.
     */
    void Restart();
    bool IsRestarting() const { return restart_pending_; }

    /**
     * The namespace this notebook's cells run in. A closed notebook calls
     * ReleaseNamespace so its variables do not stay in memory.
     */
    const std::string& NamespaceKey() const { return namespace_key_; }

    // What the notebook is doing, for the status bar and gutters (TOFIX133
    // P4 step 4.3): the cell being run in this batch (1-based) of how many,
    // the [n] of the cell that stopped the last batch (0 = none), and how
    // long the running cell has run.
    int BatchPosition() const { return batch_position_; }
    int BatchTotal() const { return batch_total_; }
    int StoppedAtCount() const { return stopped_at_count_; }
    bool StoppedByInterrupt() const { return stopped_by_interrupt_; }
    double RunningSeconds() const;
    // True once after a run changed outputs (an .ipynb stores them, so its
    // tab becomes modified).
    // Changes when a run ends or the notebook restarts (its variables changed).
    std::uint64_t RunGeneration() const { return run_generation_; }
    bool TakeOutputsChanged() {
        const bool changed = outputs_changed_;
        outputs_changed_ = false;
        return changed;
    }
    void ReleaseNamespace();

    /**
     * Check if any cell is currently running
     */
    bool IsRunning() const { return is_running_; }

    /**
     * Get current execution count
     */
    int GetExecutionCount() const { return execution_counter_; }

    /**
     * Get index of currently running cell (-1 if none)
     */
    int GetRunningCellIndex() const;

    /**
     * Every frame, on the UI thread: applies output from the worker thread
     * and starts the next queued cell (TOFIX133 P0 items 8-10).
     */
    void Pump();

    // ========== Serialization ==========

    /**
     * Parse cells from .cyx file content
     * @param content File content
     * @return True on success
     */
    bool ParseFromCyx(const std::string& content);

    /**
     * Serialize cells to .cyx file format
     * @return File content
     */
    std::string SerializeToCyx() const;

    /**
     * Jupyter .ipynb (TOFIX133 P4, D3). Outputs, ids and metadata are kept;
     * what the Engine does not draw is written back unchanged.
     */
    bool ParseFromIpynb(const std::string& content, std::string* error = nullptr);
    std::string SerializeToIpynb() const;

    /**
     * Check if content has cell markers
     */
    static bool HasCellMarkers(const std::string& content);

    // ========== Editor Theme ==========


    /**
     * Apply tab size to all code cell editors
     */
    void ApplyTabSize(int size);

    /**
     * Apply whitespace visibility to all code cell editors
     */
    void ApplyShowWhitespace(bool show);

    /**
     * Apply syntax highlighting to all code cell editors
     */
    void ApplySyntaxHighlighting(bool enabled);

private:
    // Execution helpers
    void ExecuteCellInternal(int index);
    void Enqueue(int from, int to);  // code cells with text, by id
    bool StartNext();
    int IndexOfId(const std::string& id) const;

    // What the worker thread posts; only Pump() reads it.
    struct RunEvent {
        enum class Kind { Stdout, Stderr, Add, Done } kind;
        std::uint64_t run = 0;
        std::string text;
        CellOutput output;
        bool success = false;
        bool cancelled = false;
        std::string error;
    };
    struct Mailbox {
        std::mutex mutex;
        std::vector<RunEvent> events;
    };

    // Data
    std::vector<Cell> cells_;
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;

    // Execution state
    bool is_running_ = false;
    int execution_counter_ = 0;
    std::string ipynb_metadata_;         // notebook metadata of an opened .ipynb
    int ipynb_minor_ = 5;
    std::string running_cell_id_;
    std::vector<std::string> execution_queue_;  // cell ids
    // The cell to run under the debugger, and its breakpoints (P6).
    std::string debug_cell_id_;
    std::vector<scripting::DebugBreakpoint> debug_breakpoints_;
    bool debug_stop_on_error_ = true;
    std::uint64_t run_counter_ = 0;
    std::string namespace_key_;
    bool restart_pending_ = false;
    int batch_position_ = 0;
    int batch_total_ = 0;
    int stopped_at_count_ = 0;
    bool stopped_by_interrupt_ = false;
    bool outputs_changed_ = false;
    std::uint64_t run_generation_ = 1;
    std::chrono::steady_clock::time_point run_started_{};
    bool TryRestart();
    void AppendStream(Cell& cell, const std::string& name, const std::string& text);
    std::shared_ptr<Mailbox> mailbox_ = std::make_shared<Mailbox>();

    // Thread safety
    mutable std::mutex mutex_;
};

} // namespace cyxwiz
