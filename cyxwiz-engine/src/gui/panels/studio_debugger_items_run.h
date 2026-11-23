#pragma once

// A Studio Debugger view's computation over items from the graph's data (Top
// images and Embedding, TOFIX178 6.3e / 6.4b): it waits for a reopened
// project's data to load and for the canvas to compile again, then runs on a
// worker; while it runs, a card shows its steps (approved board
// TopImagesStates). One implementation for every such view.

#include "studio_debugger_panel.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct ItemsRun {
    bool waiting_for_load = false;     // the Data Inputs load first
    bool waiting_for_compile = false;  // loaded; the canvas compiles again
    bool start_after_load = false;     // the loads ended: start on the next frame
    bool computing = false;            // on a worker
    std::string loading;               // the data being (or that was) loaded
    std::string error;
    std::chrono::steady_clock::time_point started;
    // The view's step running now: 1 its first after the load, 2 the next, ...
    std::shared_ptr<std::atomic<int>> stage = std::make_shared<std::atomic<int>>(1);
    std::shared_ptr<std::atomic<size_t>> done = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::atomic<size_t>> total = std::make_shared<std::atomic<size_t>>(0);
    std::shared_ptr<std::atomic<bool>> cancel = std::make_shared<std::atomic<bool>>(false);

    bool Busy() const { return computing || waiting_for_load; }
};

// Asks for the items job. When the data must load or the canvas compiles
// again, the run waits (Busy) and is started again later (see Retry);
// otherwise an empty job sets `error`. A returned job leaves the run ready to
// compute: `computing` set, the counters reset.
StudioDebuggerPanel::DataItemsJob BeginItems(ItemsRun& run, const StudioDebuggerPanel::DataItemsCallback& callback,
                                             const std::string& what);
// Each frame: true when the view should call its start again (the loads have
// ended, or the canvas may have compiled).
bool RetryItems(ItemsRun& run);
// The loads a waiting run asked for have ended (`failed`: what could not load).
void ItemsLoadsEnded(ItemsRun& run, const std::string& failed);
// Stop: a waiting run stops waiting; a computing one stops after its batch.
void StopItems(ItemsRun& run);

// The card while it runs: "Load the data" first (filled in here), then the
// view's steps; `bar_step` (1-based, as `stage`) shows the done / total bar.
struct ItemsStep {
    std::string title;
    std::string detail;  // shown while the step runs
};
void DrawItemsRunCard(const ItemsRun& run, const std::string& title, const std::vector<ItemsStep>& steps,
                      int bar_step);

}  // namespace cyxwiz
