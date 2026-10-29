#pragma once

// What Data Studio's Explore tabs Missing, Outliers, Correlation and
// Distribution share (TOFIX129 D1): the dataset Explore shows with its
// profile (Profile's, computed once for all of them), and a background job
// whose answer is kept only for the question it was asked (a new pick
// cancels the old job, so an answer never shows under the wrong column).

#include "../../core/async_task_manager.h"
#include "../../core/dataset_profiler.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct ExploreContext {
    std::string dataset;                              // catalog name (queries use it)
    std::string shown;                                // as the graph calls it
    std::shared_ptr<const DatasetProfile> profile;    // null while profiling or failed
    std::map<std::string, std::vector<std::string>> missing_texts;  // column -> texts marked as missing
    bool profiling = false;
    std::string error;                                // why there is no profile
};

// Draws why there is nothing to show yet (no dataset, profiling, an error);
// true when the profile is there to show.
bool ExploreProfileReady(const ExploreContext& context, const char* empty_hint);

// Label / value pairs, two to a line when the width allows, with no lines
// around them (the old KeyValueTable draws an outline and cut the labels).
void ExploreFigures(const char* id, const std::vector<std::pair<std::string, std::string>>& rows);

// Choices side by side (a segmented control) when they fit on the line, else
// a list; true when the choice changed.
bool ExploreChoices(const char* id, const char* const* labels, int count, int* selected);

// SameLine before a button of `label`, or a new line when it would not fit.
void ExploreNextButton(const char* label);

// One background answer of type T for a key (dataset + choices).
template <typename T>
class ExploreJob {
public:
    ~ExploreJob() { Cancel(); }

    // Starts `work` for `key` unless that answer is held or coming; a running
    // job for another key is cancelled.
    void Ask(const std::string& label, const std::string& key, std::function<T(LambdaTask&)> work) {
        if (key == key_ && (task_id_ != 0 || answer_ || !error_.empty())) return;  // Reset() asks again
        Cancel();
        key_ = key;
        answer_.reset();
        error_.clear();
        auto result = std::make_shared<T>();
        std::weak_ptr<int> alive = alive_;
        const uint64_t ticket = ++ticket_;
        task_id_ = AsyncTaskManager::Instance().RunAsync(
            label,
            [work = std::move(work), result](LambdaTask& task) {
                *result = work(task);
                if (result->ok()) return;
                if (result->error == "Cancelled.") task.MarkCancelled();  // asked to stop: not a failure
                else task.MarkFailed(result->error);
            },
            nullptr,
            [this, alive, result, ticket](bool, const std::string&) {
                if (alive.expired() || ticket != ticket_) return;  // cancelled or asked again since
                task_id_ = 0;
                if (result->ok()) answer_ = result;
                else error_ = result->error == "Cancelled." ? std::string() : result->error;
            },
            alive_);
    }
    // Forgets the answer, so the next Ask runs again (the data changed).
    void Reset() {
        Cancel();
        key_.clear();
        answer_.reset();
        error_.clear();
    }
    void Cancel() {
        if (task_id_) AsyncTaskManager::Instance().Cancel(task_id_);
        task_id_ = 0;
        ++ticket_;
    }
    bool Running() const { return task_id_ != 0; }
    // The answer for `key`, or null.
    const T* Answer(const std::string& key) const { return key == key_ ? answer_.get() : nullptr; }
    const std::string& Error() const { return error_; }

private:
    std::string key_;
    uint64_t task_id_ = 0;
    uint64_t ticket_ = 0;  // the job whose answer is wanted
    std::shared_ptr<T> answer_;
    std::string error_;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
