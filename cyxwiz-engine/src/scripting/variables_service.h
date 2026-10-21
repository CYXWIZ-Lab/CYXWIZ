#pragma once

// Variable Explorer and Data Viewer reads (TOFIX133 P5, boards 9-10): calls
// into python_tools/cyxwiz_vars.py run on one worker thread, so a large
// value never stalls the UI. The UI submits requests and polls results each
// frame. A read is refused (Result::busy) while a script, cell or Console
// command runs; the caller asks again when the run ends. A newer List for
// the same scope replaces one still waiting.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz {
class DataTable;
}

namespace scripting {

class ScriptingEngine;

class VariablesService {
public:
    // Evaluate and Console run in a paused debug frame (TOFIX133 P6).
    enum class Kind { List, Children, Value, Delete, SaveCsv, Table, Evaluate, Console };

    struct Request {
        Kind kind = Kind::List;
        std::uint64_t id = 0;     // set by Submit
        std::string scope;        // "" = the Python session, else a notebook key
        std::string path_json;    // [["name","df"], ...] (not for List)
        bool show_all = false;    // List: modules, functions and classes too
        std::string file;         // SaveCsv
        long long max_rows = 200000;  // Table; <= 0 for all rows
        std::vector<int> index;   // Table: leading-axis indices of an array
        std::string expression;   // Evaluate / Console
        int frame = 0;            // Evaluate / Console: the paused frame (0 innermost)
    };

    struct Result {
        Kind kind = Kind::List;
        std::uint64_t id = 0;
        std::string scope;
        bool busy = false;        // a run was active: nothing was read
        std::string json;         // List/Children/Value/Delete/SaveCsv ("" when Python is not running)
        // Table
        std::shared_ptr<cyxwiz::DataTable> table;
        std::string table_kind;   // frame, array, list
        std::vector<long long> shape;
        long long rows = 0;       // in the value
        long long shown = 0;      // read into the table
        std::vector<std::string> dtypes;  // per table column (index first for frames)
        std::vector<int> slice;
        std::string error;
    };

    explicit VariablesService(ScriptingEngine* engine);
    ~VariablesService();
    VariablesService(const VariablesService&) = delete;
    VariablesService& operator=(const VariablesService&) = delete;

    std::uint64_t Submit(Request request);
    std::vector<Result> Poll();
    void Stop();

private:
    void Run();

    ScriptingEngine* engine_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Request> queue_;
    std::vector<Result> done_;
    std::uint64_t next_id_ = 1;
    bool stop_ = false;
    std::thread worker_;
};

const char* VariablesFunctionName(VariablesService::Kind kind);

}  // namespace scripting
