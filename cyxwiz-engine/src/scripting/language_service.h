#pragma once

// Script Editor language intelligence (TOFIX133 P3, decision D2): requests
// to the bundled Jedi/pyflakes service (python_tools/cyxwiz_intel.py) run on
// one worker thread in the Engine's own Python, so typing never waits for
// them. The UI submits requests and polls results each frame; a newer
// request of the same kind replaces one still waiting, and a result for an
// older request id is dropped by the caller.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace scripting {

class ScriptingEngine;

class LanguageService {
public:
    enum class Kind { Complete, Describe, Hover, Signatures, Definition, Diagnostics };

    struct Request {
        Kind kind = Kind::Complete;
        std::uint64_t id = 0;      // set by Submit
        std::string source;
        int line = 1;              // 1-based
        int column = 0;            // 0-based, in characters
        std::string path;          // the file (empty: untitled)
        std::string project_root;
        std::string namespace_key; // a notebook's live namespace (its cells)
        std::string name;          // Describe: the completion's name
        std::vector<std::string> known_names;  // Diagnostics: names defined elsewhere (earlier cells)
    };

    struct Result {
        Kind kind = Kind::Complete;
        std::uint64_t id = 0;
        std::string json;          // what cyxwiz_intel returned ("" when it could not run)
    };

    explicit LanguageService(ScriptingEngine* engine);
    ~LanguageService();
    LanguageService(const LanguageService&) = delete;
    LanguageService& operator=(const LanguageService&) = delete;

    // Queues a request and returns its id. Python must be running (the
    // caller starts it on the UI thread); otherwise the result comes back empty.
    std::uint64_t Submit(Request request);
    // Results finished since the last call (UI thread).
    std::vector<Result> Poll();
    // Stops the worker; pending requests are dropped. Called before Python ends.
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

const char* LanguageFunctionName(LanguageService::Kind kind);

}  // namespace scripting
