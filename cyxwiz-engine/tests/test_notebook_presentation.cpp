// Notebook wording (TOFIX133 P4 step 4.3, boards 4-5): gutter, kernel chip,
// run status, not-run note, traceback frame links.
#include "../src/core/notebook_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::nbview;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    Check(FormatDuration(0.0213) == "0.02 s", "short run");
    Check(FormatDuration(2.04) == "2.04 s", "seconds");
    Check(FormatDuration(12.34) == "12.3 s", "tens of seconds");
    Check(FormatDuration(64.9) == "1 min 4 s", "minutes");

    // Board 4/5 gutters.
    Gutter g = GutterFor(CellRun::Success, 1, 0.02, 0);
    Check(g.label == "[1]" && g.detail == "\xE2\x9C\x93 0.02 s" && g.tone == Tone::Success, "done cell");
    g = GutterFor(CellRun::Error, 5, 2.04, 0);
    Check(g.label == "[5]" && g.detail == "\xE2\x9C\x95 2.04 s" && g.tone == Tone::Error, "failed cell");
    g = GutterFor(CellRun::Running, 3, -1, 3.21);
    Check(g.label == "[*]" && g.detail == "3.2 s" && g.tone == Tone::Running, "running cell");
    g = GutterFor(CellRun::Queued, 0, -1, 0);
    Check(g.label == "[ ]" && g.detail == "Queued", "queued cell");
    g = GutterFor(CellRun::NotRun, 4, -1, 0);
    Check(g.label == "[ ]" && g.detail.empty(), "not run keeps no stale count");
    g = GutterFor(CellRun::Success, 2, -1, 0);
    Check(g.label == "[2]" && g.detail.empty(), "loaded from a file: count, no time");

    KernelFacts k;
    k.python_version = "3.12.8";
    k.environment = "sentiment";
    k.started = true;
    k.busy = true;
    KernelChip chip = KernelChipFor(k);
    Check(chip.text == "Python 3.12.8 \xC2\xB7 sentiment \xC2\xB7 Busy" && chip.tone == Tone::Running, "busy chip");
    k.busy = false;
    Check(KernelChipFor(k).text == "Python 3.12.8 \xC2\xB7 sentiment \xC2\xB7 Idle", "idle chip");
    k.other_busy = true;
    Check(KernelChipFor(k).text.find("Waiting") != std::string::npos, "another script holds Python");
    KernelFacts cold;
    cold.environment = "sentiment";
    Check(KernelChipFor(cold).text == "Python \xC2\xB7 sentiment \xC2\xB7 Starts on first run", "before Python starts");

    RunFacts r;
    r.running = true;
    r.run_position = 3;
    r.run_total = 4;
    Check(RunStatusFor(r).text == "Running cell 3 of 4", "batch progress");
    r.run_total = 1;
    Check(RunStatusFor(r).text == "Running", "one cell");
    RunFacts stopped;
    stopped.stopped_at_count = 5;
    Check(RunStatusFor(stopped).text == "Stopped after an error in [5]" && RunStatusFor(stopped).tone == Tone::Error,
          "stopped by an error");
    stopped.stopped_by_interrupt = true;
    Check(RunStatusFor(stopped).text == "Interrupted in [5]", "interrupted");
    Check(RunStatusFor(RunFacts{}).text == "Ready", "ready");

    Check(NotRunMessage(5, false) == "Not run: Run All stopped after the error in [5].", "not-run note");
    Check(NotRunMessage(5, true) == "Not run: Run All was interrupted in [5].", "not-run after interrupt");

    FrameLink f = FrameLinkFor("Cell In[5]", 1);
    Check(f.label == "Cell [5], line 1" && f.cell_count == 5 && f.path.empty(), "cell frame");
    f = FrameLinkFor("D:\\tmp\\gui129\\sentiment_analysis_inference.py", 236);
    Check(f.label == "sentiment_analysis_inference.py:236" && !f.path.empty(), "file frame");
    f = FrameLinkFor("<string>", 2);
    Check(f.label == "<string>, line 2" && f.path.empty(), "no file to open");

    const std::vector<std::string> files = {
        "Cell In[4]", "D:\\tmp\\gui129\\jvenv\\Lib\\site-packages\\pandas\\core\\frame.py",
        "D:\\tmp\\gui129\\jvenv\\Lib\\site-packages\\pandas\\core\\indexes\\base.py"};
    const auto user = UserFrames(files);
    Check(user.size() == 1 && user[0] == 0, "library frames behind Show full traceback");
    const auto only_lib = UserFrames({"C:\\Python312\\Lib\\json\\decoder.py"});
    Check(only_lib.size() == 1 && only_lib[0] == 0, "keeps the innermost frame when none is the user's");
    Check(UserFrames({"D:\\work\\train.py", "Cell In[2]"}).size() == 2, "user files and cells kept");
    Check(UserFrames({"D:\\work\\lib\\model.py", "/usr/lib/python3.12/json/decoder.py"}).size() == 1,
          "a project's lib folder is the user's; /usr/lib/python3.12 is not");

    std::cout << "notebook presentation: gutters, kernel chip, run status, frames. OK\n";
    return 0;
}
