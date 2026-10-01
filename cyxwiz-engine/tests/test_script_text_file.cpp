// Script Editor file rules (TOFIX133 P0 items 1-3): failed loads are not
// writable, Save As keeps the typed extension, BOM and CRLF survive a
// round trip, and writes go through a temporary file.
#include "../src/core/script_text_file.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace cyxwiz::scriptfile;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string ReadAll(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
}  // namespace

int main() {
    // Item 1: a failed or cancelled load must not be saved over the file.
    Check(CanWriteTab(false, false, false), "loaded tab is writable");
    Check(!CanWriteTab(false, true, false), "failed load is not writable");
    Check(!CanWriteTab(true, false, false), "loading tab is not writable");
    Check(!CanWriteTab(false, false, true), "large-file view is read-only");

    // Item 14: text operations only on the visible text buffer.
    Check(UsesTextBuffer(false, false, false, false), "text mode uses the buffer");
    Check(!UsesTextBuffer(false, false, false, true), "notebook mode hides the buffer");
    Check(!UsesTextBuffer(false, false, true, false), "large-file view has no buffer");
    Check(!UsesTextBuffer(false, true, false, false), "failed load has no buffer");

    // Item 2: Save As keeps the extension the user typed.
    Check(SaveAsPath("C:/w/x.py", "Untitled1.cyx") == "C:/w/x.py", "x.py stays x.py");
    Check(SaveAsPath("C:/w/notes.cyx", "a.py") == "C:/w/notes.cyx", "typed .cyx kept");
    Check(SaveAsPath("C:/w/x", "train.py") == "C:/w/x.py", "no extension: current one");
    Check(SaveAsPath("C:/w/x", "eda.ipynb") == "C:/w/x.ipynb", "a notebook stays a notebook");
    Check(IsNotebookJson("C:/w/EDA.IPYNB") && !IsNotebookJson("C:/w/eda.cyx") && !IsNotebookJson(""), "ipynb by extension");
    Check(SaveAsPath("C:/w/x", "Untitled") == "C:/w/x.cyx", "no extension anywhere: .cyx");

    // Item 3: BOM and CRLF are kept.
    const std::string crlf_bom = "\xEF\xBB\xBFimport os\r\nprint(1)\r\n";
    Decoded d = Decode(crlf_bom);
    Check(d.text == "import os\nprint(1)\n", "decoded text has LF and no BOM");
    Check(d.format.bom && d.format.eol == LineEnding::CRLF, "format detected");
    Check(Encode(d.text, d.format) == crlf_bom, "CRLF + BOM round trip is byte exact");

    const std::string lf = "a\nb\n";
    d = Decode(lf);
    Check(!d.format.bom && d.format.eol == LineEnding::LF && Encode(d.text, d.format) == lf, "LF round trip");

    d = Decode("a\r\nb\r\nc\n");
    Check(d.format.eol == LineEnding::CRLF && d.text == "a\nb\nc\n", "mostly CRLF stays CRLF");
    d = Decode("old\rmac\r");
    Check(d.text == "old\nmac\n", "lone CR becomes a line break");

    // Atomic write: replaces the file, leaves no temporary behind.
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_test_script_text_file";
    std::filesystem::create_directories(dir);
    const auto file = dir / "s.py";
    { std::ofstream(file, std::ios::binary) << "old content that is longer"; }
    std::string error;
    Check(WriteAtomically(file.string(), crlf_bom, &error), "atomic write: " + error);
    Check(ReadAll(file) == crlf_bom, "atomic write replaced the content");
    for (const auto& e : std::filesystem::directory_iterator(dir))
        Check(e.path().filename() == "s.py", "no temporary file left: " + e.path().filename().string());
    Check(!WriteAtomically((dir / "missing_dir" / "x.py").string(), "x", &error) && !error.empty(),
          "write into a missing folder fails with a reason");
    Check(ReadAll(file) == crlf_bom, "failed write elsewhere left the file alone");
    std::filesystem::remove_all(dir);

    std::cout << "script text file: writable rule, Save As path, BOM/CRLF round trip, atomic write. OK\n";
    return 0;
}
