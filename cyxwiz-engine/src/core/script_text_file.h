// Script Editor file rules (TOFIX133 P0 items 1-3): when a tab may be
// written, which path Save As writes, and how text is read and written
// without changing its byte-order mark or line endings. No ImGui.
#pragma once

#include <string>

namespace cyxwiz::scriptfile {

enum class LineEnding { LF, CRLF };

struct TextFormat {
    bool bom = false;              // UTF-8 byte-order mark at the start
    LineEnding eol = LineEnding::LF;
};

struct Decoded {
    std::string text;              // '\n' line endings, no BOM: what the editor holds
    TextFormat format;             // how the file was written, restored on save
};

// Strips a UTF-8 BOM and turns CRLF (and lone CR) into LF. The file keeps
// CRLF when most of its line breaks were CRLF.
Decoded Decode(const std::string& raw);
// The editor's text back in the file's own format.
std::string Encode(const std::string& text, const TextFormat& format);

// Writes through a temporary file next to `path`, then replaces `path`, so a
// failed write never leaves a truncated file. `error` says why on failure.
bool WriteAtomically(const std::string& path, const std::string& bytes, std::string* error = nullptr);

// A tab may be written to its file only when its content was read in full.
// A failed or cancelled load leaves an empty editor bound to the file.
bool CanWriteTab(bool loading, bool load_failed, bool large_file_view);

// Save As keeps the extension the user typed (x.py stays x.py). Without one,
// the tab's current extension is used, else ".cyx".
std::string SaveAsPath(const std::string& chosen, const std::string& current_name);

}  // namespace cyxwiz::scriptfile
