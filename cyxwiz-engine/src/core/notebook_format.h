// Jupyter .ipynb read and write for the Script Editor's notebooks (TOFIX133
// P4, decision D3: .ipynb first-class next to .cyx). nbformat 4. Lossless
// round trip: cell and notebook metadata, ids, attachments and outputs this
// Engine does not draw are kept as JSON and written back unchanged. No ImGui.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::nb {

enum class CellKind { Code, Markdown, Raw };

struct Output {
    enum class Kind { Stream, Result, Display, Error };
    Kind kind = Kind::Stream;
    std::string stream_name;              // stdout / stderr (Stream)
    std::string text;                     // stream text, or text/plain (Result, Display)
    std::string png_base64;               // image/png (Result, Display)
    std::string html;                     // text/html (Result, Display)
    std::string ename;                    // Error
    std::string evalue;
    std::vector<std::string> traceback;   // Error lines (may hold ANSI colours)
    int execution_count = 0;              // Result
    std::string raw;                      // the output's JSON as read; written back when set
};

struct Cell {
    CellKind kind = CellKind::Code;
    std::string source;
    int execution_count = 0;              // 0 = not run (null in the file)
    std::vector<Output> outputs;
    std::string extra;                    // other keys of the cell (id, metadata, attachments) as a JSON object
};

struct Notebook {
    std::vector<Cell> cells;
    std::string metadata;                 // notebook metadata as a JSON object
    int nbformat = 4;
    int nbformat_minor = 5;
};

bool ParseIpynb(const std::string& json, Notebook& out, std::string* error = nullptr);
std::string SerializeIpynb(const Notebook& notebook);

// An error output from a traceback as text. The exception is the last
// unindented "Name: value" line whose Name is a (dotted) identifier: the
// last line of a Python traceback, or the first line of the Engine's
// "Name: value / At: / frames" form.
Output ErrorFromText(const std::string& text);

// Base64 for image/png outputs. Decoding skips the line breaks Jupyter may
// put in long values; an invalid character gives an empty result.
std::string EncodeBase64(const std::vector<unsigned char>& bytes);
std::vector<unsigned char> DecodeBase64(const std::string& text);

// Traceback lines from IPython carry ANSI colour codes; drawn text drops them.
std::string StripAnsi(const std::string& text);

// A new notebook's metadata (Python 3 kernel, this Engine's Python version).
std::string DefaultMetadata(const std::string& python_version);

}  // namespace cyxwiz::nb
