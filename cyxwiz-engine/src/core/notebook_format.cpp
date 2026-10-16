#include "notebook_format.h"

#include <nlohmann/json.hpp>

namespace cyxwiz::nb {

using nlohmann::json;
using ordered = nlohmann::ordered_json;

namespace {
// nbformat stores text as a list of lines (each ending in '\n' but the last)
// or as one string.
std::string JoinText(const ordered& value) {
    if (value.is_string()) return value.get<std::string>();
    std::string out;
    if (value.is_array())
        for (const auto& part : value)
            if (part.is_string()) out += part.get<std::string>();
    return out;
}

ordered SplitText(const std::string& text) {
    ordered lines = ordered::array();
    size_t start = 0;
    while (start < text.size()) {
        const size_t nl = text.find('\n', start);
        if (nl == std::string::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        lines.push_back(text.substr(start, nl - start + 1));
        start = nl + 1;
    }
    return lines;
}

Output ReadOutput(const ordered& o) {
    Output out;
    out.raw = o.dump();
    const std::string type = o.value("output_type", "");
    if (type == "stream") {
        out.kind = Output::Kind::Stream;
        out.stream_name = o.value("name", "stdout");
        out.text = JoinText(o.value("text", ordered()));
    } else if (type == "error") {
        out.kind = Output::Kind::Error;
        out.ename = o.value("ename", "");
        out.evalue = o.value("evalue", "");
        if (o.contains("traceback") && o["traceback"].is_array())
            for (const auto& line : o["traceback"])
                if (line.is_string()) out.traceback.push_back(line.get<std::string>());
    } else {
        out.kind = type == "execute_result" ? Output::Kind::Result : Output::Kind::Display;
        if (o.contains("execution_count") && o["execution_count"].is_number_integer()) out.execution_count = o["execution_count"].get<int>();
        if (o.contains("data") && o["data"].is_object()) {
            const auto& data = o["data"];
            if (data.contains("text/plain")) out.text = JoinText(data["text/plain"]);
            if (data.contains("image/png")) out.png_base64 = JoinText(data["image/png"]);
            if (data.contains("text/html")) out.html = JoinText(data["text/html"]);
        }
    }
    return out;
}

ordered WriteOutput(const Output& out) {
    if (!out.raw.empty()) return ordered::parse(out.raw);  // as read: keeps every MIME type
    ordered o;
    switch (out.kind) {
        case Output::Kind::Stream:
            o["name"] = out.stream_name.empty() ? "stdout" : out.stream_name;
            o["output_type"] = "stream";
            o["text"] = SplitText(out.text);
            break;
        case Output::Kind::Error: {
            o["ename"] = out.ename;
            o["evalue"] = out.evalue;
            o["output_type"] = "error";
            o["traceback"] = out.traceback;
            break;
        }
        case Output::Kind::Result:
        case Output::Kind::Display: {
            ordered data = ordered::object();
            if (!out.png_base64.empty()) data["image/png"] = out.png_base64;
            if (!out.html.empty()) data["text/html"] = SplitText(out.html);
            data["text/plain"] = SplitText(out.text);
            o["data"] = data;
            if (out.kind == Output::Kind::Result) o["execution_count"] = out.execution_count > 0 ? ordered(out.execution_count) : ordered();
            o["metadata"] = ordered::object();
            o["output_type"] = out.kind == Output::Kind::Result ? "execute_result" : "display_data";
            break;
        }
    }
    return o;
}
}  // namespace

bool ParseIpynb(const std::string& text, Notebook& out, std::string* error) {
    ordered doc;
    try {
        doc = ordered::parse(text);
    } catch (const std::exception& e) {
        if (error) *error = std::string("Not valid JSON: ") + e.what();
        return false;
    }
    if (!doc.is_object() || !doc.contains("cells") || !doc["cells"].is_array()) {
        if (error) *error = "Not a Jupyter notebook (no cell list)";
        return false;
    }
    out = Notebook{};
    out.nbformat = doc.value("nbformat", 4);
    out.nbformat_minor = doc.value("nbformat_minor", 5);
    if (out.nbformat < 4) {
        if (error) *error = "Notebook format " + std::to_string(out.nbformat) + " is older than 4; save it with Jupyter first";
        return false;
    }
    out.metadata = doc.contains("metadata") ? doc["metadata"].dump() : std::string("{}");
    for (const auto& c : doc["cells"]) {
        Cell cell;
        const std::string type = c.value("cell_type", "code");
        cell.kind = type == "markdown" ? CellKind::Markdown : (type == "raw" ? CellKind::Raw : CellKind::Code);
        cell.source = JoinText(c.value("source", ordered()));
        if (c.contains("execution_count") && c["execution_count"].is_number_integer()) cell.execution_count = c["execution_count"].get<int>();
        if (c.contains("outputs") && c["outputs"].is_array())
            for (const auto& o : c["outputs"]) cell.outputs.push_back(ReadOutput(o));
        ordered extra = ordered::object();
        for (auto it = c.begin(); it != c.end(); ++it) {
            const std::string& key = it.key();
            if (key != "cell_type" && key != "source" && key != "outputs" && key != "execution_count") extra[key] = it.value();
        }
        cell.extra = extra.dump();
        out.cells.push_back(std::move(cell));
    }
    return true;
}

std::string SerializeIpynb(const Notebook& notebook) {
    ordered doc;
    ordered cells = ordered::array();
    for (const auto& cell : notebook.cells) {
        ordered c;
        c["cell_type"] = cell.kind == CellKind::Markdown ? "markdown" : (cell.kind == CellKind::Raw ? "raw" : "code");
        ordered extra = cell.extra.empty() ? ordered::object() : ordered::parse(cell.extra);
        if (cell.kind == CellKind::Code) c["execution_count"] = cell.execution_count > 0 ? ordered(cell.execution_count) : ordered();
        if (extra.contains("id")) c["id"] = extra["id"];
        c["metadata"] = extra.contains("metadata") ? extra["metadata"] : ordered::object();
        if (cell.kind == CellKind::Code) {
            ordered outs = ordered::array();
            for (const auto& o : cell.outputs) outs.push_back(WriteOutput(o));
            c["outputs"] = outs;
        }
        c["source"] = SplitText(cell.source);
        for (auto it = extra.begin(); it != extra.end(); ++it)
            if (!c.contains(it.key())) c[it.key()] = it.value();
        cells.push_back(c);
    }
    doc["cells"] = cells;
    doc["metadata"] = notebook.metadata.empty() ? ordered::object() : ordered::parse(notebook.metadata);
    doc["nbformat"] = notebook.nbformat;
    doc["nbformat_minor"] = notebook.nbformat_minor;
    return doc.dump(1) + "\n";  // Jupyter writes one-space indentation and a final newline
}

namespace {
bool IsDottedIdentifier(const std::string& s) {
    if (s.empty()) return false;
    bool start = true;
    for (const char ch : s) {
        const bool alpha = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
        const bool digit = ch >= '0' && ch <= '9';
        if (ch == '.') {
            if (start) return false;
            start = true;
        } else if (alpha || (digit && !start)) {
            start = false;
        } else {
            return false;
        }
    }
    return !start;
}
}  // namespace

Output ErrorFromText(const std::string& text) {
    Output o;
    o.kind = Output::Kind::Error;
    std::string fallback;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        o.traceback.push_back(line);
        if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
            const size_t colon = line.find(": ");
            if (colon != std::string::npos && IsDottedIdentifier(line.substr(0, colon))) {
                o.ename = line.substr(0, colon);
                o.evalue = line.substr(colon + 2);
            } else if (line.find_first_not_of(" \t") != std::string::npos && fallback.empty()) {
                fallback = line;
            }
        }
        start = end + 1;
    }
    while (!o.traceback.empty() && o.traceback.back().empty()) o.traceback.pop_back();
    if (o.ename.empty()) {
        o.ename = "Error";
        o.evalue = fallback;
    }
    return o;
}

std::string EncodeBase64(const std::vector<unsigned char>& bytes) {
    static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const unsigned v = (unsigned(bytes[i]) << 16) | (i + 1 < bytes.size() ? unsigned(bytes[i + 1]) << 8 : 0u) |
                           (i + 2 < bytes.size() ? unsigned(bytes[i + 2]) : 0u);
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < bytes.size() ? table[(v >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? table[v & 63] : '=';
    }
    return out;
}

std::vector<unsigned char> DecodeBase64(const std::string& text) {
    std::vector<unsigned char> out;
    unsigned value = 0;
    int bits = 0;
    for (const char ch : text) {
        int digit;
        if (ch >= 'A' && ch <= 'Z') digit = ch - 'A';
        else if (ch >= 'a' && ch <= 'z') digit = ch - 'a' + 26;
        else if (ch >= '0' && ch <= '9') digit = ch - '0' + 52;
        else if (ch == '+') digit = 62;
        else if (ch == '/') digit = 63;
        else if (ch == '=') break;
        else if (ch == '\n' || ch == '\r' || ch == ' ') continue;
        else return {};
        value = (value << 6) | unsigned(digit);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((value >> bits) & 0xFF));
        }
    }
    return out;
}

std::string StripAnsi(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\x1b' && i + 1 < text.size() && text[i + 1] == '[') {
            i += 2;
            while (i < text.size() && !(text[i] >= '@' && text[i] <= '~')) ++i;
            continue;
        }
        out += text[i];
    }
    return out;
}

std::string DefaultMetadata(const std::string& python_version) {
    ordered m;
    m["kernelspec"] = {{"display_name", "Python 3"}, {"language", "python"}, {"name", "python3"}};
    m["language_info"] = {{"name", "python"}, {"version", python_version.empty() ? std::string("3") : python_version}};
    return m.dump();
}

}  // namespace cyxwiz::nb
