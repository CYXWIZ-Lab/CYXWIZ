#include "script_text_file.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace cyxwiz::scriptfile {

namespace {
constexpr char kBom[] = "\xEF\xBB\xBF";
}

Decoded Decode(const std::string& raw) {
    Decoded out;
    size_t start = 0;
    if (raw.compare(0, 3, kBom) == 0) {
        out.format.bom = true;
        start = 3;
    }
    size_t crlf = 0;
    size_t lf = 0;
    out.text.reserve(raw.size() - start);
    for (size_t i = start; i < raw.size(); ++i) {
        const char c = raw[i];
        if (c == '\r') {
            if (i + 1 < raw.size() && raw[i + 1] == '\n') {
                ++crlf;
                ++i;
            }
            out.text += '\n';
        } else {
            if (c == '\n') ++lf;
            out.text += c;
        }
    }
    out.format.eol = crlf > lf ? LineEnding::CRLF : LineEnding::LF;
    return out;
}

std::string Encode(const std::string& text, const TextFormat& format) {
    std::string out;
    out.reserve(text.size() + (format.bom ? 3 : 0) + (format.eol == LineEnding::CRLF ? text.size() / 16 : 0));
    if (format.bom) out += kBom;
    for (const char c : text) {
        if (c == '\r') continue;
        if (c == '\n' && format.eol == LineEnding::CRLF) out += '\r';
        out += c;
    }
    return out;
}

bool WriteAtomically(const std::string& path, const std::string& bytes, std::string* error) {
    namespace fs = std::filesystem;
    const fs::path target(path);
    fs::path temp = target;
    temp += ".cyxwiz-save.tmp";
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        if (!file) {
            if (error) *error = "cannot write in " + target.parent_path().string();
            return false;
        }
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.flush();
        if (!file) {
            file.close();
            std::error_code ignore;
            fs::remove(temp, ignore);
            if (error) *error = "the disk refused the write (full or removed?)";
            return false;
        }
    }
    std::error_code ec;
    fs::rename(temp, target, ec);  // replaces an existing file
    if (ec) {
        std::error_code ignore;
        fs::remove(temp, ignore);
        if (error) *error = ec.message();
        return false;
    }
    return true;
}

bool CanWriteTab(bool loading, bool load_failed, bool large_file_view) {
    return !loading && !load_failed && !large_file_view;
}

bool UsesTextBuffer(bool loading, bool load_failed, bool large_file_view, bool notebook_mode) {
    return CanWriteTab(loading, load_failed, large_file_view) && !notebook_mode;
}

std::string SaveAsPath(const std::string& chosen, const std::string& current_name) {
    namespace fs = std::filesystem;
    if (chosen.empty()) return chosen;
    if (fs::path(chosen).has_extension()) return chosen;
    const fs::path current(current_name);
    return chosen + (current.has_extension() ? current.extension().string() : std::string(".cyx"));
}

bool IsNotebookJson(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return ext == ".ipynb";
}

}  // namespace cyxwiz::scriptfile
