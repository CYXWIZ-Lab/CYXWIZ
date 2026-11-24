#include "recycle_bin.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <cstdlib>
#include <ctime>
#include <fstream>
#endif

#include <system_error>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

bool Fail(std::string* error, std::string reason) {
    if (error) *error = std::move(reason);
    return false;
}

#ifndef _WIN32
// A name not yet used in `dir`: "name", "name 2", "name 3", ...
fs::path FreeName(const fs::path& dir, const fs::path& name) {
    fs::path candidate = dir / name;
    for (int n = 2; fs::exists(candidate); ++n)
        candidate = dir / (name.stem().string() + " " + std::to_string(n) + name.extension().string());
    return candidate;
}
#endif

}  // namespace

const char* RecycleBinName() {
#ifdef _WIN32
    return "Recycle Bin";
#else
    return "Trash";
#endif
}

bool MoveToRecycleBin(const fs::path& path, std::string* error) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return Fail(error, "the file is no longer there");
    const fs::path absolute = fs::absolute(path, ec);
    if (ec) return Fail(error, ec.message());
#ifdef _WIN32
    // SHFileOperation wants a list ending in two zeros.
    std::wstring from = absolute.wstring();
    from.push_back(L'\0');
    SHFILEOPSTRUCTW op{};
    op.wFunc = FO_DELETE;
    op.pFrom = from.c_str();
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT;
    if (SHFileOperationW(&op) != 0 || op.fAnyOperationsAborted)
        return Fail(error, "Windows could not move it to the Recycle Bin");
    return true;
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    if (!home || !*home) return Fail(error, "the home folder is not known");
    const fs::path trash = fs::path(home) / ".Trash";
    fs::create_directories(trash, ec);
    fs::rename(absolute, FreeName(trash, absolute.filename()), ec);
    if (ec) return Fail(error, ec.message());
    return true;
#else
    // freedesktop.org trash: the file under files/, its origin under info/.
    fs::path base;
    if (const char* data = std::getenv("XDG_DATA_HOME"); data && *data) {
        base = data;
    } else if (const char* home = std::getenv("HOME"); home && *home) {
        base = fs::path(home) / ".local" / "share";
    } else {
        return Fail(error, "the home folder is not known");
    }
    const fs::path files = base / "Trash" / "files", info = base / "Trash" / "info";
    fs::create_directories(files, ec);
    fs::create_directories(info, ec);
    const fs::path target = FreeName(files, absolute.filename());
    const std::time_t now = std::time(nullptr);
    char stamp[32] = {};
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    {
        std::ofstream out(info / (target.filename().string() + ".trashinfo"));
        if (!out) return Fail(error, "the Trash could not be written");
        out << "[Trash Info]\nPath=" << absolute.string() << "\nDeletionDate=" << stamp << "\n";
    }
    fs::rename(absolute, target, ec);
    if (ec) {
        fs::remove(info / (target.filename().string() + ".trashinfo"), ec);
        return Fail(error, "it could not be moved to the Trash");
    }
    return true;
#endif
}

}  // namespace cyxwiz
