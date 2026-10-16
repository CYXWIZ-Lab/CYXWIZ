// Python scan cache: a start uses the last scan when nothing changed, and
// scans again when an interpreter changed, went away, none is usable, the
// saved path is not in it, or the Engine needs another Python.
#include "../src/core/python_scan_cache.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace cyxwiz::pythonsetup;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

Candidate Python(const std::string& path, const std::string& version, bool usable) {
    Candidate c;
    c.path = path;
    c.version = version;
    c.venv = true;
    c.pip = true;
    c.usable = usable;
    if (!usable) c.reason = "This Engine embeds Python 3.12; found " + version;
    c.home = path.substr(0, path.rfind('\\'));
    return c;
}
}  // namespace

int main() {
    // The owner's PC: 3.12 usable, 3.14 not.
    ScanResult scan;
    scan.scanned = true;
    scan.found = {Python("C:\\Python312\\python.exe", "3.12.8", true),
                  Python("C:\\Program Files\\Python314\\python.exe", "3.14.0", false)};
    std::map<std::string, FileStamp> disk = {{"C:\\Python312\\python.exe", {104448, 1733900000}},
                                             {"C:\\Program Files\\Python314\\python.exe", {106496, 1760000000}}};
    const StampOf stamp_of = [&](const std::string& p) -> std::optional<FileStamp> {
        auto it = disk.find(p);
        if (it == disk.end()) return std::nullopt;
        return it->second;
    };

    const ScanCache cache = CacheOf(scan, "3.12", stamp_of);
    Check(cache.entries.size() == 2, "both interpreters cached");
    const auto back = ScanCacheFromJson(ScanCacheToJson(cache));
    Check(back && back->entries.size() == 2 && back->required == "3.12" && back->entries[1].candidate.reason == scan.found[1].reason &&
              back->entries[0].stamp == disk["C:\\Python312\\python.exe"],
          "JSON round trip");
    Check(!ScanCacheFromJson("{\"version\":2,\"interpreters\":[]}") && !ScanCacheFromJson("not json"),
          "other versions and text are refused");

    // Nothing changed: the cache answers, in the scan's order.
    auto r = ResultFromCache(*back, "", "3.12", stamp_of);
    Check(r && r->from_cache && r->found.size() == 2 && r->found[0].usable && !r->configured_path_set,
          "unchanged: the last scan is used");
    // A saved path in the cache comes first and says whether it is usable.
    r = ResultFromCache(*back, "C:\\Program Files\\Python314\\python.exe", "3.12", stamp_of);
    Check(r && r->configured_path_set && !r->configured_ok && r->found[0].version == "3.14.0", "saved path first");

    // Scan again when ...
    Check(!ResultFromCache(*back, "D:\\other\\python.exe", "3.12", stamp_of), "... the saved path is not in the cache");
    Check(!ResultFromCache(*back, "", "3.13", stamp_of), "... the Engine needs another Python");
    disk["C:\\Python312\\python.exe"].modified += 1;
    Check(!ResultFromCache(*back, "", "3.12", stamp_of), "... an interpreter changed (updated)");
    disk["C:\\Python312\\python.exe"].modified -= 1;
    disk.erase("C:\\Program Files\\Python314\\python.exe");
    Check(!ResultFromCache(*back, "", "3.12", stamp_of), "... an interpreter went away");
    ScanCache none_usable = *back;
    none_usable.entries.pop_back();
    none_usable.entries[0].candidate.usable = false;
    Check(!ResultFromCache(none_usable, "", "3.12", stamp_of), "... none is usable (a new install may help)");
    Check(!ResultFromCache(ScanCache{}, "", "3.12", stamp_of), "... the cache is empty");

    // A real file's stamp.
    const auto file = std::filesystem::temp_directory_path() / "cyxwiz_scan_cache_test.bin";
    std::ofstream(file, std::ios::binary) << "1234";
    const auto stamp = StampOfFile(file.string());
    Check(stamp && stamp->size == 4 && stamp->modified != 0, "a file's size and time");
    std::filesystem::remove(file);
    Check(!StampOfFile(file.string()), "a missing file has no stamp");
    std::cout << "python scan cache: round trip, used when unchanged, scans again on change, removal, no usable, "
                 "other saved path or required Python. OK\n";
    return 0;
}
