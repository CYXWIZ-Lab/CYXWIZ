// An installed add-on (plugin pack) is checked against its signed inventory
// before it loads: a verified pack loads as a signed add-on, a changed file
// refuses it with a reason, a changed cache copy is copied again, a folder
// under the runtime's plugins/ that is not active is refused, a developer
// plugin loads as before, and the check's cost on a pack the size of the
// MuJoCo add-on is timed. Test keys only (generated here).

#include "../src/plugin/plugin_loader.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/plugin_pack_trust.h"

#include "backend_pack_hash.h"
#include "backend_pack_metadata_verifier.h"
#include "backend_pack_platform.h"
#include "runtime_layout.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <unistd.h>
#endif

#ifndef OK_PLUGIN
#error "OK_PLUGIN must name the test plugin library"
#endif

namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace cyxwiz::plugin;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

constexpr const char* kPluginId = "com.cyxwiz.tests.pack";
constexpr const char* kPackId = "test-pack-v1";

// ---- test signing (Ed25519, as the release tools sign) ----

using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;

Key GenerateKey() {
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> context(
        EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr), EVP_PKEY_CTX_free);
    EVP_PKEY* key = nullptr;
    if (!context || EVP_PKEY_keygen_init(context.get()) != 1 || EVP_PKEY_keygen(context.get(), &key) != 1)
        throw std::runtime_error("cannot generate a test key");
    return Key(key, EVP_PKEY_free);
}

std::string Base64Url(const unsigned char* bytes, std::size_t size) {
    std::string output(4 * ((size + 2) / 3), '\0');
    const auto written = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(output.data()), bytes,
                                         static_cast<int>(size));
    output.resize(static_cast<std::size_t>(written));
    while (!output.empty() && output.back() == '=') output.pop_back();
    std::replace(output.begin(), output.end(), '+', '-');
    std::replace(output.begin(), output.end(), '/', '_');
    return output;
}

std::string PublicKey(const Key& key) {
    std::array<unsigned char, 32> bytes{};
    std::size_t size = bytes.size();
    if (EVP_PKEY_get_raw_public_key(key.get(), bytes.data(), &size) != 1) throw std::runtime_error("public key");
    return Base64Url(bytes.data(), size);
}

std::string Sign(const Json& body, const Key& key) {
    const auto bytes = body.dump();
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    std::array<unsigned char, 64> signature{};
    std::size_t size = signature.size();
    if (!context || EVP_DigestSignInit(context.get(), nullptr, nullptr, nullptr, key.get()) != 1 ||
        EVP_DigestSign(context.get(), signature.data(), &size, reinterpret_cast<const unsigned char*>(bytes.data()),
                       bytes.size()) != 1)
        throw std::runtime_error("cannot sign");
    return Base64Url(signature.data(), size);
}

void WriteJson(const fs::path& path, const Json& value) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << value.dump(2) << "\n";
}

std::string ReadAll(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void FlipByteInMiddle(const fs::path& path) {
    const auto size = fs::file_size(path);
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekg(static_cast<std::streamoff>(size / 2));
    const int value = file.get();
    file.seekp(static_cast<std::streamoff>(size / 2));
    file.put(static_cast<char>(value ^ 0xff));
}

std::string Sha256(const fs::path& path) {
    std::string digest;
    std::string error;
    if (!cyxwiz::runtime::Sha256File(path, digest, error)) throw std::runtime_error(error);
    return digest;
}

// A runtime with one active add-on built from the test plugin, a large
// library beside it (8.7 MB in all, as the MuJoCo add-on), signed with a test pack key.
struct Fixture {
    fs::path root;
    fs::path runtime;
    fs::path pack;
    fs::path library;   // the plugin library in the pack
    fs::path big;       // the large library beside it
    Key pack_key = GenerateKey();

    Fixture() {
        root = fs::temp_directory_path() / ("cyxwiz-test-pack-load-" + std::to_string(GetCurrentProcessIdOrPid()));
        std::error_code ec;
        fs::remove_all(root, ec);
        runtime = root / "runtime";
        pack = cyxwiz::runtime::PluginPackDirectory(runtime, kPluginId, kPackId);
        fs::create_directories(pack / "bin");
        library = pack / "bin" / fs::path(OK_PLUGIN).filename();
        fs::copy_file(OK_PLUGIN, library);
        big = pack / "bin" / "big_dependency.dll";
        {
            std::ofstream out(big, std::ios::binary);
            std::uint32_t state = 12345u;
            std::vector<char> block(1 << 16);
            for (std::size_t written = 0; written < 8'500'000; written += block.size()) {
                for (auto& byte : block) {
                    state = state * 1664525u + 1013904223u;
                    byte = static_cast<char>(state >> 24);
                }
                out.write(block.data(), static_cast<std::streamsize>(
                                            std::min<std::size_t>(block.size(), 8'500'000 - written)));
            }
        }
        std::ofstream(pack / "plugin.json") << Json{
            {"id", kPluginId}, {"name", "Pack test"}, {"version", "1.0.0"}, {"api_version", "1.3.0"},
            {"platforms",
             {{"windows", {{"library", "bin/" + library.filename().string()}}},
              {"linux", {{"library", "bin/" + library.filename().string()}}},
              {"macos", {{"library", "bin/" + library.filename().string()}}}}}}.dump(2);
        fs::create_directories(pack / "THIRD_PARTY_LICENSES" / "Test");
        std::ofstream(pack / "THIRD_PARTY_LICENSES" / "Test" / "LICENSE.txt") << "Test licence\n";

        WriteJson(runtime / "trust" / "trusted-keys.json", Trust(false));
        WriteJson(runtime / "installed-metadata" / (std::string(kPackId) + ".json"), Manifest());
        cyxwiz::runtime::ActiveRuntimeState active;
        active.runtime_set_id = "set-v1";
        active.generation = 1;
        active.base_pack_id = "base-v1";
        cyxwiz::runtime::ActivePluginsState plugins;
        plugins.base_pack_id = "base-v1";
        plugins.plugins.push_back({kPluginId, kPackId});
        std::string error;
        if (!cyxwiz::runtime::SaveActiveRuntimeStateAtomic(runtime / "active-runtime.json", active, error) ||
            !cyxwiz::runtime::SaveActivePluginsStateAtomic(runtime / cyxwiz::runtime::kActivePluginsFileName,
                                                           plugins, error))
            throw std::runtime_error(error);
    }

    ~Fixture() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    static long GetCurrentProcessIdOrPid() {
#ifdef _WIN32
        return static_cast<long>(::GetCurrentProcessId());
#else
        return static_cast<long>(::getpid());
#endif
    }

    Json Trust(bool revoked) const {
        return {{"schema_version", 1},
                {"keys", Json::array({{{"key_id", "test-pack"}, {"algorithm", "ed25519"},
                                       {"public_key", PublicKey(pack_key)}, {"roles", Json::array({"pack"})},
                                       {"revoked", revoked}}})}};
    }

    Json Component(const std::string& path, const std::string& source) const {
        const fs::path file = pack / fs::path(path);
        return {{"path", path}, {"size", fs::file_size(file)}, {"sha256", Sha256(file)},
                {"source", source}, {"executable", file.extension() == ".dll"}};
    }

    Json Manifest() const {
        Json body = {
            {"pack_id", kPackId}, {"pack_kind", "plugin_pack"}, {"plugin_id", kPluginId},
            {"package_version", "2"},
            {"platform", std::string(cyxwiz::runtime::CurrentBackendPackPlatformId())},
            {"architecture", std::string(cyxwiz::runtime::CurrentBackendPackArchitectureId())},
            {"cyxwiz_release", {{"minimum", "0.2.0"}, {"maximum", "0.2.x"}}},
            {"companion_base_id", "base-v1"}, {"conflicts", Json::array()}, {"support_status", "supported"},
            {"components", Json::array({Component("plugin.json", "cyxwiz-plugin"),
                                        Component("bin/" + library.filename().string(), "cyxwiz-plugin"),
                                        Component("bin/big_dependency.dll", "test"),
                                        Component("THIRD_PARTY_LICENSES/Test/LICENSE.txt", "test-license")})},
            {"licenses", Json::array({{{"component", "test"}, {"path", "THIRD_PARTY_LICENSES/Test/LICENSE.txt"}}})},
            {"archive", {{"file_name", std::string(kPackId) + ".zip"}, {"size", 1},
                         {"sha256", std::string(64, '0')}}},
            {"generated_utc", "2026-11-25T00:00:00Z"}};
        return {{"schema_version", 2}, {"kind", "cyxwiz-backend-pack-manifest"}, {"signed", body},
                {"signatures", Json::array({{{"key_id", "test-pack"}, {"algorithm", "ed25519"},
                                             {"value", Sign(body, pack_key)}}})}};
    }

    cyxwiz::runtime::BackendPackMetadataVerifier Verifier(const fs::path& trust_file) const {
        std::string error;
        auto trust = cyxwiz::runtime::BackendPackTrustStore::Load(trust_file, error);
        if (!trust) throw std::runtime_error(error);
        return cyxwiz::runtime::BackendPackMetadataVerifier(
            std::move(*trust), "0.2.0", std::string(cyxwiz::runtime::CurrentBackendPackPlatformId()),
            std::string(cyxwiz::runtime::CurrentBackendPackArchitectureId()));
    }

    InstalledPluginPacks Read(const fs::path& trust_file) const {
        InstalledPluginPacks packs;
        std::vector<std::string> skipped;
        std::string error;
        if (!ReadInstalledPluginPacks(runtime, Verifier(trust_file), packs, skipped, error))
            throw std::runtime_error(error);
        return packs;
    }
};

// A developer plugin folder outside the runtime.
fs::path DeveloperFolder(const fs::path& root) {
    const fs::path folder = root / "developer" / "dev";
    fs::create_directories(folder);
    fs::copy_file(OK_PLUGIN, folder / fs::path(OK_PLUGIN).filename(), fs::copy_options::overwrite_existing);
    std::ofstream(folder / "plugin.json") << Json{
        {"id", "com.cyxwiz.tests.developer"}, {"name", "Developer test"}, {"version", "1.0.0"},
        {"api_version", "1.3.0"},
        {"platforms",
         {{"windows", {{"library", fs::path(OK_PLUGIN).filename().string()}}},
          {"linux", {{"library", fs::path(OK_PLUGIN).filename().string()}}},
          {"macos", {{"library", fs::path(OK_PLUGIN).filename().string()}}}}}}.dump(2);
    return folder;
}

}  // namespace

int main() {
    try {
        Fixture fixture;
        auto& manager = PluginManager::Instance();
        const fs::path trust = fixture.runtime / "trust" / "trusted-keys.json";

        std::cout << "a verified add-on\n";
        auto packs = fixture.Read(trust);
        Check(packs.packs.size() == 1 && packs.refusals.empty() && packs.packs[0].components.size() == 4 &&
                  packs.packs[0].package_version == "2",
              "the active add-on's signed metadata verifies with its four files");
        manager.SetPackPlugins(packs);
        std::string error;
        Check(manager.LoadPlugin(fixture.pack, &error), "the verified add-on loads" +
                                                            (error.empty() ? "" : ": " + error));
        const auto* loaded = manager.GetLoadedPlugin(kPluginId);
        Check(loaded && loaded->origin == LoadedPlugin::Origin::SignedPack && loaded->pack_id == kPackId &&
                  loaded->package_version == "2" && loaded->pack_files_checked == 4,
              "it loads as a signed add-on with its pack and version");
        const double first_ms = loaded ? loaded->pack_check_ms : 0.0;
        const fs::path copy_folder = loaded ? loaded->loaded_from.parent_path() : fs::path{};
        const auto pack_bytes = loaded ? loaded->pack_bytes_checked : 0;
        std::cout << "  check of " << (loaded ? loaded->pack_files_checked : 0) << " files, "
                  << static_cast<double>(pack_bytes) / 1e6 << " MB: " << first_ms << " ms (first, with the copy)\n";
#ifdef _WIN32
        // Loaded: the system protects the copies that run.
        {
            const HANDLE library_handle = ::CreateFileW(
                (copy_folder / fixture.library.filename()).wstring().c_str(), GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            Check(library_handle == INVALID_HANDLE_VALUE, "the loaded library copy cannot be written");
            if (library_handle != INVALID_HANDLE_VALUE) ::CloseHandle(library_handle);
        }
#endif
        manager.UnloadPlugin(kPluginId);

        std::cout << "the cache\n";
        Check(manager.LoadPlugin(fixture.pack, &error), "it loads again from its cached copies");
        loaded = manager.GetLoadedPlugin(kPluginId);
        const double second_ms = loaded ? loaded->pack_check_ms : 0.0;
        std::cout << "  check: " << second_ms << " ms (cached copies)\n";
        // Every start after the first copy: the folder and the cached copies are
        // hashed. The first load also copies the libraries once (printed above).
        Check(second_ms < 2000.0, "the check at a start stays under 2 s");
        manager.UnloadPlugin(kPluginId);

        FlipByteInMiddle(copy_folder / "big_dependency.dll");
        Check(manager.LoadPlugin(fixture.pack, &error),
              "a cached copy changed at the same size is copied again and loads" +
                  (error.empty() ? "" : ": " + error));
        Check(ReadAll(copy_folder / "big_dependency.dll") == ReadAll(fixture.big),
              "the copy that loaded is the signed one");
        manager.UnloadPlugin(kPluginId);

        // A library planted in the copies' folder would be found there first
        // (it is on the DLL search path of the plugin): it is removed.
        std::ofstream(copy_folder / "planted.dll") << "not signed";
        const bool loaded_with_planted = manager.LoadPlugin(fixture.pack, &error);
        Check(loaded_with_planted && !fs::exists(copy_folder / "planted.dll"),
              "a file planted beside the library copies is removed before the load" +
                  (error.empty() ? "" : ": " + error));
        manager.UnloadPlugin(kPluginId);

        std::cout << "a changed add-on\n";
        FlipByteInMiddle(fixture.big);
        const fs::path developer = DeveloperFolder(fixture.root);
        manager.SetSearchPaths({developer.parent_path()});
        manager.LoadAllFromSearchPaths();  // the add-on, then the developer plugin
        Check(manager.GetLoadedPlugin(kPluginId) == nullptr, "an add-on with a changed library is not loaded");
        auto refusals = manager.GetPackRefusals();
        Check(refusals.size() == 1 && refusals[0].plugin_id == kPluginId && refusals[0].issues.size() == 1 &&
                  refusals[0].issues[0].path == "bin/big_dependency.dll" &&
                  refusals[0].issues[0].problem == PackFileProblem::ContentDiffers &&
                  refusals[0].reason.find("does not match its signed package") != std::string::npos &&
                  refusals[0].files_listed == 4,
              "the refusal names the file: " + (refusals.empty() ? std::string("none") : refusals[0].reason));
        const auto* dev = manager.GetLoadedPlugin("com.cyxwiz.tests.developer");
        Check(dev && dev->origin == LoadedPlugin::Origin::Developer,
              "the other plugins still load; a developer plugin loads as before");
        manager.UnloadPlugin("com.cyxwiz.tests.developer");
        manager.SetSearchPaths({});

        FlipByteInMiddle(fixture.big);  // put back: a repaired add-on loads again
        Check(manager.LoadPlugin(fixture.pack, &error) && manager.GetPackRefusals().empty(),
              "the repaired add-on loads and its refusal is cleared");
        manager.UnloadPlugin(kPluginId);

        std::ofstream(fixture.pack / "bin" / "extra.dll") << "x";
        const bool extra_loaded = manager.LoadPlugin(fixture.pack, &error);
        Check(!extra_loaded && error.find("is not part of its signed package") != std::string::npos,
              "an unlisted library in the add-on refuses it: " + error);
        fs::remove(fixture.pack / "bin" / "extra.dll");

        std::cout << "other folders\n";
        const fs::path inactive = fixture.runtime / "plugins" / kPluginId / "test-pack-v0";
        fs::copy(fixture.pack, inactive, fs::copy_options::recursive);
        const bool inactive_loaded = manager.LoadPlugin(inactive, &error);
        Check(!inactive_loaded && error == "not an active installed add-on",
              "a folder under the runtime's plugins that is not active is refused: " + error);
        fs::remove_all(inactive);

        std::cout << "signed metadata\n";
        WriteJson(fixture.root / "revoked.json", fixture.Trust(true));
        const auto revoked = fixture.Read(fixture.root / "revoked.json");
        Check(revoked.packs.empty() && revoked.refusals.size() == 1 && revoked.refusals[0].files_listed == 0,
              "an add-on whose key is revoked is refused before any file is read");
        manager.SetPackPlugins(revoked);
        Check(!manager.LoadPlugin(fixture.pack, &error) && manager.GetPackRefusals().size() == 1,
              "and its folder does not load");

        std::cout << "fail closed\n";
        // With no trust file, the active add-on is refused, never loaded unchecked.
        fs::rename(trust, fixture.root / "trust.aside");
        const bool trusted = ConfigureInstalledPluginPacks(manager, fixture.runtime);
        const bool loaded_untrusted = manager.LoadPlugin(fixture.pack, &error);
        auto untrusted = manager.GetPackRefusals();
        Check(!trusted && !loaded_untrusted && untrusted.size() == 1 &&
                  untrusted[0].reason.find("trust file") != std::string::npos,
              "with no trust file the add-on is refused: " + error);
        fs::rename(fixture.root / "trust.aside", trust);
        // With an unreadable installation state, every folder under plugins/ is refused.
        const fs::path active_plugins = fixture.runtime / cyxwiz::runtime::kActivePluginsFileName;
        const std::string active_text = ReadAll(active_plugins);
        std::ofstream(active_plugins, std::ios::binary | std::ios::trunc) << "{ damaged";
        const bool read_damaged = ConfigureInstalledPluginPacks(manager, fixture.runtime);
        const bool loaded_damaged = manager.LoadPlugin(fixture.pack, &error);
        Check(!read_damaged && !loaded_damaged && error == "not an active installed add-on",
              "with an unreadable active state the add-on folder is refused: " + error);
        std::ofstream(active_plugins, std::ios::binary | std::ios::trunc) << active_text;
        const bool read_again = ConfigureInstalledPluginPacks(manager, fixture.runtime);
        const bool loaded_again = manager.LoadPlugin(fixture.pack, &error);
        Check(read_again && loaded_again, "with both back the add-on loads checked" +
                                              (error.empty() ? "" : ": " + error));
        manager.UnloadPlugin(kPluginId);
        manager.SetPackPlugins({});
    } catch (const std::exception& e) {
        std::cout << "  FAIL " << e.what() << "\n";
        ++g_failures;
    }
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_plugin_pack_load: all checks passed\n";
    return 0;
}
