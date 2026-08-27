#include "backend_pack_acquisition.h"
#include "backend_pack_metadata_refresh.h"
#include "installer_bundle_verifier.h"
#include "installer_setup_launcher.h"
#include "installer_setup_service.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#ifndef CYXWIZ_SETUP_VERSION
#define CYXWIZ_SETUP_VERSION "0.1.0"
#endif
#ifndef CYXWIZ_SETUP_PLATFORM
#define CYXWIZ_SETUP_PLATFORM "unknown"
#endif
#ifndef CYXWIZ_SETUP_ARCHITECTURE
#define CYXWIZ_SETUP_ARCHITECTURE "unknown"
#endif
#ifndef CYXWIZ_SETUP_BUNDLE_URL
#define CYXWIZ_SETUP_BUNDLE_URL ""
#endif

namespace {

struct Options {
    std::string descriptor_url = CYXWIZ_SETUP_BUNDLE_URL;
    std::filesystem::path descriptor_path;
    std::filesystem::path trust_store_path;
    std::filesystem::path cache_root;
    bool prepare_only = false;
};

void PrintUsage() {
    std::cout
        << "CyxWiz Setup " << CYXWIZ_SETUP_VERSION << "\n"
        << "Usage: cyxwiz-setup [--descriptor-url HTTPS_URL | --descriptor FILE]\n"
        << "                    [--trust-store FILE] [--cache-root DIRECTORY]\n"
        << "                    [--prepare-only]\n";
}

bool ParseArguments(int argc, char** argv, Options& output, std::string& error) {
    bool explicit_url = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            PrintUsage();
            return false;
        }
        if (argument == "--prepare-only") {
            output.prepare_only = true;
            continue;
        }
        if (index + 1 >= argc) {
            error = "Missing value after " + argument;
            return false;
        }
        const std::string value = argv[++index];
        if (argument == "--descriptor-url") {
            output.descriptor_url = value;
            explicit_url = true;
        } else if (argument == "--descriptor") {
            output.descriptor_path = std::filesystem::absolute(value);
        } else if (argument == "--trust-store") {
            output.trust_store_path = std::filesystem::absolute(value);
        } else if (argument == "--cache-root") {
            output.cache_root = std::filesystem::absolute(value);
        } else {
            error = "Unknown setup option: " + argument;
            return false;
        }
    }
    if (!output.descriptor_path.empty() &&
        (explicit_url || (!output.descriptor_url.empty() &&
                          output.descriptor_url != CYXWIZ_SETUP_BUNDLE_URL))) {
        error = "Choose either --descriptor-url or --descriptor";
        return false;
    }
    if (!output.descriptor_path.empty()) output.descriptor_url.clear();
    return true;
}

std::filesystem::path DefaultCacheRoot() {
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        return std::filesystem::path(local) / "CyxWiz" / "Setup";
    }
#elif defined(__APPLE__)
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / "Library" / "Caches" /
            "CyxWiz" / "Setup";
    }
#else
    if (const char* cache = std::getenv("XDG_CACHE_HOME")) {
        return std::filesystem::path(cache) / "cyxwiz" / "setup";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".cache" / "cyxwiz" / "setup";
    }
#endif
    return std::filesystem::temp_directory_path() / "cyxwiz-setup";
}

std::string CurrentUtc() {
    const auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

class RemoveTemporaryDescriptor {
public:
    ~RemoveTemporaryDescriptor() {
        if (path_.empty()) return;
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    void Set(std::filesystem::path path) { path_ = std::move(path); }

private:
    std::filesystem::path path_;
};

}  // namespace

int main(int argc, char** argv) {
    Options options;
    std::string error;
    if (!ParseArguments(argc, argv, options, error)) {
        if (error.empty()) return 0;
        std::cerr << "[ERROR] " << error << "\n";
        PrintUsage();
        return 2;
    }
    const auto executable = std::filesystem::absolute(argv[0]);
    if (options.cache_root.empty()) options.cache_root = DefaultCacheRoot();
    if (options.trust_store_path.empty()) {
        options.trust_store_path = executable.parent_path() / "trusted-keys.json";
    }
    RemoveTemporaryDescriptor temporary_descriptor;
    if (options.descriptor_path.empty()) {
        if (options.descriptor_url.rfind("https://", 0) != 0) {
            std::cerr << "[ERROR] No production HTTPS installer descriptor URL "
                         "is configured\n";
            return 2;
        }
        const auto seed = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        options.descriptor_path = options.cache_root / "metadata" /
            ("descriptor-" + std::to_string(seed) + ".json");
        cyxwiz::runtime::HttpsBackendPackMetadataSource metadata_source;
        if (!metadata_source.Fetch(
                options.descriptor_url, options.descriptor_path,
                4U * 1024U * 1024U, error)) {
            std::cerr << "[ERROR] Cannot download installer descriptor: "
                      << error << "\n";
            return 3;
        }
        temporary_descriptor.Set(options.descriptor_path);
    }

    std::unique_ptr<cyxwiz::runtime::BackendPackArtifactSource> archive_source;
    if (!options.descriptor_url.empty()) {
        std::string body;
        auto trust = cyxwiz::runtime::BackendPackTrustStore::Load(
            options.trust_store_path, error);
        if (!trust) {
            std::cerr << "[ERROR] Cannot load installer trust store: "
                      << error << "\n";
            return 4;
        }
        cyxwiz::runtime::InstallerBundleVerifier verifier(
            std::move(*trust), CYXWIZ_SETUP_VERSION,
            CYXWIZ_SETUP_PLATFORM, CYXWIZ_SETUP_ARCHITECTURE);
        cyxwiz::runtime::VerifiedInstallerBundle bundle;
        if (!verifier.Verify(
                options.descriptor_path, CurrentUtc(), bundle, error)) {
            std::cerr << "[ERROR] Installer descriptor was rejected: "
                      << error << "\n";
            return 4;
        }
        std::string archive_url;
        if (!cyxwiz::runtime::ResolveHttpsBackendPackArchiveUrl(
                options.descriptor_url, bundle.archive.file_name,
                archive_url, error)) {
            std::cerr << "[ERROR] Cannot resolve installer archive: "
                      << error << "\n";
            return 4;
        }
        archive_source = std::make_unique<
            cyxwiz::runtime::HttpsBackendPackArtifactSource>(archive_url);
    } else {
        std::string body;
        auto trust = cyxwiz::runtime::BackendPackTrustStore::Load(
            options.trust_store_path, error);
        if (!trust) {
            std::cerr << "[ERROR] Cannot load installer trust store: "
                      << error << "\n";
            return 4;
        }
        cyxwiz::runtime::InstallerBundleVerifier verifier(
            std::move(*trust), CYXWIZ_SETUP_VERSION,
            CYXWIZ_SETUP_PLATFORM, CYXWIZ_SETUP_ARCHITECTURE);
        cyxwiz::runtime::VerifiedInstallerBundle bundle;
        if (!verifier.Verify(
                options.descriptor_path, CurrentUtc(), bundle, error)) {
            std::cerr << "[ERROR] Installer descriptor was rejected: "
                      << error << "\n";
            return 4;
        }
        std::filesystem::path archive_path;
        if (!cyxwiz::runtime::ResolveOfflineBackendPackArchivePath(
                options.descriptor_path, bundle.archive.file_name,
                archive_path, error)) {
            std::cerr << "[ERROR] Cannot resolve installer archive: "
                      << error << "\n";
            return 4;
        }
        archive_source = std::make_unique<
            cyxwiz::runtime::OfflineBackendPackArtifactSource>(archive_path);
    }

    const cyxwiz::runtime::InstallerSetupRequest request{
        options.descriptor_path, options.trust_store_path, options.cache_root,
        CurrentUtc(), CYXWIZ_SETUP_VERSION, CYXWIZ_SETUP_PLATFORM,
        CYXWIZ_SETUP_ARCHITECTURE};
    const auto result = cyxwiz::runtime::PrepareInstallerBundle(
        request, *archive_source);
    if (result.status != cyxwiz::runtime::InstallerSetupStatus::Ready) {
        std::cerr << "[ERROR] " << result.message << "\n";
        return 5;
    }
    std::cout << "[OK] " << result.message << ": " << result.bundle_id << "\n";
    if (options.prepare_only) return 0;
    int child_exit = -1;
    if (!cyxwiz::runtime::LaunchInstallerAndWait(
            result.installer_path, child_exit, error)) {
        std::cerr << "[ERROR] " << error << "\n";
        return 6;
    }
    return child_exit;
}
