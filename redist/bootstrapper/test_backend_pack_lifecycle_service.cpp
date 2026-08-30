#include "backend_pack_hash.h"
#include "backend_pack_lifecycle_service.h"
#include "backend_pack_metadata_cache.h"
#include "backend_pack_metadata_refresh.h"
#include "backend_pack_platform.h"

#include <archive.h>
#include <archive_entry.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;
using namespace cyxwiz::runtime;

constexpr const char* kZeroByteSha256 =
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d";

struct KeyPair {
    explicit KeyPair(EVP_PKEY* key) : key(key, EVP_PKEY_free) {}
    std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key;
};

KeyPair GenerateKey() {
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> context(
        EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr), EVP_PKEY_CTX_free);
    EVP_PKEY* key = nullptr;
    if (!context || EVP_PKEY_keygen_init(context.get()) != 1 ||
        EVP_PKEY_keygen(context.get(), &key) != 1) {
        return KeyPair(nullptr);
    }
    return KeyPair(key);
}

std::string Base64Url(const unsigned char* bytes, std::size_t size) {
    std::string output(4 * ((size + 2) / 3), '\0');
    const auto written = EVP_EncodeBlock(
        reinterpret_cast<unsigned char*>(output.data()), bytes,
        static_cast<int>(size));
    output.resize(static_cast<std::size_t>(written));
    while (!output.empty() && output.back() == '=') output.pop_back();
    std::replace(output.begin(), output.end(), '+', '-');
    std::replace(output.begin(), output.end(), '/', '_');
    return output;
}

std::string PublicKey(const KeyPair& key) {
    std::array<unsigned char, 32> bytes{};
    std::size_t size = bytes.size();
    if (!key.key || EVP_PKEY_get_raw_public_key(
            key.key.get(), bytes.data(), &size) != 1) {
        return {};
    }
    return Base64Url(bytes.data(), size);
}

std::string Sign(const Json& body, const KeyPair& key) {
    const auto bytes = body.dump();
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    std::array<unsigned char, 64> signature{};
    std::size_t size = signature.size();
    if (!context || !key.key || EVP_DigestSignInit(
            context.get(), nullptr, nullptr, nullptr, key.key.get()) != 1 ||
        EVP_DigestSign(
            context.get(), signature.data(), &size,
            reinterpret_cast<const unsigned char*>(bytes.data()),
            bytes.size()) != 1) {
        return {};
    }
    return Base64Url(signature.data(), size);
}

std::string Hash(const std::string& bytes) {
    std::string digest;
    std::string error;
    return Sha256Bytes(bytes, digest, error) ? digest : "";
}

std::string WriteJson(const std::filesystem::path& path, const Json& value) {
    std::filesystem::create_directories(path.parent_path());
    const auto bytes = value.dump(2) + "\n";
    std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
    return bytes;
}

Json Envelope(
    const char* kind,
    Json body,
    const char* key_id,
    const KeyPair& key) {
    return {
        {"schema_version", std::uint64_t{1}},
        {"kind", kind},
        {"signed", body},
        {"signatures", Json::array({{{"key_id", key_id},
             {"algorithm", "ed25519"}, {"value", Sign(body, key)}}})}};
}

bool WriteZip(
    const std::filesystem::path& path,
    const std::vector<std::string>& entries = {
        "runtime/afopencl.dll",
        "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt"}) {
    std::filesystem::create_directories(path.parent_path());
    archive* writer = archive_write_new();
    if (!writer || archive_write_set_format_zip(writer) != ARCHIVE_OK ||
#ifdef _WIN32
        archive_write_open_filename_w(writer, path.c_str()) != ARCHIVE_OK) {
#else
        archive_write_open_filename(writer, path.c_str()) != ARCHIVE_OK) {
#endif
        if (writer) archive_write_free(writer);
        return false;
    }
    for (const auto& entry_path : entries) {
        archive_entry* entry = archive_entry_new();
        archive_entry_set_pathname_utf8(entry, entry_path.c_str());
        archive_entry_set_filetype(entry, AE_IFREG);
        archive_entry_set_perm(entry, 0644);
        archive_entry_set_size(entry, 1);
        const char value = '\0';
        const bool wrote = archive_write_header(writer, entry) == ARCHIVE_OK &&
            archive_write_data(writer, &value, 1) == 1;
        archive_entry_free(entry);
        if (!wrote) {
            archive_write_free(writer);
            return false;
        }
    }
    const int closed = archive_write_close(writer);
    archive_write_free(writer);
    return closed == ARCHIVE_OK;
}

class Fixture {
public:
    Fixture()
        : catalog_key_(GenerateKey()), pack_key_(GenerateKey()) {
        root = std::filesystem::temp_directory_path() /
            ("cyxwiz-pack-lifecycle-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        runtime = root / "runtime-root";
        archive = root / "opencl-v1.zip";
        catalog_path = root / "catalog.json";
        manifest_path = root / "manifest.json";
        trust_path = root / "trusted-keys.json";
        Touch(runtime / "base" / "base-v1" / CurrentEngineExecutableName());
        ActiveRuntimeState active;
        active.runtime_set_id = "set-v1";
        active.generation = 1;
        active.base_pack_id = "base-v1";
        std::string error;
        if (!SaveActiveRuntimeStateAtomic(
                runtime / "active-runtime.json", active, error) ||
            !WriteZip(archive)) {
            throw std::runtime_error(error.empty() ?
                "Cannot create lifecycle fixture" : error);
        }
        WriteJson(trust_path, {
            {"schema_version", std::uint64_t{1}},
            {"keys", Json::array({{
                {"key_id", "catalog-2026"}, {"algorithm", "ed25519"},
                {"public_key", PublicKey(catalog_key_)},
                {"roles", Json::array({"catalog"})}, {"revoked", false}}, {
                {"key_id", "pack-2026"}, {"algorithm", "ed25519"},
                {"public_key", PublicKey(pack_key_)},
                {"roles", Json::array({"pack"})}, {"revoked", false}}})}});
        WriteMetadata("supported");
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void WriteMetadata(const std::string& support) {
        std::error_code filesystem_error;
        const auto archive_size =
            std::filesystem::file_size(archive, filesystem_error);
        std::string archive_hash;
        std::string error;
        Sha256File(archive, archive_hash, error);
        Json body = {
            {"pack_id", "opencl-v1"}, {"pack_kind", "backend_pack"},
            {"backend", "opencl"}, {"package_version", "1.0.0"},
            {"platform", "win64"}, {"architecture", "x86_64"},
            {"runtime_set_id", "set-v1"},
            {"cyxwiz_release", {{"minimum", "0.2.0"}, {"maximum", "0.2.x"}}},
            {"arrayfire", {{"version", "3.10.0"}, {"abi", "arrayfire-3.10"}}},
            {"companion_base_id", "base-v1"}, {"conflicts", Json::array()},
            {"compatibility", {
                {"device_kinds", Json::array({"gpu"})},
                {"cpu_features", Json::array()},
                {"provider_types", Json::array({"opencl-icd"})},
                {"minimum_driver_versions", {{"intel", "31.0.101.2115"}}},
                {"tested_driver_ranges", {{"intel", ">=31.0.101.2115"}}},
                {"minimum_identity_confidence", "stable_hardware"},
                {"recommendation_targets", Json::array({"cpu"})},
                {"operation_matrix_id", "matrix-v1"},
                {"training_scope", Json::array({"dense"})},
                {"support_status", support}}},
            {"components", Json::array({{
                {"path", "runtime/afopencl.dll"}, {"size", std::uint64_t{1}},
                {"sha256", kZeroByteSha256}, {"source", "arrayfire"},
                {"executable", true}}, {
                {"path", "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt"},
                {"size", std::uint64_t{1}}, {"sha256", kZeroByteSha256},
                {"source", "arrayfire-license"}, {"executable", false}}})},
            {"licenses", Json::array({{{"component", "arrayfire"},
                {"path", "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt"}}})},
            {"archive", {{"file_name", "opencl-v1.zip"},
                {"size", archive_size}, {"sha256", archive_hash}}},
            {"generated_utc", "2026-08-13T20:00:00Z"}};
        const auto manifest_bytes = WriteJson(
            manifest_path, Envelope(
                "cyxwiz-backend-pack-manifest", std::move(body),
                "pack-2026", pack_key_));
        Json catalog = {
            {"catalog_id", "production-2026-08"},
            {"generated_utc", "2026-08-13T20:00:00Z"},
            {"expires_utc", "2026-09-13T20:00:00Z"},
            {"minimum_client_version", "0.2.0"},
            {"packs", Json::array({{{"pack_id", "opencl-v1"},
                {"manifest_url", "https://downloads.cyxwiz.com/opencl-v1.json"},
                {"manifest_sha256", Hash(manifest_bytes)},
                {"signing_key_id", "pack-2026"},
                {"support_status", support}}})}};
        WriteJson(catalog_path, Envelope(
            "cyxwiz-backend-pack-catalog", std::move(catalog),
            "catalog-2026", catalog_key_));
    }

    void PrepareBase(
        const std::string& pack_id = "base-v1",
        const std::string& runtime_set_id = "set-v1",
        bool fresh = true) {
        std::error_code error;
        if (fresh) {
            std::filesystem::remove(runtime / "active-runtime.json", error);
        }
        std::filesystem::remove_all(
            runtime / "base" / pack_id, error);
        archive = root / (pack_id + ".zip");
        manifest_path = root / "base-manifest.json";
        catalog_path = root / "base-catalog.json";
        const std::string engine(CurrentEngineExecutableName());
        const std::string launcher(
            CurrentRuntimeBootstrapperExecutableName());
        const std::string finalizer(
            CurrentProductRemovalFinalizerExecutableName());
        if (!WriteZip(
                archive, {engine, launcher, finalizer, "LICENSE"})) {
            throw std::runtime_error("Cannot create base archive");
        }
        const auto archive_size = std::filesystem::file_size(archive);
        std::string archive_hash;
        std::string hash_error;
        if (!Sha256File(archive, archive_hash, hash_error)) {
            throw std::runtime_error(hash_error);
        }
        Json body = {
            {"pack_id", pack_id}, {"pack_kind", "base"},
            {"backend", "cpu"}, {"package_version", "1.0.0"},
            {"platform", "win64"}, {"architecture", "x86_64"},
            {"runtime_set_id", runtime_set_id},
            {"cyxwiz_release", {{"minimum", "0.2.0"}, {"maximum", "0.2.x"}}},
            {"arrayfire", {{"version", "3.10.0"}, {"abi", "arrayfire-3.10"}}},
            {"companion_base_id", nullptr}, {"conflicts", Json::array()},
            {"compatibility", {
                {"device_kinds", Json::array({"cpu"})},
                {"cpu_features", Json::array()},
                {"provider_types", Json::array()},
                {"minimum_driver_versions", Json::object()},
                {"tested_driver_ranges", Json::object()},
                {"minimum_identity_confidence", "backend_local"},
                {"recommendation_targets", Json::array({"cpu"})},
                {"operation_matrix_id", "matrix-v1"},
                {"training_scope", Json::array({"dense"})},
                {"support_status", "supported"}}},
            {"components", Json::array({{
                {"path", engine}, {"size", std::uint64_t{1}},
                {"sha256", kZeroByteSha256}, {"source", "cyxwiz"},
                {"executable", true}}, {
                {"path", launcher}, {"size", std::uint64_t{1}},
                {"sha256", kZeroByteSha256}, {"source", "cyxwiz"},
                {"executable", true}}, {
                {"path", finalizer}, {"size", std::uint64_t{1}},
                {"sha256", kZeroByteSha256}, {"source", "cyxwiz"},
                {"executable", true}}, {
                {"path", "LICENSE"}, {"size", std::uint64_t{1}},
                {"sha256", kZeroByteSha256}, {"source", "cyxwiz-license"},
                {"executable", false}}})},
            {"licenses", Json::array({{{"component", "cyxwiz"},
                {"path", "LICENSE"}}})},
            {"archive", {{"file_name", pack_id + ".zip"},
                {"size", archive_size}, {"sha256", archive_hash}}},
            {"generated_utc", "2026-08-13T20:00:00Z"}};
        const auto manifest_bytes = WriteJson(
            manifest_path, Envelope(
                "cyxwiz-backend-pack-manifest", std::move(body),
                "pack-2026", pack_key_));
        Json catalog = {
            {"catalog_id", "production-2026-08"},
            {"generated_utc", "2026-08-13T20:00:00Z"},
            {"expires_utc", "2026-09-13T20:00:00Z"},
            {"minimum_client_version", "0.2.0"},
            {"packs", Json::array({{{"pack_id", pack_id},
                {"manifest_url", "https://downloads.cyxwiz.com/" + pack_id + ".json"},
                {"manifest_sha256", Hash(manifest_bytes)},
                {"signing_key_id", "pack-2026"},
                {"support_status", "supported"}}})}};
        WriteJson(catalog_path, Envelope(
            "cyxwiz-backend-pack-catalog", std::move(catalog),
            "catalog-2026", catalog_key_));
    }

    BackendPackMetadataVerifier Verifier() const {
        std::string error;
        auto trust = BackendPackTrustStore::Load(trust_path, error);
        if (!trust) throw std::runtime_error(error);
        return BackendPackMetadataVerifier(
            std::move(*trust), "0.2.0", "win64", "x86_64");
    }

    BackendPackDeliveryRequest Request() const {
        BackendPackDeliveryRequest request;
        request.catalog_path = catalog_path;
        request.manifest_path = manifest_path;
        request.current_utc = "2026-08-14T12:00:00Z";
        request.pack_id = "opencl-v1";
        request.acquisition_disk_budget_bytes = 1024 * 1024;
        request.extraction_disk_budget_bytes = 1024;
        request.installation_disk_budget_bytes = 1024;
        return request;
    }

    void PublishCatalogCache() const {
        const auto cached_catalog =
            BackendPackCurrentCatalogPath(runtime);
        const auto cached_manifest =
            BackendPackCachedManifestPath(runtime, "opencl-v1");
        std::filesystem::create_directories(cached_catalog.parent_path());
        std::filesystem::create_directories(cached_manifest.parent_path());
        std::filesystem::copy_file(
            catalog_path, cached_catalog,
            std::filesystem::copy_options::overwrite_existing);
        std::filesystem::copy_file(
            manifest_path, cached_manifest,
            std::filesystem::copy_options::overwrite_existing);
    }

    ActiveRuntimeState Active() const {
        ActiveRuntimeState state;
        std::string error;
        if (!LoadActiveRuntimeState(
                runtime / "active-runtime.json", state, error)) {
            throw std::runtime_error(error);
        }
        return state;
    }

    static void Touch(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).put('\0');
    }

    static char ReadByte(const std::filesystem::path& path) {
        char value = '\xff';
        std::ifstream(path, std::ios::binary).get(value);
        return value;
    }

    std::filesystem::path root;
    std::filesystem::path runtime;
    std::filesystem::path archive;
    std::filesystem::path catalog_path;
    std::filesystem::path manifest_path;
    std::filesystem::path trust_path;

private:
    KeyPair catalog_key_;
    KeyPair pack_key_;
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

bool HasPack(const ActiveRuntimeState& state) {
    return state.packs.size() == 1 &&
           state.packs.front().backend == "opencl" &&
           state.packs.front().pack_id == "opencl-v1";
}

class FixtureMetadataSource final : public BackendPackMetadataSource {
public:
    void Set(std::string url, std::filesystem::path source) {
        for (auto& entry : documents_) {
            if (entry.first == url) {
                entry.second = std::move(source);
                return;
            }
        }
        documents_.emplace_back(std::move(url), std::move(source));
    }

    bool Fetch(
        const std::string& url,
        const std::filesystem::path& destination,
        std::uint64_t maximum_bytes,
        std::string& error) override {
        const auto entry = std::find_if(
            documents_.begin(), documents_.end(),
            [&](const auto& candidate) { return candidate.first == url; });
        if (entry == documents_.end()) {
            error = "Fixture URL is unavailable";
            return false;
        }
        std::error_code filesystem_error;
        const auto size = std::filesystem::file_size(
            entry->second, filesystem_error);
        if (filesystem_error || size == 0 || size > maximum_bytes) {
            error = "Fixture document violates its byte bound";
            return false;
        }
        std::filesystem::create_directories(
            destination.parent_path(), filesystem_error);
        if (filesystem_error || !std::filesystem::copy_file(
                entry->second, destination,
                std::filesystem::copy_options::none, filesystem_error)) {
            error = "Cannot copy fixture metadata document";
            return false;
        }
        return true;
    }

private:
    std::vector<std::pair<std::string, std::filesystem::path>> documents_;
};

}  // namespace

int main() {
    int failures = 0;
    {
        Fixture fixture;
        constexpr const char* catalog_url =
            "https://downloads.cyxwiz.com/catalogs/current.json";
        constexpr const char* manifest_url =
            "https://downloads.cyxwiz.com/opencl-v1.json";
        FixtureMetadataSource source;
        source.Set(catalog_url, fixture.catalog_path);
        source.Set(manifest_url, fixture.manifest_path);
        const auto destination = fixture.root / "refreshed-metadata";
        BackendPackMetadataRefreshRequest request;
        request.catalog_url = catalog_url;
        request.trusted_keys_path = fixture.trust_path;
        request.destination_root = destination;
        request.current_utc = "2026-08-14T12:00:00Z";
        const auto refreshed = RefreshBackendPackMetadata(
            request, fixture.Verifier(), source);

        BackendPackLifecycleService reader(
            destination, fixture.Verifier());
        VerifiedBackendPackCatalogSnapshot snapshot;
        std::string read_error;
        const bool readable = reader.ReadCatalogSnapshot(
            request.current_utc, snapshot, read_error);
        failures += !Expect(
            refreshed.status == BackendPackMetadataRefreshStatus::Refreshed &&
                refreshed.verified_pack_count == 1 && readable &&
                snapshot.catalog.catalog_id == "production-2026-08" &&
                snapshot.records.size() == 1 &&
                snapshot.records.front().manifest.has_value(),
            "remote metadata refresh must verify and atomically publish a complete catalog snapshot");

        const auto corrupt = fixture.root / "corrupt-manifest.json";
        std::ofstream(corrupt, std::ios::binary | std::ios::trunc)
            << "{\"corrupt\":true}";
        source.Set(manifest_url, corrupt);
        const auto rejected = RefreshBackendPackMetadata(
            request, fixture.Verifier(), source);
        snapshot = {};
        read_error.clear();
        const bool previous_remains = reader.ReadCatalogSnapshot(
            request.current_utc, snapshot, read_error);
        failures += !Expect(
            rejected.status ==
                    BackendPackMetadataRefreshStatus::VerificationFailure &&
                previous_remains &&
                snapshot.catalog.catalog_id == "production-2026-08" &&
                snapshot.records.front().manifest.has_value(),
            "a corrupt remote manifest must leave the previous verified catalog readable");
    }
    {
        Fixture fixture;
        fixture.PrepareBase();
        bool saw_candidate = false;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [&](const auto& manifest, const auto& installed,
                const ActiveRuntimeState& candidate) {
                saw_candidate =
                    manifest.kind == BackendPackManifestKind::Base &&
                    manifest.backend == "cpu" &&
                    std::filesystem::is_directory(installed) &&
                    candidate.generation == 1 &&
                    candidate.base_pack_id == "base-v1" &&
                    candidate.packs.empty();
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "CPU base qualified"};
            });
        auto request = fixture.Request();
        request.pack_id = "base-v1";
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.DeliverBase(request, source);
        const auto active = fixture.Active();
        failures += !Expect(
            result.status ==
                    BackendPackLifecycleStatus::InstalledAndActivated &&
                saw_candidate && active.generation == 1 &&
                active.base_pack_id == "base-v1" && active.packs.empty() &&
                std::filesystem::is_regular_file(
                    fixture.root /
                    std::string(CurrentRuntimeBootstrapperExecutableName())) &&
                std::filesystem::is_regular_file(
                    fixture.root /
                    std::string(CurrentProductRemovalFinalizerExecutableName())) &&
                Fixture::ReadByte(
                    fixture.root /
                    std::string(CurrentRuntimeBootstrapperExecutableName())) ==
                    '\0' &&
                Fixture::ReadByte(
                    fixture.root /
                    std::string(CurrentProductRemovalFinalizerExecutableName())) ==
                    '\0',
            "fresh base delivery must verify, stage, publish stable tools, qualify, and initialize generation 1");
    }
    {
        Fixture fixture;
        fixture.PrepareBase();
        BackendPackLifecycleService initial_service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [](const auto&, const auto&, const auto&) {
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "Initial CPU base qualified"};
            });
        auto initial_request = fixture.Request();
        initial_request.pack_id = "base-v1";
        OfflineBackendPackArtifactSource initial_source(fixture.archive);
        const auto initial =
            initial_service.DeliverBase(initial_request, initial_source);

        const auto stable_launcher = fixture.root /
            std::string(CurrentRuntimeBootstrapperExecutableName());
        std::ofstream(stable_launcher, std::ios::binary | std::ios::trunc)
            .put('p');
        fixture.PrepareBase("base-v2", "set-v2", false);
        bool saw_update_candidate = false;
        BackendPackLifecycleService update_service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [&](const auto& manifest, const auto&, const auto& candidate) {
                saw_update_candidate =
                    manifest.pack_id == "base-v2" &&
                    candidate.runtime_set_id == "set-v2" &&
                    candidate.base_pack_id == "base-v2" &&
                    candidate.generation == 2 && candidate.packs.empty();
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "CPU base update qualified"};
            });
        auto update_request = fixture.Request();
        update_request.pack_id = "base-v2";
        OfflineBackendPackArtifactSource update_source(fixture.archive);
        const auto updated = update_service.DeliverBaseUpdate(
            update_request, update_source);
        const auto active = fixture.Active();
        failures += !Expect(
            initial.status ==
                    BackendPackLifecycleStatus::InstalledAndActivated &&
                updated.status ==
                    BackendPackLifecycleStatus::InstalledAndActivated &&
                saw_update_candidate && active.runtime_set_id == "set-v2" &&
                active.base_pack_id == "base-v2" &&
                active.generation == 2 && active.packs.empty() &&
                std::filesystem::is_regular_file(
                    fixture.runtime / "rollback" / "set-v2" /
                    "previous-active-runtime.json") &&
                Fixture::ReadByte(stable_launcher) == 'p',
            "base update must qualify and activate the new CPU runtime without replacing the running stable launcher");
    }
    {
        Fixture fixture;
        fixture.PrepareBase();
        const auto published_launcher = fixture.root /
            std::string(CurrentRuntimeBootstrapperExecutableName());
        std::ofstream(
            published_launcher, std::ios::binary | std::ios::trunc)
            .put('p');
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [&](const auto&, const auto& installed, const auto&) {
                std::ofstream(
                    installed /
                        std::string(
                            CurrentRuntimeBootstrapperExecutableName()),
                    std::ios::binary | std::ios::trunc)
                    .put('x');
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "CPU base qualified before launcher tamper"};
            });
        auto request = fixture.Request();
        request.pack_id = "base-v1";
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.DeliverBase(request, source);
        std::error_code active_error;
        failures += !Expect(
            result.status == BackendPackLifecycleStatus::InstallationFailure &&
                !std::filesystem::exists(
                    fixture.runtime / "active-runtime.json", active_error) &&
                !active_error && Fixture::ReadByte(published_launcher) == 'p',
            "post-qualification launcher tampering must block activation and preserve the previous app-level launcher");
    }
    {
        Fixture fixture;
        fixture.PublishCatalogCache();
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; });
        VerifiedBackendPackCatalogSnapshot snapshot;
        std::string error;
        const bool read = service.ReadCatalogSnapshot(
            "2026-08-14T12:00:00Z", snapshot, error);
        failures += !Expect(
            read && error.empty() &&
                snapshot.catalog.catalog_id == "production-2026-08" &&
                snapshot.catalog_path ==
                    BackendPackCurrentCatalogPath(fixture.runtime) &&
                snapshot.records.size() == 1 &&
                snapshot.records.front().manifest.has_value() &&
                snapshot.records.front().manifest->licenses ==
                    std::vector<std::string>{"arrayfire"} &&
                snapshot.records.front().manifest->compatibility.
                    provider_types ==
                    std::vector<std::string>{"opencl-icd"} &&
                snapshot.records.front().manifest_path ==
                    BackendPackCachedManifestPath(
                        fixture.runtime, "opencl-v1"),
            "the catalog snapshot must verify deterministic cached metadata and preserve consent details");

        std::ofstream(
            BackendPackCachedManifestPath(
                fixture.runtime, "opencl-v1"),
            std::ios::binary | std::ios::app)
            .put('x');
        const bool reread = service.ReadCatalogSnapshot(
            "2026-08-14T12:00:00Z", snapshot, error);
        failures += !Expect(
            reread && error.empty() && snapshot.records.size() == 1 &&
                !snapshot.records.front().manifest.has_value() &&
                !snapshot.records.front().manifest_error.empty(),
            "an invalid cached manifest must disable only its catalog entry");
    }
    {
        Fixture fixture;
        fixture.PublishCatalogCache();
        BackendPackLifecycleService source_service(
            fixture.runtime, fixture.Verifier(), [] { return false; });
        VerifiedBackendPackCatalogSnapshot snapshot;
        std::string error;
        const auto destination = fixture.root / "published-runtime";
        const bool read = source_service.ReadCatalogSnapshot(
            "2026-08-14T12:00:00Z", snapshot, error);
        const bool published = read && PublishVerifiedBackendPackMetadata(
            fixture.trust_path, snapshot, destination, error);
        auto trust = BackendPackTrustStore::Load(
            destination / "trust" / "trusted-keys.json", error);
        bool reread = false;
        VerifiedBackendPackCatalogSnapshot installed_snapshot;
        if (trust) {
            BackendPackLifecycleService installed_service(
                destination,
                BackendPackMetadataVerifier(
                    std::move(*trust), "0.2.0", "win64", "x86_64"),
                [] { return false; });
            reread = installed_service.ReadCatalogSnapshot(
                "2026-08-14T12:00:00Z", installed_snapshot, error);
        }
        failures += !Expect(
            published && reread && installed_snapshot.records.size() == 1 &&
                installed_snapshot.records.front().manifest.has_value() &&
                !PublishVerifiedBackendPackMetadata(
                    fixture.trust_path, snapshot,
                    std::filesystem::path("relative-runtime"), error),
            "verified setup metadata must publish atomically into a distinct absolute runtime cache");
    }
    {
        Fixture fixture;
        bool saw_candidate = false;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [&](const auto&, const auto& installed,
                const ActiveRuntimeState& candidate) {
                saw_candidate = std::filesystem::is_directory(installed) &&
                    candidate.generation == 2 && HasPack(candidate);
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "exact candidate qualified"};
            });
        auto request = fixture.Request();
        request.source = BackendPackDeliverySource::OfflineSibling;
        const auto result = service.Deliver(request);
        failures += !Expect(
            result.status == BackendPackLifecycleStatus::
                                 InstalledAndActivated &&
                saw_candidate && fixture.Active().generation == 2 &&
                HasPack(fixture.Active()) &&
                service.GetProgress().stage ==
                    BackendPackLifecycleStage::Complete,
            "verified delivery must qualify the prospective identity before activation (" +
                std::string(BackendPackLifecycleStatusName(result.status)) +
                ": " + result.message + ")");
        const auto rollback = service.Rollback();
        const auto removal = service.Remove("opencl", "opencl-v1");
        failures += !Expect(
            rollback.status == BackendPackLifecycleStatus::RolledBack &&
                fixture.Active().packs.empty() &&
                removal.status == BackendPackLifecycleStatus::Removed &&
                !std::filesystem::exists(
                    fixture.runtime / "packs" / "opencl" / "opencl-v1"),
            "the lifecycle facade must expose shared rollback and removal workflows (rollback=" +
                std::string(BackendPackLifecycleStatusName(rollback.status)) +
                ": " + rollback.message + ", removal=" +
                BackendPackLifecycleStatusName(removal.status) + ": " +
                removal.message + ")");
    }
    {
        Fixture fixture;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [](const auto&, const auto&, const auto&) {
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::InstalledUnqualified,
                    "operation matrix failed"};
            });
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.Deliver(fixture.Request(), source);
        failures += !Expect(
            result.status ==
                    BackendPackLifecycleStatus::InstalledUnqualified &&
                fixture.Active().generation == 1 &&
                fixture.Active().packs.empty() &&
                std::filesystem::is_directory(
                    fixture.runtime / "packs" / "opencl" / "opencl-v1"),
            "failed qualification must leave a complete installed pack inactive");
    }
    {
        Fixture fixture;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [](const auto&, const auto&, const auto&) {
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::RollbackRequired,
                    "release policy requires rollback"};
            });
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.Deliver(fixture.Request(), source);
        failures += !Expect(
            result.status == BackendPackLifecycleStatus::RolledBack &&
                fixture.Active().generation == 1 &&
                fixture.Active().packs.empty() &&
                !std::filesystem::exists(
                    fixture.runtime / "packs" / "opencl" / "opencl-v1"),
            "rollback-required qualification must remove a new inactive candidate and retain the old runtime");
    }
    {
        Fixture fixture;
        fixture.WriteMetadata("diagnostic");
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [](const auto&, const auto&, const auto&) {
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "diagnostic route passed"};
            });
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.Deliver(fixture.Request(), source);
        failures += !Expect(
            result.status ==
                    BackendPackLifecycleStatus::InstalledUnqualified &&
                fixture.Active().packs.empty(),
            "diagnostic catalog policy must prevent normal activation even when a local probe passes");
    }
    {
        Fixture fixture;
        fixture.WriteMetadata("blocked");
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; });
        VerifiedBackendPackCatalog catalog;
        std::string error;
        const bool read = service.ReadCatalog(
            fixture.catalog_path, "2026-08-14T12:00:00Z", catalog, error);
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.Deliver(fixture.Request(), source);
        failures += !Expect(
            read && catalog.packs.size() == 1 &&
                result.status ==
                    BackendPackLifecycleStatus::PolicyRejected &&
                !std::filesystem::exists(
                    fixture.runtime / "cache" / "artifacts"),
            "verified blocked catalog entries must remain browsable but must not acquire or install artifacts");
    }
    {
        Fixture fixture;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [](const auto&, const auto&, const auto&) {
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "candidate qualified"};
            });
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto installed = service.Deliver(fixture.Request(), source);
        const auto installed_file = fixture.runtime / "packs" / "opencl" /
            "opencl-v1" / "runtime" / "afopencl.dll";
        std::ofstream(installed_file, std::ios::binary | std::ios::trunc)
            .put('x');
        auto repair_request = fixture.Request();
        repair_request.repair = true;
        const auto repaired = service.Deliver(repair_request, source);
        failures += !Expect(
            installed.status == BackendPackLifecycleStatus::
                                    InstalledAndActivated &&
                repaired.status == BackendPackLifecycleStatus::
                                   InstalledAndActivated &&
                fixture.Active().generation == 4 &&
                HasPack(fixture.Active()) &&
                Fixture::ReadByte(installed_file) == '\0',
            "lifecycle repair must replace a corrupt active pack through a CPU-only intermediate state and requalify before reactivation (" +
                std::string(BackendPackLifecycleStatusName(repaired.status)) +
                ": " + repaired.message + ")");
    }
    {
        Fixture fixture;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; },
            [&](const auto&, const auto&, const auto&) {
                auto changed = fixture.Active();
                ++changed.generation;
                std::string error;
                SaveActiveRuntimeStateAtomic(
                    fixture.runtime / "active-runtime.json", changed, error);
                return BackendPackQualificationDecision{
                    BackendPackQualificationDisposition::Qualified,
                    "candidate qualified before concurrent state change"};
            });
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.Deliver(fixture.Request(), source);
        failures += !Expect(
            result.status ==
                    BackendPackLifecycleStatus::InstalledUnqualified &&
                fixture.Active().generation == 2 &&
                fixture.Active().packs.empty(),
            "runtime changes during qualification must make evidence stale and block activation (" +
                std::string(BackendPackLifecycleStatusName(result.status)) +
                ": " + result.message + ")");
    }
    {
        Fixture fixture;
        BackendPackLifecycleService* active_service = nullptr;
        BackendPackLifecycleService service(
            fixture.runtime, fixture.Verifier(), [] { return false; }, {},
            [&](const BackendPackLifecycleProgress& progress) {
                if (active_service &&
                    progress.stage == BackendPackLifecycleStage::Acquiring &&
                    progress.completed_bytes > 0) {
                    active_service->Cancel();
                }
            });
        active_service = &service;
        auto request = fixture.Request();
        request.discard_operation_data_on_cancel = true;
        OfflineBackendPackArtifactSource source(fixture.archive);
        const auto result = service.Deliver(request, source);
        const auto artifact = fixture.runtime / "cache" / "artifacts" /
            "opencl-v1" / "opencl-v1.zip";
        auto partial = artifact;
        partial += ".part";
        failures += !Expect(
            result.status == BackendPackLifecycleStatus::Interrupted &&
                !std::filesystem::exists(artifact) &&
                !std::filesystem::exists(partial) &&
                !std::filesystem::exists(
                    fixture.runtime / "packs" / "opencl" / "opencl-v1") &&
                fixture.Active().packs.empty(),
            "explicit installer cancellation must discard partial, cached, staged, and unpublished pack data");
    }

    if (failures == 0) {
        std::cout << "backend pack lifecycle service contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
