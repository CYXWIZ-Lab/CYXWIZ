#include "backend_pack_hash.h"
#include "backend_pack_lifecycle_service.h"

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

bool WriteZip(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    archive* writer = archive_write_new();
    if (!writer || archive_write_set_format_zip(writer) != ARCHIVE_OK ||
        archive_write_open_filename_w(writer, path.c_str()) != ARCHIVE_OK) {
        if (writer) archive_write_free(writer);
        return false;
    }
    for (const std::string entry_path : {
             "runtime/afopencl.dll",
             "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt"}) {
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
        Touch(runtime / "base" / "base-v1" / "cyxwiz-engine.exe");
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

}  // namespace

int main() {
    int failures = 0;
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
            "verified delivery must qualify the prospective identity before activation");
        const auto rollback = service.Rollback();
        const auto removal = service.Remove("opencl", "opencl-v1");
        failures += !Expect(
            rollback.status == BackendPackLifecycleStatus::RolledBack &&
                fixture.Active().packs.empty() &&
                removal.status == BackendPackLifecycleStatus::Removed &&
                !std::filesystem::exists(
                    fixture.runtime / "packs" / "opencl" / "opencl-v1"),
            "the lifecycle facade must expose shared rollback and removal workflows");
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
            "lifecycle repair must replace a corrupt active pack through a CPU-only intermediate state and requalify before reactivation");
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
            "runtime changes during qualification must make evidence stale and block activation");
    }

    if (failures == 0) {
        std::cout << "backend pack lifecycle service contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
