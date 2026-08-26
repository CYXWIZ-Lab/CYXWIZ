#include "product_removal_request.h"

#include "atomic_file_publisher.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;

constexpr std::uintmax_t kMaximumRequestBytes = 64 * 1024;
constexpr std::string_view kRequestKind = "cyxwiz-product-removal";

class RemoveTemporaryRequest {
public:
    explicit RemoveTemporaryRequest(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~RemoveTemporaryRequest() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

private:
    std::filesystem::path path_;
};

std::string PathUtf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::filesystem::path Utf8Path(const std::string& value) {
    const std::u8string utf8(
        reinterpret_cast<const char8_t*>(value.data()), value.size());
    return std::filesystem::path(utf8);
}

Json RequestDocument(const ProductRemovalAuthorization& authorization) {
    Json packs = Json::array();
    for (const auto& pack : authorization.runtime.packs) {
        packs.push_back({
            {"backend", pack.backend},
            {"pack_id", pack.pack_id},
        });
    }
    return {
        {"schema_version", std::uint64_t{1}},
        {"kind", kRequestKind},
        {"install_root", PathUtf8(authorization.install_root)},
        {"scope", ProductInstallScopeName(authorization.scope)},
        {"install_id", authorization.install_id},
        {"runtime", {
            {"runtime_set_id", authorization.runtime.runtime_set_id},
            {"generation", authorization.runtime.generation},
            {"base_pack_id", authorization.runtime.base_pack_id},
            {"packs", std::move(packs)},
        }},
    };
}

bool HasExactKeys(
    const Json& value,
    std::initializer_list<const char*> keys) {
    if (!value.is_object() || value.size() != keys.size()) return false;
    for (const char* key : keys) {
        if (!value.contains(key)) return false;
    }
    return true;
}

bool ReadRequestFile(
    const std::filesystem::path& path,
    ProductRemovalAuthorization& authorization,
    std::string& error) {
    authorization = {};
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(
        path, filesystem_error);
    if (filesystem_error ||
        status.type() != std::filesystem::file_type::regular) {
        error = "The product removal request is missing or not a regular file";
        return false;
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size == 0 || size > kMaximumRequestBytes) {
        error = "The product removal request violates its byte bound";
        return false;
    }
    std::ifstream stream(path, std::ios::binary);
    Json document = Json::parse(stream, nullptr, false);
    if (stream.bad() || document.is_discarded() ||
        !HasExactKeys(document, {
            "schema_version", "kind", "install_root", "scope",
            "install_id", "runtime"}) ||
        !document["schema_version"].is_number_unsigned() ||
        document["schema_version"].get<std::uint64_t>() != 1 ||
        !document["kind"].is_string() ||
        document["kind"].get<std::string>() != kRequestKind ||
        !document["install_root"].is_string() ||
        !document["scope"].is_string() ||
        !document["install_id"].is_string() ||
        !HasExactKeys(document["runtime"], {
            "runtime_set_id", "generation", "base_pack_id", "packs"})) {
        error = "The product removal request schema is invalid";
        return false;
    }
    const auto& runtime = document["runtime"];
    if (!runtime["runtime_set_id"].is_string() ||
        !runtime["generation"].is_number_unsigned() ||
        !runtime["base_pack_id"].is_string() ||
        !runtime["packs"].is_array()) {
        error = "The product removal runtime identity schema is invalid";
        return false;
    }

    authorization.install_root = Utf8Path(
        document["install_root"].get<std::string>());
    authorization.install_id = document["install_id"].get<std::string>();
    if (!ParseProductInstallScope(
            document["scope"].get<std::string>(), authorization.scope)) {
        error = "The product removal request scope is invalid";
        authorization = {};
        return false;
    }
    authorization.runtime.runtime_set_id =
        runtime["runtime_set_id"].get<std::string>();
    authorization.runtime.generation =
        runtime["generation"].get<std::uint64_t>();
    authorization.runtime.base_pack_id =
        runtime["base_pack_id"].get<std::string>();
    for (const auto& pack : runtime["packs"]) {
        if (!HasExactKeys(pack, {"backend", "pack_id"}) ||
            !pack["backend"].is_string() ||
            !pack["pack_id"].is_string()) {
            error = "The product removal pack identity schema is invalid";
            authorization = {};
            return false;
        }
        authorization.runtime.packs.push_back({
            pack["backend"].get<std::string>(),
            pack["pack_id"].get<std::string>(),
        });
    }
    return true;
}

bool SameAuthorization(
    const ProductRemovalAuthorization& left,
    const ProductRemovalAuthorization& right) {
    if (left.install_root != right.install_root ||
        left.scope != right.scope || left.install_id != right.install_id ||
        left.runtime.runtime_set_id != right.runtime.runtime_set_id ||
        left.runtime.generation != right.runtime.generation ||
        left.runtime.base_pack_id != right.runtime.base_pack_id ||
        left.runtime.packs.size() != right.runtime.packs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.runtime.packs.size(); ++index) {
        if (left.runtime.packs[index].backend !=
                right.runtime.packs[index].backend ||
            left.runtime.packs[index].pack_id !=
                right.runtime.packs[index].pack_id) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::filesystem::path ProductRemovalRequestPath(
    const std::filesystem::path& install_root) {
    return install_root / ".cyxwiz-removal-request.json";
}

bool LoadProductRemovalRequest(
    const std::filesystem::path& install_root,
    ProductRemovalAuthorization& authorization,
    std::string& error) {
    if (!ReadRequestFile(
            ProductRemovalRequestPath(install_root), authorization, error)) {
        return false;
    }
    if (authorization.install_root != install_root) {
        authorization = {};
        error = "The product removal request belongs to another root";
        return false;
    }
    if (!ValidateProductRemovalAuthorization(authorization, error)) {
        authorization = {};
        error = "The product removal request is stale or invalid: " + error;
        return false;
    }
    error.clear();
    return true;
}

bool QueueProductRemovalRequest(
    const std::filesystem::path& install_root,
    ProductInstallScope scope,
    ProductRemovalAuthorization& authorization,
    std::string& error) {
    authorization = {};
    ProductRemovalAuthorization captured;
    if (!CaptureProductRemovalAuthorization(
            install_root, scope, captured, error)) {
        return false;
    }
    const auto temporary = install_root /
        (".cyxwiz-removal-request-source-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    RemoveTemporaryRequest cleanup(temporary);
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = "Cannot create the product removal request source";
        return false;
    }
    stream << RequestDocument(captured).dump(2) << '\n';
    stream.flush();
    if (!stream) {
        error = "Cannot write the complete product removal request";
        return false;
    }
    stream.close();
    if (!PublishRegularFileAtomic(
            temporary, ProductRemovalRequestPath(install_root),
            kMaximumRequestBytes, error,
            [&](const std::filesystem::path& candidate,
                std::string& validation_error) {
                ProductRemovalAuthorization parsed;
                if (!ReadRequestFile(
                        candidate, parsed, validation_error)) {
                    return false;
                }
                if (!SameAuthorization(parsed, captured)) {
                    validation_error =
                        "The temporary product removal request changed unexpectedly";
                    return false;
                }
                return true;
            })) {
        return false;
    }
    if (!LoadProductRemovalRequest(install_root, authorization, error) ||
        !SameAuthorization(authorization, captured)) {
        if (error.empty()) {
            error = "The published product removal request changed unexpectedly";
        }
        authorization = {};
        return false;
    }
    return true;
}

}  // namespace cyxwiz::runtime
