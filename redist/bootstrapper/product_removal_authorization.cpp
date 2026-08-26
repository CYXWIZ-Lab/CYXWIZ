#include "product_removal_authorization.h"

#include "backend_pack_platform.h"

#include <algorithm>
#include <system_error>

namespace cyxwiz::runtime {
namespace {

bool IsExactDirectory(
    const std::filesystem::path& path,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(
        path, filesystem_error);
    if (filesystem_error ||
        status.type() != std::filesystem::file_type::directory) {
        error = "The product root is missing, redirected, or not a directory";
        return false;
    }
    const auto canonical = std::filesystem::canonical(
        path, filesystem_error);
    if (filesystem_error || canonical != path) {
        error = "The product root does not resolve to its exact normalized path";
        return false;
    }
    return true;
}

bool IsExactRegularFile(
    const std::filesystem::path& path,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(
        path, filesystem_error);
    if (filesystem_error ||
        status.type() != std::filesystem::file_type::regular) {
        error = "The verified stable CyxWiz launcher is missing or redirected";
        return false;
    }
    return true;
}

ActiveRuntimeState RuntimeIdentity(const ActiveRuntime& runtime) {
    ActiveRuntimeState identity;
    identity.runtime_set_id = runtime.runtime_set_id;
    identity.generation = runtime.generation;
    identity.base_pack_id = runtime.base_pack_id;
    for (const auto& pack : runtime.packs) {
        identity.packs.push_back({pack.backend, pack.pack_id});
    }
    return identity;
}

bool SameRuntimeIdentity(
    const ActiveRuntimeState& left,
    const ActiveRuntimeState& right) {
    return left.runtime_set_id == right.runtime_set_id &&
        left.generation == right.generation &&
        left.base_pack_id == right.base_pack_id &&
        left.packs.size() == right.packs.size() &&
        std::equal(
            left.packs.begin(), left.packs.end(), right.packs.begin(),
            [](const ActivePackState& first,
                const ActivePackState& second) {
                return first.backend == second.backend &&
                    first.pack_id == second.pack_id;
            });
}

}  // namespace

bool CaptureProductRemovalAuthorization(
    const std::filesystem::path& install_root,
    ProductInstallScope scope,
    ProductRemovalAuthorization& authorization,
    std::string& error) {
    authorization = {};
    if (!install_root.is_absolute() ||
        install_root != install_root.lexically_normal() ||
        install_root == install_root.root_path()) {
        error = "A normalized absolute non-root product path is required";
        return false;
    }
    if (!IsExactDirectory(install_root, error) ||
        !IsExactRegularFile(
            install_root /
                std::string(CurrentRuntimeBootstrapperExecutableName()),
            error)) {
        return false;
    }

    ProductInstallationReceipt receipt;
    if (!LoadProductInstallationReceipt(
            install_root, receipt, error)) {
        return false;
    }
    if (receipt.scope != scope) {
        error = "The product removal scope does not match its receipt";
        return false;
    }

    const auto runtime_root = install_root / "runtime";
    ActiveRuntime active;
    if (!ResolveActiveRuntime(runtime_root, active, error)) {
        error = "The active product runtime is invalid: " + error;
        return false;
    }
    if (active.runtime_root != runtime_root) {
        error = "The active product runtime is redirected";
        return false;
    }

    authorization.install_root = install_root;
    authorization.scope = scope;
    authorization.install_id = receipt.install_id;
    authorization.runtime = RuntimeIdentity(active);
    error.clear();
    return true;
}

bool ValidateProductRemovalAuthorization(
    const ProductRemovalAuthorization& authorization,
    std::string& error) {
    ProductRemovalAuthorization current;
    if (!CaptureProductRemovalAuthorization(
            authorization.install_root, authorization.scope,
            current, error)) {
        return false;
    }
    if (current.install_id != authorization.install_id) {
        error = "The product installation identity changed before removal";
        return false;
    }
    if (!SameRuntimeIdentity(current.runtime, authorization.runtime)) {
        error = "The active runtime changed before removal";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
