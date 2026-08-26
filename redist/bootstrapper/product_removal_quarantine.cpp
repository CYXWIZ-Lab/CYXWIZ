#include "product_removal_quarantine.h"

#include "product_installation_receipt.h"

#include <algorithm>
#include <string_view>
#include <system_error>
#include <utility>

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
        error = "The quarantined product root is missing or redirected";
        return false;
    }
    const auto canonical = std::filesystem::canonical(
        path, filesystem_error);
    if (filesystem_error || canonical != path) {
        error = "The quarantined product root is not its exact path";
        return false;
    }
    return true;
}

bool IsExpectedQuarantine(
    const QuarantinedProductInstallation& quarantined) {
    const auto valid_install_id =
        quarantined.install_id.size() == 32 &&
        std::all_of(
            quarantined.install_id.begin(), quarantined.install_id.end(),
            [](char character) {
                return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
            });
    if (!quarantined.original_root.is_absolute() ||
        quarantined.original_root !=
            quarantined.original_root.lexically_normal() ||
        quarantined.original_root == quarantined.original_root.root_path() ||
        !valid_install_id) {
        return false;
    }
    ProductRemovalAuthorization expected;
    expected.install_root = quarantined.original_root;
    expected.install_id = quarantined.install_id;
    return ProductRemovalQuarantinePath(expected) ==
        quarantined.quarantine_root;
}

}  // namespace

std::filesystem::path ProductRemovalQuarantinePath(
    const ProductRemovalAuthorization& authorization) {
    return authorization.install_root.parent_path() /
        (".cyxwiz-removing-" + authorization.install_id);
}

bool ValidateQuarantinedProductInstallation(
    const QuarantinedProductInstallation& quarantined,
    std::string& error) {
    if (!IsExpectedQuarantine(quarantined)) {
        error = "The product quarantine identity is invalid";
        return false;
    }
    if (!IsExactDirectory(quarantined.quarantine_root, error)) {
        return false;
    }
    ProductInstallationReceipt receipt;
    if (!LoadRelocatedProductInstallationReceipt(
            quarantined.quarantine_root, quarantined.original_root,
            receipt, error)) {
        return false;
    }
    if (receipt.install_id != quarantined.install_id ||
        receipt.scope != quarantined.scope) {
        error = "The quarantined product receipt identity changed";
        return false;
    }
    error.clear();
    return true;
}

bool QuarantineProductInstallation(
    const ProductRemovalAuthorization& authorization,
    QuarantinedProductInstallation& quarantined,
    std::string& error) {
    quarantined = {};
    if (!ValidateProductRemovalAuthorization(authorization, error)) {
        return false;
    }
    const auto quarantine_root =
        ProductRemovalQuarantinePath(authorization);
    std::error_code filesystem_error;
    const auto quarantine_status = std::filesystem::symlink_status(
        quarantine_root, filesystem_error);
    if ((!filesystem_error && quarantine_status.type() !=
             std::filesystem::file_type::not_found) ||
        (filesystem_error &&
         filesystem_error != std::errc::no_such_file_or_directory)) {
        error = "The product removal quarantine already exists or cannot be inspected";
        return false;
    }

    std::filesystem::rename(
        authorization.install_root, quarantine_root, filesystem_error);
    if (filesystem_error) {
        error = "Cannot atomically quarantine the product installation: " +
            filesystem_error.message();
        return false;
    }

    QuarantinedProductInstallation candidate;
    candidate.original_root = authorization.install_root;
    candidate.quarantine_root = quarantine_root;
    candidate.install_id = authorization.install_id;
    candidate.scope = authorization.scope;
    std::string validation_error;
    if (!ValidateQuarantinedProductInstallation(
            candidate, validation_error)) {
        std::error_code rollback_error;
        std::filesystem::rename(
            quarantine_root, authorization.install_root, rollback_error);
        error = "The quarantined product failed validation: " +
            validation_error;
        if (rollback_error) {
            error += "; rollback also failed: " + rollback_error.message();
            quarantined = std::move(candidate);
        }
        return false;
    }
    quarantined = std::move(candidate);
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
