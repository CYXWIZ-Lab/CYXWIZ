#include "product_removal_handoff.h"

#include "backend_pack_platform.h"
#include "product_removal_handoff_platform.h"
#include "product_removal_request.h"

#include <system_error>
#include <utility>

namespace cyxwiz::runtime {

ProductRemovalParentLifetime::ProductRemovalParentLifetime(
    std::uintptr_t native_token)
    : native_token_(native_token) {}

ProductRemovalParentLifetime::~ProductRemovalParentLifetime() {
    Close();
}

ProductRemovalParentLifetime::ProductRemovalParentLifetime(
    ProductRemovalParentLifetime&& other) noexcept
    : native_token_(std::exchange(other.native_token_, 0)) {}

ProductRemovalParentLifetime& ProductRemovalParentLifetime::operator=(
    ProductRemovalParentLifetime&& other) noexcept {
    if (this != &other) {
        Close();
        native_token_ = std::exchange(other.native_token_, 0);
    }
    return *this;
}

bool ProductRemovalParentLifetime::valid() const noexcept {
    return native_token_ != 0;
}

void ProductRemovalParentLifetime::Close() noexcept {
    if (native_token_ == 0) return;
    detail::CloseProductRemovalLifetimeToken(native_token_);
    native_token_ = 0;
}

void ProductRemovalParentLifetime::PreserveUntilProcessExit() noexcept {
    // Deliberately transfer ownership to the process. The OS closes the native
    // token only when this bootstrapper exits, which produces finalizer EOF.
    native_token_ = 0;
}

ProductRemovalHandoff SchedulePendingProductRemoval(
    const std::filesystem::path& install_root,
    std::string& error) {
    ProductRemovalHandoff output;
    std::error_code filesystem_error;
    const auto request_status = std::filesystem::symlink_status(
        ProductRemovalRequestPath(install_root), filesystem_error);
    if ((filesystem_error == std::errc::no_such_file_or_directory) ||
        (!filesystem_error && request_status.type() ==
             std::filesystem::file_type::not_found)) {
        error.clear();
        return output;
    }
    if (filesystem_error || request_status.type() !=
            std::filesystem::file_type::regular) {
        output.status = ProductRemovalHandoffStatus::Rejected;
        error = "The pending product removal request is redirected or invalid";
        return output;
    }

    ProductRemovalAuthorization authorization;
    if (!LoadProductRemovalRequest(install_root, authorization, error)) {
        output.status = ProductRemovalHandoffStatus::Rejected;
        return output;
    }
    const auto finalizer = install_root /
        std::string(CurrentProductRemovalFinalizerExecutableName());
    const auto finalizer_status = std::filesystem::symlink_status(
        finalizer, filesystem_error);
    if (filesystem_error || finalizer_status.type() !=
            std::filesystem::file_type::regular) {
        output.status = ProductRemovalHandoffStatus::Rejected;
        error = "The product removal finalizer is missing or redirected";
        return output;
    }
    if (!detail::LaunchDetachedProductRemovalFinalizer(
            finalizer, install_root, authorization.install_id,
            output, error)) {
        output = {};
        output.status = ProductRemovalHandoffStatus::Rejected;
        return output;
    }
    output.status = ProductRemovalHandoffStatus::Scheduled;
    error.clear();
    return output;
}

}  // namespace cyxwiz::runtime
