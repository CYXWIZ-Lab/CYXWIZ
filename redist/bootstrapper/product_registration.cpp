#include "product_registration.h"

#include "backend_pack_platform.h"
#include "product_registration_internal.h"

#include <algorithm>
#include <cctype>
#include <string_view>
#include <system_error>

namespace cyxwiz::runtime {
namespace {

bool IsSafeVersion(std::string_view value) {
    return !value.empty() && value.size() <= 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '.' ||
                   character == '_' || character == '+' || character == '-';
        });
}

std::string ValidateRequest(const ProductRegistrationRequest& request) {
    if (!request.install_root.is_absolute() ||
        !request.runtime_root.is_absolute() ||
        request.install_root != request.install_root.lexically_normal() ||
        request.runtime_root != request.runtime_root.lexically_normal() ||
        request.install_root == request.install_root.root_path() ||
        request.runtime_root != request.install_root / "runtime" ||
        !IsSafeVersion(request.product_version)) {
        return "A normalized absolute product root, matching runtime root, and safe version are required";
    }
#ifndef _WIN32
    const auto path_text = request.install_root.native();
    if (path_text.find('\n') != std::string::npos ||
        path_text.find('\r') != std::string::npos) {
        return "The product root contains unsupported control characters";
    }
#endif
    const auto launcher = request.install_root /
        std::string(CurrentRuntimeBootstrapperExecutableName());
    std::error_code error;
    if (std::filesystem::symlink_status(launcher, error).type() !=
            std::filesystem::file_type::regular ||
        error) {
        return "The verified stable CyxWiz launcher is missing";
    }
    return {};
}

}  // namespace

ProductRegistrationResult RegisterInstalledProduct(
    const ProductRegistrationRequest& request) {
    const auto validation = ValidateRequest(request);
    if (!validation.empty()) return {false, validation};
    return detail::RegisterPlatformProduct(request);
}

ProductUnregistrationResult UnregisterInstalledProduct(
    const ProductRegistrationRequest& request) {
    const auto validation = ValidateRequest(request);
    if (!validation.empty()) return {false, validation};
    return detail::UnregisterPlatformProduct(request);
}

}  // namespace cyxwiz::runtime
