#include "product_removal_finalizer.h"

#include "product_removal_request.h"

#include <limits>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <unistd.h>
#endif

namespace cyxwiz::runtime {
namespace {

#ifdef _WIN32
class OwnedLifetimeToken {
public:
    explicit OwnedLifetimeToken(HANDLE handle) : handle_(handle) {}
    ~OwnedLifetimeToken() {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle_);
        }
    }
    HANDLE get() const { return handle_; }

private:
    HANDLE handle_ = nullptr;
};

bool WaitForParentExit(std::uintptr_t token, std::string& error) {
    const auto handle = reinterpret_cast<HANDLE>(token);
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
        error = "The parent lifetime handle is invalid";
        return false;
    }
    OwnedLifetimeToken owned(handle);
    char byte = 0;
    DWORD bytes_read = 0;
    if (::ReadFile(owned.get(), &byte, 1, &bytes_read, nullptr)) {
        error = bytes_read == 0
            ? "The parent lifetime pipe ended without an EOF signal"
            : "The parent lifetime pipe contained unexpected data";
        return false;
    }
    const DWORD code = ::GetLastError();
    if (code != ERROR_BROKEN_PIPE) {
        error = "Waiting for the parent lifetime boundary failed; Win32 error " +
            std::to_string(code);
        return false;
    }
    return true;
}
#else
class OwnedLifetimeToken {
public:
    explicit OwnedLifetimeToken(int descriptor) : descriptor_(descriptor) {}
    ~OwnedLifetimeToken() {
        if (descriptor_ >= 0) ::close(descriptor_);
    }
    int get() const { return descriptor_; }

private:
    int descriptor_ = -1;
};

bool WaitForParentExit(std::uintptr_t token, std::string& error) {
    if (token > static_cast<std::uintptr_t>(std::numeric_limits<int>::max())) {
        error = "The parent lifetime descriptor is invalid";
        return false;
    }
    const int descriptor = static_cast<int>(token);
    if (descriptor < 0) {
        error = "The parent lifetime descriptor is invalid";
        return false;
    }
    OwnedLifetimeToken owned(descriptor);
    char byte = 0;
    for (;;) {
        const auto read_result = ::read(owned.get(), &byte, 1);
        if (read_result == 0) return true;
        if (read_result > 0) {
            error = "The parent lifetime pipe contained unexpected data";
            return false;
        }
        if (errno == EINTR) continue;
        error = "Waiting for the parent lifetime boundary failed: " +
            std::string(std::strerror(errno));
        return false;
    }
}
#endif

}  // namespace

bool AwaitAuthorizedProductRemoval(
    const std::filesystem::path& install_root,
    std::uintptr_t parent_lifetime_token,
    ProductRemovalAuthorization& authorization,
    std::string& error) {
    authorization = {};
    if (!WaitForParentExit(parent_lifetime_token, error)) return false;
    if (!LoadProductRemovalRequest(install_root, authorization, error)) {
        error = "Product removal was rejected after the parent exited: " + error;
        return false;
    }
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
