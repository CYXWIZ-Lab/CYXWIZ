#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

enum class ProductRemovalHandoffStatus {
    NoRequest,
    Scheduled,
    Rejected,
};

class ProductRemovalParentLifetime {
public:
    ProductRemovalParentLifetime() = default;
    explicit ProductRemovalParentLifetime(std::uintptr_t native_token);
    ~ProductRemovalParentLifetime();

    ProductRemovalParentLifetime(const ProductRemovalParentLifetime&) = delete;
    ProductRemovalParentLifetime& operator=(
        const ProductRemovalParentLifetime&) = delete;
    ProductRemovalParentLifetime(ProductRemovalParentLifetime&& other) noexcept;
    ProductRemovalParentLifetime& operator=(
        ProductRemovalParentLifetime&& other) noexcept;

    bool valid() const noexcept;
    void Close() noexcept;
    void PreserveUntilProcessExit() noexcept;

private:
    std::uintptr_t native_token_ = 0;
};

struct ProductRemovalHandoff {
    ProductRemovalHandoffStatus status =
        ProductRemovalHandoffStatus::NoRequest;
    std::filesystem::path staged_finalizer;
    std::filesystem::path result_path;
    ProductRemovalParentLifetime parent_lifetime;
};

ProductRemovalHandoff SchedulePendingProductRemoval(
    const std::filesystem::path& install_root,
    std::string& error);

}  // namespace cyxwiz::runtime
