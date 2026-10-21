#pragma once

#include "product_removal_handoff.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime::detail {

bool LaunchDetachedProductRemovalFinalizer(
    const std::filesystem::path& source_finalizer,
    const std::filesystem::path& install_root,
    std::string_view install_id,
    ProductRemovalHandoff& handoff,
    std::string& error);

void CloseProductRemovalLifetimeToken(std::uintptr_t token) noexcept;

}  // namespace cyxwiz::runtime::detail
