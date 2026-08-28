#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

enum class ProductInstallScope {
    CurrentUser,
    AllUsers,
};

std::string_view ProductInstallScopeName(ProductInstallScope scope);
bool ParseProductInstallScope(
    std::string_view value,
    ProductInstallScope& scope);

struct ProductInstallationReceipt {
    std::string install_id;
    std::filesystem::path install_root;
    ProductInstallScope scope = ProductInstallScope::CurrentUser;
};

std::filesystem::path ProductInstallationReceiptPath(
    const std::filesystem::path& install_root);

bool PublishProductInstallationReceipt(
    const std::filesystem::path& install_root,
    ProductInstallScope scope,
    ProductInstallationReceipt& receipt,
    std::string& error);

bool LoadProductInstallationReceipt(
    const std::filesystem::path& install_root,
    ProductInstallationReceipt& receipt,
    std::string& error);

bool LoadRelocatedProductInstallationReceipt(
    const std::filesystem::path& relocated_root,
    const std::filesystem::path& original_install_root,
    ProductInstallationReceipt& receipt,
    std::string& error);

}  // namespace cyxwiz::runtime
