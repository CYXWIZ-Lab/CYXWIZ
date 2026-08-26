#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

enum class ProductInstallScope {
    CurrentUser,
    AllUsers,
};

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

}  // namespace cyxwiz::runtime
