#include "product_installation_receipt.h"

#include "atomic_file_publisher.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;

constexpr std::uintmax_t kMaximumReceiptBytes = 16 * 1024;
constexpr std::string_view kReceiptKind = "cyxwiz-product-installation";

class RemoveTemporaryReceipt {
public:
    explicit RemoveTemporaryReceipt(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~RemoveTemporaryReceipt() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

private:
    std::filesystem::path path_;
};

bool IsNormalizedProductRoot(const std::filesystem::path& path) {
    return path.is_absolute() && path == path.lexically_normal() &&
        path != path.root_path();
}

bool IsInstallId(std::string_view value) {
    return value.size() == 32 &&
        std::all_of(value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        });
}

std::string PathUtf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::filesystem::path Utf8Path(const std::string& value) {
    const std::u8string utf8(
        reinterpret_cast<const char8_t*>(value.data()), value.size());
    return std::filesystem::path(utf8);
}

Json ReceiptDocument(const ProductInstallationReceipt& receipt) {
    return {
        {"schema_version", std::uint64_t{1}},
        {"kind", kReceiptKind},
        {"install_id", receipt.install_id},
        {"install_root", PathUtf8(receipt.install_root)},
        {"scope", ProductInstallScopeName(receipt.scope)},
    };
}

bool ReadReceiptFile(
    const std::filesystem::path& path,
    ProductInstallationReceipt& receipt,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error ||
        status.type() != std::filesystem::file_type::regular) {
        error = "The product installation receipt is missing or not a regular file";
        return false;
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size == 0 || size > kMaximumReceiptBytes) {
        error = "The product installation receipt violates its byte bound";
        return false;
    }
    std::ifstream stream(path, std::ios::binary);
    Json document = Json::parse(stream, nullptr, false);
    if (stream.bad() || document.is_discarded() || !document.is_object() ||
        document.size() != 5 || !document.contains("schema_version") ||
        !document.contains("kind") || !document.contains("install_id") ||
        !document.contains("install_root") || !document.contains("scope") ||
        !document["schema_version"].is_number_unsigned() ||
        document["schema_version"].get<std::uint64_t>() != 1 ||
        !document["kind"].is_string() ||
        document["kind"].get<std::string>() != kReceiptKind ||
        !document["install_id"].is_string() ||
        !document["install_root"].is_string()) {
        error = "The product installation receipt schema is invalid";
        return false;
    }
    ProductInstallationReceipt parsed;
    parsed.install_id = document["install_id"].get<std::string>();
    parsed.install_root = Utf8Path(
        document["install_root"].get<std::string>());
    if (!IsInstallId(parsed.install_id) ||
        !IsNormalizedProductRoot(parsed.install_root) ||
        !document["scope"].is_string() ||
        !ParseProductInstallScope(
            document["scope"].get<std::string>(), parsed.scope)) {
        error = "The product installation receipt values are invalid";
        return false;
    }
    receipt = std::move(parsed);
    return true;
}

bool GenerateInstallId(std::string& output, std::string& error) {
    try {
        std::random_device random;
        if (random.entropy() <= 0.0) {
            error = "The platform random source cannot issue an installation identity";
            return false;
        }
        std::ostringstream encoded;
        encoded << std::hex << std::setfill('0');
        for (int index = 0; index < 4; ++index) {
            encoded << std::setw(8) <<
                static_cast<std::uint32_t>(random());
        }
        output = encoded.str();
    } catch (const std::exception& exception) {
        error = "Cannot issue an installation identity: " +
            std::string(exception.what());
        return false;
    }
    if (!IsInstallId(output)) {
        error = "The platform random source returned an invalid installation identity";
        return false;
    }
    return true;
}

}  // namespace

std::string_view ProductInstallScopeName(ProductInstallScope scope) {
    switch (scope) {
    case ProductInstallScope::CurrentUser:
        return "current_user";
    case ProductInstallScope::AllUsers:
        return "all_users";
    }
    return {};
}

bool ParseProductInstallScope(
    std::string_view value,
    ProductInstallScope& scope) {
    if (value == "current_user") {
        scope = ProductInstallScope::CurrentUser;
        return true;
    }
    if (value == "all_users") {
        scope = ProductInstallScope::AllUsers;
        return true;
    }
    return false;
}

std::filesystem::path ProductInstallationReceiptPath(
    const std::filesystem::path& install_root) {
    return install_root / ".cyxwiz-installation.json";
}

bool LoadProductInstallationReceipt(
    const std::filesystem::path& install_root,
    ProductInstallationReceipt& receipt,
    std::string& error) {
    receipt = {};
    if (!IsNormalizedProductRoot(install_root)) {
        error = "A normalized absolute product root is required";
        return false;
    }
    if (!ReadReceiptFile(
            ProductInstallationReceiptPath(install_root), receipt, error)) {
        return false;
    }
    if (receipt.install_root != install_root) {
        receipt = {};
        error = "The product installation receipt belongs to another root";
        return false;
    }
    error.clear();
    return true;
}

bool LoadRelocatedProductInstallationReceipt(
    const std::filesystem::path& relocated_root,
    const std::filesystem::path& original_install_root,
    ProductInstallationReceipt& receipt,
    std::string& error) {
    receipt = {};
    if (!IsNormalizedProductRoot(relocated_root) ||
        !IsNormalizedProductRoot(original_install_root) ||
        relocated_root.parent_path() != original_install_root.parent_path() ||
        relocated_root == original_install_root) {
        error = "Normalized sibling product roots are required";
        return false;
    }
    if (!ReadReceiptFile(
            ProductInstallationReceiptPath(relocated_root), receipt, error)) {
        return false;
    }
    if (receipt.install_root != original_install_root) {
        receipt = {};
        error = "The relocated product receipt belongs to another root";
        return false;
    }
    error.clear();
    return true;
}

bool PublishProductInstallationReceipt(
    const std::filesystem::path& install_root,
    ProductInstallScope scope,
    ProductInstallationReceipt& receipt,
    std::string& error) {
    receipt = {};
    if (!IsNormalizedProductRoot(install_root)) {
        error = "A normalized absolute product root is required";
        return false;
    }
    const auto destination = ProductInstallationReceiptPath(install_root);
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(
        destination, filesystem_error);
    if (!filesystem_error &&
        status.type() != std::filesystem::file_type::not_found) {
        if (!LoadProductInstallationReceipt(install_root, receipt, error)) {
            return false;
        }
        if (receipt.scope != scope) {
            receipt = {};
            error = "The existing product receipt has another install scope";
            return false;
        }
        return true;
    }
    if (filesystem_error &&
        filesystem_error != std::errc::no_such_file_or_directory) {
        error = "Cannot inspect the product installation receipt: " +
            filesystem_error.message();
        return false;
    }

    ProductInstallationReceipt issued;
    issued.install_root = install_root;
    issued.scope = scope;
    if (!GenerateInstallId(issued.install_id, error)) return false;
    const auto temporary = install_root /
        (".cyxwiz-installation-source-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    RemoveTemporaryReceipt cleanup(temporary);
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = "Cannot create the product installation receipt source";
        return false;
    }
    stream << ReceiptDocument(issued).dump(2) << '\n';
    stream.flush();
    if (!stream) {
        error = "Cannot write the complete product installation receipt";
        return false;
    }
    stream.close();
    if (!PublishRegularFileAtomic(
            temporary, destination, kMaximumReceiptBytes, error,
            [&](const std::filesystem::path& candidate,
                std::string& validation_error) {
                ProductInstallationReceipt validated;
                if (!ReadReceiptFile(
                        candidate, validated, validation_error)) {
                    return false;
                }
                if (validated.install_id != issued.install_id ||
                    validated.install_root != issued.install_root ||
                    validated.scope != issued.scope) {
                    validation_error =
                        "The temporary product receipt changed unexpectedly";
                    return false;
                }
                return true;
            })) {
        return false;
    }
    if (!LoadProductInstallationReceipt(install_root, receipt, error) ||
        receipt.install_id != issued.install_id || receipt.scope != scope) {
        if (error.empty()) error = "The published product receipt changed unexpectedly";
        receipt = {};
        return false;
    }
    return true;
}

}  // namespace cyxwiz::runtime
