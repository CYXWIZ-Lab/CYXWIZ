#include "product_installation_receipt.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
            ("cyxwiz-installation-receipt-test-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

void WriteText(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << content;
    Check(static_cast<bool>(stream), "Receipt fixture write must succeed");
}

void TestPublishLoadAndPreserveIdentity() {
    TemporaryDirectory temporary;
    const auto root = temporary.path() / "CyxWiz Product";
    std::filesystem::create_directories(root);
    cyxwiz::runtime::ProductInstallationReceipt issued;
    std::string error;
    Check(cyxwiz::runtime::PublishProductInstallationReceipt(
              root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
              issued, error) &&
              issued.install_id.size() == 32 && issued.install_root == root,
          "A normalized product root must receive an installation identity");

    cyxwiz::runtime::ProductInstallationReceipt loaded;
    Check(cyxwiz::runtime::LoadProductInstallationReceipt(
              root, loaded, error) &&
              loaded.install_id == issued.install_id &&
              loaded.scope ==
                  cyxwiz::runtime::ProductInstallScope::CurrentUser,
          "The exact installation receipt must load successfully");

    cyxwiz::runtime::ProductInstallationReceipt repeated;
    Check(cyxwiz::runtime::PublishProductInstallationReceipt(
              root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
              repeated, error) &&
              repeated.install_id == issued.install_id,
          "Repair registration must preserve the installation identity");

    cyxwiz::runtime::ProductInstallationReceipt wrong_scope;
    Check(!cyxwiz::runtime::PublishProductInstallationReceipt(
              root, cyxwiz::runtime::ProductInstallScope::AllUsers,
              wrong_scope, error) &&
              error.find("another install scope") != std::string::npos,
          "An existing installation receipt must lock its install scope");
}

void TestRejectsRedirectedAndCorruptReceipts() {
    TemporaryDirectory temporary;
    const auto first = temporary.path() / "First";
    const auto second = temporary.path() / "Second";
    std::filesystem::create_directories(first);
    std::filesystem::create_directories(second);
    cyxwiz::runtime::ProductInstallationReceipt issued;
    std::string error;
    Check(cyxwiz::runtime::PublishProductInstallationReceipt(
              first, cyxwiz::runtime::ProductInstallScope::CurrentUser,
              issued, error),
          "Redirect protection fixture must publish a receipt");
    std::filesystem::copy_file(
        cyxwiz::runtime::ProductInstallationReceiptPath(first),
        cyxwiz::runtime::ProductInstallationReceiptPath(second));
    cyxwiz::runtime::ProductInstallationReceipt redirected;
    Check(!cyxwiz::runtime::LoadProductInstallationReceipt(
              second, redirected, error) &&
              error.find("another root") != std::string::npos,
          "A copied receipt must not authorize another product root");

    const auto receipt_path =
        cyxwiz::runtime::ProductInstallationReceiptPath(first);
    const std::string corrupt = "{\"schema_version\":1,\"unknown\":true}\n";
    WriteText(receipt_path, corrupt);
    cyxwiz::runtime::ProductInstallationReceipt replacement;
    Check(!cyxwiz::runtime::PublishProductInstallationReceipt(
              first, cyxwiz::runtime::ProductInstallScope::CurrentUser,
              replacement, error) &&
              ReadText(receipt_path) == corrupt,
          "A corrupt existing receipt must fail closed without replacement");
}

void TestRejectsUnsafeRoots() {
    cyxwiz::runtime::ProductInstallationReceipt receipt;
    std::string error;
    Check(!cyxwiz::runtime::PublishProductInstallationReceipt(
              "relative", cyxwiz::runtime::ProductInstallScope::CurrentUser,
              receipt, error),
          "A relative product root must not receive a deletion identity");
    Check(!cyxwiz::runtime::LoadProductInstallationReceipt(
              std::filesystem::path("relative"), receipt, error),
          "A relative product root must not load a deletion identity");
}

}  // namespace

int main() {
    TestPublishLoadAndPreserveIdentity();
    TestRejectsRedirectedAndCorruptReceipts();
    TestRejectsUnsafeRoots();
    std::cout << "Product installation receipt contracts passed\n";
    return 0;
}
