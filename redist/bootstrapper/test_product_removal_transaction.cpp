#include "product_removal_transaction_internal.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

struct TransactionFixture {
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    cyxwiz::runtime::detail::ProductRemovalTransactionOperations operations;
    std::vector<std::string> calls;
    bool validate = true;
    bool unregister = true;
    bool restore = true;
    bool quarantine = true;
    bool cleanup = true;

    TransactionFixture() {
        authorization.install_root = "/fixture/CyxWiz";
        authorization.product_version = "0.2.0";
        operations.validate = [&](const auto&, std::string& error) {
            calls.push_back("validate");
            if (!validate) error = "stale";
            return validate;
        };
        operations.unregister_product = [&](const auto&) {
            calls.push_back("unregister");
            return cyxwiz::runtime::ProductUnregistrationResult{
                unregister, unregister ? "removed" : "unmanaged"};
        };
        operations.register_product = [&](const auto&) {
            calls.push_back("restore");
            return cyxwiz::runtime::ProductRegistrationResult{
                restore, restore ? "restored" : "restore failed"};
        };
        operations.quarantine = [&](const auto&, auto& moved, std::string& error) {
            calls.push_back("quarantine");
            if (!quarantine) {
                error = "rename failed";
                return false;
            }
            moved.original_root = authorization.install_root;
            moved.quarantine_root = "/fixture/.cyxwiz-removing-id";
            return true;
        };
        operations.cleanup = [&](const auto&, auto& result, std::string& error) {
            calls.push_back("cleanup");
            result.removed_entries = 12;
            result.complete = cleanup;
            if (!cleanup) error = "locked payload";
            return cleanup;
        };
    }

    bool Execute(
        cyxwiz::runtime::ProductRemovalTransactionResult& result,
        std::string& error) {
        return cyxwiz::runtime::detail::
            ExecuteProductRemovalTransactionWithOperations(
                authorization, operations, result, error);
    }
};

void TestCompletesInExactOrder() {
    TransactionFixture fixture;
    cyxwiz::runtime::ProductRemovalTransactionResult result;
    std::string error;
    Check(fixture.Execute(result, error) &&
              result.stage ==
                  cyxwiz::runtime::ProductRemovalTransactionStage::Complete &&
              result.removed_entries == 12 &&
              fixture.calls == std::vector<std::string>{
                  "validate", "unregister", "quarantine", "cleanup"},
          "A valid transaction must unregister, quarantine, then clean: " + error);
}

void TestUnregistrationFailureDoesNotMutateProduct() {
    TransactionFixture fixture;
    fixture.unregister = false;
    cyxwiz::runtime::ProductRemovalTransactionResult result;
    std::string error;
    Check(!fixture.Execute(result, error) &&
              result.stage ==
                  cyxwiz::runtime::ProductRemovalTransactionStage::None &&
              fixture.calls == std::vector<std::string>{
                  "validate", "unregister"},
          "Failed native preflight must not quarantine or overwrite registration");
}

void TestQuarantineFailureRestoresRegistration() {
    TransactionFixture fixture;
    fixture.quarantine = false;
    cyxwiz::runtime::ProductRemovalTransactionResult result;
    std::string error;
    Check(!fixture.Execute(result, error) &&
              result.stage ==
                  cyxwiz::runtime::ProductRemovalTransactionStage::None &&
              fixture.calls == std::vector<std::string>{
                  "validate", "unregister", "quarantine", "restore"} &&
              error.find("was restored") != std::string::npos,
          "A quarantine failure must restore exact native registration");
}

void TestCleanupFailureRetainsQuarantineForRecovery() {
    TransactionFixture fixture;
    fixture.cleanup = false;
    cyxwiz::runtime::ProductRemovalTransactionResult result;
    std::string error;
    Check(!fixture.Execute(result, error) &&
              result.stage ==
                  cyxwiz::runtime::ProductRemovalTransactionStage::Quarantined &&
              result.removed_entries == 12 &&
              fixture.calls == std::vector<std::string>{
                  "validate", "unregister", "quarantine", "cleanup"},
          "A cleanup failure must retain quarantine without restoring a partial product");
}

}  // namespace

int main() {
    TestCompletesInExactOrder();
    TestUnregistrationFailureDoesNotMutateProduct();
    TestQuarantineFailureRestoresRegistration();
    TestCleanupFailureRetainsQuarantineForRecovery();
    std::cout << "Product removal transaction contracts passed\n";
    return 0;
}
