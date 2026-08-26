#include "product_removal_transaction.h"

#include "product_registration.h"
#include "product_removal_cleanup.h"
#include "product_removal_quarantine.h"
#include "product_removal_transaction_internal.h"

namespace cyxwiz::runtime {
namespace {

ProductRegistrationRequest RegistrationRequest(
    const ProductRemovalAuthorization& authorization) {
    ProductRegistrationRequest request;
    request.install_root = authorization.install_root;
    request.runtime_root = authorization.install_root / "runtime";
    request.scope = authorization.scope;
    request.product_version = authorization.product_version;
    return request;
}

}  // namespace

namespace detail {

bool ExecuteProductRemovalTransactionWithOperations(
    const ProductRemovalAuthorization& authorization,
    const ProductRemovalTransactionOperations& operations,
    ProductRemovalTransactionResult& result,
    std::string& error) {
    result = {};
    if (!operations.validate || !operations.unregister_product ||
        !operations.register_product || !operations.quarantine ||
        !operations.cleanup) {
        error = "The product removal transaction operations are incomplete";
        return false;
    }
    if (!operations.validate(authorization, error)) {
        error = "Product removal authorization changed: " + error;
        return false;
    }

    const auto registration = RegistrationRequest(authorization);
    const auto unregistered = operations.unregister_product(registration);
    if (!unregistered.unregistered) {
        error = "Native product unregistration failed before quarantine: " +
            unregistered.message;
        return false;
    }
    result.stage = ProductRemovalTransactionStage::Unregistered;

    QuarantinedProductInstallation quarantined;
    std::string quarantine_error;
    if (!operations.quarantine(
            authorization, quarantined, quarantine_error)) {
        const auto restored = operations.register_product(registration);
        if (restored.registered) {
            result.stage = ProductRemovalTransactionStage::None;
            error = "Product quarantine failed and native registration was restored: " +
                quarantine_error;
        } else {
            error = "Product quarantine failed after native unregistration: " +
                quarantine_error + "; native registration rollback failed: " +
                restored.message;
        }
        return false;
    }
    result.stage = ProductRemovalTransactionStage::Quarantined;

    ProductRemovalCleanupResult cleanup_result;
    if (!operations.cleanup(quarantined, cleanup_result, error) ||
        !cleanup_result.complete) {
        if (error.empty()) {
            error = "Quarantined product cleanup did not complete";
        } else {
            error = "Quarantined product cleanup failed: " + error;
        }
        result.removed_entries = cleanup_result.removed_entries;
        return false;
    }

    result.stage = ProductRemovalTransactionStage::Complete;
    result.removed_entries = cleanup_result.removed_entries;
    error.clear();
    return true;
}

}  // namespace detail

bool ExecuteProductRemovalTransaction(
    const ProductRemovalAuthorization& authorization,
    ProductRemovalTransactionResult& result,
    std::string& error) {
    detail::ProductRemovalTransactionOperations operations;
    operations.validate = ValidateProductRemovalAuthorization;
    operations.unregister_product = UnregisterInstalledProduct;
    operations.register_product = RegisterInstalledProduct;
    operations.quarantine = QuarantineProductInstallation;
    operations.cleanup = CleanupQuarantinedProductInstallation;
    return detail::ExecuteProductRemovalTransactionWithOperations(
        authorization, operations, result, error);
}

}  // namespace cyxwiz::runtime
