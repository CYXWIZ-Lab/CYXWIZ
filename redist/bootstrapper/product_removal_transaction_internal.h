#pragma once

#include "product_registration.h"
#include "product_removal_cleanup.h"
#include "product_removal_transaction.h"

#include <functional>

namespace cyxwiz::runtime::detail {

struct ProductRemovalTransactionOperations {
    std::function<bool(
        const ProductRemovalAuthorization&, std::string&)> validate;
    std::function<ProductUnregistrationResult(
        const ProductRegistrationRequest&)> unregister_product;
    std::function<ProductRegistrationResult(
        const ProductRegistrationRequest&)> register_product;
    std::function<bool(
        const ProductRemovalAuthorization&,
        QuarantinedProductInstallation&,
        std::string&)> quarantine;
    std::function<bool(
        const QuarantinedProductInstallation&,
        ProductRemovalCleanupResult&,
        std::string&)> cleanup;
};

bool ExecuteProductRemovalTransactionWithOperations(
    const ProductRemovalAuthorization& authorization,
    const ProductRemovalTransactionOperations& operations,
    ProductRemovalTransactionResult& result,
    std::string& error);

}  // namespace cyxwiz::runtime::detail
