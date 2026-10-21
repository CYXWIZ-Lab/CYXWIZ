#pragma once

#include "product_removal_authorization.h"

#include <cstdint>
#include <string>

namespace cyxwiz::runtime {

enum class ProductRemovalTransactionStage {
    None,
    Unregistered,
    Quarantined,
    Complete,
};

struct ProductRemovalTransactionResult {
    ProductRemovalTransactionStage stage = ProductRemovalTransactionStage::None;
    std::uint64_t removed_entries = 0;
};

bool ExecuteProductRemovalTransaction(
    const ProductRemovalAuthorization& authorization,
    ProductRemovalTransactionResult& result,
    std::string& error);

}  // namespace cyxwiz::runtime
