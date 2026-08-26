#include "product_removal_cleanup.h"

#include "product_removal_cleanup_platform.h"

namespace cyxwiz::runtime {

bool CleanupQuarantinedProductInstallation(
    const QuarantinedProductInstallation& quarantined,
    ProductRemovalCleanupResult& result,
    std::string& error) {
    result = {};
    if (!ValidateQuarantinedProductInstallation(quarantined, error)) {
        error = "Product cleanup rejected its quarantine identity: " + error;
        return false;
    }
    if (!detail::CleanupQuarantineNoFollow(quarantined, result, error)) {
        result.complete = false;
        return false;
    }
    result.complete = true;
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
