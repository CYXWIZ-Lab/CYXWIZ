// The data budget leaves the compute device room to start (TOFIX157 A1):
// whatever is set, data may use at most this PC's memory less the device's
// start-up reserve; a budget below that is used as set; 0 is half the memory.
#include "../src/core/data_budget.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

}  // namespace

int main() {
    const uint64_t total = cyxwiz::PhysicalMemoryBytes();
    const uint64_t most = cyxwiz::MaxDataBudgetBytes();
    Check(total > 0 && most < total, "the most data may use is under this PC's memory");
    if (total > 2 * cyxwiz::kDeviceStartupReserveBytes)
        Check(most == total - cyxwiz::kDeviceStartupReserveBytes, "the device's start-up memory is kept out");

    cyxwiz::SetDataBudgetBytes(0);
    Check(cyxwiz::DataBudgetBytes() == cyxwiz::DefaultDataBudgetBytes() && cyxwiz::DataBudgetIsDefault(),
          "0: half of this PC's memory");

    cyxwiz::SetDataBudgetBytes(512ULL << 20);
    Check(cyxwiz::DataBudgetBytes() == (512ULL << 20) && cyxwiz::RequestedDataBudgetBytes() == (512ULL << 20),
          "a budget under the most is used as set");

    // All of this PC's memory, or more: data gets the most, the device keeps its share.
    cyxwiz::SetDataBudgetBytes(total);
    Check(cyxwiz::RequestedDataBudgetBytes() == total && cyxwiz::DataBudgetBytes() == most,
          "a budget of all the memory leaves the device its share");
    cyxwiz::SetDataBudgetBytes(total * 4);
    Check(cyxwiz::DataBudgetBytes() == most, "more than the PC has: the most");
    Check(cyxwiz::DataResultThresholdBytes() == most / 4, "the on-disk threshold follows what data may use");

    cyxwiz::SetDataBudgetBytes(0);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("data budget: default, as set, capped for the device (%s of %s). OK\n",
                cyxwiz::FormatDataBytes(most).c_str(), cyxwiz::FormatDataBytes(total).c_str());
    return 0;
}
