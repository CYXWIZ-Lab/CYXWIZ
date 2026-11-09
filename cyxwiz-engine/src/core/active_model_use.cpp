#include "active_model_use.h"

#include <mutex>

namespace cyxwiz {
namespace {

std::mutex& HolderMutex() {
    static std::mutex mutex;
    return mutex;
}

std::string& HolderName() {
    static std::string name;  // Guarded by HolderMutex.
    return name;
}

std::string BusyLocked(bool training, bool testing) {
    if (training) return "Train is running on the model";
    if (testing) return "A test is running on the model";
    if (!HolderName().empty()) return HolderName() + " is running on the model";
    return {};
}

}  // namespace

ActiveModelUse::~ActiveModelUse() {
    std::lock_guard<std::mutex> lock(HolderMutex());
    HolderName().clear();
}

std::shared_ptr<ActiveModelUse> ActiveModelUse::TryClaim(const std::string& what, bool training, bool testing,
                                                         std::string* why) {
    std::lock_guard<std::mutex> lock(HolderMutex());
    if (auto busy = BusyLocked(training, testing); !busy.empty()) {
        if (why) *why = std::move(busy);
        return nullptr;
    }
    HolderName() = what.empty() ? std::string("A job") : what;
    return std::shared_ptr<ActiveModelUse>(new ActiveModelUse());
}

std::string ActiveModelBusy(bool training, bool testing) {
    std::lock_guard<std::mutex> lock(HolderMutex());
    return BusyLocked(training, testing);
}

}  // namespace cyxwiz
