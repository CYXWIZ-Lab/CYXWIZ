#include "backend_pack_platform.h"
#include "product_removal_request.h"
#include "product_removal_transaction_internal.h"
#include "runtime_operation_lock.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace cyxwiz::runtime;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Touch(const fs::path &path, const std::string &contents = "fixture") {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path);
  stream << contents;
  Check(static_cast<bool>(stream), "Cannot create fixture");
}

struct Fixture {
  fs::path parent =
      fs::temp_directory_path() /
      ("cyxwiz-removal-ownership-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::path root;
  ProductRemovalAuthorization authorization;
  detail::ProductRemovalTransactionOperations operations;
  int validations = 0;
  int unregistered = 0;
  int restored = 0;
  int cleaned = 0;
  bool reject_second_validation = false;
  bool fail_quarantine = false;
  bool throw_quarantine = false;
  bool fail_cleanup = false;
  bool recreate_original = false;

  Fixture() {
    fs::create_directories(parent);
    parent = fs::canonical(parent);
    root = parent / "CyxWiz";
    const auto base = root / "runtime/base/base-v1";
    Touch(root / CurrentRuntimeBootstrapperExecutableName());
    Touch(base / CurrentEngineExecutableName());
    Touch(base / "RUNTIME_VERSIONS.json",
          R"({"arrayfire":"3.10.0","cyxwiz":"0.2.0","python":"3.12.0"})");
    ActiveRuntimeState active;
    active.runtime_set_id = "set-v1";
    active.generation = 1;
    active.base_pack_id = "base-v1";
    std::string error;
    Check(SaveActiveRuntimeStateAtomic(root / "runtime/active-runtime.json",
                                       active, error),
          error);
    ProductInstallationReceipt receipt;
    Check(PublishProductInstallationReceipt(
              root, ProductInstallScope::CurrentUser, receipt, error),
          error);
    Check(QueueProductRemovalRequest(root, ProductInstallScope::CurrentUser,
                                     authorization, error),
          error);
    operations.validate = [&](const auto &request, std::string &reason) {
      if (++validations == 2 && reject_second_validation) {
        reason = "Injected identity change after lock acquisition";
        return false;
      }
      return ValidateProductRemovalAuthorization(request, reason);
    };
    // Native registration is substituted; this contract never changes the
    // host registry/shortcuts. Authorization, locking, rename and cleanup are
    // real.
    operations.unregister_product = [&](const auto &) {
      RequireBusy(root / "runtime");
      ++unregistered;
      return ProductUnregistrationResult{true, "fixture unregistered"};
    };
    operations.register_product = [&](const auto &) {
      RequireBusy(root / "runtime");
      ++restored;
      return ProductRegistrationResult{true, "fixture restored"};
    };
    operations.quarantine = [&](const auto &request, auto &moved,
                                std::string &reason) {
      RequireBusy(root / "runtime");
      if (throw_quarantine)
        throw std::runtime_error("injected quarantine exception");
      if (fail_quarantine) {
        reason = "injected rename failure";
        return false;
      }
      const bool result = QuarantineProductInstallation(request, moved, reason);
      if (result)
        RequireBusy(root / "runtime");
      return result;
    };
    operations.cleanup = [&](const auto &moved, auto &result,
                             std::string &reason) {
      RequireAvailable(moved.quarantine_root / "runtime");
      RequireBusy(root / "runtime");
      ++cleaned;
      if (fail_cleanup) {
        reason = "injected cleanup failure";
        return false;
      }
      if (recreate_original) {
        Touch(root / "new-install-marker", "preserve");
      }
      return CleanupQuarantinedProductInstallation(moved, result, reason);
    };
  }
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(parent, ignored);
  }
  static void RequireBusy(const fs::path &runtime) {
    RuntimeOperationLock contender;
    std::string error;
    Check(contender.Acquire(runtime, error) == RuntimeOperationLockStatus::Busy,
          "Helper/repair must remain excluded until quarantine completes: " +
              error);
  }
  static void RequireAvailable(const fs::path &runtime) {
    RuntimeOperationLock contender;
    std::string error;
    Check(contender.Acquire(runtime, error) ==
              RuntimeOperationLockStatus::Acquired,
          "Removal must release ownership before cleanup/retry: " + error);
  }
  bool Execute(ProductRemovalTransactionResult &result, std::string &error) {
    return detail::ExecuteGuardedProductRemovalTransactionWithOperations(
        authorization, operations, result, error);
  }
};

void Run() {
  {
    Fixture f;
    RuntimeOperationLock engine;
    ProductRemovalTransactionResult result;
    std::string error;
    Check(engine.AcquireEngineUse(f.root / "runtime", error) == RuntimeOperationLockStatus::Acquired, error);
    Check(!f.Execute(result, error) && !f.unregistered && !f.cleaned && fs::exists(f.root),
          "An Engine session must block uninstall before product mutation");
    Check(error.find("Cannot uninstall yet. CyxWiz is in use.") == 0 &&
              error.find("Save your work and close CyxWiz Engine") !=
                  std::string::npos &&
              error.find("then retry") != std::string::npos,
          "Busy Engine feedback must explain how to safely retry: " + error);
  }
  {
    Fixture f;
    ProductRemovalTransactionResult result;
    std::string error;
    {
      RuntimeOperationLock helper;
      Check(helper.Acquire(f.root / "runtime", error) ==
                RuntimeOperationLockStatus::Acquired,
            error);
      Check(!f.Execute(result, error) &&
                error.find("If another installation or update is running") !=
                    std::string::npos &&
                error.find("wait for it to finish") != std::string::npos &&
                result.stage == ProductRemovalTransactionStage::None &&
                !f.unregistered && !f.cleaned &&
                fs::exists(f.root / ".cyxwiz-installation.json"),
            "Busy helper must prevent unregister, quarantine and cleanup");
    }
    const bool completed = f.Execute(result, error);
    Check(completed &&
              result.stage == ProductRemovalTransactionStage::Complete &&
              f.unregistered == 1 && f.cleaned == 1 && !fs::exists(f.root) &&
              !fs::exists(ProductRemovalQuarantinePath(f.authorization)),
          "Retry after helper release must fully remove the fixture: " + error);
  }
  {
    Fixture f;
    f.fail_quarantine = true;
    ProductRemovalTransactionResult result;
    std::string error;
    Check(!f.Execute(result, error) && f.restored == 1 && !f.cleaned &&
              fs::exists(f.root),
          "Rename failure must restore registration under ownership");
    Fixture::RequireAvailable(f.root / "runtime");
  }
  {
    Fixture f;
    f.reject_second_validation = true;
    ProductRemovalTransactionResult result;
    std::string error;
    Check(!f.Execute(result, error) && f.validations == 2 && !f.unregistered &&
              !f.cleaned,
          "Revalidate authorization under ownership before unregistering");
    Fixture::RequireAvailable(f.root / "runtime");
  }
  {
    Fixture f;
    f.fail_cleanup = true;
    ProductRemovalTransactionResult result;
    std::string error;
    Check(!f.Execute(result, error) &&
              result.stage == ProductRemovalTransactionStage::Quarantined &&
              fs::exists(ProductRemovalQuarantinePath(f.authorization) /
                         ".cyxwiz-installation.json") &&
              !fs::exists(f.root),
          "Cleanup failure must preserve quarantine evidence");
  }
  {
    Fixture f;
    f.recreate_original = true;
    ProductRemovalTransactionResult result;
    std::string error;
    const bool completed = f.Execute(result, error);
    Check(completed && fs::file_size(f.root / "new-install-marker") == 8 &&
              !fs::exists(ProductRemovalQuarantinePath(f.authorization)),
          "Cleanup must not touch a fresh installation at the original "
          "location: " +
              error);
  }
  {
    Fixture f;
    const auto invalid = f.parent / "unauthorized";
    f.authorization.install_root = invalid;
    ProductRemovalTransactionResult result;
    std::string error;
    Check(!f.Execute(result, error) && !fs::exists(invalid) && !f.unregistered,
          "Failed authorization must not create a lock directory");
  }
  {
    Fixture f;
    f.throw_quarantine = true;
    ProductRemovalTransactionResult result;
    std::string error;
    bool threw = false;
    try {
      f.Execute(result, error);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    Check(threw, "Exception fixture must execute");
    Fixture::RequireAvailable(f.root / "runtime");
  }
}
} // namespace

int main() {
  try {
    Run();
    std::cout << "Product removal ownership contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
