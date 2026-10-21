#include "backend_pack_platform.h"
#include "engine_runtime_ownership.h"
#include "runtime_layout.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace cyxwiz::runtime;
void Check(bool ok, const std::string &error) {
  if (!ok)
    throw std::runtime_error(error);
}
void Touch(const fs::path &path) {
  fs::create_directories(path.parent_path());
  std::ofstream(path).put('x');
}
void Token(const char *value) {
#ifdef _WIN32
  ::_putenv_s("CYXWIZ_RUNTIME_USE_TOKEN", value);
#else
  if (*value)
    ::setenv("CYXWIZ_RUNTIME_USE_TOKEN", value, 1);
  else
    ::unsetenv("CYXWIZ_RUNTIME_USE_TOKEN");
#endif
}
struct Fixture {
  fs::path parent =
      fs::temp_directory_path() /
      ("cyxwiz-engine-ownership-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::path root = parent / "product/runtime";
  fs::path engine = root / "base/base-v1" / CurrentEngineExecutableName();
  Fixture() {
    Token("");
    Touch(engine);
    ActiveRuntimeState state;
    state.base_pack_id = "base-v1";
    state.runtime_set_id = "set-v1";
    state.generation = 1;
    std::string error;
    Check(SaveActiveRuntimeStateAtomic(root / "active-runtime.json", state,
                                       error),
          error);
  }
  ~Fixture() {
    Token("");
    std::error_code ignored;
    fs::remove_all(parent, ignored);
  }
};
void Run() {
  Fixture f;
  std::string error;
  {
    RuntimeOperationLock engine, maintenance;
    Check(AcquirePackagedEngineOwnership(f.engine, {}, engine, error), error);
    Check(maintenance.Acquire(f.root, error) ==
              RuntimeOperationLockStatus::Busy,
          "Direct installed Engine must exclude maintenance without an "
          "environment hint");
  }
  {
    RuntimeOperationLock maintenance, engine;
    Check(maintenance.Acquire(f.root, error) ==
              RuntimeOperationLockStatus::Acquired,
          error);
    Check(!AcquirePackagedEngineOwnership(f.engine, f.root, engine, error),
          "Installed Engine must not start during maintenance");
  }
  {
    const auto inactive =
        f.root / "base/base-old" / CurrentEngineExecutableName();
    Touch(inactive);
    RuntimeOperationLock engine;
    Check(!AcquirePackagedEngineOwnership(inactive, {}, engine, error),
          "Inactive base must not execute against the current activation");
  }
  {
    const auto developer =
        f.parent / "build/bin" / CurrentEngineExecutableName();
    Touch(developer);
    RuntimeOperationLock engine, maintenance;
    Check(AcquirePackagedEngineOwnership(developer, {}, engine, error) &&
              maintenance.Acquire(f.root, error) ==
                  RuntimeOperationLockStatus::Acquired,
          "Unpackaged developer binaries must not lock unrelated installed "
          "products");
  }
  {
    RuntimeOperationLock engine;
    Token("not-a-handle");
    Check(!AcquirePackagedEngineOwnership(f.engine, f.root, engine, error),
          "Invalid inherited ownership must fail closed");
    Token("");
  }
  {
    std::ofstream(f.root / "active-runtime.json", std::ios::trunc) << "{}";
    RuntimeOperationLock engine;
    Check(!AcquirePackagedEngineOwnership(f.engine, {}, engine, error),
          "Corrupt installed activation must not be treated as a developer "
          "build");
  }
}
} // namespace
int main() {
  try {
    Run();
    std::cout << "Engine runtime ownership contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
