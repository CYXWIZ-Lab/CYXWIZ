#include "backend_pack_update_policy.h"

#include <iostream>
#include <string>

using namespace cyxwiz::runtime;

int main() {
  int failures = 0;
  const auto check = [&](bool passed, const char *message) {
    if (!passed) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  };
  check(CompareBackendPackVersions("10", "2.0.0") == 1,
        "Order numeric, not lexical");
  check(CompareBackendPackVersions("1.0.0", "1") == 0,
        "Normalize trailing zeroes");
  for (const auto value : {"", "1.", "1..2", "01", "-1", "+1", "1.x", "1-beta",
                           "4294967296", "1.2.3.4.5.6.7.8.9"}) {
    check(!CompareBackendPackVersions(value, "1"),
          "Reject ambiguous or unbounded versions");
  }
  VerifiedBackendPackManifest installed;
  installed.kind = BackendPackManifestKind::Base;
  installed.backend = "cpu";
  installed.platform = "win64";
  installed.architecture = "x86_64";
  installed.pack_id = "base-a";
  installed.runtime_set_id = "set-a";
  installed.package_version = "2";
  installed.minimum_cyxwiz_release = installed.maximum_cyxwiz_release = "0.2.0";
  installed.archive.sha256 = std::string(64, 'a');
  auto candidate = installed;
  using D = BackendPackUpdateDisposition;
  const auto expect = [&](D result, const char *message) {
    check(EvaluateBackendPackUpdate(installed, candidate).disposition == result,
          message);
  };
  expect(D::SamePackage, "Same identity is a no-op");
  candidate.archive.sha256 = std::string(64, 'b');
  expect(D::Ambiguous, "Same ID with changed content is blocked");
  candidate = installed;
  candidate.pack_id = "base-b";
  expect(D::Ambiguous, "Equal versions with different IDs are blocked");
  candidate.package_version = "1";
  expect(D::Downgrade, "Lower revision is blocked");
  candidate.package_version = "10";
  expect(D::Upgrade, "Higher revision updates the same engine release");
  candidate.minimum_cyxwiz_release = candidate.maximum_cyxwiz_release = "0.1.0";
  expect(D::Downgrade, "Higher revision cannot downgrade the engine");
  candidate.minimum_cyxwiz_release = candidate.maximum_cyxwiz_release = "0.3.0";
  candidate.package_version = "1";
  expect(D::Upgrade, "Engine release takes precedence over package revision");
  candidate.maximum_cyxwiz_release = "0.3.x";
  expect(D::Unknown,
         "A compatibility range is not an installed engine version");
  candidate = installed;
  candidate.pack_id = "base-b";
  candidate.architecture = "arm64";
  expect(D::Unknown, "Never compare different targets");
  installed.kind = BackendPackManifestKind::BackendPack;
  installed.backend = "opencl";
  installed.companion_base_id = "base-a";
  candidate = installed;
  candidate.pack_id = "opencl-b";
  candidate.package_version = "3";
  expect(D::Upgrade, "Optional packs update within the same base closure");
  candidate.companion_base_id = "base-b";
  expect(D::Unknown,
         "Different companion bases need a coordinated base update");
  candidate = installed;
  candidate.pack_id = "opencl-b";
  candidate.package_version = "bad";
  expect(D::Unknown, "Unorderable revisions fail closed");
  return failures == 0 ? 0 : 1;
}
