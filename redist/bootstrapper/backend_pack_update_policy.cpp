#include "backend_pack_update_policy.h"

#include <array>
#include <charconv>
#include <cstdint>

namespace cyxwiz::runtime {
namespace {
std::optional<std::array<std::uint32_t, 8>> Parse(std::string_view value) {
  if (value.empty() || value.size() > 128 || value.back() == '.')
    return std::nullopt;
  std::array<std::uint32_t, 8> parts{};
  std::size_t index = 0;
  while (!value.empty()) {
    if (index == parts.size())
      return std::nullopt;
    const auto end = value.find('.');
    const auto part = value.substr(0, end);
    if (part.empty() || (part.size() > 1 && part.front() == '0'))
      return std::nullopt;
    const auto result =
        std::from_chars(part.data(), part.data() + part.size(), parts[index++]);
    if (result.ec != std::errc{} || result.ptr != part.data() + part.size())
      return std::nullopt;
    if (end == std::string_view::npos)
      break;
    value.remove_prefix(end + 1);
  }
  return parts;
}
} // namespace

std::optional<int> CompareBackendPackVersions(std::string_view left,
                                              std::string_view right) {
  const auto a = Parse(left), b = Parse(right);
  if (!a || !b)
    return std::nullopt;
  return *a < *b ? -1 : (*a > *b ? 1 : 0);
}

BackendPackUpdateDecision
EvaluateBackendPackUpdate(const VerifiedBackendPackManifest &installed,
                          const VerifiedBackendPackManifest &candidate) {
  using D = BackendPackUpdateDisposition;
  if (installed.kind != candidate.kind ||
      installed.backend != candidate.backend ||
      installed.platform != candidate.platform ||
      installed.architecture != candidate.architecture)
    return {D::Unknown, "Installed and offered package targets differ"};
  if (installed.pack_id == candidate.pack_id) {
    if (installed.archive.sha256 != candidate.archive.sha256 ||
        installed.archive.size != candidate.archive.size ||
        installed.package_version != candidate.package_version ||
        installed.minimum_cyxwiz_release != candidate.minimum_cyxwiz_release ||
        installed.maximum_cyxwiz_release != candidate.maximum_cyxwiz_release ||
        installed.companion_base_id != candidate.companion_base_id ||
        installed.arrayfire_abi != candidate.arrayfire_abi ||
        installed.runtime_set_id != candidate.runtime_set_id)
      return {D::Ambiguous,
              "The same package ID has conflicting signed content"};
    return {D::SamePackage, "Already installed; no package download is needed"};
  }
  int release_order = 0;
  if (candidate.kind == BackendPackManifestKind::Base) {
    // Schema-1 base producers pin the payload's engine release to an exact
    // minimum/maximum. Broad compatibility ranges are not version evidence.
    if (installed.minimum_cyxwiz_release != installed.maximum_cyxwiz_release ||
        candidate.minimum_cyxwiz_release != candidate.maximum_cyxwiz_release)
      return {
          D::Unknown,
          "An exact signed engine release is required to compare CPU bases"};
    const auto order = CompareBackendPackVersions(
        candidate.minimum_cyxwiz_release, installed.minimum_cyxwiz_release);
    if (!order)
      return {D::Unknown, "The engine release cannot be ordered safely"};
    release_order = *order;
  } else if (installed.companion_base_id != candidate.companion_base_id ||
             installed.runtime_set_id != candidate.runtime_set_id) {
    return {D::Unknown, "Backend updates require the same companion CPU base"};
  }
  const auto revision = CompareBackendPackVersions(candidate.package_version,
                                                   installed.package_version);
  if (!revision)
    return {D::Unknown, "The package revision cannot be ordered safely"};
  const int order = release_order == 0 ? *revision : release_order;
  if (order < 0)
    return {D::Downgrade,
            "Older package blocked; automatic downgrade is not allowed"};
  if (order == 0)
    return {D::Ambiguous, "Equal versions have different package IDs; "
                          "automatic replacement is blocked"};
  return {D::Upgrade, "A newer signed package is available"};
}

} // namespace cyxwiz::runtime
