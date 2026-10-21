#include "backend_pack_installed_metadata.h"
#include "atomic_file_publisher.h"
#include "backend_pack_metadata_limits.h"
#include "backend_pack_path.h"

namespace cyxwiz::runtime {
namespace {
bool ValidRootAndId(const std::filesystem::path &root, const std::string &id) {
  return root.is_absolute() && IsCanonicalBackendPackRelativePath(id) &&
         id.find('/') == std::string::npos;
}
std::filesystem::path EvidencePath(const std::filesystem::path &root,
                                   const std::string &id) {
  return root / "installed-metadata" / (id + ".json");
}
} // namespace

bool ReadInstalledBackendPackMetadata(
    const std::filesystem::path &root, const std::string &id,
    BackendPackManifestKind kind, const BackendPackMetadataVerifier &verifier,
    VerifiedBackendPackManifest &output, std::string &error) {
  if (!ValidRootAndId(root, id)) {
    error = "Invalid installed metadata identity";
    return false;
  }
  auto path = EvidencePath(root, id);
  std::error_code ec;
  const bool retained = std::filesystem::exists(path, ec);
  if (ec) {
    error = "Cannot inspect installed metadata: " + ec.message();
    return false;
  }
  // Migration for older installs: use their cached, independently signed
  // manifest when available. Never infer versions from a directory/pack name.
  if (!retained)
    path = root / "catalogs/manifests" / (id + ".json");
  return verifier.VerifyInstalledManifest(path, id, kind, output, error);
}

bool RetainInstalledBackendPackMetadata(
    const std::filesystem::path &root, const std::filesystem::path &source,
    const BackendPackCatalogEntry &entry, BackendPackManifestKind kind,
    const BackendPackMetadataVerifier &verifier, std::string &error) {
  if (!ValidRootAndId(root, entry.pack_id)) {
    error = "Invalid installed metadata identity";
    return false;
  }
  return PublishRegularFileAtomic(
      source, EvidencePath(root, entry.pack_id),
      kMaximumBackendPackMetadataBytes, error,
      [&](const std::filesystem::path &staged, std::string &reason) {
        VerifiedBackendPackManifest checked;
        return verifier.VerifyManifest(staged, entry, checked, reason, kind);
      });
}
} // namespace cyxwiz::runtime
