#include "backend_pack_platform.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using KeyContext = std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)>;
using DigestContext = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

Key GenerateKey() {
  KeyContext context(EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr),
                     &EVP_PKEY_CTX_free);
  EVP_PKEY *raw_key = nullptr;
  if (!context || EVP_PKEY_keygen_init(context.get()) != 1 ||
      EVP_PKEY_keygen(context.get(), &raw_key) != 1) {
    throw std::runtime_error("Cannot generate the fixture signing key");
  }
  return {raw_key, &EVP_PKEY_free};
}

std::string Base64Url(const unsigned char *bytes, std::size_t size) {
  std::string encoded(4 * ((size + 2) / 3), '\0');
  const auto written =
      EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()), bytes,
                      static_cast<int>(size));
  if (written <= 0) {
    throw std::runtime_error("Cannot encode fixture signing material");
  }
  encoded.resize(static_cast<std::size_t>(written));
  for (auto &character : encoded) {
    if (character == '+')
      character = '-';
    else if (character == '/')
      character = '_';
  }
  while (!encoded.empty() && encoded.back() == '=')
    encoded.pop_back();
  return encoded;
}

std::string PublicKey(const Key &key) {
  std::array<unsigned char, 32> bytes{};
  std::size_t size = bytes.size();
  if (EVP_PKEY_get_raw_public_key(key.get(), bytes.data(), &size) != 1 ||
      size != bytes.size()) {
    throw std::runtime_error("Cannot export the fixture public key");
  }
  return Base64Url(bytes.data(), bytes.size());
}

std::string Sign(const Json &body, const Key &key) {
  const auto payload = body.dump();
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  std::array<unsigned char, 64> signature{};
  std::size_t size = signature.size();
  if (!context ||
      EVP_DigestSignInit(context.get(), nullptr, nullptr, nullptr, key.get()) !=
          1 ||
      EVP_DigestSign(context.get(), signature.data(), &size,
                     reinterpret_cast<const unsigned char *>(payload.data()),
                     payload.size()) != 1 ||
      size != signature.size()) {
    throw std::runtime_error("Cannot sign the fixture metadata");
  }
  return Base64Url(signature.data(), signature.size());
}

std::string Sha256(std::string_view bytes) {
  std::array<unsigned char, 32> digest{};
  unsigned int size = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, EVP_sha256(),
                 nullptr) != 1 ||
      size != digest.size()) {
    throw std::runtime_error("Cannot hash the fixture metadata");
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(64, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    output[2 * index] = kHex[digest[index] >> 4];
    output[2 * index + 1] = kHex[digest[index] & 0x0f];
  }
  return output;
}

Json Envelope(const char *kind, Json body, const char *key_id, const Key &key) {
  const auto signature = Sign(body, key);
  return {{"schema_version", std::uint64_t{1}},
          {"kind", kind},
          {"signed", std::move(body)},
          {"signatures", Json::array({{{"key_id", key_id},
                                       {"algorithm", "ed25519"},
                                       {"value", signature}}})}};
}

Json ManifestBody(const std::string &pack_id, const std::string &backend,
                  const std::optional<std::string> &companion_base_id) {
  const bool base = backend == "cpu";
  const auto device_kinds = [&]() {
    if (base)
      return Json::array({"cpu"});
    if (backend == "cuda")
      return Json::array({"gpu"});
    return Json::array({"cpu", "gpu", "accelerator"});
  }();
  const auto provider_types = [&]() {
    if (base)
      return Json::array({"arrayfire-cpu"});
    if (backend == "cuda")
      return Json::array({"nvidia-driver"});
    return Json::array({"opencl-icd"});
  }();
  const auto recommendation_targets = [&]() {
    if (base)
      return Json::array();
    if (backend == "cuda")
      return Json::array({"opencl", "cpu"});
    return Json::array({"cuda", "oneapi", "cpu"});
  }();
  const auto component_name = base                ? "ci-afcpu.bin"
                              : backend == "cuda" ? "ci-afcuda.bin"
                                                  : "ci-afopencl.bin";
  return {
      {"pack_id", pack_id},
      {"pack_kind", base ? "base" : "backend_pack"},
      {"backend", backend},
      {"package_version", "1.0.0"},
      {"platform", cyxwiz::runtime::CurrentBackendPackPlatformId()},
      {"architecture", cyxwiz::runtime::CurrentBackendPackArchitectureId()},
      {"runtime_set_id", "ci-runtime-set-v1"},
      {"cyxwiz_release", {{"minimum", "0.2.0"}, {"maximum", "0.2.x"}}},
      {"arrayfire", {{"version", "3.10.0"}, {"abi", "arrayfire-3.10"}}},
      {"companion_base_id",
       companion_base_id ? Json(*companion_base_id) : Json(nullptr)},
      {"conflicts", Json::array()},
      {"compatibility",
       {{"device_kinds", device_kinds},
        {"cpu_features", Json::array()},
        {"provider_types", provider_types},
        {"minimum_driver_versions", Json::object()},
        {"tested_driver_ranges", Json::object()},
        {"minimum_identity_confidence",
         base ? "backend_local" : "stable_hardware"},
        {"recommendation_targets", recommendation_targets},
        {"operation_matrix_id", "ci-matrix-v1"},
        {"training_scope", Json::array({"dense"})},
        {"support_status", "supported"}}},
      {"components",
       Json::array({{{"path", "runtime/" + std::string(component_name)},
                     {"size", std::uint64_t{1}},
                     {"sha256", "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738"
                                "768511a30617afa01d"},
                     {"source", "ci-fixture"},
                     {"executable", false}},
                    {{"path", "THIRD_PARTY_LICENSES/fixture/LICENSE.txt"},
                     {"size", std::uint64_t{1}},
                     {"sha256", "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738"
                                "768511a30617afa01d"},
                     {"source", "ci-fixture-license"},
                     {"executable", false}}})},
      {"licenses",
       Json::array({{{"component", "ci-fixture"},
                     {"path", "THIRD_PARTY_LICENSES/fixture/LICENSE.txt"}}})},
      {"archive",
       {{"file_name", pack_id + ".zip"},
        {"size", std::uint64_t{4096}},
        {"sha256", std::string(64, 'a')}}},
      {"generated_utc", "2026-01-01T00:00:00Z"}};
}

std::string WriteJson(const std::filesystem::path &path, const Json &value) {
  const auto bytes = value.dump(2) + "\n";
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream << bytes;
  return bytes;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: generate_installer_metadata_fixture <output-root>\n";
      return 64;
    }
    const auto output_root = std::filesystem::absolute(argv[1]);
    const auto trust_directory = output_root / "trust";
    const auto manifest_directory = output_root / "catalogs" / "manifests";
    std::filesystem::create_directories(trust_directory);
    std::filesystem::create_directories(manifest_directory);

    const auto catalog_key = GenerateKey();
    const auto pack_key = GenerateKey();
    constexpr auto kBaseId = "ci-base-v1";
    constexpr auto kOpenClId = "ci-opencl-v1";
    constexpr auto kCudaId = "ci-cuda-v1";

    const auto base_manifest = Envelope(
        "cyxwiz-backend-pack-manifest",
        ManifestBody(kBaseId, "cpu", std::nullopt), "ci-pack-key", pack_key);
    const auto opencl_manifest =
        Envelope("cyxwiz-backend-pack-manifest",
                 ManifestBody(kOpenClId, "opencl", std::string(kBaseId)),
                 "ci-pack-key", pack_key);
    const auto base_bytes = WriteJson(
        manifest_directory / (std::string(kBaseId) + ".json"), base_manifest);
    const auto opencl_bytes =
        WriteJson(manifest_directory / (std::string(kOpenClId) + ".json"),
                  opencl_manifest);

    std::vector<std::pair<std::string, std::string>> catalog_entries = {
        {kBaseId, Sha256(base_bytes)}, {kOpenClId, Sha256(opencl_bytes)}};
    if (cyxwiz::runtime::CurrentBackendPackPlatformId() != "macos") {
      const auto cuda_manifest =
          Envelope("cyxwiz-backend-pack-manifest",
                   ManifestBody(kCudaId, "cuda", std::string(kBaseId)),
                   "ci-pack-key", pack_key);
      const auto cuda_bytes = WriteJson(
          manifest_directory / (std::string(kCudaId) + ".json"), cuda_manifest);
      catalog_entries.emplace_back(kCudaId, Sha256(cuda_bytes));
    }

    Json catalog_body = {{"catalog_id", "ci-installer-catalog-v1"},
                         {"generated_utc", "2026-01-01T00:00:00Z"},
                         {"expires_utc", "2099-01-01T00:00:00Z"},
                         {"minimum_client_version", "0.2.0"},
                         {"packs", Json::array()}};
    for (const auto &entry : catalog_entries) {
      catalog_body["packs"].push_back(
          {{"pack_id", entry.first},
           {"manifest_url",
            "https://packages.invalid/cyxwiz/" + entry.first + ".json"},
           {"manifest_sha256", entry.second},
           {"signing_key_id", "ci-pack-key"},
           {"support_status", "supported"}});
    }
    WriteJson(output_root / "catalogs" / "current.json",
              Envelope("cyxwiz-backend-pack-catalog", std::move(catalog_body),
                       "ci-catalog-key", catalog_key));
    WriteJson(trust_directory / "trusted-keys.json",
              {{"schema_version", std::uint64_t{1}},
               {"keys", Json::array({{{"key_id", "ci-catalog-key"},
                                      {"algorithm", "ed25519"},
                                      {"public_key", PublicKey(catalog_key)},
                                      {"roles", Json::array({"catalog"})},
                                      {"revoked", false}},
                                     {{"key_id", "ci-pack-key"},
                                      {"algorithm", "ed25519"},
                                      {"public_key", PublicKey(pack_key)},
                                      {"roles", Json::array({"pack"})},
                                      {"revoked", false}}})}});

    std::cout << "Generated signed installer metadata fixture: "
              << output_root.string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Cannot generate installer metadata fixture: " << error.what()
              << '\n';
    return 1;
  }
}
