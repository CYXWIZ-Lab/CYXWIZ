#pragma once

// The files of a pack against its signed inventory: the one check Repair,
// the Engine's load of an installed add-on and the add-on cards share.

#include "backend_pack_hash.h"
#include "backend_pack_metadata_verifier.h"
#include "backend_pack_payload.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

// What is wrong with one file of a pack, for words on screen.
enum class PackFileFault {
    None,
    Missing,         // missing, a link, or not a regular file
    SizeDiffers,
    ContentDiffers,  // same size, another SHA-256
    Unreadable,
    NotListed        // a file the signed inventory does not list
};

// A regular file (not a link) of the component's signed size and SHA-256.
// `fault`, when given, says what failed.
bool VerifyPackComponentFile(
    const std::filesystem::path& path,
    const VerifiedPackComponent& component,
    std::string& error,
    const Sha256FileProgress& hash_progress = {},
    PackFileFault* fault = nullptr);

// The inventory's relative paths, folded (FoldBackendPackPath).
std::set<std::string> FoldedPackInventory(
    const std::vector<VerifiedPackComponent>& components);

// The files under `root` are exactly `expected` (folded relative paths): no
// link, no special file, no file missing and none more. `unlisted` and
// `missing`, when given, receive the folded paths that differ.
bool VerifyExactPackInventory(
    const std::filesystem::path& root,
    const std::set<std::string>& expected,
    std::string& error,
    std::vector<std::string>* unlisted = nullptr,
    std::vector<std::string>* missing = nullptr);

// The signed, installed metadata of an installed plugin pack
// (installed-metadata/<pack id>.json, read as Repair reads it), for exactly
// this plugin and pack.
bool VerifyInstalledPluginPack(
    const std::filesystem::path& runtime_root,
    const std::string& plugin_id,
    const std::string& pack_id,
    const BackendPackMetadataVerifier& verifier,
    VerifiedBackendPackManifest& output,
    std::string& error);

struct PackFileFinding {
    std::string relative_path;  // the inventory's spelling; folded for an unlisted file
    PackFileFault fault = PackFileFault::None;
};

struct PackFilesCheck {
    std::size_t files_checked = 0;     // inventory files hashed
    std::uint64_t bytes_checked = 0;
    std::vector<PackFileFinding> findings;
    std::string error;                 // the first failure, in the helpers' words
    bool ok() const { return findings.empty() && error.empty(); }
};

// Every file of an installed pack folder against the inventory: the exact
// file set and each component's size and SHA-256. Unlike Repair, which stops
// at the first difference, it lists them all (a card says which files).
PackFilesCheck CheckInstalledPackFiles(
    const std::filesystem::path& folder,
    const std::vector<VerifiedPackComponent>& components);

}  // namespace cyxwiz::runtime
