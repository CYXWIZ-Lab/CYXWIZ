#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz::installer {

// Cancellation uses the same unguessable, path-safe identity as the helper
// progress channel. The request lives outside the privileged installation
// root so a non-elevated GUI can cooperatively stop an elevated helper.
std::filesystem::path InstallerCancellationPath(const std::string &token);
bool RequestInstallerCancellation(const std::string &token);
bool IsInstallerCancellationRequested(const std::string &token);
void ClearInstallerCancellation(const std::string &token);

} // namespace cyxwiz::installer
