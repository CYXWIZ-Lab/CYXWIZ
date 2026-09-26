#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz::core {

// Package smoke for the Python runtime bundled with a release install:
// initializes an isolated interpreter whose home is <engine_dir>/python,
// imports a few stdlib modules that depend on native extensions, and checks
// that the interpreter really runs from the bundled tree.
//
// Returns the interpreter version ("3.12.14") on success, "disabled" when
// this Engine is built without Python scripting, or an empty string on
// failure with the reason in *error.
std::string RunBundledPythonSmoke(const std::filesystem::path& engine_dir,
                                  std::string* error);

}  // namespace cyxwiz::core
