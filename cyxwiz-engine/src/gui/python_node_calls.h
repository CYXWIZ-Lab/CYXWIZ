#pragma once

// The Python calls that make a node bundle, shared by Script Editor > Create
// node from script and Make a Python node: a pycyxwiz.nodes function run on a
// worker thread under the interpreter lock, and what follows a bundle written
// by the user (approved as created, picked up at once).

#include <filesystem>
#include <future>
#include <string>
#include <utility>

namespace cyxwiz::python_calls {

// (ok, the function's text result) or (false, the error).
using Result = std::pair<bool, std::string>;

// pycyxwiz.nodes.<function>(script) or <function>(script, folder, name), on a
// worker thread. The caller polls the future once per frame.
std::future<Result> CallNodes(const char* function, std::string script, std::string folder = {},
                              std::string name = {});

// A bundle the user just wrote at `folder`: read it, approve its code as
// created by the user and rescan, so the node is in the Nodes panel now.
// False, with the reason, when the written bundle cannot be read.
bool FinishCreatedBundle(const std::filesystem::path& folder, std::string* error);

}  // namespace cyxwiz::python_calls
