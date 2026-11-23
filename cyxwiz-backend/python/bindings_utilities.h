#pragma once

// pycyxwiz.utilities (TOFIX129 D12): the work of Tools > Utilities from Python
// (units, hashes, JSON, regex, seeded random values, UUIDs, the calculator),
// checked against hashlib, json, re and numpy by test_python_utilities.py.

#include <pybind11/pybind11.h>

namespace pycyxwiz_bindings {

void BindUtilities(pybind11::module_& m);

}  // namespace pycyxwiz_bindings
