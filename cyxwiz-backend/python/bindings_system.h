#pragma once

// pycyxwiz.system (TOFIX129 D13): the Engine's own memory figures from Python
// (what it holds on the compute device, what the device has free, host
// tensors) and the release of unused device memory. The same functions the
// Monitor window reads; checked by test_python_system.py.

#include <pybind11/pybind11.h>

namespace pycyxwiz_bindings {

void BindSystem(pybind11::module_& m);

}  // namespace pycyxwiz_bindings
