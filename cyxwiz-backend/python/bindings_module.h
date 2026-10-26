#pragma once

// Python subclasses of cyxwiz.Module (TOFIX125 P3 step 3.2, Part C3).
//
// A Python class derived from pycyxwiz.Module is a real cyxwiz::Module: the
// trampoline below forwards Forward/Backward and the parameter methods to the
// Python overrides. Sequential.add(module) keeps the Python object alive for
// as long as the model holds it.

#include <pybind11/pybind11.h>

#include "cyxwiz/sequential.h"

namespace pycyxwiz_bindings {

// Binds pycyxwiz.Module (the base every module class derives from).
void BindModuleBase(pybind11::module_& m);

// Binds pycyxwiz.Loss, subclassable in Python like Module (TOFIX125 P7):
// forward(predictions, targets) -> one value, backward(predictions, targets)
// -> dLoss/dpredictions. Call before any built-in loss class is bound.
void BindLossBase(pybind11::module_& m);

// pycyxwiz.gradcheck and its result type: the backend's gradient checker,
// the one the Engine runs (TOFIX125 P3 step 3.3).
void BindGradientCheck(pybind11::module_& m);

// pycyxwiz.gradcheck_loss: the backend's loss gradient checker (P7).
void BindLossGradientCheck(pybind11::module_& m);

// pycyxwiz.nodes: the Python source of pycyxwiz_nodes.py, compiled in and
// run as a submodule (also importable as `import pycyxwiz.nodes`).
void InstallNodesSubmodule(pybind11::module_& m);

// Sequential.add(module): appends any pycyxwiz module, Python-defined or
// built in, without taking it from Python.
void AddModuleToSequential(cyxwiz::SequentialModel& model, pybind11::object module);

}  // namespace pycyxwiz_bindings
