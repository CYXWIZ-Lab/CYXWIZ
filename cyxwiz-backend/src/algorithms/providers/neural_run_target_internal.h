#pragma once

// Internal to the backend (neural_provider_registry.cpp keeps the state,
// neural_provider_dispatch.cpp reads it); not part of the plugin SDK.

namespace cyxwiz::detail {

// Whether this thread's ArrayFire device was read and found to be the bound
// device of its innermost run scope (ScopedNeuralRunTarget). Every scope
// starts and ends unverified, so a scope that switches the thread's device
// (PolicyDeviceScope) is read again, and so is the scope around it after it.
bool NeuralRunThreadVerified();
void MarkNeuralRunThreadVerified();

} // namespace cyxwiz::detail
