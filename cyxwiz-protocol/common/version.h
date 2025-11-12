#pragma once

#include <string>
#include "common.pb.h"

namespace cyxwiz {
namespace protocol {

// Get version as string
std::string GetVersionString();

// Get version as protobuf message
Version GetVersion();

} // namespace protocol
} // namespace cyxwiz
