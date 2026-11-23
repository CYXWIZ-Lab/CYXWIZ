#pragma once

// Whether this node is in a training group. A node in a group gives its
// compute device to the group: it takes no single job until the group ends.

#include <atomic>

namespace cyxwiz::servernode {

inline std::atomic<bool>& NodeInGroup() {
    static std::atomic<bool> in_group{false};
    return in_group;
}

inline const char* kNodeInGroupRefusal =
    "This node is training in a group and takes no other job until the group ends";

}  // namespace cyxwiz::servernode
