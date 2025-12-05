// node_editor_connection.cpp
// Connection Tracking Module for NodeEditor
//
// This module contains connection tracking and validation functions for variadic pins
// in the visual node editor. It provides utilities to query pin connections, validate
// new connections, and manage connection limits.

#include "node_editor.h"
#include <vector>
#include <string>

namespace gui {

// ========== Connection Tracking for Variadic Pins ==========

const NodePin* NodeEditor::FindPinById(int pin_id) const {
    for (const auto& node : nodes_) {
        for (const auto& pin : node.inputs) {
            if (pin.id == pin_id) {
                return &pin;
            }
        }
        for (const auto& pin : node.outputs) {
            if (pin.id == pin_id) {
                return &pin;
            }
        }
    }
    return nullptr;
}

int NodeEditor::GetConnectionCount(int pin_id) const {
    int count = 0;
    for (const auto& link : links_) {
        if (link.from_pin == pin_id || link.to_pin == pin_id) {
            count++;
        }
    }
    return count;
}

std::vector<int> NodeEditor::GetConnectedPins(int pin_id) const {
    std::vector<int> connected;
    for (const auto& link : links_) {
        if (link.from_pin == pin_id) {
            connected.push_back(link.to_pin);
        } else if (link.to_pin == pin_id) {
            connected.push_back(link.from_pin);
        }
    }
    return connected;
}

bool NodeEditor::IsPinFull(int pin_id) const {
    const NodePin* pin = FindPinById(pin_id);
    if (!pin) return true;  // Unknown pin is considered full

    if (pin->max_connections == PIN_UNLIMITED) {
        return false;  // Never full
    }

    return GetConnectionCount(pin_id) >= pin->max_connections;
}

bool NodeEditor::IsPinConnected(int pin_id) const {
    return GetConnectionCount(pin_id) > 0;
}

bool NodeEditor::CanAcceptConnection(int pin_id) const {
    return !IsPinFull(pin_id);
}

std::vector<NodeLink> NodeEditor::GetLinksToPin(int pin_id) const {
    std::vector<NodeLink> result;
    for (const auto& link : links_) {
        if (link.to_pin == pin_id) {
            result.push_back(link);
        }
    }
    return result;
}

std::vector<NodeLink> NodeEditor::GetLinksFromPin(int pin_id) const {
    std::vector<NodeLink> result;
    for (const auto& link : links_) {
        if (link.from_pin == pin_id) {
            result.push_back(link);
        }
    }
    return result;
}

bool NodeEditor::ValidateLink(int from_pin, int to_pin, std::string& error) const {
    const NodePin* source = FindPinById(from_pin);
    const NodePin* target = FindPinById(to_pin);

    if (!source || !target) {
        error = "Invalid pin ID";
        return false;
    }

    // Source must be output, target must be input
    if (source->is_input) {
        error = "Source must be an output pin";
        return false;
    }
    if (!target->is_input) {
        error = "Target must be an input pin";
        return false;
    }

    // Type compatibility check
    if (source->type != target->type) {
        // Allow some compatible type combinations
        bool compatible = false;
        // Tensor is compatible with most data types for flexibility
        if (source->type == PinType::Tensor || target->type == PinType::Tensor) {
            compatible = true;
        }
        if (!compatible) {
            error = "Incompatible pin types";
            return false;
        }
    }

    // Check if target pin can accept more connections
    if (IsPinFull(to_pin)) {
        error = "Target pin has reached maximum connections";
        return false;
    }

    // Check for duplicate links
    for (const auto& link : links_) {
        if (link.from_pin == from_pin && link.to_pin == to_pin) {
            error = "Connection already exists";
            return false;
        }
    }

    // Find parent nodes to check for self-connection
    int from_node_id = -1, to_node_id = -1;
    for (const auto& node : nodes_) {
        for (const auto& pin : node.outputs) {
            if (pin.id == from_pin) from_node_id = node.id;
        }
        for (const auto& pin : node.inputs) {
            if (pin.id == to_pin) to_node_id = node.id;
        }
    }

    if (from_node_id == to_node_id && from_node_id != -1) {
        error = "Cannot connect node to itself";
        return false;
    }

    return true;
}

}  // namespace gui
