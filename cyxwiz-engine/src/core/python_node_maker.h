#pragma once

// Make a Python node (TOFIX231): the form behind the window, without ImGui.
// The form names a node and its settings and holds the bodies of the methods
// the user writes; Generate turns it into the Python source pycyxwiz.nodes
// bundles (the decorator and settings follow the form), Read takes such a
// source back into the form, Validate says in plain words what blocks it.

#include <map>
#include <string>
#include <vector>

namespace cyxwiz::node_maker {

// The kind decides the pins and the methods: an Activation is f and its
// derivative df (no settings); a Layer is forward and backward on one input;
// a Loss takes predictions and targets.
enum class Kind { Activation, Layer, Loss };

enum class SettingType { Number, WholeNumber, OnOff, Text, Choice };

struct Setting {
    std::string name;
    SettingType type = SettingType::Number;
    std::string default_value;  // as typed: "1.0", "16", "true", "relu"
    std::string lowest;         // Number / WholeNumber; empty = no bound
    std::string highest;
    std::string description;
    std::vector<std::string> choices;  // Choice
};

struct NodeForm {
    Kind kind = Kind::Layer;
    std::string display_name;  // the name in the Nodes panel
    std::string class_name;    // the Python class and the node's name in the bundle
    std::string version = "1.0.0";
    std::string category;      // the Nodes panel group
    std::string description;
    bool shown_as_activation = false;  // Layer: listed with the activations
    bool computes_shape = false;       // Layer: output_shape(self, input_shape) is written
    bool continuous_targets = false;   // Loss: target values instead of one-hot classes
    std::vector<Setting> settings;     // never for an Activation
    std::map<std::string, std::string> bodies;  // method name -> body, not indented
};

struct Pin {
    std::string name;
    std::string type;  // "Tensor", "Labels", "Loss"
};

const char* KindLabel(Kind kind);          // "Activation", "Layer", "Loss"
const char* KindNote(Kind kind);           // one line on what the kind is
const char* SettingTypeLabel(SettingType type);  // "Number", "Whole number", ...
std::vector<Pin> InputPins(Kind kind);
std::vector<Pin> OutputPins(Kind kind);

// The methods the user writes, in order, and each one's Python signature
// ("def forward(self, x):", a loss's "def forward(self, predictions, targets):").
std::vector<std::string> MethodNames(const NodeForm& form);
std::string MethodSignature(const NodeForm& form, const std::string& method);

// A working node of each kind to start from.
NodeForm Starter(Kind kind);
// The form switched to another kind. A form still showing a starter becomes
// the other kind's starter; one the user named keeps its names, group,
// description, version and (Layer <-> Loss) settings. The method bodies come
// from the new kind's starter: a Layer's forward does not fit a Loss.
NodeForm WithKind(const NodeForm& form, Kind kind);

// "Swish (beta)" -> "SwishBeta"; empty when nothing usable is left.
std::string ClassNameFrom(const std::string& display_name);
// "SoftClip" -> "soft_clip": the source file stem and the bundle folder name.
std::string FileStemFor(const std::string& class_name);
// The bundle id of one of a project's nodes: "project.<project>.<file stem>".
// One per node (the Engine loads one bundle per id), the same on every
// Create, so graphs that use the node keep finding it.
std::string BundleIdFor(const std::string& project_name, const std::string& file_stem);

// What blocks Create, in plain words; empty when the form can be written.
std::vector<std::string> Validate(const NodeForm& form);

// The node's Python source.
std::string Generate(const NodeForm& form, const std::string& bundle_id);

// A source Generate wrote, back into the form. False, with the reason, for a
// file it did not write (that one opens in the Script Editor).
bool Read(const std::string& source, NodeForm& form, std::string* error);

}  // namespace cyxwiz::node_maker
