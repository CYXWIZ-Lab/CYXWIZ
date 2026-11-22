#include "tutorial_content.h"

#include <algorithm>

namespace cyxwiz::tutorial {
namespace {

// The CyxWiz Studio window's id (NodeEditor::kWindowId)
const char* const kStudio = "###CyxWiz Studio";

}  // namespace

std::vector<Tutorial> BuiltInTutorials() {
    std::vector<Tutorial> out;

    out.push_back({"getting_started", "Getting Started",
                   "Learn the basics of CyxWiz Engine: the windows, the menus and where your project lives.",
                   "Getting Started",
                   {
                       {"welcome", "Welcome to CyxWiz Engine",
                        "CyxWiz Engine designs, trains and deploys machine learning models as graphs of nodes. "
                        "This tour shows the main windows.",
                        "", "Press Next to begin"},
                       {"menu_bar", "The menu bar",
                        "Every command is in the menu bar at the top: File, Edit, View, Nodes, Train, Data, Tools, "
                        "Script, Deploy, Simulation, Apps and Help. Ctrl+P opens the Command Palette, which finds any "
                        "of them by name.",
                        "", "Open a menu, or press Next"},
                       {"asset_browser", "Asset Browser",
                        "The Asset Browser lists your project's files by folder. Click a file to preview it; "
                        "double-click a table to open it in the Table Viewer.",
                        "Asset Browser", "Click a file to preview it"},
                       {"studio", "CyxWiz Studio",
                        "CyxWiz Studio is where you build a model as a graph. Right-click or use Search nodes to add a "
                        "node, drag from one pin to another to connect two, Ctrl+G groups the selected nodes and "
                        "Ctrl+F finds a node.",
                        kStudio, "Right-click the canvas to see the nodes"},
                       {"properties", "Properties",
                        "Properties shows the selected node's settings: layer sizes, activations, the data it reads. "
                        "A node with a dialog opens it on a double-click.",
                        "Properties", "Click a node to see its settings"},
                       {"console", "Console",
                        "The Console holds the Engine's logs and its terminals: the Python REPL, Commands and your "
                        "system shells (the + button adds one).",
                        "Console", "Try a Python line in the REPL"},
                       {"script_editor", "Script Editor",
                        "The Script Editor edits Python scripts and notebooks, with completion and problems shown as "
                        "you type. F5 runs the script while the editor has the focus.",
                        "Script Editor", "Open a script to edit"},
                       {"viewport", "Viewport",
                        "The Viewport shows this machine and the compute device the last run used (CUDA, OpenCL or "
                        "the CPU).",
                        "Viewport", "Check your compute devices"},
                       {"training_dashboard", "Training Dashboard",
                        "The Training Dashboard follows a run as it trains: the loss and accuracy curves, the epoch, "
                        "the speed and where it runs.",
                        "Training Dashboard", "Watch a run here"},
                       {"complete", "You're ready",
                        "You know the main windows. Next: Your First Model, or open an example graph from the empty "
                        "canvas and press Train.",
                        "", "Press Complete to finish"},
                   }});

    out.push_back({"first_model", "Your First Model",
                   "Build a small neural network for classification, step by step.", "Getting Started",
                   {
                       {"intro", "Your first model",
                        "You will build a small classifier: the data comes in, a hidden layer learns from it and an "
                        "output layer gives the answer.",
                        "", "Press Next to begin"},
                       {"add_input", "Add the data",
                        "In CyxWiz Studio type Data Input in Search nodes and add it, then double-click it to choose "
                        "a table. Data > Import Dataset... does the same.",
                        kStudio, "Add a Data Input node"},
                       {"add_dense", "Add a Dense layer",
                        "Add a Dense node the same way. A Dense layer learns patterns in your data.", kStudio,
                        "Add a Dense node"},
                       {"connect", "Connect the nodes",
                        "Drag from the Data Input's output pin to the Dense node's input pin. The link carries the "
                        "data from one to the other.",
                        kStudio, "Drag from an output pin to an input pin"},
                       {"add_output", "Add the output",
                        "Add another Dense node with one unit per class (10 for digits), then an Output node after "
                        "it.",
                        kStudio, "Add a Dense node and an Output node"},
                       {"complete", "Model built",
                        "Your network is built. Training Basics adds the loss and the optimizer and trains it.", "",
                        "Press Complete to finish"},
                   }});

    out.push_back({"training_basics", "Training Basics",
                   "Learn how to set up training, follow its progress and evaluate your model.", "Training",
                   {
                       {"intro", "Training your model",
                        "Training teaches the network to find patterns. You need data, a loss that scores its guesses "
                        "and an optimizer that improves the weights.",
                        "", "Press Next to begin"},
                       {"dataset", "Load a dataset",
                        "Use Data > Import Dataset... It adds a Data Input node; choose your file there. The Asset "
                        "Browser lists the files in your project.",
                        "Asset Browser", "Choose a table to train on"},
                       {"optimizer", "Add an optimizer",
                        "In CyxWiz Studio type Adam in Search nodes (or find it in the Nodes panel under Training) and "
                        "add it. The optimizer decides how the model learns.",
                        kStudio, "Add an Adam node"},
                       {"loss", "Add a loss",
                        "Add a loss the same way: CrossEntropy for classes, MSE Loss for numbers. Connect the Output "
                        "node and the labels to it.",
                        kStudio, "Add a loss node"},
                       {"start", "Start training",
                        "Press Train in the Studio bar, or F5. The Training Dashboard shows the loss and accuracy as "
                        "the model learns.",
                        "Training Dashboard", "Press F5"},
                       {"complete", "Well done",
                        "You know the basics of training. Try other optimizers, learning rates and layers.", "",
                        "Press Complete to finish"},
                   }});
    return out;
}

ListRow DescribeListRow(const Tutorial& tutorial, bool done, int current_step) {
    ListRow row;
    row.done = done;
    const int total = static_cast<int>(tutorial.steps.size());
    if (current_step >= 0 && current_step < total) {
        row.status = "In progress, step " + std::to_string(current_step + 1) + " of " + std::to_string(total);
        row.action = "Resume";
    } else if (done) {
        row.status = "Done";
        row.action = "Start again";
    } else {
        row.status = std::to_string(total) + (total == 1 ? " step" : " steps");
        row.action = "Start";
    }
    return row;
}

Point PlaceCard(const Box* target, float w, float h, float sw, float sh, float gap) {
    const float m = 8.0f;  // kept from the screen's edge
    const auto clamp_x = [&](float x) { return std::clamp(x, m, std::max(m, sw - w - m)); };
    const auto clamp_y = [&](float y) { return std::clamp(y, m, std::max(m, sh - h - m)); };
    if (!target || target->Width() <= 0.0f || target->Height() <= 0.0f)
        return {clamp_x((sw - w) * 0.5f), clamp_y((sh - h) * 0.5f)};
    const Box& t = *target;
    if (t.x1 + gap + w <= sw - m) return {t.x1 + gap, clamp_y(t.y0)};
    if (t.x0 - gap - w >= m) return {t.x0 - gap - w, clamp_y(t.y0)};
    if (t.y1 + gap + h <= sh - m) return {clamp_x(t.x0), t.y1 + gap};
    if (t.y0 - gap - h >= m) return {clamp_x(t.x0), t.y0 - gap - h};
    return {clamp_x(t.x0 + gap), clamp_y(t.y0 + gap)};  // a window filling the screen: inside its corner
}

}  // namespace cyxwiz::tutorial
