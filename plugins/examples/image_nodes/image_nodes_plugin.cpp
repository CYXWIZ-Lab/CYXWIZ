#include "image_nodes_plugin.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin::examples {

// =============================================================================
// IPlugin Lifecycle
// =============================================================================

bool ImageNodesPlugin::OnLoad(PluginContext& ctx) {
    ctx.LogInfo("Image Nodes: OnLoad");
    state_ = PluginState::Loaded;
    return true;
}

bool ImageNodesPlugin::OnInitialize(PluginContext& ctx) {
    ctx.LogInfo("Image Nodes: OnInitialize");

    // Register node types — requires UIModify permission
    if (!ctx.RegisterNodeProvider(this)) {
        ctx.LogError("Image Nodes: Failed to register node provider");
        state_ = PluginState::Failed;
        return false;
    }

    // Register settings panel — requires UIModify permission
    if (!ctx.RegisterPanelProvider(this)) {
        ctx.LogError("Image Nodes: Failed to register panel provider");
        state_ = PluginState::Failed;
        return false;
    }

    ctx.LogInfo("Image Nodes: Registered 3 node types and 1 panel");
    state_ = PluginState::Active;
    return true;
}

void ImageNodesPlugin::OnShutdown(PluginContext& ctx) {
    ctx.LogInfo("Image Nodes: OnShutdown");
    state_ = PluginState::Loaded;
}

void ImageNodesPlugin::OnUnload(PluginContext& ctx) {
    ctx.LogInfo("Image Nodes: OnUnload");
    state_ = PluginState::Unloaded;
}

// =============================================================================
// INodeProvider - Custom Node Types
// =============================================================================

std::vector<PluginNodeTypeInfo> ImageNodesPlugin::GetNodeTypes() {
    std::vector<PluginNodeTypeInfo> nodes;

    // --- Gaussian Blur ---
    {
        PluginNodeTypeInfo info;
        info.type_name = "GaussianBlur";
        info.display_name = "Gaussian Blur";
        info.category = "Image Processing";
        info.description = "Apply Gaussian blur filter to an image tensor";
        info.color = 0xFF88AA44;  // ABGR: green-ish

        info.pins.push_back({"input",  "Tensor", true});   // input pin
        info.pins.push_back({"output", "Tensor", false});   // output pin

        info.default_parameters["kernel_size"] = "5";
        info.default_parameters["sigma"] = "1.0";

        nodes.push_back(std::move(info));
    }

    // --- Edge Detection ---
    {
        PluginNodeTypeInfo info;
        info.type_name = "EdgeDetect";
        info.display_name = "Edge Detection";
        info.category = "Image Processing";
        info.description = "Detect edges using Sobel or Canny operator";
        info.color = 0xFF4488CC;  // ABGR: blue-ish

        info.pins.push_back({"input",  "Tensor", true});
        info.pins.push_back({"edges",  "Tensor", false});

        info.default_parameters["method"] = "sobel";  // "sobel" or "canny"
        info.default_parameters["threshold"] = "0.5";

        nodes.push_back(std::move(info));
    }

    // --- Normalize ---
    {
        PluginNodeTypeInfo info;
        info.type_name = "Normalize";
        info.display_name = "Normalize";
        info.category = "Image Processing";
        info.description = "Normalize tensor with mean and std (e.g. ImageNet normalization)";
        info.color = 0xFFCC8844;  // ABGR: orange

        info.pins.push_back({"input",      "Tensor", true});
        info.pins.push_back({"normalized", "Tensor", false});

        info.default_parameters["mean"] = "0.485,0.456,0.406";
        info.default_parameters["std"]  = "0.229,0.224,0.225";

        nodes.push_back(std::move(info));
    }

    return nodes;
}

std::string ImageNodesPlugin::GenerateCode(
    const std::string& node_type_name,
    const std::map<std::string, std::string>& parameters,
    const std::string& framework)
{
    if (framework != "pytorch") {
        return "# " + node_type_name + ": only PyTorch code generation is supported\n";
    }

    auto get_param = [&](const std::string& key, const std::string& fallback) -> std::string {
        auto it = parameters.find(key);
        return (it != parameters.end()) ? it->second : fallback;
    };

    if (node_type_name == "GaussianBlur") {
        std::string ks = get_param("kernel_size", "5");
        std::string sigma = get_param("sigma", "1.0");
        return "import torchvision.transforms.functional as TF\n"
               "output = TF.gaussian_blur(input, kernel_size=[" + ks + ", " + ks + "], "
               "sigma=" + sigma + ")\n";
    }

    if (node_type_name == "EdgeDetect") {
        std::string method = get_param("method", "sobel");
        std::string threshold = get_param("threshold", "0.5");
        if (method == "sobel") {
            return "import torch.nn.functional as F\n"
                   "sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)\n"
                   "sobel_y = sobel_x.transpose(2, 3)\n"
                   "gx = F.conv2d(input, sobel_x, padding=1)\n"
                   "gy = F.conv2d(input, sobel_y, padding=1)\n"
                   "edges = torch.sqrt(gx**2 + gy**2)\n"
                   "edges = (edges > " + threshold + ").float()\n";
        }
        return "# Canny edge detection requires OpenCV\n"
               "import cv2, torch, numpy as np\n"
               "np_img = (input.squeeze().cpu().numpy() * 255).astype(np.uint8)\n"
               "edges = torch.from_numpy(cv2.Canny(np_img, 100, 200).astype(np.float32) / 255.0)\n";
    }

    if (node_type_name == "Normalize") {
        std::string mean = get_param("mean", "0.485,0.456,0.406");
        std::string std_val = get_param("std", "0.229,0.224,0.225");
        return "import torchvision.transforms as T\n"
               "normalize = T.Normalize(mean=[" + mean + "], std=[" + std_val + "])\n"
               "normalized = normalize(input)\n";
    }

    return "# Unknown node type: " + node_type_name + "\n";
}

// =============================================================================
// IPanelProvider - Settings Panel
// =============================================================================

std::vector<PluginPanelInfo> ImageNodesPlugin::GetPanels() {
    PluginPanelInfo info;
    info.panel_id = "com.cyxwiz.examples.image-nodes.settings";
    info.title = "Image Nodes Settings";
    info.category = "Tools";
    info.icon = "";  // Use default plug icon
    info.show_by_default = false;

    return {info};
}

void ImageNodesPlugin::RenderPanel(const std::string& panel_id, bool* visible) {
    if (panel_id != "com.cyxwiz.examples.image-nodes.settings") return;

    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Image Nodes Settings", visible)) {
        ImGui::End();
        return;
    }

    ImGui::Text("Default Settings");
    ImGui::Separator();

    ImGui::SliderInt("Default Kernel Size", &default_kernel_size_, 1, 31);
    if (default_kernel_size_ % 2 == 0) default_kernel_size_++;  // Must be odd

    ImGui::Checkbox("Use GPU Acceleration", &use_gpu_);

    ImGui::Spacing();
    ImGui::TextDisabled("These settings apply to newly created nodes.");

    ImGui::End();
}

} // namespace cyxwiz::plugin::examples

// =============================================================================
// Plugin Entry Point
// =============================================================================

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::ImageNodesPlugin)
