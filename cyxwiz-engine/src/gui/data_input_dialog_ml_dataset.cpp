// DataInputDialog ML dataset source rendering.

#include "node_config_dialog.h"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace gui {

void DataInputDialog::RenderMLDatasetOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Dataset Options");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Data Split:");
    ImGui::Spacing();

    static int split_mode = 0;
    ImGui::RadioButton("Use default split", &split_mode, 0);
    ImGui::RadioButton("Custom split", &split_mode, 1);

    if (split_mode == 1) {
        static float train_ratio = 0.8f;
        static float val_ratio = 0.1f;
        ImGui::Text("Train:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("##train", &train_ratio, 0.1f, 0.9f, "%.0f%%");
        ImGui::Text("Val:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("##val", &val_ratio, 0.0f, 0.5f, "%.0f%%");
        ImGui::Text("Test:");
        ImGui::SameLine(80);
        float test_ratio = 1.0f - train_ratio - val_ratio;
        ImGui::Text("%.0f%%", test_ratio * 100);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(accent, "Preprocessing");
    ImGui::Spacing();

    if (ImGui::Checkbox("Normalize to [0, 1]", &normalize_images_)) {
        has_changes_ = true;
    }

    static bool shuffle = true;
    if (ImGui::Checkbox("Shuffle on load", &shuffle)) {
        has_changes_ = true;
    }
}

void DataInputDialog::RenderMLDatasetSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "ML DATASET");
    ImGui::Spacing();

    int ml_idx = static_cast<int>(ml_dataset_type_);
    const char* ml_items[] = {
        "MNIST", "CIFAR-10", "CIFAR-100", "Fashion-MNIST",
        "ImageNet", "Image Folder", "HuggingFace", "Kaggle", "Custom"
    };

    ImGui::Text("Dataset:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##mldataset", &ml_idx, ml_items, 9)) {
        ml_dataset_type_ = static_cast<MLDatasetType>(ml_idx);
        has_changes_ = true;
        preview_loaded_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    switch (ml_dataset_type_) {
        case MLDatasetType::MNIST:
        case MLDatasetType::CIFAR10:
        case MLDatasetType::CIFAR100:
        case MLDatasetType::FashionMNIST:
            RenderBuiltinDatasets();
            break;
        case MLDatasetType::ImageNet:
        case MLDatasetType::ImageFolder:
            RenderImageFolderPicker();
            break;
        case MLDatasetType::HuggingFace:
            RenderHuggingFaceConfig();
            break;
        case MLDatasetType::Kaggle:
            RenderKaggleConfig();
            break;
        case MLDatasetType::Custom:
            ImGui::TextDisabled("Use File source for custom datasets");
            break;
    }
}

void DataInputDialog::RenderBuiltinDatasets() {
    const char* dataset_names[] = {"mnist", "cifar10", "cifar100", "fashion_mnist"};
    const char* descriptions[] = {
        "Handwritten digits (28x28, 10 classes, 60K train)",
        "Color images (32x32x3, 10 classes, 50K train)",
        "Color images (32x32x3, 100 classes, 50K train)",
        "Fashion items (28x28, 10 classes, 60K train)"
    };

    int idx = static_cast<int>(ml_dataset_type_);
    if (idx < 4) {
        strncpy(dataset_name_, dataset_names[idx], sizeof(dataset_name_) - 1);
        ImGui::TextWrapped("%s", descriptions[idx]);
    }

    ImGui::Spacing();

    ImGui::Text("Cache dir:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
    ImGui::InputText("##cachedir", cache_dir_, sizeof(cache_dir_));
    ImGui::SameLine();
    if (ImGui::Button("...##cachebrowse", ImVec2(80, 0))) {
        BrowseFolder();
        strncpy(cache_dir_, folder_path_, sizeof(cache_dir_) - 1);
    }

    if (strlen(cache_dir_) == 0) {
        ImGui::TextDisabled("Default: ~/.cyxwiz/datasets/");
    }

    ImGui::Spacing();

    ImGui::BeginDisabled();
    ImGui::Button("Download Dataset", ImVec2(-1, 30));
    ImGui::EndDisabled();
    ImGui::TextDisabled("ML dataset downloads are planned but not wired yet.");
}

void DataInputDialog::RenderImageFolderPicker() {
    ImGui::Text("Root folder:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    if (ImGui::InputText("##imgroot", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("...##rootbrowse", ImVec2(90, 0))) {
        BrowseFolder();
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Structure: root/class_name/image.jpg");

    if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        std::vector<std::string> classes;
        for (const auto& entry : fs::directory_iterator(folder_path_)) {
            if (entry.is_directory()) {
                classes.push_back(entry.path().filename().string());
            }
        }

        if (!classes.empty()) {
            ImGui::Text("Detected %zu classes:", classes.size());
            ImGui::BeginChild("ClassList", ImVec2(0, 100), true);
            for (const auto& c : classes) {
                ImGui::BulletText("%s", c.c_str());
            }
            ImGui::EndChild();
        }
    }
}

void DataInputDialog::RenderHuggingFaceConfig() {
    ImGui::Text("Dataset:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    if (ImGui::InputText("##hfname", dataset_name_, sizeof(dataset_name_))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("e.g., mnist, imdb, squad, coco");

    ImGui::Spacing();

    ImGui::Text("Subset:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##hfsubset", dataset_subset_, sizeof(dataset_subset_));

    ImGui::Text("Split:");
    ImGui::SameLine(90);
    static int split_idx = 0;
    const char* splits[] = {"train", "validation", "test"};
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("##hfsplit", &split_idx, splits, 3);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Auth token:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    ImGui::InputText("##hftoken", hf_token_, sizeof(hf_token_), ImGuiInputTextFlags_Password);
    ImGui::TextDisabled("Required for private/gated datasets");

    ImGui::Spacing();

    ImGui::BeginDisabled();
    ImGui::Button("Load from HuggingFace", ImVec2(-1, 30));
    ImGui::EndDisabled();
    ImGui::TextDisabled("HuggingFace dataset loading is planned but not wired yet.");
}

void DataInputDialog::RenderKaggleConfig() {
    ImGui::Text("Dataset slug:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##kaggleslug", kaggle_slug_, sizeof(kaggle_slug_));
    ImGui::TextDisabled("e.g., zalando-research/fashionmnist");

    ImGui::Spacing();

    ImGui::Text("or Competition:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    static char competition[128] = "";
    ImGui::InputText("##kagglecomp", competition, sizeof(competition));
    ImGui::TextDisabled("e.g., titanic, digit-recognizer");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextDisabled("API credentials from ~/.kaggle/kaggle.json");

    ImGui::BeginDisabled();
    ImGui::Button("Download from Kaggle", ImVec2(-1, 30));
    ImGui::EndDisabled();
    ImGui::TextDisabled("Kaggle dataset loading is planned but not wired yet.");
}

} // namespace gui
