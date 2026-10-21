// file_dialogs.cpp - Cross-platform native file dialog implementation
#include "file_dialogs.h"
#include "project_manager.h"
#include <spdlog/spdlog.h>
#include <nfd.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <nfd_glfw3.h>

#include <filesystem>

namespace cyxwiz {

namespace {

// The folder a dialog opens in (TOFIX157): the caller's folder, the folder of
// a file path, or the nearest folder above it that exists (the native dialog
// ignores a folder that does not exist); with none given, the open project's
// folder. In the platform's own form (backslashes on Windows). The native
// dialog is built to open exactly here, not where Windows last was.
std::string StartFolder(const char* requested) {
    namespace fs = std::filesystem;
    fs::path folder;
    if (requested && *requested) {
        folder = fs::path(requested);
    } else if (ProjectManager::Instance().HasActiveProject()) {
        folder = fs::path(ProjectManager::Instance().GetProjectRoot());
    } else {
        return {};
    }
    std::error_code error;
    if (fs::is_regular_file(folder, error)) folder = folder.parent_path();
    while (!folder.empty() && !fs::is_directory(folder, error)) {
        const fs::path parent = folder.parent_path();
        if (parent == folder) return {};
        folder = parent;
    }
    folder = fs::absolute(folder, error);
    return folder.make_preferred().string();
}

const char* OrNull(const std::string& folder) { return folder.empty() ? nullptr : folder.c_str(); }

// The Engine window the dialog belongs to (the UI thread's GLFW window): the
// dialog opens centred on it and stays in front of it, instead of where
// Windows last put a dialog (even on a screen no longer attached).
nfdwindowhandle_t ParentWindow() {
    nfdwindowhandle_t window{};
    if (GLFWwindow* current = glfwGetCurrentContext()) NFD_GetNativeWindowFromGLFWWindow(current, &window);
    return window;
}

}  // namespace

bool FileDialogs::initialized_ = false;

void FileDialogs::EnsureInitialized() {
    if (!initialized_) {
        if (NFD_Init() != NFD_OKAY) {
            spdlog::error("FileDialogs: Failed to initialize NFD: {}", NFD_GetError());
        } else {
            initialized_ = true;
            spdlog::debug("FileDialogs: NFD initialized");
        }
    }
}

std::optional<std::string> FileDialogs::OpenFile(
    const char* /*title*/,
    const FilterList& filters,
    const char* default_path
) {
    EnsureInitialized();

    // Build filter list
    std::vector<nfdfilteritem_t> nfd_filters;
    for (const auto& [name, spec] : filters) {
        nfd_filters.push_back({name.c_str(), spec.c_str()});
    }

    const std::string start = StartFolder(default_path);
    nfdopendialogu8args_t args{};
    args.filterList = nfd_filters.empty() ? nullptr : nfd_filters.data();
    args.filterCount = static_cast<nfdfiltersize_t>(nfd_filters.size());
    args.defaultPath = OrNull(start);
    args.parentWindow = ParentWindow();
    nfdchar_t* out_path = nullptr;
    nfdresult_t result = NFD_OpenDialogU8_With(&out_path, &args);

    if (result == NFD_OKAY && out_path) {
        std::string path(out_path);
        NFD_FreePath(out_path);
        spdlog::debug("FileDialogs: OpenFile selected: {}", path);
        return path;
    } else if (result == NFD_CANCEL) {
        spdlog::debug("FileDialogs: OpenFile cancelled");
        return std::nullopt;
    } else {
        spdlog::error("FileDialogs: OpenFile error: {}", NFD_GetError());
        return std::nullopt;
    }
}

std::vector<std::string> FileDialogs::OpenMultiple(
    const char* /*title*/,
    const FilterList& filters,
    const char* default_path
) {
    EnsureInitialized();

    std::vector<std::string> paths;

    // Build filter list
    std::vector<nfdfilteritem_t> nfd_filters;
    for (const auto& [name, spec] : filters) {
        nfd_filters.push_back({name.c_str(), spec.c_str()});
    }

    const nfdpathset_t* out_paths = nullptr;
    const std::string start = StartFolder(default_path);
    nfdopendialogu8args_t args{};
    args.filterList = nfd_filters.empty() ? nullptr : nfd_filters.data();
    args.filterCount = static_cast<nfdfiltersize_t>(nfd_filters.size());
    args.defaultPath = OrNull(start);
    args.parentWindow = ParentWindow();
    nfdresult_t result = NFD_OpenDialogMultipleU8_With(&out_paths, &args);

    if (result == NFD_OKAY && out_paths) {
        nfdpathsetsize_t count;
        if (NFD_PathSet_GetCount(out_paths, &count) == NFD_OKAY) {
            for (nfdpathsetsize_t i = 0; i < count; ++i) {
                nfdchar_t* path = nullptr;
                if (NFD_PathSet_GetPath(out_paths, i, &path) == NFD_OKAY && path) {
                    paths.emplace_back(path);
                    NFD_FreePath(path);
                }
            }
        }
        NFD_PathSet_Free(out_paths);
        spdlog::debug("FileDialogs: OpenMultiple selected {} files", paths.size());
    } else if (result == NFD_CANCEL) {
        spdlog::debug("FileDialogs: OpenMultiple cancelled");
    } else {
        spdlog::error("FileDialogs: OpenMultiple error: {}", NFD_GetError());
    }

    return paths;
}

std::optional<std::string> FileDialogs::SaveFile(
    const char* /*title*/,
    const FilterList& filters,
    const char* default_path,
    const char* default_name
) {
    EnsureInitialized();

    // Build filter list
    std::vector<nfdfilteritem_t> nfd_filters;
    for (const auto& [name, spec] : filters) {
        nfd_filters.push_back({name.c_str(), spec.c_str()});
    }

    nfdchar_t* out_path = nullptr;
    const std::string start = StartFolder(default_path);
    nfdsavedialogu8args_t args{};
    args.filterList = nfd_filters.empty() ? nullptr : nfd_filters.data();
    args.filterCount = static_cast<nfdfiltersize_t>(nfd_filters.size());
    args.defaultPath = OrNull(start);
    args.defaultName = default_name;
    args.parentWindow = ParentWindow();
    nfdresult_t result = NFD_SaveDialogU8_With(&out_path, &args);

    if (result == NFD_OKAY && out_path) {
        std::string path(out_path);
        NFD_FreePath(out_path);
        spdlog::debug("FileDialogs: SaveFile selected: {}", path);
        return path;
    } else if (result == NFD_CANCEL) {
        spdlog::debug("FileDialogs: SaveFile cancelled");
        return std::nullopt;
    } else {
        spdlog::error("FileDialogs: SaveFile error: {}", NFD_GetError());
        return std::nullopt;
    }
}

std::optional<std::string> FileDialogs::SelectFolder(
    const char* /*title*/,
    const char* default_path
) {
    EnsureInitialized();

    nfdchar_t* out_path = nullptr;
    const std::string start = StartFolder(default_path);
    nfdpickfolderu8args_t args{};
    args.defaultPath = OrNull(start);
    args.parentWindow = ParentWindow();
    nfdresult_t result = NFD_PickFolderU8_With(&out_path, &args);

    if (result == NFD_OKAY && out_path) {
        std::string path(out_path);
        NFD_FreePath(out_path);
        spdlog::debug("FileDialogs: SelectFolder selected: {}", path);
        return path;
    } else if (result == NFD_CANCEL) {
        spdlog::debug("FileDialogs: SelectFolder cancelled");
        return std::nullopt;
    } else {
        spdlog::error("FileDialogs: SelectFolder error: {}", NFD_GetError());
        return std::nullopt;
    }
}

// =========================================================================
// Convenience methods
// =========================================================================

std::optional<std::string> FileDialogs::OpenProject(const char* default_path) {
    return OpenFile("Open Project", {
        {"CyxWiz Projects", "cyxwiz"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::SaveProject(const char* default_path) {
    return SaveFile("Save Project", {
        {"CyxWiz Projects", "cyxwiz"}
    }, default_path, "project.cyxwiz");
}

std::optional<std::string> FileDialogs::OpenGraph(const char* default_path) {
    return OpenFile("Open Graph", {
        {"CyxWiz Graphs", "cyxgraph"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::SaveGraph(const char* default_path) {
    return SaveFile("Save Graph", {
        {"CyxWiz Graphs", "cyxgraph"}
    }, default_path, "graph.cyxgraph");
}

std::optional<std::string> FileDialogs::OpenScript(const char* default_path) {
    return OpenFile("Open Script", {
        {"Python Scripts", "py"},
        {"CyxWiz Scripts", "cyx"},
        {"Jupyter Notebooks", "ipynb"},
        {"All Scripts", "py,cyx,ipynb"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::SaveScript(const char* default_path) {
    return SaveFile("Save Script", {
        {"Python Scripts", "py"},
        {"CyxWiz Scripts", "cyx"},
        {"Jupyter Notebooks", "ipynb"}
    }, default_path, "script.py");
}

std::optional<std::string> FileDialogs::OpenModel(const char* default_path) {
    return OpenFile("Open Model", {
        {"CyxWiz Models", "cyxmodel"},
        {"ONNX Models", "onnx"},
        {"Safetensors", "safetensors"},
        {"GGUF Models", "gguf"},
        {"PyTorch Models", "pt,pth"},
        {"TensorFlow Models", "pb,h5"},
        {"Keras Models", "keras,h5"},
        {"All Models", "cyxmodel,onnx,safetensors,gguf,pt,pth,pb,h5,keras"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::SaveModel(const char* default_path) {
    return SaveFile("Save Model", {
        {"CyxWiz Models", "cyxmodel"},
        {"ONNX Models", "onnx"},
        {"Safetensors", "safetensors"},
        {"GGUF Models", "gguf"},
        {"PyTorch Models", "pt"},
        {"All Files", "*"}
    }, default_path, "model.cyxmodel");
}

std::optional<std::string> FileDialogs::OpenDataset(const char* default_path) {
    return OpenFile("Open Dataset", {
        {"CSV Files", "csv"},
        {"HDF5 Files", "h5,hdf5"},
        {"NumPy Files", "npy,npz"},
        {"All Datasets", "csv,h5,hdf5,npy,npz"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::OpenImage(const char* default_path) {
    return OpenFile("Open Image", {
        {"Images", "png,jpg,jpeg,bmp,gif,tiff"},
        {"PNG Files", "png"},
        {"JPEG Files", "jpg,jpeg"},
        {"All Files", "*"}
    }, default_path);
}

std::vector<std::string> FileDialogs::OpenImages(const char* default_path) {
    return OpenMultiple("Open Images", {
        {"Images", "png,jpg,jpeg,bmp,gif,tiff"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::SelectDatasetFolder(const char* default_path) {
    return SelectFolder("Select Dataset Folder", default_path);
}

std::optional<std::string> FileDialogs::SelectOutputFolder(const char* default_path) {
    return SelectFolder("Select Output Folder", default_path);
}

std::optional<std::string> FileDialogs::OpenTheme(const char* default_path) {
    return OpenFile("Open Theme", {
        {"Theme Files", "json"},
        {"All Files", "*"}
    }, default_path);
}

std::optional<std::string> FileDialogs::SaveTheme(const char* default_path, const char* default_name) {
    return SaveFile("Save Theme", {
        {"Theme Files", "json"},
        {"All Files", "*"}
    }, default_path, default_name);
}

} // namespace cyxwiz
