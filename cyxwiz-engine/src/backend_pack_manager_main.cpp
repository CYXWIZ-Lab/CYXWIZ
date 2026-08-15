#include "core/backend_pack_manager_model.h"
#include "installer/backend_pack_installer_platform.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace {

struct Arguments {
    std::filesystem::path runtime_root;
    std::string selected_pack;
};

struct InstallBatchResult {
    bool succeeded = true;
    std::string message;
};

bool ParseArguments(
    const std::vector<std::string>& values,
    const std::filesystem::path& executable_directory,
    Arguments& output,
    std::string& error) {
    output.runtime_root = executable_directory / "runtime";
    bool runtime_seen = false;
    bool selection_seen = false;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] == "--runtime-root" && !runtime_seen &&
            index + 1 < values.size()) {
            output.runtime_root = values[++index];
            runtime_seen = true;
        } else if (values[index] == "--select" && !selection_seen &&
                   index + 1 < values.size()) {
            output.selected_pack = values[++index];
            selection_seen = true;
        } else {
            error = "Unsupported, duplicate, or incomplete installer argument";
            return false;
        }
    }
    output.runtime_root = std::filesystem::absolute(output.runtime_root);
    return true;
}

void ApplyTheme() {
    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 7.0f;
    style.FrameRounding = 4.0f;
    style.ChildRounding = 5.0f;
    style.WindowPadding = ImVec2(18.0f, 16.0f);
    style.FramePadding = ImVec2(9.0f, 6.0f);
    style.ItemSpacing = ImVec2(9.0f, 8.0f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.12f, 0.40f, 0.72f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.16f, 0.49f, 0.86f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.12f, 0.40f, 0.72f, 0.75f);
}

const char* PackState(const cyxwiz::BackendPackManagerRecord& record) {
    if (!record.delivery_metadata_available && !record.installed) {
        return "Unavailable";
    }
    if (record.update_available) return "Update available";
    if (record.active) return "Installed and active";
    if (record.installed) return "Installed";
    return "Available";
}

std::vector<std::string> SelectedPackIds(
    const std::set<std::string>& selected) {
    return {selected.begin(), selected.end()};
}

std::string Join(const std::vector<std::string>& values) {
    if (values.empty()) return "Not specified";
    std::string result;
    for (const auto& value : values) {
        if (!result.empty()) result += ", ";
        result += value;
    }
    return result;
}

void ShowFatal(const std::string& message) {
#ifdef _WIN32
    ::MessageBoxA(
        nullptr, message.c_str(), "CyxWiz Installer",
        MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
#else
    std::cerr << "CyxWiz Installer: " << message << '\n';
#endif
}

int RunInstaller(
    const std::vector<std::string>& arguments,
    const std::filesystem::path& executable_directory) {
    Arguments parsed;
    std::string argument_error;
    if (!ParseArguments(
            arguments, executable_directory, parsed, argument_error)) {
        ShowFatal(argument_error);
        return 78;
    }

    auto platform = cyxwiz::installer::CreateBackendPackInstallerPlatform(
        parsed.runtime_root, executable_directory);
    auto catalog = platform->Refresh();
    std::set<std::string> custom_selection;
    int choice = static_cast<int>(
        cyxwiz::BackendPackInstallChoice::Recommended);
    if (!parsed.selected_pack.empty()) {
        custom_selection.insert(parsed.selected_pack);
        choice = static_cast<int>(cyxwiz::BackendPackInstallChoice::Custom);
    }

    glfwSetErrorCallback([](int, const char* description) {
        std::cerr << "GLFW: " << description << '\n';
    });
    if (!glfwInit()) {
        ShowFatal("Cannot initialize the graphical window service");
        return 1;
    }
#ifdef __APPLE__
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif
    GLFWwindow* window = glfwCreateWindow(
        1020, 720, "CyxWiz Installer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        ShowFatal("Cannot create the CyxWiz Installer window");
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader(
            reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(window);
        glfwTerminate();
        ShowFatal("Cannot initialize the graphical renderer");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ApplyTheme();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    std::future<InstallBatchResult> operation;
    bool operation_running = false;
    std::string operation_message;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (operation_running && operation.valid() &&
            operation.wait_for(std::chrono::milliseconds(0)) ==
                std::future_status::ready) {
            const auto result = operation.get();
            operation_running = false;
            operation_message = result.message;
            catalog = platform->Refresh();
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::Begin(
            "CyxWiz Installer", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings);

        ImGui::Text("CyxWiz Installer");
        ImGui::SameLine();
        ImGui::TextDisabled("%s", platform->PlatformName().c_str());
        ImGui::TextWrapped(
            "Install only the compute packages this machine needs. Every package is signature-verified and locally qualified before it can authorize training.");
        ImGui::Separator();

        if (ImGui::Button("Refresh signed catalog") && !operation_running) {
            catalog = platform->Refresh();
        }
        ImGui::SameLine();
        ImGui::TextWrapped("%s", catalog.message.c_str());
        ImGui::Spacing();

        ImGui::Text("Installation choice");
        ImGui::RadioButton("Recommended", &choice, 0);
        ImGui::SameLine();
        ImGui::RadioButton("CPU only", &choice, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Custom backend packs", &choice, 2);

        if (choice == 0) {
            ImGui::TextDisabled(
                "Recommended uses detected display hardware and only catalog-supported packs.");
        } else if (choice == 1) {
            ImGui::TextDisabled(
                "CPU only keeps the required base and downloads no optional backend pack.");
        } else {
            ImGui::TextDisabled(
                "Choose individual optional packs. Diagnostic-only packs require explicit consent and still cannot authorize normal training without qualification.");
        }

        ImGui::Spacing();
        if (ImGui::BeginTable(
                "BackendPacks", 7,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                ImVec2(0.0f, -150.0f))) {
            ImGui::TableSetupColumn("Select", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("Backend");
            ImGui::TableSetupColumn("Version");
            ImGui::TableSetupColumn("Status");
            ImGui::TableSetupColumn("Support");
            ImGui::TableSetupColumn("Download");
            ImGui::TableSetupColumn("Requirements");
            ImGui::TableHeadersRow();
            for (const auto& record : catalog.records) {
                ImGui::PushID(record.pack_id.c_str());
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                const bool selectable =
                    choice == 2 && record.delivery_metadata_available &&
                    (record.catalog_support ==
                         cyxwiz::BackendPackCatalogSupport::Supported ||
                     record.catalog_support ==
                         cyxwiz::BackendPackCatalogSupport::Diagnostic);
                bool checked = choice == 0
                    ? record.recommended
                    : custom_selection.contains(record.pack_id);
                ImGui::BeginDisabled(!selectable || operation_running);
                if (ImGui::Checkbox("##selected", &checked) && choice == 2) {
                    if (checked) custom_selection.insert(record.pack_id);
                    else custom_selection.erase(record.pack_id);
                }
                ImGui::EndDisabled();
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(record.backend.c_str());
                if (record.recommended) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("Recommended");
                }
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(
                    record.package_version.empty()
                        ? "Unavailable" : record.package_version.c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::TextUnformatted(PackState(record));
                ImGui::TableSetColumnIndex(4);
                ImGui::TextUnformatted(
                    cyxwiz::BackendPackCatalogSupportName(
                        record.catalog_support));
                ImGui::TableSetColumnIndex(5);
                const auto size = cyxwiz::FormatBackendPackByteSize(
                    record.download_size_bytes);
                ImGui::TextUnformatted(size.c_str());
                ImGui::TableSetColumnIndex(6);
                const auto providers = Join(record.provider_requirements);
                ImGui::TextUnformatted(providers.c_str());
                if (ImGui::IsItemHovered()) {
                    const auto licenses = Join(record.licenses);
                    ImGui::BeginTooltip();
                    ImGui::Text("Pack: %s", record.pack_id.c_str());
                    ImGui::TextWrapped("Providers: %s", providers.c_str());
                    ImGui::TextWrapped("Licenses: %s", licenses.c_str());
                    if (!record.delivery_metadata_error.empty()) {
                        ImGui::TextWrapped(
                            "Unavailable: %s",
                            record.delivery_metadata_error.c_str());
                    }
                    ImGui::EndTooltip();
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        const auto mode = static_cast<cyxwiz::BackendPackInstallChoice>(choice);
        const auto selection = cyxwiz::ResolveBackendPackInstallerSelection(
            mode, catalog.records, SelectedPackIds(custom_selection));
        const auto plan = cyxwiz::BuildBackendPackInstallerPlan(
            selection, catalog.records);
        ImGui::TextWrapped("%s", plan.message.c_str());
        if (plan.download_size_bytes > 0) {
            ImGui::Text(
                "Required download: %s",
                cyxwiz::FormatBackendPackByteSize(
                    plan.download_size_bytes).c_str());
        }
        ImGui::TextDisabled(
            "Close CyxWiz Engine before applying package changes. Installation does not modify global PATH or system driver settings.");

        const bool can_apply = catalog.available && plan.valid &&
            !plan.pack_ids.empty() && !operation_running;
        ImGui::BeginDisabled(!can_apply);
        if (ImGui::Button("Apply selected packages")) {
            operation_running = true;
            operation_message = "Downloading and verifying signed packages...";
            const auto pack_ids = plan.pack_ids;
            auto* worker = platform.get();
            operation = std::async(
                std::launch::async, [worker, pack_ids]() {
                    InstallBatchResult batch;
                    for (const auto& pack_id : pack_ids) {
                        const auto result = worker->InstallOrUpdate(pack_id);
                        if (!batch.message.empty()) batch.message += "\n";
                        batch.message += result.message;
                        if (!result.succeeded) {
                            batch.succeeded = false;
                            break;
                        }
                    }
                    return batch;
                });
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Close") && !operation_running) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (operation_running) {
            ImGui::SameLine();
            ImGui::TextUnformatted("Working...");
        }
        if (!operation_message.empty()) {
            ImGui::TextWrapped("%s", operation_message.c_str());
        }

        ImGui::End();
        ImGui::Render();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.055f, 0.065f, 0.085f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (operation_running && operation.valid()) {
        operation.wait();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

#ifdef _WIN32
std::string Utf8(const wchar_t* value) {
    if (!value || !*value) return {};
    const int length = ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, nullptr, 0, nullptr,
        nullptr);
    if (length <= 1) return {};
    std::string output(static_cast<std::size_t>(length), '\0');
    ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, output.data(), length,
        nullptr, nullptr);
    output.pop_back();
    return output;
}
#else
std::filesystem::path ExecutableDirectory(const char* argument_zero) {
#if defined(__APPLE__)
    std::uint32_t size = 0;
    ::_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (size > 0 && ::_NSGetExecutablePath(buffer.data(), &size) == 0) {
        std::error_code error;
        const auto executable = std::filesystem::weakly_canonical(
            std::filesystem::path(buffer.data()), error);
        if (!error) return executable.parent_path();
    }
#elif defined(__linux__)
    std::vector<char> buffer(4096);
    const auto length = ::readlink(
        "/proc/self/exe", buffer.data(), buffer.size());
    if (length > 0 &&
        static_cast<std::size_t>(length) < buffer.size()) {
        return std::filesystem::path(
            std::string(buffer.data(), static_cast<std::size_t>(length)))
            .parent_path();
    }
#endif
    std::error_code error;
    const auto executable = std::filesystem::weakly_canonical(
        std::filesystem::absolute(argument_zero), error);
    return error ? std::filesystem::path{} : executable.parent_path();
}
#endif

}  // namespace

#ifdef _WIN32
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    int count = 0;
    wchar_t** wide_arguments = ::CommandLineToArgvW(
        ::GetCommandLineW(), &count);
    if (!wide_arguments) return 78;
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        arguments.push_back(Utf8(wide_arguments[index]));
    }
    ::LocalFree(wide_arguments);
    std::vector<wchar_t> path(32768);
    const DWORD length = ::GetModuleFileNameW(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) return 78;
    const auto executable_directory = std::filesystem::path(
        std::wstring(path.data(), length)).parent_path();
    return RunInstaller(arguments, executable_directory);
}
#else
int main(int argc, char** argv) {
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }
    const auto executable_directory = ExecutableDirectory(argv[0]);
    if (executable_directory.empty()) return 78;
    return RunInstaller(arguments, executable_directory);
}
#endif
