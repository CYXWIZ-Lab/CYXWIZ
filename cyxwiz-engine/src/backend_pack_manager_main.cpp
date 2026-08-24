#include "core/backend_pack_manager_model.h"
#include "installer/backend_pack_installer_platform.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
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
    std::filesystem::path metadata_root;
    cyxwiz::CyxWizInstallScope scope =
        cyxwiz::CyxWizInstallScope::CurrentUser;
    std::string selected_pack;
    bool package_smoke = false;
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
    const auto bundled_runtime = executable_directory / "runtime";
    std::error_code state_error;
    output.runtime_root = std::filesystem::is_regular_file(
        bundled_runtime / "active-runtime.json", state_error)
        ? bundled_runtime
        : cyxwiz::installer::DefaultCyxWizInstallRoot(output.scope) /
              "runtime";
    output.metadata_root = bundled_runtime;
    bool runtime_seen = false;
    bool metadata_seen = false;
    bool selection_seen = false;
    bool package_smoke_seen = false;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] == "--runtime-root" && !runtime_seen &&
            index + 1 < values.size()) {
            output.runtime_root = values[++index];
            runtime_seen = true;
        } else if (values[index] == "--metadata-root" && !metadata_seen &&
                   index + 1 < values.size()) {
            output.metadata_root = values[++index];
            metadata_seen = true;
        } else if (values[index] == "--all-users" &&
                   output.scope == cyxwiz::CyxWizInstallScope::CurrentUser) {
            output.scope = cyxwiz::CyxWizInstallScope::AllUsers;
        } else if (values[index] == "--select" && !selection_seen &&
                   index + 1 < values.size()) {
            output.selected_pack = values[++index];
            selection_seen = true;
        } else if (values[index] == "--package-smoke" &&
                   !package_smoke_seen) {
            output.package_smoke = true;
            package_smoke_seen = true;
        } else {
            error = "Unsupported, duplicate, or incomplete installer argument";
            return false;
        }
    }
    if (output.package_smoke &&
        (runtime_seen || metadata_seen || selection_seen ||
         output.scope == cyxwiz::CyxWizInstallScope::AllUsers)) {
        error = "--package-smoke cannot be combined with installer arguments";
        return false;
    }
    if (!runtime_seen &&
        output.scope == cyxwiz::CyxWizInstallScope::AllUsers) {
        output.runtime_root =
            cyxwiz::installer::DefaultCyxWizInstallRoot(output.scope) /
            "runtime";
    }
    output.runtime_root = std::filesystem::absolute(output.runtime_root);
    if (runtime_seen && !metadata_seen) {
        output.metadata_root = output.runtime_root;
    }
    output.metadata_root = std::filesystem::absolute(output.metadata_root);
    return true;
}

std::string PathText(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::filesystem::path PathFromText(const char* value) {
    const std::string_view text(value ? value : "");
    return std::filesystem::path(std::u8string(
        reinterpret_cast<const char8_t*>(text.data()), text.size()));
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

void RenderVerificationSummary(
    const cyxwiz::InstallerVerificationSummary& summary) {
    if (!ImGui::CollapsingHeader(
            "Verification results", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::TextWrapped("%s", summary.headline.c_str());
    ImGui::TextWrapped("%s", summary.performance_message.c_str());
    if (!summary.evidence_matches_runtime || summary.routes.empty()) return;

    if (ImGui::BeginTable(
            "VerificationRoutes", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Route");
        ImGui::TableSetupColumn("Result");
        ImGui::TableSetupColumn("Reason");
        ImGui::TableSetupColumn("Benchmark");
        ImGui::TableHeadersRow();
        for (const auto& route : summary.routes) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s device %d", route.backend.c_str(), route.device_id);
            if (!route.display_name.empty()) {
                ImGui::TextDisabled("%s", route.display_name.c_str());
            }
            if (!route.active) ImGui::TextDisabled("Not active");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(
                cyxwiz::InstallerRouteVerificationStatusName(route.status));
            ImGui::TableSetColumnIndex(2);
            ImGui::TextWrapped("%s", route.reason.c_str());
            if (!route.recommended_action.empty()) {
                ImGui::TextDisabled("%s", route.recommended_action.c_str());
            }
            ImGui::TableSetColumnIndex(3);
            if (route.benchmark_available) {
                ImGui::Text("%.3f ms median", route.benchmark_median_iteration_ms);
                if (route.best_measured) ImGui::TextDisabled("Best measured");
            } else {
                ImGui::TextDisabled("Not available");
            }
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled(
        "Performance comparisons use only active routes that passed the same CyxWiz benchmark contract.");
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
    if (parsed.package_smoke) {
        std::cout << "CyxWiz installer package smoke passed\n";
        return 0;
    }

    auto install_location = cyxwiz::ResolveCyxWizInstallLocation(
        parsed.runtime_root.parent_path(), parsed.scope);
    if (!install_location.valid ||
        install_location.runtime_root != parsed.runtime_root.lexically_normal()) {
        ShowFatal(install_location.valid
            ? "The runtime root must be the runtime directory below the installation location"
            : install_location.message);
        return 78;
    }
    auto platform = cyxwiz::installer::CreateBackendPackInstallerPlatform(
        install_location.runtime_root, parsed.metadata_root,
        executable_directory, install_location.scope);
    auto catalog = platform->Refresh();
    std::array<char, 2048> install_path_text{};
    const auto initial_install_path = PathText(install_location.install_root);
    if (initial_install_path.size() >= install_path_text.size()) {
        ShowFatal("The installation location is too long");
        return 78;
    }
    std::memcpy(
        install_path_text.data(), initial_install_path.c_str(),
        initial_install_path.size() + 1);
    int install_scope = static_cast<int>(install_location.scope);
    bool install_location_dirty = false;
    std::string install_location_message = install_location.message;
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
    cyxwiz::BackendPackInstallerPlan pending_plan;

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
        ImGui::SameLine();
        ImGui::TextDisabled(
            catalog.mode == cyxwiz::CyxWizInstallerMode::FreshInstall
                ? "Fresh installation" : "Modify installation");
        ImGui::TextWrapped(
            "Install only the compute packages this machine needs. Every package is signature-verified and locally qualified before it can authorize training.");
        ImGui::Separator();

        if (ImGui::Button("Refresh signed catalog") && !operation_running) {
            catalog = platform->Refresh();
        }
        ImGui::SameLine();
        ImGui::TextWrapped("%s", catalog.message.c_str());
        ImGui::Spacing();

        ImGui::Text("Installation location");
        if (catalog.mode == cyxwiz::CyxWizInstallerMode::FreshInstall) {
            ImGui::SetNextItemWidth(-145.0f);
            if (ImGui::InputText(
                    "##install-location", install_path_text.data(),
                    install_path_text.size())) {
                install_location_dirty = true;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(operation_running);
            if (ImGui::Button("Use location")) {
                const auto candidate = cyxwiz::ResolveCyxWizInstallLocation(
                    PathFromText(install_path_text.data()),
                    static_cast<cyxwiz::CyxWizInstallScope>(install_scope));
                install_location_message = candidate.message;
                if (candidate.valid) {
                    install_location = candidate;
                    platform = cyxwiz::installer::
                        CreateBackendPackInstallerPlatform(
                            install_location.runtime_root,
                            parsed.metadata_root, executable_directory,
                            install_location.scope);
                    catalog = platform->Refresh();
                    custom_selection.clear();
                    install_location_dirty = false;
                }
            }
            ImGui::EndDisabled();
            const int previous_scope = install_scope;
            ImGui::BeginDisabled(operation_running);
            ImGui::RadioButton("Install for me", &install_scope, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Install for all users", &install_scope, 1);
            ImGui::EndDisabled();
            if (install_scope != previous_scope) {
                const auto default_root =
                    cyxwiz::installer::DefaultCyxWizInstallRoot(
                        static_cast<cyxwiz::CyxWizInstallScope>(
                            install_scope));
                const auto default_text = PathText(default_root);
                if (default_text.size() < install_path_text.size()) {
                    install_path_text.fill('\0');
                    std::memcpy(
                        install_path_text.data(), default_text.c_str(),
                        default_text.size() + 1);
                }
                install_location_dirty = true;
                install_location_message = install_scope == 1
                    ? "System-wide installation requires platform authorization"
                    : "Current-user installation is the recommended least-privilege choice";
            }
            ImGui::TextDisabled(
                "%s%s", install_location_message.c_str(),
                install_location_dirty
                    ? "; select Use location to inspect this destination"
                    : "");
        } else {
            const auto installed_path = PathText(install_location.install_root);
            ImGui::TextUnformatted(installed_path.c_str());
            ImGui::TextDisabled(
                "The installation location is fixed while modifying an existing runtime.");
        }
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
                "CPU only keeps the required base, deactivates optional routes, and leaves their package files installed.");
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
                    record.backend != "cpu" && choice == 2 &&
                    record.delivery_metadata_available &&
                    (record.catalog_support ==
                         cyxwiz::BackendPackCatalogSupport::Supported ||
                     record.catalog_support ==
                         cyxwiz::BackendPackCatalogSupport::Diagnostic);
                bool checked = record.backend == "cpu" || (choice == 0
                    ? record.recommended
                    : custom_selection.contains(record.pack_id));
                ImGui::BeginDisabled(!selectable || operation_running);
                if (ImGui::Checkbox("##selected", &checked) && choice == 2) {
                    if (checked) custom_selection.insert(record.pack_id);
                    else custom_selection.erase(record.pack_id);
                }
                ImGui::EndDisabled();
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(record.backend.c_str());
                if (record.backend == "cpu") {
                    ImGui::SameLine();
                    ImGui::TextDisabled("Required");
                }
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
            selection, catalog.records, catalog.mode);
        ImGui::TextWrapped("%s", plan.message.c_str());
        if (plan.download_size_bytes > 0) {
            ImGui::Text(
                "Required download: %s",
                cyxwiz::FormatBackendPackByteSize(
                    plan.download_size_bytes).c_str());
        }
        ImGui::TextDisabled(
            "Close CyxWiz Engine before applying package changes. Installation does not modify global PATH or system driver settings.");

        const bool has_changes = plan.install_base || !plan.pack_ids.empty() ||
            !plan.deactivate_backends.empty();
        const bool can_apply = plan.valid && has_changes &&
            (plan.pack_ids.empty() || catalog.available) &&
            !install_location_dirty && install_location.valid &&
            !operation_running;
        ImGui::BeginDisabled(!can_apply);
        if (ImGui::Button("Review changes")) {
            pending_plan = plan;
            ImGui::OpenPopup("Confirm backend changes");
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
        RenderVerificationSummary(catalog.verification);

        if (ImGui::BeginPopupModal(
                "Confirm backend changes", nullptr,
                ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped(
                "Review the exact changes CyxWiz will make to this runtime.");
            if (pending_plan.install_base) {
                ImGui::Spacing();
                ImGui::Text("Install location:");
                const auto reviewed_path =
                    PathText(install_location.install_root);
                ImGui::BulletText("%s", reviewed_path.c_str());
                ImGui::BulletText(
                    "%s", install_location.requires_elevation
                        ? "All users (authorization required)"
                        : "Current user");
                ImGui::Text("Install required product component:");
                ImGui::BulletText("%s", pending_plan.base_pack_id.c_str());
            }
            if (!pending_plan.pack_ids.empty()) {
                ImGui::Spacing();
                ImGui::Text("Download, verify, and locally qualify:");
                for (const auto& pack_id : pending_plan.pack_ids) {
                    ImGui::BulletText("%s", pack_id.c_str());
                }
                ImGui::Text(
                    "Download total: %s",
                    cyxwiz::FormatBackendPackByteSize(
                        pending_plan.download_size_bytes).c_str());
            }
            if (!pending_plan.deactivate_backends.empty()) {
                ImGui::Spacing();
                ImGui::Text("Deactivate compute routes (files are kept):");
                for (const auto& backend :
                     pending_plan.deactivate_backends) {
                    ImGui::BulletText("%s", backend.c_str());
                }
            }
            ImGui::Spacing();
            ImGui::TextWrapped(
                "Close CyxWiz Engine before continuing. A package that fails local qualification will not be activated, and later changes will stop.");
            if (ImGui::Button("Apply changes")) {
                const auto pack_ids = pending_plan.pack_ids;
                const auto base_pack_id = pending_plan.base_pack_id;
                const bool install_base = pending_plan.install_base;
                const auto deactivate_backends =
                    pending_plan.deactivate_backends;
                auto* worker = platform.get();
                operation_running = true;
                operation_message = install_base
                    ? "Installing and CPU-qualifying the required CyxWiz Engine base..."
                    : (pack_ids.empty()
                          ? "Applying CPU-only configuration..."
                          : "Downloading and verifying signed packages...");
                operation = std::async(
                    std::launch::async,
                    [worker, install_base, base_pack_id, pack_ids,
                     deactivate_backends]() {
                        InstallBatchResult batch;
                        if (install_base) {
                            const auto result =
                                worker->InstallBase(base_pack_id);
                            batch.message = result.message;
                            if (!result.succeeded || !result.activated) {
                                batch.succeeded = false;
                                return batch;
                            }
                        }
                        for (const auto& pack_id : pack_ids) {
                            const auto result =
                                worker->InstallOrUpdate(pack_id);
                            if (!batch.message.empty()) batch.message += "\n";
                            batch.message += result.message;
                            if (!result.succeeded || !result.activated) {
                                batch.succeeded = false;
                                return batch;
                            }
                        }
                        for (const auto& backend : deactivate_backends) {
                            const auto result =
                                worker->DeactivateBackend(backend);
                            if (!batch.message.empty()) batch.message += "\n";
                            batch.message += result.message;
                            if (!result.succeeded) {
                                batch.succeeded = false;
                                return batch;
                            }
                        }
                        return batch;
                    });
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
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
