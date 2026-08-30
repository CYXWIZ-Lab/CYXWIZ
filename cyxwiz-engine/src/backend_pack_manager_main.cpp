#include "core/backend_pack_manager_model.h"
#include "installer/backend_pack_installer_platform.h"
#include "installer/installer_frame_pacing.h"
#include "installer/installer_operation.h"
#include "installer/installer_product_removal.h"
#include "installer/installer_theme.h"
#include "installer/installer_view.h"
#include "product_removal_protocol.h"

// glad must own the OpenGL declarations before GLFW includes platform headers.
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>
#include <shellapi.h>
// clang-format on
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

#ifndef CYXWIZ_INSTALLER_DEFAULT_CATALOG_URL
#define CYXWIZ_INSTALLER_DEFAULT_CATALOG_URL ""
#endif

namespace {

enum class AsyncOperation { None, InstallPlan, CatalogRefresh };

struct Arguments {
  std::filesystem::path runtime_root;
  std::filesystem::path metadata_root;
  cyxwiz::CyxWizInstallScope scope = cyxwiz::CyxWizInstallScope::CurrentUser;
  std::string selected_pack;
  std::string catalog_url = CYXWIZ_INSTALLER_DEFAULT_CATALOG_URL;
  cyxwiz::installer::InstallerPackageSource package_source =
      cyxwiz::installer::InstallerPackageSource::CatalogHttps;
  bool product_removal_host = false;
  bool package_smoke = false;
};

struct SharedInstallProgress {
  std::mutex mutex;
  cyxwiz::installer::InstallerPlanExecutionProgress value;
};

bool ParseArguments(const std::vector<std::string> &values,
                    const std::filesystem::path &executable_directory,
                    Arguments &output, std::string &error) {
  const auto bundled_runtime = executable_directory / "runtime";
  std::error_code state_error;
  output.runtime_root =
      std::filesystem::is_regular_file(bundled_runtime / "active-runtime.json",
                                       state_error)
          ? bundled_runtime
          : cyxwiz::installer::DefaultCyxWizInstallRoot(output.scope) /
                "runtime";
  output.metadata_root = bundled_runtime;
  bool runtime_seen = false;
  bool metadata_seen = false;
  bool selection_seen = false;
  bool catalog_url_seen = false;
  bool offline_seen = false;
  bool product_removal_host_seen = false;
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
    } else if (values[index] == "--catalog-url" && !catalog_url_seen &&
               index + 1 < values.size()) {
      output.catalog_url = values[++index];
      catalog_url_seen = true;
    } else if (values[index] == "--offline" && !offline_seen) {
      output.package_source =
          cyxwiz::installer::InstallerPackageSource::OfflineSibling;
      offline_seen = true;
    } else if (values[index] == "--product-removal-host" &&
               !product_removal_host_seen) {
      output.product_removal_host = true;
      product_removal_host_seen = true;
    } else if (values[index] == "--package-smoke" && !package_smoke_seen) {
      output.package_smoke = true;
      package_smoke_seen = true;
    } else {
      error = "Unsupported, duplicate, or incomplete installer argument";
      return false;
    }
  }
  if (output.package_smoke &&
      (runtime_seen || metadata_seen || selection_seen || catalog_url_seen ||
       offline_seen || product_removal_host_seen ||
       output.scope == cyxwiz::CyxWizInstallScope::AllUsers)) {
    error = "--package-smoke cannot be combined with installer arguments";
    return false;
  }
  if (offline_seen && !metadata_seen) {
    error = "--offline requires an explicit --metadata-root";
    return false;
  }
  if (!runtime_seen && output.scope == cyxwiz::CyxWizInstallScope::AllUsers) {
    output.runtime_root =
        cyxwiz::installer::DefaultCyxWizInstallRoot(output.scope) / "runtime";
  }
  output.runtime_root = std::filesystem::absolute(output.runtime_root);
  if (runtime_seen && !metadata_seen) {
    output.metadata_root = output.runtime_root;
  }
  output.metadata_root = std::filesystem::absolute(output.metadata_root);
  return true;
}

std::string PathText(const std::filesystem::path &path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char *>(value.data()), value.size()};
}

std::future<cyxwiz::installer::InstallerPlanExecutionResult>
ApplyPlanAsync(cyxwiz::installer::BackendPackInstallerPlatform &platform,
               cyxwiz::BackendPackInstallerPlan plan,
               std::shared_ptr<SharedInstallProgress> progress) {
  return std::async(
      std::launch::async,
      [&platform, plan = std::move(plan), progress = std::move(progress)]() {
        return cyxwiz::installer::ExecuteInstallerPlan(
            platform, plan,
            [progress](
                const cyxwiz::installer::InstallerPlanExecutionProgress
                    &snapshot) {
              const std::scoped_lock lock(progress->mutex);
              auto monotonic = snapshot;
              monotonic.overall_fraction = std::max(
                  progress->value.overall_fraction,
                  snapshot.overall_fraction);
              progress->value = std::move(monotonic);
            });
      });
}

cyxwiz::CyxWizInstallScope InstallerScope(
    cyxwiz::runtime::ProductInstallScope scope) {
  return scope == cyxwiz::runtime::ProductInstallScope::AllUsers
             ? cyxwiz::CyxWizInstallScope::AllUsers
             : cyxwiz::CyxWizInstallScope::CurrentUser;
}

void ShowFatal(const std::string &message) {
#ifdef _WIN32
  ::MessageBoxA(nullptr, message.c_str(), "CyxWiz Installer",
                MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
#else
  std::cerr << "CyxWiz Installer: " << message << '\n';
#endif
}

int RunInstaller(const std::vector<std::string> &arguments,
                 const std::filesystem::path &executable_directory) {
  Arguments parsed;
  std::string argument_error;
  if (!ParseArguments(arguments, executable_directory, parsed,
                      argument_error)) {
    ShowFatal(argument_error);
    return 78;
  }
  if (parsed.package_smoke) {
    const auto metadata_root =
        std::filesystem::absolute(executable_directory / "runtime");
    auto smoke_platform = cyxwiz::installer::CreateBackendPackInstallerPlatform(
        metadata_root, metadata_root, executable_directory,
        cyxwiz::CyxWizInstallScope::CurrentUser);
    const auto smoke_catalog = smoke_platform->Refresh();
    const bool base_available =
        std::any_of(smoke_catalog.records.begin(), smoke_catalog.records.end(),
                    [](const auto &record) {
                      return record.backend == "cpu" &&
                             record.delivery_metadata_available &&
                             record.catalog_support ==
                                 cyxwiz::BackendPackCatalogSupport::Supported;
                    });
    if (!smoke_catalog.available || !base_available ||
        !cyxwiz::HasSelectableCustomBackendPack(smoke_catalog.records)) {
      std::cerr << "CyxWiz installer package smoke failed: "
                << smoke_catalog.message << '\n';
      for (const auto &record : smoke_catalog.records) {
        std::cerr << "  pack=" << record.pack_id
                  << " backend="
                  << (record.backend.empty() ? "unknown" : record.backend)
                  << " metadata="
                  << (record.delivery_metadata_available ? "verified"
                                                         : "unavailable")
                  << " catalog_support="
                  << static_cast<int>(record.catalog_support);
        if (record.compatibility) {
          std::cerr << " eligibility="
                    << static_cast<int>(record.compatibility->eligibility)
                    << " recommendation="
                    << static_cast<int>(
                           record.compatibility->install_recommendation)
                    << " rule="
                    << static_cast<int>(record.compatibility->rule);
        }
        if (!record.delivery_metadata_error.empty()) {
          std::cerr << " error='" << record.delivery_metadata_error << "'";
        }
        std::cerr << '\n';
      }
      return 1;
    }
    std::cout << "CyxWiz installer package smoke passed: "
              << smoke_catalog.catalog_id << " with "
              << smoke_catalog.records.size() << " verified packs\n";
    return 0;
  }

  auto product_removal =
      cyxwiz::installer::InspectInstallerProductRemoval(
          parsed.runtime_root, parsed.product_removal_host);
  if (product_removal.installed) {
    parsed.scope = InstallerScope(product_removal.scope);
  }
  auto install_location = cyxwiz::ResolveCyxWizInstallLocation(
      parsed.runtime_root.parent_path(), parsed.scope);
  if (!install_location.valid ||
      install_location.runtime_root != parsed.runtime_root.lexically_normal()) {
    ShowFatal(install_location.valid
                  ? "The runtime root must be the runtime directory below the "
                    "installation location"
                  : install_location.message);
    return 78;
  }
  auto platform = cyxwiz::installer::CreateBackendPackInstallerPlatform(
      install_location.runtime_root, parsed.metadata_root, executable_directory,
      install_location.scope, parsed.catalog_url, parsed.package_source);
  auto catalog = platform->Refresh();
  cyxwiz::installer::gui::InstallerViewState view_state;
  const auto initial_install_path = PathText(install_location.install_root);
  if (initial_install_path.size() >= view_state.install_path_text.size()) {
    ShowFatal("The installation location is too long");
    return 78;
  }
  std::memcpy(view_state.install_path_text.data(), initial_install_path.c_str(),
              initial_install_path.size() + 1);
  view_state.install_scope = install_location.scope;
  view_state.install_location_message = install_location.message;
  if (!parsed.selected_pack.empty()) {
    view_state.custom_selection.insert(parsed.selected_pack);
    view_state.choice = cyxwiz::BackendPackInstallChoice::Custom;
  }

  glfwSetErrorCallback([](int, const char *description) {
    std::cerr << "GLFW: " << description << '\n';
  });
  if (!glfwInit()) {
    ShowFatal("Cannot initialize the graphical window service");
    return 1;
  }
#ifdef __APPLE__
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif
  GLFWwindow *window =
      glfwCreateWindow(1240, 800, "CyxWiz Installer", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    ShowFatal("Cannot create the CyxWiz Installer window");
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSetWindowSizeLimits(window, 1040, 700, GLFW_DONT_CARE, GLFW_DONT_CARE);
  glfwSwapInterval(1);
  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    glfwDestroyWindow(window);
    glfwTerminate();
    ShowFatal("Cannot initialize the graphical renderer");
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  float content_scale_x = 1.0f;
  float content_scale_y = 1.0f;
  glfwGetWindowContentScale(window, &content_scale_x, &content_scale_y);
  const float content_scale =
      std::clamp(std::max(content_scale_x, content_scale_y), 1.0f, 2.0f);
  cyxwiz::installer::gui::ApplyInstallerTheme(content_scale);
  std::string visual_warning;
  auto visual_assets = cyxwiz::installer::gui::LoadInstallerVisualAssets(
      window, executable_directory, content_scale, visual_warning);
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  std::future<cyxwiz::installer::InstallerPlanExecutionResult> operation;
  std::future<cyxwiz::installer::InstallerCatalogRefreshResult>
      catalog_refresh;
  std::shared_ptr<SharedInstallProgress> shared_progress;
  bool operation_running = false;
  bool launch_when_complete = false;
  bool close_when_complete = false;
  AsyncOperation async_operation = AsyncOperation::None;
  std::string operation_message;
  if (!visual_warning.empty())
    operation_message = visual_warning;
  int requested_exit_code = 0;

  while (!glfwWindowShouldClose(window)) {
    cyxwiz::installer::gui::WaitForInstallerFrame(
        window, operation_running);
    if (operation_running && glfwWindowShouldClose(window)) {
      glfwSetWindowShouldClose(window, GLFW_FALSE);
      view_state.close_confirmation_requested = true;
    }
    if (operation_running && async_operation == AsyncOperation::InstallPlan &&
        operation.valid() &&
        operation.wait_for(std::chrono::milliseconds(0)) ==
            std::future_status::ready) {
      const auto result = operation.get();
      operation_running = false;
      async_operation = AsyncOperation::None;
      view_state.cancellation_requested = false;
      operation_message = result.message;
      catalog = platform->Refresh();
      product_removal = cyxwiz::installer::InspectInstallerProductRemoval(
          install_location.runtime_root,
          parsed.product_removal_host &&
              install_location.runtime_root == parsed.runtime_root);
      view_state.install_completed = result.succeeded;
      if (result.succeeded && launch_when_complete) {
        const auto launched = platform->LaunchEngine();
        if (!operation_message.empty()) operation_message += '\n';
        operation_message += launched.message;
        view_state.engine_launched = launched.succeeded;
      }
      launch_when_complete = false;
      if (close_when_complete) {
        close_when_complete = false;
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
    } else if (operation_running &&
               async_operation == AsyncOperation::CatalogRefresh &&
               catalog_refresh.valid() &&
               catalog_refresh.wait_for(std::chrono::milliseconds(0)) ==
                   std::future_status::ready) {
      const auto result = catalog_refresh.get();
      operation_running = false;
      async_operation = AsyncOperation::None;
      operation_message = result.message;
      if (shared_progress) {
        const std::scoped_lock lock(shared_progress->mutex);
        shared_progress->value.completed_steps = 1;
        shared_progress->value.overall_fraction = 1.0f;
        shared_progress->value.activity = result.succeeded
            ? "Signed catalog refresh completed"
            : "Signed catalog refresh failed; local catalog retained";
      }
      catalog = platform->Refresh();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    cyxwiz::installer::InstallerPlanExecutionProgress operation_progress;
    if (shared_progress) {
      const std::scoped_lock lock(shared_progress->mutex);
      operation_progress = shared_progress->value;
    }

    const auto action = cyxwiz::installer::gui::RenderInstallerView(
        view_state, catalog, install_location, product_removal,
        platform->PlatformName(), operation_running,
        operation_running && async_operation == AsyncOperation::InstallPlan,
        operation_message,
        operation_progress,
        visual_assets);
    switch (action.kind) {
    case cyxwiz::installer::gui::InstallerViewActionKind::RefreshCatalog:
      operation_message.clear();
      shared_progress = std::make_shared<SharedInstallProgress>();
      shared_progress->value.total_steps = 1;
      shared_progress->value.activity =
          "Downloading and verifying the signed catalog";
      catalog_refresh = std::async(
          std::launch::async,
          [&platform] { return platform->RefreshOnline(); });
      async_operation = AsyncOperation::CatalogRefresh;
      operation_running = true;
      break;
    case cyxwiz::installer::gui::InstallerViewActionKind::UseInstallLocation: {
      const auto candidate = cyxwiz::ResolveCyxWizInstallLocation(
          action.install_root, action.scope);
      view_state.install_location_message = candidate.message;
      if (candidate.valid) {
        install_location = candidate;
        platform = cyxwiz::installer::CreateBackendPackInstallerPlatform(
            install_location.runtime_root, parsed.metadata_root,
            executable_directory, install_location.scope,
            parsed.catalog_url, parsed.package_source);
        catalog = platform->Refresh();
        product_removal =
            cyxwiz::installer::InspectInstallerProductRemoval(
                install_location.runtime_root,
                parsed.product_removal_host &&
                    install_location.runtime_root == parsed.runtime_root);
        view_state.custom_selection.clear();
        view_state.choice = cyxwiz::BackendPackInstallChoice::Recommended;
        view_state.install_location_dirty = false;
      }
      break;
    }
    case cyxwiz::installer::gui::InstallerViewActionKind::ApplyPlan:
      operation_message.clear();
      view_state.install_completed = false;
      view_state.engine_launched = false;
      view_state.cancellation_requested = false;
      launch_when_complete = action.launch_after_install;
      shared_progress = std::make_shared<SharedInstallProgress>();
      shared_progress->value.activity = "Preparing installation changes";
      operation = ApplyPlanAsync(*platform, action.plan, shared_progress);
      async_operation = AsyncOperation::InstallPlan;
      operation_running = true;
      break;
    case cyxwiz::installer::gui::InstallerViewActionKind::CancelOperation: {
      const auto cancellation = platform->RequestCancellation();
      operation_message = cancellation.message;
      view_state.cancellation_requested = cancellation.succeeded;
      break;
    }
    case cyxwiz::installer::gui::InstallerViewActionKind::CancelAndClose: {
      const auto cancellation = platform->RequestCancellation();
      operation_message = cancellation.message;
      view_state.cancellation_requested = cancellation.succeeded;
      close_when_complete = cancellation.succeeded;
      break;
    }
    case cyxwiz::installer::gui::InstallerViewActionKind::LaunchEngine: {
      const auto launched = platform->LaunchEngine();
      operation_message = launched.message;
      view_state.engine_launched = launched.succeeded;
      break;
    }
    case cyxwiz::installer::gui::InstallerViewActionKind::OpenInstalledManager: {
      const auto opened = platform->OpenInstalledManager();
      operation_message = opened.message;
      if (opened.succeeded) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      break;
    }
    case cyxwiz::installer::gui::InstallerViewActionKind::RemoveProduct:
      if (cyxwiz::installer::QueueInstallerProductRemoval(
              product_removal, operation_message)) {
        requested_exit_code =
            cyxwiz::runtime::kProductRemovalRequestedExitCode;
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      break;
    case cyxwiz::installer::gui::InstallerViewActionKind::Close:
      if (operation_running) {
        view_state.close_confirmation_requested = true;
      } else {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
      break;
    case cyxwiz::installer::gui::InstallerViewActionKind::None:
      break;
    }

    ImGui::Render();
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    const auto canvas = cyxwiz::installer::gui::InstallerCanvasColor();
    glClearColor(canvas.x, canvas.y, canvas.z, canvas.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  if (async_operation == AsyncOperation::InstallPlan && operation.valid()) {
    operation.wait();
  } else if (async_operation == AsyncOperation::CatalogRefresh &&
             catalog_refresh.valid()) {
    catalog_refresh.wait();
  }
  cyxwiz::installer::gui::DestroyInstallerVisualAssets(visual_assets);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return requested_exit_code;
}

#ifdef _WIN32
std::string Utf8(const wchar_t *value) {
  if (!value || !*value)
    return {};
  const int length = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value,
                                           -1, nullptr, 0, nullptr, nullptr);
  if (length <= 1)
    return {};
  std::string output(static_cast<std::size_t>(length), '\0');
  ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, output.data(),
                        length, nullptr, nullptr);
  output.pop_back();
  return output;
}
#else
std::filesystem::path ExecutableDirectory(const char *argument_zero) {
#if defined(__APPLE__)
  std::uint32_t size = 0;
  ::_NSGetExecutablePath(nullptr, &size);
  std::vector<char> buffer(size);
  if (size > 0 && ::_NSGetExecutablePath(buffer.data(), &size) == 0) {
    std::error_code error;
    const auto executable = std::filesystem::weakly_canonical(
        std::filesystem::path(buffer.data()), error);
    if (!error)
      return executable.parent_path();
  }
#elif defined(__linux__)
  std::vector<char> buffer(4096);
  const auto length =
      ::readlink("/proc/self/exe", buffer.data(), buffer.size());
  if (length > 0 && static_cast<std::size_t>(length) < buffer.size()) {
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

} // namespace

#ifdef _WIN32
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
  int count = 0;
  wchar_t **wide_arguments = ::CommandLineToArgvW(::GetCommandLineW(), &count);
  if (!wide_arguments)
    return 78;
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    arguments.push_back(Utf8(wide_arguments[index]));
  }
  ::LocalFree(wide_arguments);
  std::vector<wchar_t> path(32768);
  const DWORD length = ::GetModuleFileNameW(nullptr, path.data(),
                                            static_cast<DWORD>(path.size()));
  if (length == 0 || length >= path.size())
    return 78;
  const auto executable_directory =
      std::filesystem::path(std::wstring(path.data(), length)).parent_path();
  return RunInstaller(arguments, executable_directory);
}
#else
int main(int argc, char **argv) {
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  const auto executable_directory = ExecutableDirectory(argv[0]);
  if (executable_directory.empty())
    return 78;
  return RunInstaller(arguments, executable_directory);
}
#endif
