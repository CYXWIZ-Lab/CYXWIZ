#include "../src/core/async_task_manager.h"
#include "../src/core/hdf5_input_settings.h"
#include "../src/gui/hdf5_source_inspector.h"

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/highfive.hpp>
#endif

#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class ManagerFixture {
public:
    ManagerFixture() {
        auto& manager = cyxwiz::AsyncTaskManager::Instance();
        manager.Shutdown(std::chrono::milliseconds(500));
        manager.Initialize(1);
    }

    ~ManagerFixture() {
        cyxwiz::AsyncTaskManager::Instance().Shutdown(std::chrono::seconds(5));
    }
};

class ImGuiFixture {
public:
    ImGuiFixture() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DeltaTime = 1.0f / 60.0f;
        unsigned char* pixels = nullptr;
        int width = 0;
        int height = 0;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    }

    ~ImGuiFixture() {
        ImGui::DestroyContext();
    }

    template <typename RenderBody>
    void Frame(float width, RenderBody&& body) {
        auto& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(width, 720.0f);
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(width, 680.0f), ImGuiCond_Always);
        ImGui::Begin("HDF5 source inspector test");
        body();
        ImGui::End();
        ImGui::Render();
        const auto* draw = ImGui::GetDrawData();
        Check(draw != nullptr && draw->TotalVtxCount > 0,
              "headless inspector render should produce draw data");
    }
};

bool WaitUntil(const std::function<bool()>& predicate,
               std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    return predicate();
}

void PollUntilIdle(gui::Hdf5SourceInspector& inspector,
                   std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    Check(WaitUntil([&] {
        inspector.Poll();
        return !inspector.Busy();
    }, timeout), "inspector task should finish within the bounded wait");
    inspector.Poll();
}

bool AnyInspectorTaskRunning() {
    for (const auto& task : cyxwiz::AsyncTaskManager::Instance().GetActiveTasks()) {
        if (task.name == "Inspect HDF5 source" &&
            task.state == cyxwiz::TaskState::Running) {
            return true;
        }
    }
    return false;
}

std::optional<uint64_t> FindInspectorTaskId() {
    for (const auto& task : cyxwiz::AsyncTaskManager::Instance().GetActiveTasks()) {
        if (task.name == "Inspect HDF5 source") {
            return task.id;
        }
    }
    return std::nullopt;
}

void WaitForManagerIdle(std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    Check(WaitUntil([] {
        return cyxwiz::AsyncTaskManager::Instance().GetActiveTaskCount() == 0;
    }, timeout), "async task manager should drain within the bounded wait");
}

class WorkerBlocker {
public:
    explicit WorkerBlocker(const std::string& name) {
        auto started = std::make_shared<std::promise<void>>();
        auto started_future = started->get_future();
        release_ = release_promise_.get_future().share();
        cyxwiz::AsyncTaskManager::Instance().Submit(std::make_shared<cyxwiz::LambdaTask>(
            name,
            [started, release = release_](cyxwiz::LambdaTask& task) mutable {
                started->set_value();
                while (release.wait_for(std::chrono::milliseconds(2)) !=
                       std::future_status::ready) {
                    if (task.ShouldStop()) {
                        task.MarkCancelled("blocker cancelled");
                        return;
                    }
                }
                task.MarkCompleted("blocker released");
            },
            true));
        Check(started_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
              name + " should occupy the only worker");
    }

    ~WorkerBlocker() {
        Release();
    }

    void Release() {
        if (!released_) {
            release_promise_.set_value();
            released_ = true;
        }
    }

private:
    std::promise<void> release_promise_;
    std::shared_future<void> release_;
    bool released_ = false;
};

void TestRestoreSettingsWithoutTasks() {
    auto& manager = cyxwiz::AsyncTaskManager::Instance();
    const auto recent_before = manager.GetRecentTasks();
    const auto active_before = manager.GetActiveTaskCount();
    const auto check_idle = [&](const gui::Hdf5SourceInspector& inspector) {
        Check(!inspector.Busy() && !inspector.Page().ok && inspector.Page().rows.empty(),
              "restoring settings must not load a preview or schedule inspection");
        Check(manager.GetActiveTaskCount() == active_before,
              "restoring settings must not add active tasks");
        const auto recent = manager.GetRecentTasks();
        Check(recent.size() == recent_before.size() &&
                  (recent.empty() || recent.front().id == recent_before.front().id),
              "restoring settings must not run a task to completion");
    };

    cyxwiz::Hdf5InputSettings settings;
    settings.selection = {"/saved/" + std::string(160, 'f'), "/saved/labels"};
    settings.numeric_policy = cyxwiz::Hdf5NumericPolicy::Float64;
    settings.max_materialized_bytes = 1024;
    std::map<std::string, std::string> canonical{{"file_path", "saved-source.h5"}};
    std::string error;
    Check(cyxwiz::WriteHdf5InputSettings(settings, canonical, error),
          "canonical restore fixture should serialize: " + error);
    const auto saved = canonical;

    gui::Hdf5SourceInspector inspector;
    inspector.SetSource(canonical.at("file_path"));
    inspector.RestoreSettings(canonical);
    Check(inspector.Selection().data_path == settings.selection.data_path &&
              inspector.Selection().label_path == settings.selection.label_path &&
              inspector.Error().empty(),
          "canonical restoration should preserve both paths without legacy-buffer truncation");
    check_idle(inspector);
    Check(canonical == saved, "restoration must not change saved graph parameters");

    const std::map<std::string, std::string> legacy{{"hdf5_dataset", "group/features"}};
    inspector.RestoreSettings(legacy);
    Check(inspector.Selection().data_path == "/group/features" &&
              inspector.Selection().label_path.empty() && inspector.Error().empty(),
          "legacy restoration should normalize the data path and clear old labels");
    check_idle(inspector);

    std::vector<std::map<std::string, std::string>> invalid_settings;
    auto invalid = canonical;
    invalid["hdf5_selection_version"] = "unsupported";
    invalid_settings.push_back(invalid);
    invalid = canonical;
    invalid["hdf5_data_path"].clear();
    invalid_settings.push_back(invalid);
    invalid = canonical;
    invalid["hdf5_data_path"] = "/" + std::string(4096, 'x');
    invalid_settings.push_back(invalid);
    invalid_settings.push_back({{"hdf5_dataset", ""}});
    for (const auto& parameters : invalid_settings) {
        inspector.RestoreSettings(canonical);
        const auto parsed = cyxwiz::ReadHdf5InputSettings(parameters);
        Check(!parsed.ok && !parsed.error.empty(), "invalid restore fixture should fail parsing");
        inspector.RestoreSettings(parameters);
        Check(inspector.Selection().data_path.empty() && inspector.Selection().label_path.empty(),
              "invalid saved settings must clear selection without falling back to /data");
        Check(inspector.Error() == parsed.error, "restoration must expose the settings parse error");
        inspector.Poll();
        check_idle(inspector);
    }

    inspector.SetSource("edited-source.h5");
    inspector.SetSelection({"/edited/features", "/edited/labels"});
    inspector.Reset();
    inspector.SetSource(canonical.at("file_path"));
    inspector.RestoreSettings(canonical);
    inspector.SetSource(canonical.at("file_path"));
    Check(inspector.Selection().data_path == settings.selection.data_path &&
              inspector.Selection().label_path == settings.selection.label_path &&
              inspector.Error().empty(),
          "reset must restore the original source before selection and retain it on later sync");
    check_idle(inspector);

    gui::Hdf5SourceInspector reopened;
    reopened.SetSource(canonical.at("file_path"));
    reopened.RestoreSettings(canonical);
    Check(reopened.Selection().data_path == inspector.Selection().data_path &&
              reopened.Selection().label_path == inspector.Selection().label_path,
          "recreated inspector should restore the same saved selection after source setup");
    check_idle(reopened);
}

#ifdef CYXWIZ_HAS_HDF5
struct Workspace {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyxwiz_hdf5_source_inspector_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    Workspace() {
        Check(std::filesystem::create_directory(path), "create unique HDF5 test workspace");
    }

    ~Workspace() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

template <typename T>
void WriteVector(HighFive::File& file, const std::string& path,
                 const std::vector<T>& values) {
    file.createDataSet<T>(path, HighFive::DataSpace::From(values)).write(values);
}

std::string CreateFixtureFile(const std::filesystem::path& path, int base) {
    const auto file_path = path.string();
    HighFive::File file(file_path, HighFive::File::Overwrite);
    file.createGroup("/group");
    const std::vector<std::vector<int32_t>> data{
        {base + 1, base + 2, base + 3},
        {base + 4, base + 5, base + 6},
        {base + 7, base + 8, base + 9},
        {base + 10, base + 11, base + 12},
    };
    file.createDataSet<int32_t>("/group/features", HighFive::DataSpace::From(data)).write(data);
    WriteVector<uint8_t>(file, "/group/labels", {2, 1, 0, 1});
    WriteVector<int32_t>(file, "/other", {base + 100, base + 101});
    for (int i = 0; i < 96; ++i) {
        WriteVector<int32_t>(file, "/bulk_" + std::to_string(i), {base + i});
    }
    return file_path;
}

const cyxwiz::Hdf5BrowseEntry* FindEntry(const cyxwiz::Hdf5BrowsePage& page,
                                         const std::string& path) {
    for (const auto& entry : page.entries) {
        if (entry.path == path) {
            return &entry;
        }
    }
    return nullptr;
}

void BumpModificationTime(const std::string& file_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path path(file_path);
    const auto before = fs::last_write_time(path, ec);
    Check(!ec, "fixture mtime should be readable before source-stamp invalidation");
    const auto target = before + std::chrono::seconds(10);
    fs::last_write_time(path, target, ec);
    Check(!ec, "fixture mtime should be writable for source-stamp invalidation");
    const auto after = fs::last_write_time(path, ec);
    Check(!ec && after != before, "fixture mtime should change deterministically");
}

void CheckPreviewPage(const cyxwiz::DataPreviewPage& page, int base) {
    Check(page.ok && page.status == cyxwiz::DataPreviewStatus::Ready,
          "preview page should be ready");
    Check(page.backend == "HDF5 source", "preview backend should identify HDF5 source");
    Check(page.total_rows == 4 && page.total_columns == 4,
          "preview should report full source shape plus label column");
    Check(page.rows_returned == 4 && page.rows.size() == 4,
          "preview should return all fixture rows");
    Check(page.schema.size() == 4, "preview schema should include data columns and labels");
    Check(page.schema[0].name == "col_0" && page.schema[0].type == "int32",
          "first data column schema should be preserved");
    Check(page.schema[2].name == "col_2" && page.schema[2].type == "int32",
          "last data column schema should be preserved");
    Check(page.schema[3].name == "label" && page.schema[3].type == "uint8",
          "label schema should append last with native type");
    Check(page.rows[0][0] == std::to_string(base + 1) &&
              page.rows[0][2] == std::to_string(base + 3) &&
              page.rows[0][3] == "2",
          "first preview row should contain exact values");
    Check(page.rows[3][0] == std::to_string(base + 10) &&
              page.rows[3][2] == std::to_string(base + 12) &&
              page.rows[3][3] == "1",
          "last preview row should contain exact values");
}

void TestBrowsePreviewAndInvalidation(const std::string& file_path) {
    gui::Hdf5SourceInspector inspector;
    std::string mutable_source = file_path;
    inspector.SetSource(mutable_source);
    mutable_source.assign("mutated-after-setsource.h5");

    cyxwiz::Hdf5TableSelection selection{"/group/features", "/group/labels"};
    inspector.SetSelection(selection);
    selection.data_path = "/other";
    selection.label_path.clear();
    Check(inspector.Selection().data_path == "/group/features" &&
              inspector.Selection().label_path == "/group/labels",
          "inspector should copy selection strings");

    inspector.Browse("/group");
    PollUntilIdle(inspector);
    Check(inspector.Error().empty(), "browse should not report an error");
    const auto& hierarchy = inspector.Hierarchy();
    Check(hierarchy.status == cyxwiz::Hdf5TableStatus::Ok &&
              hierarchy.group_path == "/group",
          "browse should return the selected group");
    const auto* features = FindEntry(hierarchy, "/group/features");
    const auto* labels = FindEntry(hierarchy, "/group/labels");
    Check(features && labels, "browse should list feature and label datasets");
    Check(features->dataset.shape == std::vector<uint64_t>({4, 3}) &&
              features->dataset.source_type == "int32",
          "browse should preserve feature shape and dtype");
    Check(labels->dataset.shape == std::vector<uint64_t>({4}) &&
              labels->dataset.source_type == "uint8",
          "browse should preserve label shape and dtype");

    inspector.Preview();
    PollUntilIdle(inspector);
    Check(inspector.Error().empty(), "preview should not report an error");
    CheckPreviewPage(inspector.Page(), 10);

    inspector.SetSelection({"/other", ""});
    Check(!inspector.Page().ok && inspector.Page().rows.empty(),
          "SetSelection should invalidate a stale preview page");
    Check(inspector.Error().empty(), "valid SetSelection should clear stale errors");
}

void TestSourceStampInvalidationAndRefresh(const std::string& file_path) {
    gui::Hdf5SourceInspector inspector;
    inspector.SetSource(file_path);
    inspector.SetSelection({"/group/features", "/group/labels"});
    inspector.Browse("/group");
    PollUntilIdle(inspector);
    Check(!inspector.Hierarchy().entries.empty(), "stamp test should start with hierarchy");
    inspector.Preview();
    PollUntilIdle(inspector);
    CheckPreviewPage(inspector.Page(), 10);

    BumpModificationTime(file_path);

    inspector.Preview();
    PollUntilIdle(inspector);
    Check(!inspector.Page().ok && inspector.Hierarchy().entries.empty(),
          "source-stamp mismatch should clear stale hierarchy and preview");
    Check(inspector.Error().find("changed") != std::string::npos,
          "source-stamp mismatch should explain that the source changed");

    inspector.Browse("/group", 0, true);
    PollUntilIdle(inspector);
    Check(inspector.Error().empty() && !inspector.Hierarchy().entries.empty(),
          "refresh browse should accept the changed source stamp");
    inspector.Preview();
    PollUntilIdle(inspector);
    CheckPreviewPage(inspector.Page(), 10);
}

void TestRestoreInvalidSettingsClearsPreviewAndCancels(const std::string& file_path) {
    gui::Hdf5SourceInspector inspector;
    inspector.SetSource(file_path);
    inspector.SetSelection({"/group/features", "/group/labels"});
    inspector.Preview();
    PollUntilIdle(inspector);
    CheckPreviewPage(inspector.Page(), 10);

    WorkerBlocker blocker("HDF5 restore queued cancellation blocker");
    inspector.Browse("/group");
    const auto task_id = FindInspectorTaskId();
    Check(task_id.has_value(), "restoration cancellation fixture should have a queued inspection");
    const auto task = cyxwiz::AsyncTaskManager::Instance().GetTask(*task_id);
    inspector.RestoreSettings({{"hdf5_dataset", ""}});
    Check(task && task->IsCancelRequested() && !inspector.Busy(),
          "invalid settings restoration must cancel the previous inspection");
    Check(inspector.Selection().data_path.empty() && inspector.Selection().label_path.empty() &&
              !inspector.Page().ok && inspector.Page().rows.empty() && !inspector.Error().empty(),
          "invalid restoration must clear a loaded preview and selection");
    const auto error = inspector.Error();
    blocker.Release();
    WaitForManagerIdle();
    inspector.Poll();
    Check(!inspector.Page().ok && inspector.Hierarchy().entries.empty() && inspector.Error() == error,
          "late cancelled inspection must not replace the saved-settings error or preview");
}

void TestSwitchSourceResetCancelAndLateCompletion(
    const std::string& first_path, const std::string& second_path) {
    gui::Hdf5SourceInspector inspector;
    inspector.SetSource(first_path);
    inspector.SetSelection({"/group/features", "/group/labels"});
    inspector.Preview();
    PollUntilIdle(inspector);
    CheckPreviewPage(inspector.Page(), 10);

    inspector.SetSource(second_path);
    Check(!inspector.Page().ok && inspector.Hierarchy().entries.empty() &&
              inspector.Error().empty(),
          "SetSource should reset stale page, hierarchy and errors");
    Check(inspector.Selection().data_path == "/data" &&
              inspector.Selection().label_path.empty(),
          "SetSource reset should restore default selection");
    inspector.SetSelection({"/group/features", "/group/labels"});
    inspector.Preview();
    PollUntilIdle(inspector);
    CheckPreviewPage(inspector.Page(), 100);

    {
        WorkerBlocker blocker("HDF5 source inspector queued cancel blocker");
        inspector.Browse("/group");
        Check(inspector.Busy(), "queued browse should leave inspector busy");
        inspector.Cancel();
        Check(!inspector.Busy() && inspector.Page().ok,
              "Cancel should drop the queued task without clearing the loaded page");
        blocker.Release();
        WaitForManagerIdle();
        inspector.Poll();
        CheckPreviewPage(inspector.Page(), 100);
    }

    {
        WorkerBlocker blocker("HDF5 source inspector queued source switch blocker");
        inspector.SetSource(first_path);
        inspector.SetSelection({"/group/features", "/group/labels"});
        inspector.Browse("/group");
        Check(inspector.Busy(), "queued first-source browse should be busy");
        inspector.SetSource(second_path);
        Check(!inspector.Busy() && !inspector.Page().ok &&
                  inspector.Hierarchy().entries.empty(),
              "SetSource should cancel and clear queued first-source state");
        blocker.Release();
        WaitForManagerIdle();
        inspector.Poll();
        Check(!inspector.Page().ok && inspector.Hierarchy().entries.empty() &&
                  inspector.Error().empty(),
              "queued first-source completion must not cross-deliver after source switch");
        inspector.SetSelection({"/group/features", "/group/labels"});
        inspector.Preview();
        PollUntilIdle(inspector);
        CheckPreviewPage(inspector.Page(), 100);
    }

    {
        WorkerBlocker blocker("HDF5 source inspector queued reset blocker");
        inspector.Browse("/group");
        Check(inspector.Busy(), "queued reset browse should be busy");
        inspector.Reset();
        Check(!inspector.Busy() && !inspector.Page().ok &&
                  inspector.Hierarchy().entries.empty(),
              "Reset should cancel and clear queued inspection state");
        blocker.Release();
        WaitForManagerIdle();
        inspector.Poll();
        Check(!inspector.Page().ok && inspector.Hierarchy().entries.empty() &&
                  inspector.Error().empty(),
              "queued completion must not repopulate after Reset");
        inspector.SetSource(second_path);
        inspector.SetSelection({"/group/features", "/group/labels"});
        inspector.Preview();
        PollUntilIdle(inspector);
        CheckPreviewPage(inspector.Page(), 100);
    }

    {
        WorkerBlocker blocker("HDF5 source inspector manager-cancel blocker");
        inspector.Browse("/group");
        const auto task_id = FindInspectorTaskId();
        Check(task_id.has_value(), "queued inspector task should be discoverable");
        Check(cyxwiz::AsyncTaskManager::Instance().Cancel(*task_id),
              "manager.Cancel should accept the queued inspector task id");
        blocker.Release();
        Check(WaitUntil([&] {
            inspector.Poll();
            return !inspector.Busy();
        }), "Poll should release a queued task cancelled by manager.Cancel");
        Check(!inspector.Busy(), "manager-cancelled queued task should not remain busy");
    }

    inspector.Browse("/");
    WaitUntil([&] {
        return !inspector.Busy() || AnyInspectorTaskRunning();
    }, std::chrono::milliseconds(500));
    inspector.Reset();
    Check(!inspector.Busy() && !inspector.Page().ok && inspector.Hierarchy().entries.empty(),
          "Reset should drop any running inspection state");
    WaitForManagerIdle();
    inspector.Poll();
    Check(!inspector.Page().ok && inspector.Hierarchy().entries.empty() &&
              inspector.Error().empty(),
          "late completion after Reset must not cross-deliver old results");

    {
        gui::Hdf5SourceInspector dying;
        dying.SetSource(first_path);
        dying.SetSelection({"/group/features", "/group/labels"});
        dying.Preview();
    }
    WaitForManagerIdle();
}

void TestRenderSmoke(const std::string& file_path) {
    ImGuiFixture imgui;
    gui::Hdf5SourceInspector inspector;
    inspector.SetSource(file_path);
    inspector.SetSelection({"/group/features", "/group/labels"});
    inspector.Browse("/group");
    PollUntilIdle(inspector);
    inspector.Preview();
    PollUntilIdle(inspector);
    Check(inspector.Page().ok, "render smoke should start from a loaded page");

    imgui.Frame(1120.0f, [] {
        ImGui::TextUnformatted("warmup");
    });
    for (float width : {1120.0f, 600.0f}) {
        imgui.Frame(width, [&] {
            inspector.RenderSettings();
        });
        imgui.Frame(width, [&] {
            inspector.RenderPreview();
        });
    }

    const std::string long_path = "/" + std::string(4000, 'a');
    inspector.SetSelection({long_path, long_path});
    for (float width : {1120.0f, 600.0f}) {
        imgui.Frame(width, [&] {
            inspector.RenderSettings();
        });
        imgui.Frame(width, [&] {
            inspector.RenderPreview();
        });
    }
}
#endif

} // namespace

int main() try {
    ManagerFixture manager;
    TestRestoreSettingsWithoutTasks();
#ifdef CYXWIZ_HAS_HDF5
    Workspace workspace;
    const auto first = CreateFixtureFile(workspace.path / "first.h5", 10);
    const auto second = CreateFixtureFile(workspace.path / "second.h5", 100);

    TestBrowsePreviewAndInvalidation(first);
    TestSourceStampInvalidationAndRefresh(first);
    TestRestoreInvalidSettingsClearsPreviewAndCancels(first);
    TestSwitchSourceResetCancelAndLateCompletion(first, second);
    TestRenderSmoke(first);
#else
    ImGuiFixture imgui;
    gui::Hdf5SourceInspector inspector;
    inspector.SetSource("disabled.h5");
    inspector.Browse("/");
    PollUntilIdle(inspector);
    Check(inspector.Error().find("HDF5 support is not compiled") != std::string::npos,
          "disabled HDF5 browse should report dependency absence");
    imgui.Frame(1120.0f, [&] {
        inspector.RenderSettings();
    });
    imgui.Frame(1120.0f, [&] {
        inspector.RenderPreview();
    });
#endif
    std::cout << "HDF5 source inspector: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
