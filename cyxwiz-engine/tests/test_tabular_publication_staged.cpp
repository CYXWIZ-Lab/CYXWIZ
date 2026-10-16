#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"

#include <arrow/api.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {
using Registry = cyxwiz::DataRegistry;
int checks = 0;

void Check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

struct Fixture {
    Registry& registry = Registry::Instance();
    Fixture() {
        registry.SetOnDatasetLoaded({});
        registry.ClearAllTabularDatasets();
    }
    ~Fixture() {
        registry.SetOnDatasetLoaded({});
        registry.ClearAllTabularDatasets();
    }
};

std::shared_ptr<cyxwiz::ArrowDataset> Candidate(const std::string& name) {
    arrow::Int64Builder builder;
    if (!builder.AppendValues({1, 2, 3}).ok()) throw std::runtime_error("append failed");
    auto array = builder.Finish().ValueOrDie();
    auto table = arrow::Table::Make(arrow::schema({arrow::field("value", arrow::int64())}), {array});
    return std::make_shared<cyxwiz::ArrowDataset>(table, name);
}

void TestDiscardAndMove() {
    Fixture fixture;
    auto& registry = fixture.registry;
    std::string error;
    auto token = registry.CaptureTabularPublication("discard");
    auto candidate = Candidate("discard");
    std::weak_ptr<cyxwiz::ArrowDataset> observer = candidate;
    auto staged = registry.PrepareArrowTablePublication(token, std::move(candidate), "source.h5", error);
    Check(staged != nullptr, "prepare private candidate");
    Check(!registry.IsArrowDataset("discard"), "preparation must not register");
    staged->Notify();
    staged.reset();
    Check(observer.expired(), "discard releases unpublished candidate");
    Check(!registry.IsArrowDataset("discard"), "discard must not register");
    staged = registry.PrepareArrowTablePublication(token, Candidate("discard"), "", error);
    Check(staged != nullptr, "discard does not consume token");
    Registry::PreparedArrowPublication moved(std::move(*staged));
    Check(!registry.TryPublishPreparedArrowTable(*staged, error), "moved-from publication rejects");
    staged->Notify();
    Check(registry.TryPublishPreparedArrowTable(moved, error), "moved publication commits");
    Check(!registry.TryPublishPreparedArrowTable(moved, error), "commit is single use");
    moved.Notify();
    Registry::PreparedArrowPublication empty;
    Check(!registry.TryPublishPreparedArrowTable(empty, error), "default publication rejects");
    empty.Notify();
}

void TestWorkerHandoffAndNotification() {
    Fixture fixture;
    auto& registry = fixture.registry;
    std::string error;
    const auto token = registry.CaptureTabularPublication("handoff");
    std::unique_ptr<Registry::PreparedArrowPublication> staged;
    // join is the publication barrier for this test; production uses task done.
    std::thread worker([&] {
        staged = registry.PrepareArrowTablePublication(token, Candidate("handoff"), "relative.h5", error);
    });
    worker.join();
    Check(staged != nullptr, "worker prepares handoff");
    int old_notifications = 0;
    int notifications = 0;
    bool node_committed = false;
    bool observed_state = false;
    bool observed_metadata = false;
    bool queried_registry = false;
    registry.SetOnDatasetLoaded([&](const std::string&, const cyxwiz::DatasetInfo&) {
        ++old_notifications;
    });
    staged->Notify();
    Check(old_notifications == 0, "notification before commit is a no-op");
    registry.SetOnDatasetLoaded([&](const std::string& name, const cyxwiz::DatasetInfo& info) {
        ++notifications;
        observed_state = node_committed;
        observed_metadata = name == "handoff" && info.name == name && info.num_samples == 3;
        queried_registry = registry.IsArrowDataset(name);
        staged->Notify();
    });
    Check(registry.TryPublishPreparedArrowTable(*staged, error), "commit worker handoff");
    Check(notifications == 0 && old_notifications == 0, "commit never notifies synchronously");
    registry.SetOnDatasetLoaded({});
    node_committed = true;
    staged->Notify();
    staged->Notify();
    Check(notifications == 1 && old_notifications == 0, "uses callback at commit exactly once");
    Check(observed_state, "callback observes committed dependent state");
    Check(observed_metadata, "notification preserves prepared metadata");
    Check(queried_registry, "callback can query registry");
    const auto expected = std::filesystem::weakly_canonical(
        std::filesystem::absolute("relative.h5")).lexically_normal().string();
    Check(registry.GetTabularSourcePath("handoff") == expected, "prepared source association is retained");
}

void TestStaleAndRetiredBacking() {
    Fixture fixture;
    auto& registry = fixture.registry;
    std::string error;
    auto original = Candidate("retained");
    std::weak_ptr<cyxwiz::ArrowDataset> retired = original;
    Check(registry.TryPublishArrowTable(registry.CaptureTabularPublication("retained"),
                                       original, "old.h5", error), "publish original backing");
    auto staged = registry.PrepareArrowTablePublication(
        registry.CaptureTabularPublication("retained"), Candidate("retained"), "new.h5", error);
    original.reset();
    Check(staged != nullptr && !retired.expired(), "preparation retains original registration");
    Check(registry.TryPublishPreparedArrowTable(*staged, error), "replace original backing");
    Check(!retired.expired(), "commit does not destroy retired backing");
    staged->Notify();
    Check(!retired.expired(), "notification does not destroy retired backing");
    staged.reset();
    Check(retired.expired(), "handoff owner chooses retired backing release time");

    staged = registry.PrepareArrowTablePublication(
        registry.CaptureTabularPublication("retained"), Candidate("retained"), "stale.h5", error);
    const auto current = registry.GetArrowDataset("retained");
    registry.RestoreTabularDataset("retained", current, nullptr, "restored.h5");
    const auto source = registry.GetTabularSourcePath("retained");
    Check(staged && !registry.TryPublishPreparedArrowTable(*staged, error), "mutation after preparation rejects");
    Check(registry.GetArrowDataset("retained") == current && registry.GetTabularSourcePath("retained") == source,
          "stale staged commit preserves backing and source");
    staged->Notify();
}

struct ThrowingCopy {
    std::shared_ptr<bool> fail;
    int* notifications;
    ThrowingCopy(std::shared_ptr<bool> flag, int& count) : fail(std::move(flag)), notifications(&count) {}
    ThrowingCopy(const ThrowingCopy& other) : fail(other.fail), notifications(other.notifications) {
        if (*fail) throw std::runtime_error("injected callback-copy failure");
    }
    void operator()(const std::string&, const cyxwiz::DatasetInfo&) const { ++*notifications; }
};

void TestCallbackCopyFailure() {
    Fixture fixture;
    auto& registry = fixture.registry;
    std::string error;
    auto original = Candidate("copy_failure");
    Check(registry.TryPublishArrowTable(registry.CaptureTabularPublication("copy_failure"),
                                       original, "original.h5", error), "publish copy-failure baseline");
    const auto source = registry.GetTabularSourcePath("copy_failure");
    auto staged = registry.PrepareArrowTablePublication(
        registry.CaptureTabularPublication("copy_failure"), Candidate("copy_failure"), "new.h5", error);
    Check(staged != nullptr, "prepare callback-copy fixture");
    int notifications = 0;
    auto fail = std::make_shared<bool>(false);
    registry.SetOnDatasetLoaded(ThrowingCopy(fail, notifications));
    *fail = true;
    Check(!registry.TryPublishPreparedArrowTable(*staged, error), "callback-copy failure rejects commit");
    Check(error.find("callback-copy") != std::string::npos, "copy failure has diagnostic");
    Check(registry.GetArrowDataset("copy_failure") == original &&
              registry.GetTabularSourcePath("copy_failure") == source,
          "copy failure leaves prior backing and source untouched");
    staged->Notify();
    Check(notifications == 0, "failed commit cannot notify");
    *fail = false;
    Check(registry.TryPublishPreparedArrowTable(*staged, error), "copy failure leaves handoff/token retryable");
    staged->Notify();
    Check(notifications == 1, "successful retry notifies once");
}

void TestCallbackDestroysHandoff() {
    Fixture fixture;
    auto& registry = fixture.registry;
    std::string error;
    auto staged = registry.PrepareArrowTablePublication(
        registry.CaptureTabularPublication("destroy"), Candidate("destroy"), "", error);
    Check(staged != nullptr, "prepare self-destroying callback fixture");
    int notifications = 0;
    registry.SetOnDatasetLoaded([&](const std::string&, const cyxwiz::DatasetInfo&) {
        ++notifications;
        staged.reset();
        throw std::runtime_error("injected error after owner destruction");
    });
    Check(registry.TryPublishPreparedArrowTable(*staged, error), "commit self-destroying callback fixture");
    staged->Notify();
    Check(!staged && notifications == 1 && registry.IsArrowDataset("destroy"),
          "callback may destroy handoff and throw without undoing publication");
}
} // namespace

int RunStagedPublicationTests() {
    TestDiscardAndMove();
    TestWorkerHandoffAndNotification();
    TestStaleAndRetiredBacking();
    TestCallbackCopyFailure();
    TestCallbackDestroysHandoff();
    return checks;
}
