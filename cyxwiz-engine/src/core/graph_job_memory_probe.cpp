#include "graph_job_memory_probe.h"

#include "training_trace_collector.h"

#include <arrayfire.h>

#include <atomic>
#include <filesystem>

namespace cyxwiz {
namespace {

std::uint64_t AllocatedDeviceBytes() {
    size_t alloc_bytes = 0, alloc_buffers = 0, lock_bytes = 0, lock_buffers = 0;
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    return alloc_bytes;
}

}  // namespace

GraphJobMemoryProbe ProbeGraphJobMemory(GraphTrainingJobRequest request) {
    GraphJobMemoryProbe probe;
    std::error_code ec;
    const auto scratch = std::filesystem::temp_directory_path(ec) / "cyxwiz-memory-probe";
    request.epochs_override = 1;
    request.checkpoint_dir_override = scratch.string();

    std::uint64_t before = 0;
    std::uint64_t after = 0;
    std::atomic<int> steps{0};
    GraphTrainingJobCallbacks callbacks;
    // The model and optimizer are built when training starts, so the
    // baseline is taken after a device garbage collection at on_start and
    // the peak after the first step (the memory manager keeps its buffers).
    callbacks.on_start = [&](int, int) {
        af::deviceGC();
        before = AllocatedDeviceBytes();
    };
    callbacks.on_batch = [&](int, int, int, float, float) {
        if (steps.fetch_add(1) == 0) {
            af::sync();
            after = AllocatedDeviceBytes();
        }
    };
    callbacks.should_cancel = [&] { return steps.load() >= 1; };

    const auto run = RunGraphTrainingJob(request, callbacks);
    std::filesystem::remove_all(scratch, ec);
    if (!run.ok && !run.cancelled) {
        probe.error = run.error;
        probe.failure = run.failure;
        return probe;
    }
    if (steps.load() == 0) {
        probe.error = "the job trained no step to measure";
        probe.failure = TrainingFailureKind::DataError;
        return probe;
    }
    probe.training_bytes = after > before ? after - before : 0;
    probe.backend = TrainingTraceCollector::Instance().Snapshot().effective_backend;
    probe.ok = true;
    return probe;
}

}  // namespace cyxwiz
