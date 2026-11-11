// One training run per GPU on this machine (TOFIX206 3.2 E): a run on a GPU
// route holds its device's lease; another run waits, is told why, and can be
// stopped while it waits; once the lease is free the next run takes it. A CPU
// route takes no lease. No GPU needed: the lease reads the run's context.

#include "../src/core/run_execution_device.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

cyxwiz::ExecutionDeviceContext Gpu(const std::string& name) {
    cyxwiz::ExecutionDeviceContext c;
    c.valid = true;
    c.effective_backend = "arrayfire_cuda";
    c.device_name = name;
    return c;
}

}  // namespace

int main() {
    using namespace std::chrono_literals;
    // The test's own device names: it never meets a real run's lease (and
    // leaves the same two lock files each time where leases are files)
    const std::string name = "Lease Test GPU";

    cyxwiz::ExecutionDeviceContext cpu;
    cpu.valid = true;
    cpu.effective_backend = "arrayfire_cpu";
    const auto none = cyxwiz::AcquireGpuRunLease(cpu, {}, {});
    Check(!none.lease && !none.cancelled && none.error.empty(), "a CPU route takes no lease");

    std::atomic<bool> held{false}, release{false};
    std::thread first([&]() {
        auto lease = cyxwiz::AcquireGpuRunLease(Gpu(name), {}, {});
        Check(lease.lease != nullptr && lease.error.empty(), "the first run takes the GPU's lease: " + lease.error);
        held = true;
        while (!release) std::this_thread::sleep_for(10ms);
    });  // the lease ends with the thread's scope
    while (!held) std::this_thread::sleep_for(10ms);

    // A second run waits, says why, and Stop ends its wait
    std::string told;
    std::atomic<bool> stop{false};
    std::thread second([&]() {
        auto lease = cyxwiz::AcquireGpuRunLease(
            Gpu(name), [&]() { return stop.load(); }, [&](const std::string& message) { told = message; });
        Check(!lease.lease && lease.cancelled, "the second run waits and is stopped while waiting");
    });
    std::this_thread::sleep_for(600ms);
    stop = true;
    second.join();
    Check(told == "Waiting for the GPU: another training run is using " + name, "it says why: " + told);

    // Another GPU is not this one's lease
    std::thread other([&]() {
        auto lease = cyxwiz::AcquireGpuRunLease(Gpu(name + " other"), {}, {});
        Check(lease.lease != nullptr, "a run on another GPU does not wait");
    });
    other.join();

    // Free: the next run takes it
    release = true;
    first.join();
    std::thread third([&]() {
        bool waited = false;
        auto lease = cyxwiz::AcquireGpuRunLease(Gpu(name), {}, [&](const std::string&) { waited = true; });
        Check(lease.lease != nullptr && !waited, "once free, the next run takes it at once");
    });
    third.join();

    if (failures == 0) std::printf("test_gpu_run_lease: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
