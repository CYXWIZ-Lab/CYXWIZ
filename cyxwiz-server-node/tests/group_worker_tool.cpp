// A group worker on its own, without the rest of the daemon: for running a
// whole group on one machine (several of these against one coordinator).
//
//   group_worker_tool --coordinator=127.0.0.1:50061 --ticket-file=PATH [--cpu] [--slow-ms=N]
//
// --cpu trains on the ArrayFire CPU route with test route evidence, so the
// tool needs no verified device. Without it the machine's saved compute
// preference is used, as the daemon does.
// Exit code: 0 the job ended, 2 this node was removed, 1 anything else.

#include <grpcpp/grpcpp.h>

#include "group_worker.h"

#include "core/execution_device_preferences.h"
#include "core/machine_compute_preference.h"
#include "core/route_qualification_snapshot.h"
#include "core/compute_runtime_paths.h"
#include "core/secret_file.h"
#include "../../cyxwiz-engine/tests/route_qualification_test_fixture.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/device.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string coordinator;
    std::string ticket_file;
    bool cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--coordinator=", 14) == 0) coordinator = argv[i] + 14;
        else if (std::strncmp(argv[i], "--ticket-file=", 14) == 0) ticket_file = argv[i] + 14;
        else if (std::strcmp(argv[i], "--cpu") == 0) cpu = true;
    }
    if (coordinator.empty() || ticket_file.empty()) {
        std::cerr << "usage: group_worker_tool --coordinator=HOST:PORT --ticket-file=PATH [--cpu]\n";
        return 1;
    }
    const auto ticket = cyxwiz::servernode::core::ReadSecretFile(ticket_file);
    if (!ticket.error.empty()) {
        std::cerr << "ticket: " << ticket.error << "\n";
        return 1;
    }

    cyxwiz::Initialize();
    cyxwiz::servernode::GroupWorkerOptions options;
    options.coordinator_address = coordinator;
    options.ticket = ticket.secret;
    options.node_version = cyxwiz::GetVersionString();
    if (cpu) {
        options.select_device = [] {
            const auto devices = cyxwiz::Device::GetAvailableDevices();
            const auto found = std::find_if(devices.begin(), devices.end(), [](const cyxwiz::DeviceInfo& device) {
                return device.type == cyxwiz::DeviceType::CPU;
            });
            if (found == devices.end()) throw std::runtime_error("no ArrayFire CPU route on this machine");
            cyxwiz::test::InstallQualifiedRouteSnapshot(devices);
            cyxwiz::ClearPendingExecutionDeviceSelection();
            cyxwiz::SetPendingExecutionDeviceSelection(cyxwiz::DeviceType::CPU, found->device_id);
            cyxwiz::ClearNextRunExecutionPolicy();
            cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
        };
    } else {
        // As the daemon does at start-up: this machine's route evidence and preference.
        cyxwiz::LoadAndInstallRouteQualificationSnapshot(cyxwiz::GetRouteQualificationCachePath());
        cyxwiz::ApplyMachineComputePreference();
    }

    cyxwiz::servernode::GroupWorker worker(std::move(options));
    worker.Run();
    const auto status = worker.Status();
    std::cout << cyxwiz::servernode::GroupWorkerStateName(status.state) << ": " << status.text << " (rounds joined "
              << status.rounds_joined << ", updates refused " << status.updates_refused << ")" << std::endl;
    cyxwiz::Shutdown();
    if (status.state == cyxwiz::servernode::GroupWorkerState::Finished) return 0;
    return status.state == cyxwiz::servernode::GroupWorkerState::Removed ? 2 : 1;
}
