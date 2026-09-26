#pragma once

// A training job's time split and the environment that ran it, as reported
// to the Engine and the central server (TOFIX118 P3 S4).

#include "common.pb.h"
#include "core/graph_training_job.h"
#include "core/machine_capability.h"

#include <algorithm>

namespace cyxwiz::servernode {

struct NodeJobPhases {
    double wall_seconds = 0.0;      // accepted -> result ready
    double transfer_seconds = 0.0;  // dataset files fetched (remote jobs)
    double save_seconds = 0.0;      // final weights saved
    GraphTrainingJobTiming run;     // the shared runner's prepare/train split
};

// Queue time is the part of the wall time no phase accounts for (waiting for
// the Engine's stream, a device, the worker thread).
inline protocol::JobTiming MakeJobTiming(const NodeJobPhases& phases) {
    protocol::JobTiming timing;
    const double accounted =
        phases.transfer_seconds + phases.run.prepare_seconds + phases.run.train_seconds + phases.save_seconds;
    const double wall = std::max(phases.wall_seconds, accounted);
    timing.set_queue_seconds(wall - accounted);
    timing.set_transfer_seconds(phases.transfer_seconds);
    timing.set_prepare_seconds(phases.run.prepare_seconds);
    timing.set_train_seconds(phases.run.train_seconds);
    timing.set_save_seconds(phases.save_seconds);
    timing.set_wall_seconds(wall);
    timing.set_goodput(wall > 0.0 ? phases.run.train_seconds / wall : 0.0);
    const long long tokens = phases.run.samples_trained * phases.run.tokens_per_sample;
    timing.set_samples_trained(phases.run.samples_trained);
    timing.set_tokens_trained(tokens);
    if (phases.run.train_seconds > 0.0) {
        timing.set_samples_per_second(static_cast<double>(phases.run.samples_trained) / phases.run.train_seconds);
        timing.set_tokens_per_second(static_cast<double>(tokens) / phases.run.train_seconds);
    }
    return timing;
}

inline void FillEnvironmentFingerprint(const MachineEnvironment& environment,
                                       protocol::EnvironmentFingerprint* out) {
    out->set_cyxwiz_build(environment.cyxwiz_build);
    out->set_os(environment.os);
    out->set_route_matrix_id(environment.route_matrix_id);
    out->set_compute_contract_id(environment.compute_contract_id);
    out->set_fingerprint(environment.fingerprint);
}

}  // namespace cyxwiz::servernode
