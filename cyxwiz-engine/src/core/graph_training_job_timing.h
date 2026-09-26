#pragma once

// Where a headless training job's time went (TOFIX118 P3 S4). Hosts add their
// own phases (queue, dataset transfer, result upload). Kept apart from
// graph_training_job.h so light headers can hold it.

namespace cyxwiz {

struct GraphTrainingJobTiming {
    double prepare_seconds = 0.0;  // load inputs, compile, build the batcher
    double train_seconds = 0.0;    // training and validation until the run ends
    long long batches_trained = 0;
    // batches x batch size (an epoch's last batch may be partial).
    long long samples_trained = 0;
    int tokens_per_sample = 0;     // sequence jobs: the window length; 0 otherwise
};

}  // namespace cyxwiz
