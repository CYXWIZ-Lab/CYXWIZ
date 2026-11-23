#pragma once

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <vector>

namespace cyxwiz {

// Host-to-device uploads a batcher causes, counted in the active run's
// training trace (TrainingTraceSummary::host_upload_groups) beside the
// device-to-host reads the backend's host-sync observer counts. Together they
// are the run's copy budget (TOFIX228 P6: counted in the batchers, no backend
// hook). A Tensor built from host memory is uploaded when the model first
// reads it, so a batcher records it where it builds it; data that already
// lives on the device costs nothing per batch.
namespace BatchUploadReason {
// A batch assembled on the host: one upload per batch.
inline constexpr const char* Batch = "batch";
// The whole set copied to the device once for the run.
inline constexpr const char* Resident = "resident";
// A batch's row positions, uploaded for a gather from a resident set.
inline constexpr const char* Index = "index";
}  // namespace BatchUploadReason

// Records one upload of `shape` x `dtype` by `batcher` (its part: "features",
// "labels", "csr", ...). Nothing when no run is recording.
void RecordBatchUpload(const char* batcher, const char* part, const char* reason,
                       const std::vector<size_t>& shape, DataType dtype);

// A batch tensor copied from host memory, recorded as one upload.
Tensor HostBatchTensor(const char* batcher, const char* part, const std::vector<size_t>& shape,
                       const void* data, DataType dtype = DataType::Float32);

}  // namespace cyxwiz
