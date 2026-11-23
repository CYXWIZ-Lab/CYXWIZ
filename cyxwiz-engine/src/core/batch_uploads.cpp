#include "batch_uploads.h"

#include "training_trace_collector.h"

#include <cstdint>

namespace cyxwiz {
namespace {

// The names the host-sync groups use for the same types.
const char* UploadDtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
    }
    return "unknown";
}

size_t UploadDtypeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::Float64:
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
        case DataType::Float32:
        case DataType::Int32: return 4;
    }
    return 0;
}

}  // namespace

void RecordBatchUpload(const char* batcher, const char* part, const char* reason,
                       const std::vector<size_t>& shape, DataType dtype) {
    uint64_t elements = 1;
    for (size_t d : shape) elements *= d;
    TrainingTraceHostSyncGroup upload;
    upload.category = batcher;
    upload.operation = part;
    upload.reason = reason;
    upload.shape = shape;
    upload.dtype = UploadDtypeName(dtype);
    upload.layout = "row_major";
    upload.event_count = 1;
    upload.bytes = elements * UploadDtypeBytes(dtype);
    TrainingTraceCollector::Instance().RecordHostUpload(upload);
}

Tensor HostBatchTensor(const char* batcher, const char* part, const std::vector<size_t>& shape,
                       const void* data, DataType dtype) {
    RecordBatchUpload(batcher, part, BatchUploadReason::Batch, shape, dtype);
    return Tensor(shape, data, dtype);
}

}  // namespace cyxwiz
