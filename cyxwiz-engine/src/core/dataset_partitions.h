#pragma once

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace cyxwiz {

enum class DatasetStorageKind {
    Unknown,
    InMemoryArrow,
    DiskBackedParquet,
    ImageCached,
    AudioCached,
    TextCached,
    AdapterDefined,
};

enum class DatasetModality {
    Unknown,
    Tabular,
    Image,
    Audio,
    Text,
    TimeSeries,
};

enum class PartitionOrigin {
    Unresolved,
    External,
    Derived,
};

enum class PartitionSplitMethod {
    None,
    Random,
    Stratified,
    TimeOrdered,
};

enum class PartitionCompatibility {
    Unknown,
    Compatible,
    Incompatible,
};

enum class PartitionLeakageStatus {
    NotChecked,
    Passed,
    Failed,
    Unavailable,
};

/**
 * A physical Dataset artifact selected by graph topology.
 *
 * The reference deliberately identifies registered data rather than a file
 * path. Source and schema fingerprints are populated as provenance becomes
 * available; an empty fingerprint means that identity has not been proven.
 */
struct DatasetSourceRef {
    std::string dataset_name;
    std::string label_column;
    int source_node_id = -1;
    bool externally_supplied = false;
    DatasetStorageKind storage_kind = DatasetStorageKind::Unknown;
    DatasetModality modality = DatasetModality::Unknown;
    std::string source_fingerprint;
    std::string schema_fingerprint;
    std::string feature_schema_fingerprint;
    int64_t row_count = -1;

    bool IsSupplied() const { return !dataset_name.empty(); }
};

struct SplitPolicy {
    float train_ratio = 0.8f;
    float dev_ratio = 0.1f;
    float test_ratio = 0.1f;
    int seed = 42;
    bool shuffle = true;
    bool stratified = false;
    PartitionSplitMethod method = PartitionSplitMethod::Random;
};

struct PartitionManifest {
    int version = 2;
    std::string training_source_fingerprint;
    std::string validation_source_fingerprint;
    std::string test_source_fingerprint;
    PartitionOrigin train_origin = PartitionOrigin::External;
    PartitionOrigin dev_origin = PartitionOrigin::Derived;
    PartitionOrigin test_origin = PartitionOrigin::Derived;
    std::string label_column;
    std::string feature_schema_fingerprint;
    PartitionSplitMethod split_method = PartitionSplitMethod::Random;
    float train_ratio = 0.8f;
    float dev_ratio = 0.1f;
    float test_ratio = 0.1f;
    int seed = 42;
    bool shuffle = true;
    bool stratified = false;
    PartitionCompatibility dev_compatibility = PartitionCompatibility::Unknown;
    PartitionCompatibility test_compatibility = PartitionCompatibility::Unknown;
    PartitionLeakageStatus dev_leakage = PartitionLeakageStatus::NotChecked;
    PartitionLeakageStatus test_leakage = PartitionLeakageStatus::NotChecked;
    std::string dev_status_reason;
    std::string test_status_reason;
    std::string dev_resolution_reason;
    std::string test_resolution_reason;
    int64_t train_rows = -1;
    int64_t dev_rows = -1;
    int64_t test_rows = -1;
    std::string manifest_fingerprint;
};

/**
 * Compiler-owned Train/Dev/Test resolution.
 *
 * Empty Dev/Test source references mean those partitions are derived from
 * Train according to policy. Supplied references are preserved in full.
 */
struct ResolvedDatasetPartitions {
    DatasetSourceRef train;
    DatasetSourceRef dev;
    DatasetSourceRef test;
    SplitPolicy policy;
    PartitionManifest manifest;
};

// Compatibility names for the existing tabular launcher while it is moved to
// consume ResolvedDatasetPartitions directly.
using ResolvedDatasetRole = DatasetSourceRef;
using ResolvedDatasetRoles = ResolvedDatasetPartitions;

inline const char* PartitionOriginName(PartitionOrigin origin) {
    switch (origin) {
        case PartitionOrigin::External: return "external";
        case PartitionOrigin::Derived: return "derived";
        default: return "unresolved";
    }
}

inline const char* PartitionSplitMethodName(PartitionSplitMethod method) {
    switch (method) {
        case PartitionSplitMethod::None: return "none";
        case PartitionSplitMethod::Random: return "random";
        case PartitionSplitMethod::Stratified: return "stratified";
        case PartitionSplitMethod::TimeOrdered: return "time_ordered";
        default: return "unknown";
    }
}

inline const char* PartitionCompatibilityName(
    PartitionCompatibility compatibility) {
    switch (compatibility) {
        case PartitionCompatibility::Compatible: return "compatible";
        case PartitionCompatibility::Incompatible: return "incompatible";
        default: return "unknown";
    }
}

inline const char* PartitionLeakageStatusName(
    PartitionLeakageStatus status) {
    switch (status) {
        case PartitionLeakageStatus::Passed: return "passed";
        case PartitionLeakageStatus::Failed: return "failed";
        case PartitionLeakageStatus::Unavailable: return "unavailable";
        default: return "not checked";
    }
}

inline std::string StablePartitionFingerprint(const std::string& text) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char ch : text) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

inline std::string BuildPartitionManifestFingerprint(
    const PartitionManifest& manifest) {
    std::ostringstream out;
    out << std::setprecision(9);
    out << "partition_manifest_schema=track70.v2\n";
    out << "train_source=" << manifest.training_source_fingerprint << '\n';
    out << "dev_source=" << manifest.validation_source_fingerprint << '\n';
    out << "test_source=" << manifest.test_source_fingerprint << '\n';
    out << "train_origin=" << PartitionOriginName(manifest.train_origin) << '\n';
    out << "dev_origin=" << PartitionOriginName(manifest.dev_origin) << '\n';
    out << "test_origin=" << PartitionOriginName(manifest.test_origin) << '\n';
    out << "label=" << manifest.label_column << '\n';
    out << "feature_schema=" << manifest.feature_schema_fingerprint << '\n';
    out << "split_method=" << PartitionSplitMethodName(manifest.split_method) << '\n';
    out << "train_ratio=" << manifest.train_ratio << '\n';
    out << "dev_ratio=" << manifest.dev_ratio << '\n';
    out << "test_ratio=" << manifest.test_ratio << '\n';
    out << "split_seed=" << manifest.seed << '\n';
    out << "shuffle=" << (manifest.shuffle ? "true" : "false") << '\n';
    out << "stratified=" << (manifest.stratified ? "true" : "false") << '\n';
    out << "train_rows=" << manifest.train_rows << '\n';
    out << "dev_rows=" << manifest.dev_rows << '\n';
    out << "test_rows=" << manifest.test_rows << '\n';
    return StablePartitionFingerprint(out.str());
}

inline void FinalizePartitionManifest(
    ResolvedDatasetPartitions& partitions,
    int64_t train_rows,
    int64_t dev_rows,
    int64_t test_rows) {
    auto& manifest = partitions.manifest;
    manifest.train_rows = train_rows;
    manifest.dev_rows = dev_rows;
    manifest.test_rows = test_rows;
    manifest.manifest_fingerprint =
        BuildPartitionManifestFingerprint(manifest);
}

} // namespace cyxwiz
