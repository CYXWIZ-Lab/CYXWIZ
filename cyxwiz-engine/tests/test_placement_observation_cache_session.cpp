// Computation-truth test for the placement observation cache session
// lifecycle (tofix67 slice 1): evidence recorded in one engine run must be
// visible to the next run's compile via the persisted cache, and a clean run
// must never truncate previously persisted evidence.

#include "core/placement_observation_cache_session.h"

#include <cyxwiz/backend_placement_observation.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::BackendPlacementObservation MakeObservation() {
    cyxwiz::BackendPlacementObservation observation;
    observation.op_type = "lstm_forward";
    observation.backend = "cuda";
    observation.device = "af_device=0;name=test-gpu;platform=CUDA";
    observation.dtype = "float32";
    observation.shape_signature = "batch=8;seq=16;input=32;hidden=64;dir=1";
    observation.reason_code =
        cyxwiz::BackendPlacementObservationReason::CudaJitParamOverflow;
    observation.source = cyxwiz::BackendPlacementObservationSource::Test;
    observation.detail = "session round-trip fixture";
    observation.timestamp = "2026-09-22T00:00:00Z";
    return observation;
}

bool StoreHasFixture() {
    const auto expected = MakeObservation();
    cyxwiz::BackendPlacementObservation found;
    return cyxwiz::TryGetBackendPlacementObservation(
               expected.op_type, expected.backend, expected.device,
               expected.dtype, expected.shape_signature, found) &&
           found.reason_code == expected.reason_code &&
           found.source == expected.source;
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      "cyxwiz-placement-cache-session-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    {
        cyxwiz::ScopedComputeRuntimeRootOverrideForTesting scoped_root(root);
        const auto cache_path = cyxwiz::GetPlacementObservationCachePath();

        // A missing cache file is a normal first run.
        cyxwiz::ClearBackendPlacementObservationCacheForTesting();
        cyxwiz::LoadPlacementObservationCacheAtStartup();
        Check(cyxwiz::SnapshotBackendPlacementObservations().empty(),
              "missing cache file must load nothing");

        // An empty store must not create (or later truncate) a cache file.
        cyxwiz::SavePlacementObservationCache();
        Check(!std::filesystem::exists(cache_path),
              "empty store must not write a cache file");

        // Run N: record runtime evidence and persist it.
        cyxwiz::RecordBackendPlacementObservation(MakeObservation());
        cyxwiz::SavePlacementObservationCache();
        Check(std::filesystem::exists(cache_path),
              "save must write the cache file");

        // Run N+1: fresh process state; startup load must restore evidence.
        cyxwiz::ClearBackendPlacementObservationCacheForTesting();
        Check(!StoreHasFixture(), "clear must empty the store");
        cyxwiz::LoadPlacementObservationCacheAtStartup();
        Check(StoreHasFixture(),
              "startup load must restore persisted evidence");

        // A later clean run (empty store) must not clobber the file.
        cyxwiz::ClearBackendPlacementObservationCacheForTesting();
        cyxwiz::SavePlacementObservationCache();
        cyxwiz::LoadPlacementObservationCacheAtStartup();
        Check(StoreHasFixture(),
              "empty-store save must preserve prior evidence");

        cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    }
    std::filesystem::remove_all(root, ec);
    std::cout << "Placement observation cache session checks passed\n";
    return 0;
}
