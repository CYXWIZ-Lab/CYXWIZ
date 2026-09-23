#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/backend_placement_observation.h>
#include <cyxwiz/gpu_execution_contracts.h>

#include <string>

namespace {

cyxwiz::BackendPlacementObservation MakeProbeObservation() {
    cyxwiz::BackendPlacementObservation observation;
    observation.op_type = "LSTM";
    observation.backend = "cuda";
    observation.device = "af_device=0;name=typed-contract-test;platform=CUDA";
    observation.dtype = "float32";
    observation.shape_signature =
        "kind=LSTM;batch=4;seq=8;input=16;hidden=32;dir=1";
    observation.reason_code =
        cyxwiz::BackendPlacementObservationReason::CudaJitParamOverflow;
    observation.source = cyxwiz::BackendPlacementObservationSource::Test;
    observation.detail = "typed contract fixture";
    observation.timestamp = "2026-09-22T00:00:00Z";
    observation.probe_outcome = "unsafe";
    observation.probe_scope =
        cyxwiz::BackendPlacementProbeScope::NormalCompile;
    return observation;
}

} // namespace

TEST_CASE("GpuExecutionKey matches the observation store's key formation",
          "[gpu_execution][taxonomy]") {
    const auto observation = MakeProbeObservation();
    const auto key = cyxwiz::GpuExecutionKey::FromObservation(observation);

    CHECK(key.ToStoreKey() ==
          cyxwiz::BuildBackendPlacementObservationKey(
              observation.op_type, observation.backend, observation.device,
              observation.dtype, observation.shape_signature));

    // The typed key must address the real store: record via the existing
    // API, then look up using the typed key's fields.
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecordBackendPlacementObservation(observation);
    cyxwiz::BackendPlacementObservation found;
    REQUIRE(cyxwiz::TryGetBackendPlacementObservation(
        key.op_type, key.backend, key.device_signature, key.dtype,
        key.shape_signature, found));
    CHECK(found.reason_code == observation.reason_code);
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
}

TEST_CASE("GpuExecutionEvidence round-trips a placement observation",
          "[gpu_execution][taxonomy]") {
    const auto observation = MakeProbeObservation();
    const auto key = cyxwiz::GpuExecutionKey::FromObservation(observation);
    const auto evidence =
        cyxwiz::GpuExecutionEvidence::FromObservation(observation);

    CHECK(evidence.source == cyxwiz::GpuExecutionEvidenceSource::Test);
    REQUIRE(evidence.outcome.has_value());
    CHECK(*evidence.outcome == cyxwiz::GpuExecutionOutcome::Unsafe);
    CHECK(evidence.reason_code == observation.reason_code);

    const auto round_tripped = evidence.ToObservation(key);
    CHECK(round_tripped.op_type == observation.op_type);
    CHECK(round_tripped.backend == observation.backend);
    CHECK(round_tripped.device == observation.device);
    CHECK(round_tripped.dtype == observation.dtype);
    CHECK(round_tripped.shape_signature == observation.shape_signature);
    CHECK(round_tripped.source == observation.source);
    CHECK(round_tripped.reason_code == observation.reason_code);
    CHECK(round_tripped.detail == observation.detail);
    CHECK(round_tripped.timestamp == observation.timestamp);
    CHECK(round_tripped.probe_outcome == observation.probe_outcome);
    CHECK(round_tripped.probe_scope == observation.probe_scope);

    // A runtime fallback carries no probe verdict; the optional stays empty
    // both ways.
    auto runtime_observation = observation;
    runtime_observation.source =
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback;
    runtime_observation.probe_outcome = "";
    runtime_observation.probe_scope = "";
    const auto runtime_evidence =
        cyxwiz::GpuExecutionEvidence::FromObservation(runtime_observation);
    CHECK(runtime_evidence.source ==
          cyxwiz::GpuExecutionEvidenceSource::RuntimeFallback);
    CHECK_FALSE(runtime_evidence.outcome.has_value());
    CHECK(runtime_evidence.ToObservation(key).probe_outcome.empty());
}

TEST_CASE("GpuExecution contract names pin persisted and UI spellings",
          "[gpu_execution][taxonomy]") {
    using namespace cyxwiz;

    // Evidence sources: the last three spellings already exist in persisted
    // caches (BackendPlacementObservationSource) and must not drift.
    CHECK(std::string(GpuExecutionEvidenceSourceName(
              GpuExecutionEvidenceSource::StaticPolicy)) == "static_policy");
    CHECK(std::string(GpuExecutionEvidenceSourceName(
              GpuExecutionEvidenceSource::RuntimeSuccess)) ==
          "runtime_success");
    CHECK(std::string(GpuExecutionEvidenceSourceName(
              GpuExecutionEvidenceSource::PreflightProbe)) ==
          BackendPlacementObservationSource::PreflightProbe);
    CHECK(std::string(GpuExecutionEvidenceSourceName(
              GpuExecutionEvidenceSource::RuntimeFallback)) ==
          BackendPlacementObservationSource::RuntimeFallback);
    CHECK(std::string(GpuExecutionEvidenceSourceName(
              GpuExecutionEvidenceSource::Test)) ==
          BackendPlacementObservationSource::Test);

    // Outcomes align with the persisted BackendPlacementProbeOutcome
    // spellings ("Supported" is stored as "safe").
    CHECK(std::string(GpuExecutionOutcomeName(
              GpuExecutionOutcome::Supported)) ==
          BackendPlacementProbeOutcomeName(BackendPlacementProbeOutcome::Safe));
    CHECK(std::string(GpuExecutionOutcomeName(GpuExecutionOutcome::Unsafe)) ==
          BackendPlacementProbeOutcomeName(
              BackendPlacementProbeOutcome::Unsafe));
    CHECK(std::string(GpuExecutionOutcomeName(GpuExecutionOutcome::Timeout)) ==
          BackendPlacementProbeOutcomeName(
              BackendPlacementProbeOutcome::Timeout));
    CHECK(std::string(GpuExecutionOutcomeName(
              GpuExecutionOutcome::Unsupported)) ==
          BackendPlacementProbeOutcomeName(
              BackendPlacementProbeOutcome::Unsupported));
    CHECK(std::string(GpuExecutionOutcomeName(
              GpuExecutionOutcome::Inconclusive)) ==
          BackendPlacementProbeOutcomeName(
              BackendPlacementProbeOutcome::Inconclusive));

    // Parsers accept exactly the emitted names.
    CHECK(TryParseGpuExecutionEvidenceSource("preflight_probe") ==
          GpuExecutionEvidenceSource::PreflightProbe);
    CHECK_FALSE(TryParseGpuExecutionEvidenceSource("not_a_source")
                    .has_value());
    CHECK(TryParseGpuExecutionOutcome("safe") ==
          GpuExecutionOutcome::Supported);
    CHECK_FALSE(TryParseGpuExecutionOutcome("").has_value());

    // Decision actions (ticket tofix67 architecture decision 1).
    CHECK(std::string(GpuExecutionDecisionActionName(
              GpuExecutionDecisionAction::AttemptGpu)) == "attempt_gpu");
    CHECK(std::string(GpuExecutionDecisionActionName(
              GpuExecutionDecisionAction::StagedArrayFire)) ==
          "staged_arrayfire");
    CHECK(std::string(GpuExecutionDecisionActionName(
              GpuExecutionDecisionAction::NativeProvider)) ==
          "native_provider");
    CHECK(std::string(GpuExecutionDecisionActionName(
              GpuExecutionDecisionAction::Cpu)) == "cpu");
    CHECK(std::string(GpuExecutionDecisionActionName(
              GpuExecutionDecisionAction::FailClosed)) == "fail_closed");
}
