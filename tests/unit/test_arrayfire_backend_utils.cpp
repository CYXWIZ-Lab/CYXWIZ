#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";

void SetEnvVar(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void ClearEnvVar(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value == nullptr) {
            ClearEnvVar(name_);
        } else {
            SetEnvVar(name_, value);
        }
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            SetEnvVar(name_, previous_.c_str());
        } else {
            ClearEnvVar(name_);
        }
    }

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};

} // namespace

TEST_CASE("ArrayFire fallback reasons classify backend failures", "[arrayfire][fallback]") {
    const char* overflow =
        "NVRTC_ERROR_COMPILATION: Formal parameter space overflowed "
        "(4097 bytes required, max 4096 bytes allowed)";
    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(overflow) ==
            cyxwiz::BackendFallbackReason::CudaJitParamOverflow);
    REQUIRE(std::string(cyxwiz::BackendFallbackReasonName(
                cyxwiz::BackendFallbackReason::CudaJitParamOverflow)) ==
            "cuda_jit_param_overflow");

    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(
                "ArrayFire JIT compile failed") ==
            cyxwiz::BackendFallbackReason::ArrayFireJitCompileFailure);
    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(
                "device lost") ==
            cyxwiz::BackendFallbackReason::GpuBackendException);
}

TEST_CASE("ArrayFire fallback log gate is once per operation reason and context",
          "[arrayfire][fallback]") {
    const std::string context_a =
        cyxwiz::BuildArrayFireBackendFallbackContext(
            cyxwiz::BuildTensorShapeContext(
                "input", std::vector<size_t>{2, 3}),
            "test-backend");
    const std::string context_b =
        cyxwiz::BuildArrayFireBackendFallbackContext(
            cyxwiz::BuildTensorShapeContext(
                "input", std::vector<size_t>{4, 3}),
            "test-backend");

    REQUIRE(cyxwiz::ShouldLogArrayFireBackendFallbackOnce(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::GpuBackendException,
        context_a));
    REQUIRE_FALSE(cyxwiz::ShouldLogArrayFireBackendFallbackOnce(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::GpuBackendException,
        context_a));
    REQUIRE(cyxwiz::ShouldLogArrayFireBackendFallbackOnce(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::GpuBackendException,
        context_b));
}

TEST_CASE("ArrayFire fallback messages can suppress compiler dumps",
          "[arrayfire][fallback]") {
    const std::string context =
        cyxwiz::BuildArrayFireBackendFallbackContext("batch=8", "cuda");
    const std::string message = cyxwiz::BuildArrayFireBackendFallbackMessage(
        "UnitTestArrayFireFallback::Backward",
        cyxwiz::BackendFallbackReason::CudaJitParamOverflow,
        false,
        "very long NVRTC compiler output",
        context);

    REQUIRE(message.find("reason=cuda_jit_param_overflow") != std::string::npos);
    REQUIRE(message.find("Training continues") != std::string::npos);
    REQUIRE(message.find("batch=8") != std::string::npos);
    REQUIRE(message.find("very long NVRTC compiler output") == std::string::npos);
}

TEST_CASE("ArrayFire forced fallback test hook parses requested operations",
          "[arrayfire][fallback]") {
    {
        ScopedEnvVar env(kForceFallbackEnv, nullptr);
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }

#ifdef NDEBUG
    {
        ScopedEnvVar env(kForceFallbackEnv, "*");
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }
#else
    {
        ScopedEnvVar env(kForceFallbackEnv, "*");
        REQUIRE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }
    {
        ScopedEnvVar env(
            kForceFallbackEnv,
            " DenseLayer::Backward ; AttentionLayer::Forward ");
        REQUIRE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Backward"));
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }
    {
        ScopedEnvVar env(kForceFallbackEnv, ", ; DenseLayer::Forward");
        REQUIRE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Backward"));
    }
#endif
}
