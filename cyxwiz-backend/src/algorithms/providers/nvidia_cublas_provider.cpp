// tofix68 phase 3: the "cyxwiz.nvidia-cublas-cell" provider executes
// vanilla RNN forward (pilot) and LSTM forward (P1, inference) as cuBLAS
// GEMMs plus small fused cell kernels. Kernels are NVRTC-compiled at
// runtime with fixed, compact argument lists (the compact-launch principle
// that eliminates the 4096-byte JIT parameter-overflow class by
// construction) and cached per device; NVRTC also sidesteps
// host-compiler/NVCC version coupling. Everything outside the declared
// contract stays a truthful Unsupported.
//
// Kernel provenance: the cell structure (one GEMM pass for all gate
// projections + a fused pointwise cell update, gate order i,f,g,o)
// follows PyTorch ATen's fused RNN/LSTM cells (BSD-3) and lmnt/haste
// (Apache-2.0), matching CyxWiz's native CPU reference in
// lstm_direction_helpers.cpp.

#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER

#include "cyxwiz/neural_provider.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <spdlog/spdlog.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

constexpr const char* kProviderId = "cyxwiz.nvidia-cublas-cell";
constexpr const char* kProviderSemver = "0.6.0-stacked";

// Fused cell kernels. Layout conventions shared with the CyxWiz CPU
// references: x is [batch, seq, features] row-major, so the row index of
// the pre-computed input projections for (b, t) is b*seq + t; recurrent
// projections and states are [batch, hidden(*gates)] contiguous. LSTM
// gate order is i, f, g, o.
constexpr const char* kCellKernelSource = R"(
__device__ __forceinline__ float cyxwiz_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void cyxwiz_fill(float* __restrict__ p, float v,
                                       int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) p[idx] = v;
}

extern "C" __global__ void cyxwiz_rnn_cell(
    const float* __restrict__ gates_ih,
    const float* __restrict__ gates_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ hidden_out,
    float* __restrict__ sequence_out,
    int t,
    int seq,
    int hidden,
    int total,          // batch * hidden
    int use_tanh) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int b = idx / hidden;
  int j = idx - b * hidden;
  float a = gates_ih[(b * seq + t) * hidden + j] + gates_hh[idx] +
            bias_ih[j] + bias_hh[j];
  float v = use_tanh ? tanhf(a) : (a > 0.0f ? a : 0.0f);
  hidden_out[idx] = v;
  sequence_out[(b * seq + t) * hidden + j] = v;
}

extern "C" __global__ void cyxwiz_lstm_cell(
    const float* __restrict__ gates_ih,   // [(b*seq+t), 4*hidden]
    const float* __restrict__ gates_hh,   // [batch, 4*hidden]
    const float* __restrict__ bias_ih,    // [4*hidden]
    const float* __restrict__ bias_hh,    // [4*hidden]
    float* __restrict__ cell_state,       // [batch, hidden], updated in place
    float* __restrict__ hidden_out,       // [batch, hidden]
    float* __restrict__ sequence_out,     // [batch, seq, hidden]
    float* __restrict__ act_gate_cache,   // [batch, seq, 4*hidden] or 0
    float* __restrict__ cell_cache,       // [batch, seq, hidden] or 0
    int t,
    int seq,
    int hidden,
    int total) {        // batch * hidden
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int b = idx / hidden;
  int j = idx - b * hidden;
  int row_ih = (b * seq + t) * 4 * hidden;
  int row_hh = b * 4 * hidden;
  float a_i = gates_ih[row_ih + j] + gates_hh[row_hh + j] +
              bias_ih[j] + bias_hh[j];
  float a_f = gates_ih[row_ih + hidden + j] + gates_hh[row_hh + hidden + j] +
              bias_ih[hidden + j] + bias_hh[hidden + j];
  float a_g = gates_ih[row_ih + 2 * hidden + j] +
              gates_hh[row_hh + 2 * hidden + j] +
              bias_ih[2 * hidden + j] + bias_hh[2 * hidden + j];
  float a_o = gates_ih[row_ih + 3 * hidden + j] +
              gates_hh[row_hh + 3 * hidden + j] +
              bias_ih[3 * hidden + j] + bias_hh[3 * hidden + j];
  float i_gate = cyxwiz_sigmoid(a_i);
  float f_gate = cyxwiz_sigmoid(a_f);
  float g_gate = tanhf(a_g);
  float o_gate = cyxwiz_sigmoid(a_o);
  float c = f_gate * cell_state[idx] + i_gate * g_gate;
  cell_state[idx] = c;
  float h = o_gate * tanhf(c);
  hidden_out[idx] = h;
  sequence_out[(b * seq + t) * hidden + j] = h;
  if (act_gate_cache) {
    act_gate_cache[row_ih + j] = i_gate;
    act_gate_cache[row_ih + hidden + j] = f_gate;
    act_gate_cache[row_ih + 2 * hidden + j] = g_gate;
    act_gate_cache[row_ih + 3 * hidden + j] = o_gate;
    cell_cache[(b * seq + t) * hidden + j] = c;
  }
}

// BPTT cell for one timestep (executed t = seq-1 .. 0). Consumes the
// activated-gate and cell caches written by the training forward, the
// upstream dY, the recurrent dh from t+1 (GEMM of da_{t+1} with W_hh),
// and the running dc (in place). Emits da_t into the [batch, seq, 4H]
// gradient-gate buffer that the weight/input GEMMs consume afterwards.
extern "C" __global__ void cyxwiz_lstm_backward_cell(
    const float* __restrict__ grad_y,        // [batch, seq, hidden]
    const float* __restrict__ dh_recurrent,  // [batch, hidden]
    float* __restrict__ dc,                  // [batch, hidden], in place
    const float* __restrict__ act_gate_cache,// [batch, seq, 4*hidden]
    const float* __restrict__ cell_cache,    // [batch, seq, hidden]
    float* __restrict__ da,                  // [batch, seq, 4*hidden]
    int t,
    int seq,
    int hidden,
    int total) {        // batch * hidden
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int b = idx / hidden;
  int j = idx - b * hidden;
  int row = (b * seq + t) * 4 * hidden;
  float i_gate = act_gate_cache[row + j];
  float f_gate = act_gate_cache[row + hidden + j];
  float g_gate = act_gate_cache[row + 2 * hidden + j];
  float o_gate = act_gate_cache[row + 3 * hidden + j];
  float c_t = cell_cache[(b * seq + t) * hidden + j];
  float c_prev = t > 0 ? cell_cache[(b * seq + t - 1) * hidden + j] : 0.0f;
  float tanh_c = tanhf(c_t);
  float dh = grad_y[(b * seq + t) * hidden + j] + dh_recurrent[idx];
  float d_o = dh * tanh_c;
  float d_c = dh * o_gate * (1.0f - tanh_c * tanh_c) + dc[idx];
  float d_i = d_c * g_gate;
  float d_f = d_c * c_prev;
  float d_g = d_c * i_gate;
  da[row + j] = d_i * i_gate * (1.0f - i_gate);
  da[row + hidden + j] = d_f * f_gate * (1.0f - f_gate);
  da[row + 2 * hidden + j] = d_g * (1.0f - g_gate * g_gate);
  da[row + 3 * hidden + j] = d_o * o_gate * (1.0f - o_gate);
  dc[idx] = d_c * f_gate;
}

// GRU cell (gate order r, z, n — PyTorch convention, matches gru.cpp):
//   r = sig(x_r + h_r), z = sig(x_z + h_z), hn_pre = W_hn h_prev + b_hn,
//   n = tanh(x_n + r * hn_pre), h = (1 - z) * n + z * h_prev.
// gate_cache (training only) stores [r, z, n, hn_pre] per (b, t) — 4H —
// exactly the cache layout the CPU reference BPTT consumes.
extern "C" __global__ void cyxwiz_gru_cell(
    const float* __restrict__ gates_ih,   // [(b*seq+t), 3*hidden]
    const float* __restrict__ gates_hh,   // [batch, 3*hidden]
    const float* __restrict__ bias_ih,    // [3*hidden]
    const float* __restrict__ bias_hh,    // [3*hidden]
    float* __restrict__ hidden_state,     // [batch, hidden], in/out
    float* __restrict__ sequence_out,     // [batch, seq, hidden]
    float* __restrict__ gate_cache,       // [batch, seq, 4*hidden] or 0
    int t,
    int seq,
    int hidden,
    int total) {        // batch * hidden
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int b = idx / hidden;
  int j = idx - b * hidden;
  int row_ih = (b * seq + t) * 3 * hidden;
  int row_hh = b * 3 * hidden;
  float r = cyxwiz_sigmoid(gates_ih[row_ih + j] + bias_ih[j] +
                           gates_hh[row_hh + j] + bias_hh[j]);
  float z = cyxwiz_sigmoid(gates_ih[row_ih + hidden + j] +
                           bias_ih[hidden + j] +
                           gates_hh[row_hh + hidden + j] +
                           bias_hh[hidden + j]);
  float hn_pre = gates_hh[row_hh + 2 * hidden + j] + bias_hh[2 * hidden + j];
  float n = tanhf(gates_ih[row_ih + 2 * hidden + j] +
                  bias_ih[2 * hidden + j] + r * hn_pre);
  float h_prev = hidden_state[idx];
  float h = (1.0f - z) * n + z * h_prev;
  hidden_state[idx] = h;
  sequence_out[(b * seq + t) * hidden + j] = h;
  if (gate_cache) {
    int row_c = (b * seq + t) * 4 * hidden;
    gate_cache[row_c + j] = r;
    gate_cache[row_c + hidden + j] = z;
    gate_cache[row_c + 2 * hidden + j] = n;
    gate_cache[row_c + 3 * hidden + j] = hn_pre;
  }
}

// GRU BPTT cell for one timestep (t = seq-1 .. 0). dh_recurrent holds the
// complete dh arriving from t+1 on entry and is overwritten with the
// DIRECT part dh_total * z on exit; the caller's accumulating GEMM then
// adds dgates_h_t @ W_hh to complete dh for t-1. x-side and h-side gate
// gradients are emitted separately because they differ in the n slot
// (dn_pre vs dn_pre * r) — the reason GRU is not "LSTM with 3 gates".
extern "C" __global__ void cyxwiz_gru_backward_cell(
    const float* __restrict__ grad_y,        // [batch, seq, hidden]
    float* __restrict__ dh_recurrent,        // [batch, hidden], in/out
    const float* __restrict__ gate_cache,    // [batch, seq, 4*hidden]
    const float* __restrict__ sequence_out,  // [batch, seq, hidden] = h_t
    float* __restrict__ dgates_x,            // [batch, seq, 3*hidden]
    float* __restrict__ dgates_h,            // [batch, seq, 3*hidden]
    int t,
    int seq,
    int hidden,
    int total) {        // batch * hidden
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int b = idx / hidden;
  int j = idx - b * hidden;
  int row_c = (b * seq + t) * 4 * hidden;
  int row_g = (b * seq + t) * 3 * hidden;
  float r = gate_cache[row_c + j];
  float z = gate_cache[row_c + hidden + j];
  float n = gate_cache[row_c + 2 * hidden + j];
  float hn_pre = gate_cache[row_c + 3 * hidden + j];
  float h_prev = t > 0 ? sequence_out[(b * seq + t - 1) * hidden + j] : 0.0f;
  float dh = grad_y[(b * seq + t) * hidden + j] + dh_recurrent[idx];
  float dn = dh * (1.0f - z);
  float dz = dh * (h_prev - n);
  float dn_pre = dn * (1.0f - n * n);
  float dr = dn_pre * hn_pre;
  float d_hn_pre = dn_pre * r;
  float d_r_pre = dr * r * (1.0f - r);
  float d_z_pre = dz * z * (1.0f - z);
  dgates_x[row_g + j] = d_r_pre;
  dgates_x[row_g + hidden + j] = d_z_pre;
  dgates_x[row_g + 2 * hidden + j] = dn_pre;
  dgates_h[row_g + j] = d_r_pre;
  dgates_h[row_g + hidden + j] = d_z_pre;
  dgates_h[row_g + 2 * hidden + j] = d_hn_pre;
  dh_recurrent[idx] = dh * z;
}
)";

struct NvidiaRuntimeProbe {
    bool ok = false;
    int device_count = 0;
    int cuda_runtime_version = 0;
    int cublas_version = 0;
    int cc_major = 0;
    int cc_minor = 0;
    std::string failure;
};

NvidiaRuntimeProbe ProbeNvidiaRuntime() {
    NvidiaRuntimeProbe probe;
    const cudaError_t count_status = cudaGetDeviceCount(&probe.device_count);
    if (count_status != cudaSuccess || probe.device_count <= 0) {
        probe.failure = std::string("no usable CUDA device: ") +
                        cudaGetErrorString(count_status);
        return probe;
    }
    if (cudaRuntimeGetVersion(&probe.cuda_runtime_version) != cudaSuccess) {
        probe.failure = "CUDA runtime version query failed";
        return probe;
    }
    cudaDeviceGetAttribute(&probe.cc_major, cudaDevAttrComputeCapabilityMajor,
                           0);
    cudaDeviceGetAttribute(&probe.cc_minor, cudaDevAttrComputeCapabilityMinor,
                           0);
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        probe.failure = "cuBLAS handle creation failed";
        return probe;
    }
    cublasGetVersion(handle, &probe.cublas_version);
    cublasDestroy(handle);
    probe.ok = true;
    return probe;
}

class ExecutionState {
public:
    ~ExecutionState() {
        if (cublas_) cublasDestroy(cublas_);
        if (module_) cuModuleUnload(module_);
    }

    bool Ensure(int cc_major, int cc_minor, std::string& failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_) return true;
        if (cudaFree(nullptr) != cudaSuccess) {  // establish primary context
            failure = "CUDA context initialization failed";
            return false;
        }
        if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
            failure = "cuBLAS handle creation failed";
            return false;
        }
        nvrtcProgram program = nullptr;
        if (nvrtcCreateProgram(&program, kCellKernelSource,
                               "cyxwiz_cells.cu", 0, nullptr,
                               nullptr) != NVRTC_SUCCESS) {
            failure = "NVRTC program creation failed";
            return false;
        }
        const std::string arch = "--gpu-architecture=compute_" +
                                 std::to_string(cc_major) +
                                 std::to_string(cc_minor);
        const char* options[] = {arch.c_str()};
        const nvrtcResult compile_status =
            nvrtcCompileProgram(program, 1, options);
        if (compile_status != NVRTC_SUCCESS) {
            size_t log_size = 0;
            nvrtcGetProgramLogSize(program, &log_size);
            std::string log(log_size, '\0');
            nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program);
            failure = "NVRTC compile failed: " + log;
            return false;
        }
        size_t ptx_size = 0;
        nvrtcGetPTXSize(program, &ptx_size);
        std::string ptx(ptx_size, '\0');
        nvrtcGetPTX(program, ptx.data());
        nvrtcDestroyProgram(&program);

        if (cuModuleLoadData(&module_, ptx.c_str()) != CUDA_SUCCESS ||
            cuModuleGetFunction(&rnn_kernel_, module_, "cyxwiz_rnn_cell") !=
                CUDA_SUCCESS ||
            cuModuleGetFunction(&lstm_kernel_, module_, "cyxwiz_lstm_cell") !=
                CUDA_SUCCESS ||
            cuModuleGetFunction(&lstm_backward_kernel_, module_,
                                "cyxwiz_lstm_backward_cell") !=
                CUDA_SUCCESS ||
            cuModuleGetFunction(&gru_kernel_, module_, "cyxwiz_gru_cell") !=
                CUDA_SUCCESS ||
            cuModuleGetFunction(&gru_backward_kernel_, module_,
                                "cyxwiz_gru_backward_cell") !=
                CUDA_SUCCESS ||
            cuModuleGetFunction(&fill_kernel_, module_, "cyxwiz_fill") !=
                CUDA_SUCCESS) {
            failure = "cell kernel module load failed";
            return false;
        }
        ready_ = true;
        return true;
    }

    cublasHandle_t Cublas() const { return cublas_; }
    CUfunction RnnKernel() const { return rnn_kernel_; }
    CUfunction LstmKernel() const { return lstm_kernel_; }
    CUfunction LstmBackwardKernel() const { return lstm_backward_kernel_; }
    CUfunction GruKernel() const { return gru_kernel_; }
    CUfunction GruBackwardKernel() const { return gru_backward_kernel_; }
    CUfunction FillKernel() const { return fill_kernel_; }
    std::mutex& ExecuteMutex() { return mutex_; }

private:
    std::mutex mutex_;
    bool ready_ = false;
    cublasHandle_t cublas_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction rnn_kernel_ = nullptr;
    CUfunction lstm_kernel_ = nullptr;
    CUfunction lstm_backward_kernel_ = nullptr;
    CUfunction gru_kernel_ = nullptr;
    CUfunction gru_backward_kernel_ = nullptr;
    CUfunction fill_kernel_ = nullptr;
};

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }
    bool Allocate(size_t bytes) {
        return cudaMalloc(&ptr, bytes) == cudaSuccess;
    }
    float* Float() const { return static_cast<float*>(ptr); }
};

// Row-major C[m x n] = A[m x k] * B[n x k]^T via column-major cuBLAS:
// sgemm(OP_T, OP_N, n, m, k, B, ldb=k, A, lda=k, C, ldc=n).
bool GemmRowMajorABt(cublasHandle_t handle, int m, int n, int k,
                     const float* A, const float* B, float* C) {
    const float one = 1.0f;
    const float zero = 0.0f;
    return cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, B, k,
                       A, k, &zero, C, n) == CUBLAS_STATUS_SUCCESS;
}

// Row-major C[m x n] = A[m x k] * B[k x n], with row strides (leading
// dimensions in row-major elements) so batch-strided timestep slices work.
bool GemmRowMajorAB(cublasHandle_t handle, int m, int n, int k,
                    const float* A, int lda_rows, const float* B,
                    int ldb_rows, float* C, int ldc_rows) {
    const float one = 1.0f;
    const float zero = 0.0f;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, B,
                       ldb_rows, A, lda_rows, &zero, C,
                       ldc_rows) == CUBLAS_STATUS_SUCCESS;
}

// Row-major C[m x n] += A[m x k] * B[k x n] (accumulating), with row
// strides; used where a recurrent gradient has a direct term already in C.
bool GemmRowMajorABAccum(cublasHandle_t handle, int m, int n, int k,
                         const float* A, int lda_rows, const float* B,
                         int ldb_rows, float* C, int ldc_rows) {
    const float one = 1.0f;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, B,
                       ldb_rows, A, lda_rows, &one, C,
                       ldc_rows) == CUBLAS_STATUS_SUCCESS;
}

// Row-major C[k x n] += A[m x k]^T * B[m x n] (accumulating), with row
// strides, used for weight-gradient accumulation across timesteps.
bool GemmRowMajorAtBAccum(cublasHandle_t handle, int m, int k, int n,
                          const float* A, int lda_rows, const float* B,
                          int ldb_rows, float* C, int ldc_rows) {
    const float one = 1.0f;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &one, B,
                       ldb_rows, A, lda_rows, &one, C,
                       ldc_rows) == CUBLAS_STATUS_SUCCESS;
}

class NvidiaCublasProvider final : public INeuralNetworkProvider {
public:
    explicit NvidiaCublasProvider(NvidiaRuntimeProbe probe)
        : probe_(std::move(probe)) {}

    const char* ProviderId() const override { return kProviderId; }
    DeviceType Platform() const override { return DeviceType::CUDA; }

    std::string Version() const override {
        std::ostringstream out;
        out << kProviderId << " " << kProviderSemver << " / CUDA runtime "
            << probe_.cuda_runtime_version << " / cuBLAS "
            << probe_.cublas_version << " / devices " << probe_.device_count
            << " / sm_" << probe_.cc_major << probe_.cc_minor;
        return out.str();
    }

    NeuralCapability QueryCapability(
        const NeuralOpRequest& request) const override {
        NeuralCapability capability;
        capability.reason =
            BackendFallbackReason::NvidiaProviderUnsupportedContract;
        // Device-keyed dispatch: the registry already filters on
        // Platform(); answer truthfully for direct callers too.
        if (request.target.platform != DeviceType::CUDA) {
            capability.detail =
                std::string("request targets ") +
                NeuralDevicePlatformName(request.target.platform) +
                "; this provider serves cuda only";
            return capability;
        }
        const bool is_rnn = request.op == NeuralOp::RnnForward;
        const bool is_lstm = request.op == NeuralOp::LstmForward;
        const bool is_lstm_backward = request.op == NeuralOp::LstmBackward;
        const bool is_gru = request.op == NeuralOp::GruForward ||
                            request.op == NeuralOp::GruBackward;
        if (!is_rnn && !is_lstm && !is_lstm_backward && !is_gru) {
            capability.detail =
                std::string(NeuralOpName(request.op)) +
                " is not implemented yet (supported: rnn_forward, "
                "lstm_forward, lstm_backward, gru_forward, gru_backward)";
            return capability;
        }
        // rnn_forward remains inference-only; LSTM (P2) and GRU (P3)
        // support training: their backward ops are self-contained
        // recompute-forward + BPTT calls.
        if (is_rnn && request.training) {
            capability.detail =
                "rnn_forward training is not supported yet";
            return capability;
        }
        if (request.dtype != DataType::Float32) {
            capability.detail = "contract is Float32 only";
            return capability;
        }
        if (request.directions != 1) {
            capability.detail = "contract is single-direction";
            return capability;
        }
        if (request.layers < 1) {
            capability.detail = "layers must be positive";
            return capability;
        }
        if (is_rnn && request.layers != 1) {
            capability.detail = "rnn_forward is single-layer";
            return capability;
        }
        if (is_rnn && request.activation != NeuralActivation::Tanh &&
            request.activation != NeuralActivation::Relu) {
            capability.detail = "rnn_forward supports tanh or relu";
            return capability;
        }
        if ((is_lstm || is_lstm_backward || is_gru) &&
            request.activation != NeuralActivation::None) {
            capability.detail =
                "LSTM/GRU own their gate activations; request.activation "
                "must be None";
            return capability;
        }
        if (request.batch == 0 || request.seq == 0 || request.input == 0 ||
            request.hidden == 0) {
            capability.detail = "all dimensions must be positive";
            return capability;
        }
        capability.supported = true;
        capability.reason = BackendFallbackReason::BackendInternalError;
        capability.detail.clear();
        return capability;
    }

    NeuralResourceEstimate EstimateResources(
        const NeuralOpRequest& request) const override {
        // Mirrors the DeviceBuffer lists in the Execute* bodies (per layer
        // where applicable); keep them in step. reserve_bytes stays 0: the
        // backward ops are self-contained recompute-forward + BPTT calls,
        // so nothing is retained between forward and backward.
        NeuralResourceEstimate estimate;
        const size_t f = sizeof(float);
        const bool is_lstm = request.op == NeuralOp::LstmForward ||
                             request.op == NeuralOp::LstmBackward;
        const bool is_gru = request.op == NeuralOp::GruForward ||
                            request.op == NeuralOp::GruBackward;
        const bool backward = request.op == NeuralOp::LstmBackward ||
                              request.op == NeuralOp::GruBackward;
        const size_t gates = is_lstm ? 4 : (is_gru ? 3 : 1);
        const size_t layers = std::max<size_t>(1, request.layers);
        const size_t rows = request.batch * request.seq;
        const size_t gate_width = gates * request.hidden;
        size_t weight_elems = 0;
        size_t input_elems = 0;  // per-layer input sequences (dx staging)
        for (size_t l = 0; l < layers; ++l) {
            const size_t in = l == 0 ? request.input : request.hidden;
            weight_elems += gate_width * in + gate_width * request.hidden +
                            2 * gate_width;
            input_elems += rows * in;
        }
        // x, weights, input projection, recurrent projection, h, c, one
        // output sequence per layer.
        estimate.workspace_bytes =
            rows * request.input * f + weight_elems * f +
            rows * gate_width * f + request.batch * gate_width * f +
            2 * request.batch * request.hidden * f +
            layers * rows * request.hidden * f;
        if (backward) {
            // dy, per-layer gate caches (4H) and LSTM cell caches, gate
            // gradients (LSTM one buffer, GRU two), dh_rec, per-layer dx,
            // weight/bias gradients, ones.
            estimate.workspace_bytes +=
                rows * request.hidden * f +
                layers * rows * 4 * request.hidden * f +
                (is_lstm ? layers * rows * request.hidden * f : 0) +
                (is_lstm ? 1 : 2) * rows * gate_width * f +
                request.batch * request.hidden * f + input_elems * f +
                weight_elems * f + rows * f;
        } else {
            // final-state staging for the optional state outputs
            estimate.workspace_bytes +=
                (is_lstm ? 2 : 1) * layers * request.batch * request.hidden * f;
        }
        return estimate;
    }

    NeuralOpStatus Execute(const NeuralOpRequest& request,
                           NeuralOpBuffers& buffers) override {
        NeuralOpStatus status;
        const NeuralCapability capability = QueryCapability(request);
        if (!capability.supported) {
            status.reason = capability.reason;
            status.detail = capability.detail;
            return status;
        }
        switch (request.op) {
        case NeuralOp::RnnForward:
            return ExecuteRnnForward(request, buffers);
        case NeuralOp::LstmForward:
            return ExecuteStackedForward(request, buffers, CellKind::Lstm);
        case NeuralOp::LstmBackward:
            return ExecuteStackedBackward(request, buffers, CellKind::Lstm);
        case NeuralOp::GruForward:
            return ExecuteStackedForward(request, buffers, CellKind::Gru);
        case NeuralOp::GruBackward:
            return ExecuteStackedBackward(request, buffers, CellKind::Gru);
        default:
            break;
        }
        status.reason = BackendFallbackReason::NvidiaProviderUnsupportedContract;
        status.detail = "op is not executable by this provider";
        return status;
    }

private:
    // ------------------------------------------------------------ helpers
    enum class CellKind { Lstm, Gru };
    static size_t GatesFor(CellKind kind) {
        return kind == CellKind::Lstm ? 4 : 3;
    }

    static NeuralOpStatus Fail(BackendFallbackReason reason,
                               const std::string& detail) {
        NeuralOpStatus status;
        status.ok = false;
        status.reason = reason;
        status.detail = detail;
        return status;
    }

    static NeuralOpStatus Ok() {
        NeuralOpStatus status;
        status.ok = true;
        status.reason = BackendFallbackReason::BackendInternalError;
        return status;
    }

    struct LayerDeviceWeights {
        DeviceBuffer w_ih, w_hh, b_ih, b_hh;
        size_t in_features = 0;
    };

    // Stacked weight list contract: weights = { W_ih, W_hh, b_ih, b_hh } per
    // layer, layer 0 first; layer l>0 consumes the previous layer's
    // [batch, seq, hidden] output, so its W_ih is [G, hidden].
    static bool StackedWeightsMatch(const std::vector<const Tensor*>& weights,
                                    size_t layers, size_t gate_width,
                                    size_t input, size_t hidden) {
        if (weights.size() != 4 * layers) return false;
        for (size_t l = 0; l < layers; ++l) {
            const size_t in = l == 0 ? input : hidden;
            const Tensor* w_ih = weights[4 * l];
            const Tensor* w_hh = weights[4 * l + 1];
            const Tensor* b_ih = weights[4 * l + 2];
            const Tensor* b_hh = weights[4 * l + 3];
            if (!w_ih || !w_hh || !b_ih || !b_hh ||
                w_ih->NumElements() != gate_width * in ||
                w_hh->NumElements() != gate_width * hidden ||
                b_ih->NumElements() != gate_width ||
                b_hh->NumElements() != gate_width) {
                return false;
            }
        }
        return true;
    }

    static bool UploadStackedWeights(const std::vector<const Tensor*>& weights,
                                     size_t layers, size_t gate_width,
                                     size_t input, size_t hidden,
                                     std::vector<LayerDeviceWeights>& out,
                                     NeuralOpStatus& failure) {
        const size_t f = sizeof(float);
        out.clear();
        out.resize(layers);
        for (size_t l = 0; l < layers; ++l) {
            auto& w = out[l];
            w.in_features = l == 0 ? input : hidden;
            if (!w.w_ih.Allocate(gate_width * w.in_features * f) ||
                !w.w_hh.Allocate(gate_width * hidden * f) ||
                !w.b_ih.Allocate(gate_width * f) ||
                !w.b_hh.Allocate(gate_width * f)) {
                failure = Fail(BackendFallbackReason::NvidiaProviderWorkspaceExhausted,
                               "device workspace allocation failed (weights)");
                return false;
            }
            const bool uploaded =
                cudaMemcpy(w.w_ih.ptr, weights[4 * l]->ReadData<float>(),
                           gate_width * w.in_features * f,
                           cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(w.w_hh.ptr, weights[4 * l + 1]->ReadData<float>(),
                           gate_width * hidden * f,
                           cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(w.b_ih.ptr, weights[4 * l + 2]->ReadData<float>(),
                           gate_width * f, cudaMemcpyHostToDevice) ==
                    cudaSuccess &&
                cudaMemcpy(w.b_hh.ptr, weights[4 * l + 3]->ReadData<float>(),
                           gate_width * f, cudaMemcpyHostToDevice) ==
                    cudaSuccess;
            if (!uploaded) {
                failure = Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                               "host-to-device transfer failed (weights)");
                return false;
            }
        }
        return true;
    }

    // One layer's forward over the whole sequence from a ZERO initial
    // state: one input-projection GEMM over the stacked sequence, then per
    // timestep a recurrent GEMM + fused cell. gate_cache/cell_cache are
    // null for inference and written for the training recompute.
    bool RunStackedLayerForward(CellKind kind, size_t batch, size_t seq,
                                size_t hidden, const float* d_input,
                                const LayerDeviceWeights& w,
                                DeviceBuffer& d_gih, DeviceBuffer& d_ghh,
                                DeviceBuffer& d_h, DeviceBuffer& d_c,
                                float* d_y, float* gate_cache,
                                float* cell_cache, std::string& detail) {
        const size_t f = sizeof(float);
        const size_t gate_width = GatesFor(kind) * hidden;
        const size_t rows = batch * seq;
        if (cudaMemset(d_h.ptr, 0, batch * hidden * f) != cudaSuccess ||
            (kind == CellKind::Lstm &&
             cudaMemset(d_c.ptr, 0, batch * hidden * f) != cudaSuccess)) {
            detail = "state reset failed";
            return false;
        }
        if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(rows),
                             static_cast<int>(gate_width),
                             static_cast<int>(w.in_features), d_input,
                             w.w_ih.Float(), d_gih.Float())) {
            detail = "input-projection GEMM failed";
            return false;
        }
        const int total = static_cast<int>(batch * hidden);
        const int block = 256;
        const int grid = (total + block - 1) / block;
        int seq_arg = static_cast<int>(seq);
        int hidden_arg = static_cast<int>(hidden);
        int total_arg = total;
        for (size_t t = 0; t < seq; ++t) {
            if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(batch),
                                 static_cast<int>(gate_width),
                                 static_cast<int>(hidden), d_h.Float(),
                                 w.w_hh.Float(), d_ghh.Float())) {
                detail = "recurrent-projection GEMM failed";
                return false;
            }
            int t_arg = static_cast<int>(t);
            void* gih_ptr = d_gih.ptr;
            void* ghh_ptr = d_ghh.ptr;
            void* bih_ptr = w.b_ih.ptr;
            void* bhh_ptr = w.b_hh.ptr;
            void* c_ptr = d_c.ptr;
            void* h_ptr = d_h.ptr;
            void* y_ptr = d_y;
            void* gate_cache_ptr = gate_cache;
            void* cell_cache_ptr = cell_cache;
            CUresult launch;
            if (kind == CellKind::Lstm) {
                void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                                &c_ptr,   &h_ptr,   &y_ptr,   &gate_cache_ptr,
                                &cell_cache_ptr, &t_arg, &seq_arg,
                                &hidden_arg, &total_arg};
                launch = cuLaunchKernel(state_.LstmKernel(), grid, 1, 1,
                                        block, 1, 1, 0, nullptr, args,
                                        nullptr);
            } else {
                void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                                &h_ptr,   &y_ptr,   &gate_cache_ptr, &t_arg,
                                &seq_arg, &hidden_arg, &total_arg};
                launch = cuLaunchKernel(state_.GruKernel(), grid, 1, 1,
                                        block, 1, 1, 0, nullptr, args,
                                        nullptr);
            }
            if (launch != CUDA_SUCCESS) {
                detail = "cell kernel launch failed";
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------- RNN forward
    // rnn_forward (pilot op, inference-only, single layer):
    //   inputs[0] = x [batch, seq, input]; weights = {W_ih [H, input],
    //   W_hh [H, H], b_ih [H], b_hh [H]}; outputs[0] = h_seq [batch, seq, H].
    NeuralOpStatus ExecuteRnnForward(const NeuralOpRequest& request,
                                     NeuralOpBuffers& buffers) {
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t rows = batch * seq;
        if (buffers.inputs.size() != 1 || buffers.outputs.size() != 1 ||
            !buffers.inputs[0] || !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != rows * input ||
            buffers.outputs[0]->NumElements() != rows * hidden ||
            !StackedWeightsMatch(buffers.weights, 1, hidden, input, hidden)) {
            return Fail(BackendFallbackReason::NvidiaProviderUnsupportedContract,
                        "recurrent forward buffer contract violated");
        }
        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            return Fail(BackendFallbackReason::NvidiaProviderUnavailable, failure);
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        NeuralOpStatus weight_failure;
        std::vector<LayerDeviceWeights> weights;
        if (!UploadStackedWeights(buffers.weights, 1, hidden, input, hidden,
                                  weights, weight_failure)) {
            return weight_failure;
        }
        DeviceBuffer d_x, d_gih, d_ghh, d_h, d_out;
        if (!d_x.Allocate(rows * input * f) ||
            !d_gih.Allocate(rows * hidden * f) ||
            !d_ghh.Allocate(batch * hidden * f) ||
            !d_h.Allocate(batch * hidden * f) ||
            !d_out.Allocate(rows * hidden * f)) {
            return Fail(BackendFallbackReason::NvidiaProviderWorkspaceExhausted,
                        "device workspace allocation failed");
        }
        if (cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       rows * input * f,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemset(d_h.ptr, 0, batch * hidden * f) != cudaSuccess) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "host-to-device transfer failed");
        }
        if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(rows),
                             static_cast<int>(hidden),
                             static_cast<int>(input), d_x.Float(),
                             weights[0].w_ih.Float(), d_gih.Float())) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "input-projection GEMM failed");
        }
        const int total = static_cast<int>(batch * hidden);
        const int block = 256;
        const int grid = (total + block - 1) / block;
        int seq_arg = static_cast<int>(seq);
        int hidden_arg = static_cast<int>(hidden);
        int total_arg = total;
        int tanh_arg = request.activation == NeuralActivation::Tanh ? 1 : 0;
        for (size_t t = 0; t < seq; ++t) {
            if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(batch),
                                 static_cast<int>(hidden),
                                 static_cast<int>(hidden), d_h.Float(),
                                 weights[0].w_hh.Float(), d_ghh.Float())) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "recurrent-projection GEMM failed");
            }
            int t_arg = static_cast<int>(t);
            void* gih_ptr = d_gih.ptr;
            void* ghh_ptr = d_ghh.ptr;
            void* bih_ptr = weights[0].b_ih.ptr;
            void* bhh_ptr = weights[0].b_hh.ptr;
            void* h_ptr = d_h.ptr;
            void* out_ptr = d_out.ptr;
            void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                            &h_ptr,   &out_ptr, &t_arg,   &seq_arg,
                            &hidden_arg, &total_arg, &tanh_arg};
            if (cuLaunchKernel(state_.RnnKernel(), grid, 1, 1, block, 1, 1,
                               0, nullptr, args, nullptr) != CUDA_SUCCESS) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "cell kernel launch failed");
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(buffers.outputs[0]->Data<float>(), d_out.ptr,
                       rows * hidden * f,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "device execution or readback failed");
        }
        return Ok();
    }

    // ----------------------------------------------- stacked forward
    // lstm_forward / gru_forward, `layers` stacked layers (0.6.0):
    //   inputs[0]  = x [batch, seq, input]
    //   weights    = { W_ih, W_hh, b_ih, b_hh } per layer (see
    //                StackedWeightsMatch)
    //   outputs    = { y [batch, seq, hidden] }                       or
    //                { y, h_n [layers, batch, hidden] }                (GRU) or
    //                { y, h_n, c_n [layers, batch, hidden] }          (LSTM)
    // Initial state is zero (layers are stateless per call). Layer l>0
    // consumes layer l-1's full output sequence on the device — no host
    // round trip between layers.
    NeuralOpStatus ExecuteStackedForward(const NeuralOpRequest& request,
                                         NeuralOpBuffers& buffers,
                                         CellKind kind) {
        const size_t layers = request.layers;
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t gate_width = GatesFor(kind) * hidden;
        const size_t rows = batch * seq;
        const size_t state_count = layers * batch * hidden;
        const size_t state_outputs = kind == CellKind::Lstm ? 2 : 1;
        const bool wants_states = buffers.outputs.size() == 1 + state_outputs;
        bool states_ok = true;
        if (wants_states) {
            for (size_t i = 1; i <= state_outputs; ++i) {
                states_ok = states_ok && buffers.outputs[i] &&
                            buffers.outputs[i]->NumElements() == state_count;
            }
        }
        if (buffers.inputs.size() != 1 || !buffers.inputs[0] ||
            (buffers.outputs.size() != 1 && !wants_states) || !states_ok ||
            !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != rows * input ||
            buffers.outputs[0]->NumElements() != rows * hidden ||
            !StackedWeightsMatch(buffers.weights, layers, gate_width, input,
                                 hidden)) {
            return Fail(BackendFallbackReason::NvidiaProviderUnsupportedContract,
                        std::string(NeuralOpName(request.op)) +
                            " buffer contract violated");
        }
        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            return Fail(BackendFallbackReason::NvidiaProviderUnavailable, failure);
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        NeuralOpStatus weight_failure;
        std::vector<LayerDeviceWeights> weights;
        if (!UploadStackedWeights(buffers.weights, layers, gate_width, input,
                                  hidden, weights, weight_failure)) {
            return weight_failure;
        }
        DeviceBuffer d_x, d_gih, d_ghh, d_h, d_c, d_hn, d_cn;
        std::vector<DeviceBuffer> d_y(layers);
        bool allocated = d_x.Allocate(rows * input * f) &&
                         d_gih.Allocate(rows * gate_width * f) &&
                         d_ghh.Allocate(batch * gate_width * f) &&
                         d_h.Allocate(batch * hidden * f) &&
                         d_hn.Allocate(state_count * f) &&
                         (kind != CellKind::Lstm ||
                          (d_c.Allocate(batch * hidden * f) &&
                           d_cn.Allocate(state_count * f)));
        for (size_t l = 0; allocated && l < layers; ++l) {
            allocated = d_y[l].Allocate(rows * hidden * f);
        }
        if (!allocated) {
            return Fail(BackendFallbackReason::NvidiaProviderWorkspaceExhausted,
                        "device workspace allocation failed");
        }
        if (cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       rows * input * f,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "host-to-device transfer failed");
        }
        for (size_t l = 0; l < layers; ++l) {
            const float* d_input = l == 0 ? d_x.Float() : d_y[l - 1].Float();
            std::string detail;
            if (!RunStackedLayerForward(kind, batch, seq, hidden, d_input,
                                        weights[l], d_gih, d_ghh, d_h, d_c,
                                        d_y[l].Float(), nullptr, nullptr,
                                        detail)) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "layer " + std::to_string(l) + ": " + detail);
            }
            if (cudaMemcpy(d_hn.Float() + l * batch * hidden, d_h.ptr,
                           batch * hidden * f,
                           cudaMemcpyDeviceToDevice) != cudaSuccess ||
                (kind == CellKind::Lstm &&
                 cudaMemcpy(d_cn.Float() + l * batch * hidden, d_c.ptr,
                            batch * hidden * f,
                            cudaMemcpyDeviceToDevice) != cudaSuccess)) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "final state capture failed");
            }
        }
        bool downloaded =
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(buffers.outputs[0]->Data<float>(),
                       d_y[layers - 1].ptr, rows * hidden * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess;
        if (downloaded && wants_states) {
            downloaded =
                cudaMemcpy(buffers.outputs[1]->Data<float>(), d_hn.ptr,
                           state_count * f,
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                (kind != CellKind::Lstm ||
                 cudaMemcpy(buffers.outputs[2]->Data<float>(), d_cn.ptr,
                            state_count * f,
                            cudaMemcpyDeviceToHost) == cudaSuccess);
        }
        if (!downloaded) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "device execution or readback failed");
        }
        return Ok();
    }

    // ---------------------------------------------- stacked backward
    // lstm_backward / gru_backward, `layers` stacked layers (0.6.0):
    //   inputs    = { x [batch, seq, input], dy [batch, seq, hidden] }
    //   weights   = { W_ih, W_hh, b_ih, b_hh } per layer
    //   outputs   = { dx [batch, seq, input] }
    //   gradients = { dW_ih, dW_hh, db_ih, db_hh } per layer, same order
    // Self-contained: recomputes every layer's forward with caches on the
    // device, then runs BPTT top-down; layer l's dx is layer l-1's dy.
    // LSTM: db_ih == db_hh (identical column sums, matching the CPU
    // reference). GRU: db_ih and db_hh differ in the n slot.
    NeuralOpStatus ExecuteStackedBackward(const NeuralOpRequest& request,
                                          NeuralOpBuffers& buffers,
                                          CellKind kind) {
        const size_t layers = request.layers;
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t gate_width = GatesFor(kind) * hidden;
        const size_t rows = batch * seq;
        bool grads_ok = buffers.gradients.size() == 4 * layers;
        for (size_t l = 0; grads_ok && l < layers; ++l) {
            const size_t in = l == 0 ? input : hidden;
            grads_ok = buffers.gradients[4 * l] && buffers.gradients[4 * l + 1] &&
                       buffers.gradients[4 * l + 2] && buffers.gradients[4 * l + 3] &&
                       buffers.gradients[4 * l]->NumElements() == gate_width * in &&
                       buffers.gradients[4 * l + 1]->NumElements() ==
                           gate_width * hidden &&
                       buffers.gradients[4 * l + 2]->NumElements() == gate_width &&
                       buffers.gradients[4 * l + 3]->NumElements() == gate_width;
        }
        if (buffers.inputs.size() != 2 || buffers.outputs.size() != 1 ||
            !buffers.inputs[0] || !buffers.inputs[1] || !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != rows * input ||
            buffers.inputs[1]->NumElements() != rows * hidden ||
            buffers.outputs[0]->NumElements() != rows * input || !grads_ok ||
            !StackedWeightsMatch(buffers.weights, layers, gate_width, input,
                                 hidden)) {
            return Fail(BackendFallbackReason::NvidiaProviderUnsupportedContract,
                        std::string(NeuralOpName(request.op)) +
                            " buffer contract violated");
        }
        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            return Fail(BackendFallbackReason::NvidiaProviderUnavailable, failure);
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        const bool lstm = kind == CellKind::Lstm;
        NeuralOpStatus weight_failure;
        std::vector<LayerDeviceWeights> weights;
        if (!UploadStackedWeights(buffers.weights, layers, gate_width, input,
                                  hidden, weights, weight_failure)) {
            return weight_failure;
        }
        // Shared per-layer scratch + per-layer caches for the recompute.
        DeviceBuffer d_x, d_dy, d_gih, d_ghh, d_h, d_c, d_dhrec, d_dgx, d_dgh,
            d_ones;
        std::vector<DeviceBuffer> d_y(layers), d_gates(layers),
            d_ccache(layers), d_dx(layers), d_dwih(layers), d_dwhh(layers),
            d_dbih(layers), d_dbhh(layers);
        bool allocated = d_x.Allocate(rows * input * f) &&
                         d_dy.Allocate(rows * hidden * f) &&
                         d_gih.Allocate(rows * gate_width * f) &&
                         d_ghh.Allocate(batch * gate_width * f) &&
                         d_h.Allocate(batch * hidden * f) &&
                         d_c.Allocate(batch * hidden * f) &&
                         d_dhrec.Allocate(batch * hidden * f) &&
                         d_dgx.Allocate(rows * gate_width * f) &&
                         (lstm || d_dgh.Allocate(rows * gate_width * f)) &&
                         d_ones.Allocate(rows * f);
        for (size_t l = 0; allocated && l < layers; ++l) {
            const size_t in = l == 0 ? input : hidden;
            allocated = d_y[l].Allocate(rows * hidden * f) &&
                        d_gates[l].Allocate(rows * 4 * hidden * f) &&
                        (!lstm || d_ccache[l].Allocate(rows * hidden * f)) &&
                        d_dx[l].Allocate(rows * in * f) &&
                        d_dwih[l].Allocate(gate_width * in * f) &&
                        d_dwhh[l].Allocate(gate_width * hidden * f) &&
                        d_dbih[l].Allocate(gate_width * f) &&
                        d_dbhh[l].Allocate(gate_width * f);
        }
        if (!allocated) {
            return Fail(BackendFallbackReason::NvidiaProviderWorkspaceExhausted,
                        "device workspace allocation failed");
        }
        if (cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       rows * input * f,
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_dy.ptr, buffers.inputs[1]->ReadData<float>(),
                       rows * hidden * f,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "host-to-device transfer failed");
        }

        // Recompute every layer's forward with caches.
        for (size_t l = 0; l < layers; ++l) {
            const float* d_input = l == 0 ? d_x.Float() : d_y[l - 1].Float();
            std::string detail;
            if (!RunStackedLayerForward(kind, batch, seq, hidden, d_input,
                                        weights[l], d_gih, d_ghh, d_h, d_c,
                                        d_y[l].Float(), d_gates[l].Float(),
                                        lstm ? d_ccache[l].Float() : nullptr,
                                        detail)) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "training forward layer " + std::to_string(l) +
                                ": " + detail);
            }
        }

        const int total = static_cast<int>(batch * hidden);
        const int block = 256;
        const int grid = (total + block - 1) / block;
        int seq_arg = static_cast<int>(seq);
        int hidden_arg = static_cast<int>(hidden);
        int total_arg = total;
        const int ld_gates = static_cast<int>(seq * gate_width);
        {
            void* ones_ptr = d_ones.ptr;
            float one_value = 1.0f;
            int count = static_cast<int>(rows);
            const int fill_grid = (count + block - 1) / block;
            void* args[] = {&ones_ptr, &one_value, &count};
            if (cuLaunchKernel(state_.FillKernel(), fill_grid, 1, 1, block,
                               1, 1, 0, nullptr, args,
                               nullptr) != CUDA_SUCCESS) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "ones fill launch failed");
            }
        }

        // BPTT, top layer first; each layer's dx feeds the layer below.
        const float* d_dy_cur = d_dy.Float();
        for (size_t li = layers; li-- > 0;) {
            const LayerDeviceWeights& w = weights[li];
            const size_t in = w.in_features;
            const float* d_input = li == 0 ? d_x.Float() : d_y[li - 1].Float();
            if (cudaMemset(d_dhrec.ptr, 0, batch * hidden * f) != cudaSuccess ||
                cudaMemset(d_c.ptr, 0, batch * hidden * f) != cudaSuccess ||
                cudaMemset(d_dwih[li].ptr, 0, gate_width * in * f) != cudaSuccess ||
                cudaMemset(d_dwhh[li].ptr, 0, gate_width * hidden * f) !=
                    cudaSuccess) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "gradient reset failed");
            }
            for (size_t t = seq; t-- > 0;) {
                int t_arg = static_cast<int>(t);
                void* dy_ptr = const_cast<float*>(d_dy_cur);
                void* dhrec_ptr = d_dhrec.ptr;
                void* dc_ptr = d_c.ptr;
                void* gates_ptr = d_gates[li].ptr;
                void* ccache_ptr = d_ccache[li].ptr;
                void* y_ptr = d_y[li].ptr;
                void* dgx_ptr = d_dgx.ptr;
                void* dgh_ptr = d_dgh.ptr;
                CUresult launch;
                if (lstm) {
                    void* args[] = {&dy_ptr, &dhrec_ptr, &dc_ptr, &gates_ptr,
                                    &ccache_ptr, &dgx_ptr, &t_arg, &seq_arg,
                                    &hidden_arg, &total_arg};
                    launch = cuLaunchKernel(state_.LstmBackwardKernel(), grid,
                                            1, 1, block, 1, 1, 0, nullptr,
                                            args, nullptr);
                } else {
                    void* args[] = {&dy_ptr,  &dhrec_ptr, &gates_ptr, &y_ptr,
                                    &dgx_ptr, &dgh_ptr,   &t_arg,     &seq_arg,
                                    &hidden_arg, &total_arg};
                    launch = cuLaunchKernel(state_.GruBackwardKernel(), grid,
                                            1, 1, block, 1, 1, 0, nullptr,
                                            args, nullptr);
                }
                if (launch != CUDA_SUCCESS) {
                    return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                                "backward cell launch failed");
                }
                // x-side gate gradients drive dx/dW_ih; h-side drive
                // dh_rec/dW_hh (they coincide for LSTM).
                const float* dgx_t = d_dgx.Float() + t * gate_width;
                const float* dgh_t =
                    lstm ? dgx_t : d_dgh.Float() + t * gate_width;
                const bool dh_ok =
                    lstm ? GemmRowMajorAB(state_.Cublas(),
                                          static_cast<int>(batch),
                                          static_cast<int>(hidden),
                                          static_cast<int>(gate_width), dgh_t,
                                          ld_gates, w.w_hh.Float(),
                                          static_cast<int>(hidden),
                                          d_dhrec.Float(),
                                          static_cast<int>(hidden))
                         : GemmRowMajorABAccum(state_.Cublas(),
                                               static_cast<int>(batch),
                                               static_cast<int>(hidden),
                                               static_cast<int>(gate_width),
                                               dgh_t, ld_gates, w.w_hh.Float(),
                                               static_cast<int>(hidden),
                                               d_dhrec.Float(),
                                               static_cast<int>(hidden));
                if (!dh_ok) {
                    return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                                "recurrent gradient GEMM failed");
                }
                if (!GemmRowMajorAtBAccum(
                        state_.Cublas(), static_cast<int>(batch),
                        static_cast<int>(gate_width), static_cast<int>(in),
                        dgx_t, ld_gates, d_input + t * in,
                        static_cast<int>(seq * in), d_dwih[li].Float(),
                        static_cast<int>(in))) {
                    return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                                "input weight-gradient GEMM failed");
                }
                if (t > 0 &&
                    !GemmRowMajorAtBAccum(
                        state_.Cublas(), static_cast<int>(batch),
                        static_cast<int>(gate_width), static_cast<int>(hidden),
                        dgh_t, ld_gates, d_y[li].Float() + (t - 1) * hidden,
                        static_cast<int>(seq * hidden), d_dwhh[li].Float(),
                        static_cast<int>(hidden))) {
                    return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                                "hidden weight-gradient GEMM failed");
                }
            }
            // dx_l = DGX * W_ih over the stacked sequence.
            if (!GemmRowMajorAB(state_.Cublas(), static_cast<int>(rows),
                                static_cast<int>(in),
                                static_cast<int>(gate_width), d_dgx.Float(),
                                static_cast<int>(gate_width), w.w_ih.Float(),
                                static_cast<int>(in), d_dx[li].Float(),
                                static_cast<int>(in))) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "input gradient GEMM failed");
            }
            // Bias gradients = column sums via ones-vector GEMVs.
            const float alpha = 1.0f;
            const float beta = 0.0f;
            if (cublasSgemv(state_.Cublas(), CUBLAS_OP_N,
                            static_cast<int>(gate_width),
                            static_cast<int>(rows), &alpha, d_dgx.Float(),
                            static_cast<int>(gate_width), d_ones.Float(), 1,
                            &beta, d_dbih[li].Float(),
                            1) != CUBLAS_STATUS_SUCCESS) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "bias gradient GEMV failed");
            }
            if (lstm) {
                if (cudaMemcpy(d_dbhh[li].ptr, d_dbih[li].ptr, gate_width * f,
                               cudaMemcpyDeviceToDevice) != cudaSuccess) {
                    return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                                "bias gradient copy failed");
                }
            } else if (cublasSgemv(state_.Cublas(), CUBLAS_OP_N,
                                   static_cast<int>(gate_width),
                                   static_cast<int>(rows), &alpha,
                                   d_dgh.Float(), static_cast<int>(gate_width),
                                   d_ones.Float(), 1, &beta,
                                   d_dbhh[li].Float(),
                                   1) != CUBLAS_STATUS_SUCCESS) {
                return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                            "bias gradient GEMV failed");
            }
            d_dy_cur = d_dx[li].Float();
        }

        bool downloaded =
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(buffers.outputs[0]->Data<float>(), d_dx[0].ptr,
                       rows * input * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess;
        for (size_t l = 0; downloaded && l < layers; ++l) {
            const size_t in = l == 0 ? input : hidden;
            downloaded =
                cudaMemcpy(buffers.gradients[4 * l]->Data<float>(),
                           d_dwih[l].ptr, gate_width * in * f,
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(buffers.gradients[4 * l + 1]->Data<float>(),
                           d_dwhh[l].ptr, gate_width * hidden * f,
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(buffers.gradients[4 * l + 2]->Data<float>(),
                           d_dbih[l].ptr, gate_width * f,
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(buffers.gradients[4 * l + 3]->Data<float>(),
                           d_dbhh[l].ptr, gate_width * f,
                           cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (!downloaded) {
            return Fail(BackendFallbackReason::NvidiaProviderExecutionFailed,
                        "device execution or readback failed");
        }
        return Ok();
    }

    NvidiaRuntimeProbe probe_;
    ExecutionState state_;
};

} // namespace

bool NvidiaProviderDeviceMemoryForTesting(size_t& free_bytes,
                                          size_t& total_bytes) {
    free_bytes = 0;
    total_bytes = 0;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
}

void RegisterNvidiaCublasNeuralProvider(NeuralProviderRegistry& registry) {
    const NvidiaRuntimeProbe probe = ProbeNvidiaRuntime();
    if (!probe.ok) {
        spdlog::info(
            "NVIDIA neural provider not registered (reason={}): {}",
            BackendFallbackReasonName(
                BackendFallbackReason::NvidiaProviderUnavailable),
            probe.failure);
        return;
    }
    auto provider = std::make_shared<NvidiaCublasProvider>(probe);
    spdlog::info("NVIDIA neural provider registered: {}",
                 provider->Version());
    registry.Register(std::move(provider));
}

} // namespace cyxwiz

#endif // CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
