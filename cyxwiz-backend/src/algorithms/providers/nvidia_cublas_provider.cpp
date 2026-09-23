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
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

constexpr const char* kProviderId = "cyxwiz.nvidia-cublas-cell";
constexpr const char* kProviderSemver = "0.5.1-lstm-cell-state";

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
        if (request.layers != 1 || request.directions != 1) {
            capability.detail = "contract is single-layer, single-direction";
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
        // Mirrors the DeviceBuffer lists in the Execute* bodies exactly;
        // keep them in step. reserve_bytes stays 0: the backward op is a
        // self-contained recompute-forward + BPTT call, so nothing is
        // retained between forward and backward (no reserve space).
        NeuralResourceEstimate estimate;
        const size_t f = sizeof(float);
        const bool is_lstm = request.op == NeuralOp::LstmForward ||
                             request.op == NeuralOp::LstmBackward;
        const bool is_gru = request.op == NeuralOp::GruForward ||
                            request.op == NeuralOp::GruBackward;
        const size_t gates = is_lstm ? 4 : (is_gru ? 3 : 1);
        const size_t rows = request.batch * request.seq;
        const size_t gate_width = gates * request.hidden;
        // Shared by every op: x, W_ih, W_hh, b_ih, b_hh, input projection,
        // recurrent projection, h, c/state.
        estimate.workspace_bytes =
            rows * request.input * f +
            gate_width * request.input * f +
            gate_width * request.hidden * f +
            2 * gate_width * f +
            rows * gate_width * f +
            request.batch * gate_width * f +
            2 * request.batch * request.hidden * f;
        if (request.op == NeuralOp::LstmBackward) {
            // y, activated gates, cell cache, dy, dA, dh_rec, dx, dW_ih,
            // dW_hh, db, ones.
            estimate.workspace_bytes +=
                rows * request.hidden * f +
                rows * gate_width * f +
                rows * request.hidden * f +
                rows * request.hidden * f +
                rows * gate_width * f +
                request.batch * request.hidden * f +
                rows * request.input * f +
                gate_width * request.input * f +
                gate_width * request.hidden * f +
                gate_width * f +
                rows * f;
        } else if (request.op == NeuralOp::GruBackward) {
            // y, [r,z,n,hn_pre] cache (4H), dy, dgates_x, dgates_h,
            // dh_rec, dx, dW_ih, dW_hh, db_ih, db_hh, ones.
            estimate.workspace_bytes +=
                rows * request.hidden * f +
                rows * 4 * request.hidden * f +
                rows * request.hidden * f +
                2 * rows * gate_width * f +
                request.batch * request.hidden * f +
                rows * request.input * f +
                gate_width * request.input * f +
                gate_width * request.hidden * f +
                2 * gate_width * f +
                rows * f;
        } else {
            // output sequence
            estimate.workspace_bytes += rows * request.hidden * f;
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
        if (request.op == NeuralOp::LstmBackward) {
            return ExecuteLstmBackward(request, buffers);
        }
        if (request.op == NeuralOp::GruForward) {
            return ExecuteGruForward(request, buffers);
        }
        if (request.op == NeuralOp::GruBackward) {
            return ExecuteGruBackward(request, buffers);
        }
        // Shared recurrent forward buffer contract:
        //   inputs[0]  = x      [batch, seq, input]
        //   weights    = { W_ih [gates*hidden, input],
        //                  W_hh [gates*hidden, hidden],
        //                  b_ih [gates*hidden], b_hh [gates*hidden] }
        //   outputs[0] = h_seq  [batch, seq, hidden]
        //   outputs[1] = c_n    [batch, hidden]   (LSTM only, OPTIONAL:
        //                final cell state, so the layer's c_n_ is not
        //                stale on the provider path)
        // gates = 1 (RNN) or 4 (LSTM, gate order i,f,g,o).
        const size_t gates = request.op == NeuralOp::LstmForward ? 4 : 1;
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t gate_width = gates * hidden;
        const bool wants_cell_state = buffers.outputs.size() == 2;
        if (buffers.inputs.size() != 1 || buffers.weights.size() != 4 ||
            (buffers.outputs.size() != 1 && !wants_cell_state) ||
            (wants_cell_state &&
             (gates != 4 || !buffers.outputs[1] ||
              buffers.outputs[1]->NumElements() != batch * hidden)) ||
            !buffers.inputs[0] || !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != batch * seq * input ||
            buffers.weights[0]->NumElements() != gate_width * input ||
            buffers.weights[1]->NumElements() != gate_width * hidden ||
            buffers.weights[2]->NumElements() != gate_width ||
            buffers.weights[3]->NumElements() != gate_width ||
            buffers.outputs[0]->NumElements() != batch * seq * hidden) {
            status.reason =
                BackendFallbackReason::NvidiaProviderUnsupportedContract;
            status.detail = "recurrent forward buffer contract violated";
            return status;
        }

        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            status.reason = BackendFallbackReason::NvidiaProviderUnavailable;
            status.detail = failure;
            return status;
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        DeviceBuffer d_x, d_wih, d_whh, d_bih, d_bhh, d_gih, d_ghh, d_h,
            d_c, d_out;
        const bool need_cell = gates == 4;
        if (!d_x.Allocate(batch * seq * input * f) ||
            !d_wih.Allocate(gate_width * input * f) ||
            !d_whh.Allocate(gate_width * hidden * f) ||
            !d_bih.Allocate(gate_width * f) ||
            !d_bhh.Allocate(gate_width * f) ||
            !d_gih.Allocate(batch * seq * gate_width * f) ||
            !d_ghh.Allocate(batch * gate_width * f) ||
            !d_h.Allocate(batch * hidden * f) ||
            (need_cell && !d_c.Allocate(batch * hidden * f)) ||
            !d_out.Allocate(batch * seq * hidden * f)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderWorkspaceExhausted;
            status.detail = "device workspace allocation failed";
            return status;
        }
        const bool uploaded =
            cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       batch * seq * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_wih.ptr, buffers.weights[0]->ReadData<float>(),
                       gate_width * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_whh.ptr, buffers.weights[1]->ReadData<float>(),
                       gate_width * hidden * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_bih.ptr, buffers.weights[2]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(d_bhh.ptr, buffers.weights[3]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemset(d_h.ptr, 0, batch * hidden * f) == cudaSuccess &&
            (!need_cell ||
             cudaMemset(d_c.ptr, 0, batch * hidden * f) == cudaSuccess);
        if (!uploaded) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "host-to-device transfer failed";
            return status;
        }

        // All input projections in one GEMM over [batch*seq, input].
        if (!GemmRowMajorABt(state_.Cublas(),
                             static_cast<int>(batch * seq),
                             static_cast<int>(gate_width),
                             static_cast<int>(input), d_x.Float(),
                             d_wih.Float(), d_gih.Float())) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "input-projection GEMM failed";
            return status;
        }

        const int total = static_cast<int>(batch * hidden);
        const int block = 256;
        const int grid = (total + block - 1) / block;
        const int use_tanh =
            request.activation == NeuralActivation::Tanh ? 1 : 0;
        for (size_t t = 0; t < seq; ++t) {
            if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(batch),
                                 static_cast<int>(gate_width),
                                 static_cast<int>(hidden), d_h.Float(),
                                 d_whh.Float(), d_ghh.Float())) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "recurrent-projection GEMM failed";
                return status;
            }
            int t_arg = static_cast<int>(t);
            int seq_arg = static_cast<int>(seq);
            int hidden_arg = static_cast<int>(hidden);
            int total_arg = total;
            int tanh_arg = use_tanh;
            void* gih_ptr = d_gih.ptr;
            void* ghh_ptr = d_ghh.ptr;
            void* bih_ptr = d_bih.ptr;
            void* bhh_ptr = d_bhh.ptr;
            void* c_ptr = d_c.ptr;
            void* h_ptr = d_h.ptr;
            void* out_ptr = d_out.ptr;
            CUresult launch;
            if (need_cell) {
                void* null_cache = nullptr;
                void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                                &c_ptr,   &h_ptr,   &out_ptr, &null_cache,
                                &null_cache, &t_arg, &seq_arg, &hidden_arg,
                                &total_arg};
                launch = cuLaunchKernel(state_.LstmKernel(), grid, 1, 1,
                                        block, 1, 1, 0, nullptr, args,
                                        nullptr);
            } else {
                void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                                &h_ptr,   &out_ptr, &t_arg,   &seq_arg,
                                &hidden_arg, &total_arg, &tanh_arg};
                launch = cuLaunchKernel(state_.RnnKernel(), grid, 1, 1,
                                        block, 1, 1, 0, nullptr, args,
                                        nullptr);
            }
            if (launch != CUDA_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "cell kernel launch failed";
                return status;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(buffers.outputs[0]->Data<float>(), d_out.ptr,
                       batch * seq * hidden * f,
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            (wants_cell_state &&
             cudaMemcpy(buffers.outputs[1]->Data<float>(), d_c.ptr,
                        batch * hidden * f,
                        cudaMemcpyDeviceToHost) != cudaSuccess)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "device execution or readback failed";
            return status;
        }
        status.ok = true;
        status.reason = BackendFallbackReason::BackendInternalError;
        status.detail.clear();
        return status;
    }

private:
    // Self-contained training step gradient (recompute-forward + BPTT).
    // Contract:
    //   inputs    = { x [b,s,in], grad_output dY [b,s,H] }
    //   weights   = { W_ih [4H,in], W_hh [4H,H], b_ih [4H], b_hh [4H] }
    //   outputs   = { dx [b,s,in] }
    //   gradients = { dW_ih [4H,in], dW_hh [4H,H], db_ih [4H], db_hh [4H] }
    // db_ih and db_hh receive the same column-sum (bias gradients are
    // identical for this cell, matching the CPU reference).

    // ---------------------------------------------------------------- GRU
    // Shared GRU device pipeline: one input-projection GEMM over the
    // stacked sequence, then per timestep a recurrent GEMM + fused cell.
    // With gate_cache non-null (training) the cell also records
    // [r, z, n, hn_pre] for BPTT. Returns false with `detail` set on
    // failure; the caller maps it to a typed status.
    bool RunGruForwardOnDevice(size_t batch, size_t seq, size_t input,
                               size_t hidden, const DeviceBuffer& d_x,
                               const DeviceBuffer& d_wih,
                               const DeviceBuffer& d_whh,
                               const DeviceBuffer& d_bih,
                               const DeviceBuffer& d_bhh,
                               DeviceBuffer& d_gih, DeviceBuffer& d_ghh,
                               DeviceBuffer& d_h, DeviceBuffer& d_y,
                               void* gate_cache_ptr, std::string& detail) {
        const size_t gate_width = 3 * hidden;
        const size_t rows = batch * seq;
        if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(rows),
                             static_cast<int>(gate_width),
                             static_cast<int>(input), d_x.Float(),
                             d_wih.Float(), d_gih.Float())) {
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
                                 d_whh.Float(), d_ghh.Float())) {
                detail = "recurrent-projection GEMM failed";
                return false;
            }
            int t_arg = static_cast<int>(t);
            void* gih_ptr = d_gih.ptr;
            void* ghh_ptr = d_ghh.ptr;
            void* bih_ptr = d_bih.ptr;
            void* bhh_ptr = d_bhh.ptr;
            void* h_ptr = d_h.ptr;
            void* y_ptr = d_y.ptr;
            void* cache_ptr = gate_cache_ptr;
            void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                            &h_ptr,   &y_ptr,   &cache_ptr, &t_arg,
                            &seq_arg, &hidden_arg, &total_arg};
            if (cuLaunchKernel(state_.GruKernel(), grid, 1, 1, block, 1, 1,
                               0, nullptr, args, nullptr) != CUDA_SUCCESS) {
                detail = "gru cell launch failed";
                return false;
            }
        }
        return true;
    }

    // gru_forward buffer contract (same shape as the LSTM/RNN forward):
    //   inputs[0] = x [batch, seq, input]; weights = {W_ih [3H, input],
    //   W_hh [3H, H], b_ih [3H], b_hh [3H]}; outputs[0] = h_seq
    //   [batch, seq, hidden]. Gate order r, z, n. Initial hidden state is
    //   zero (see the P2/P3 note on hidden-state carry-over in track68).
    NeuralOpStatus ExecuteGruForward(const NeuralOpRequest& request,
                                     NeuralOpBuffers& buffers) {
        NeuralOpStatus status;
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t gate_width = 3 * hidden;
        const size_t rows = batch * seq;
        if (buffers.inputs.size() != 1 || buffers.weights.size() != 4 ||
            buffers.outputs.size() != 1 || !buffers.inputs[0] ||
            !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != rows * input ||
            buffers.weights[0]->NumElements() != gate_width * input ||
            buffers.weights[1]->NumElements() != gate_width * hidden ||
            buffers.weights[2]->NumElements() != gate_width ||
            buffers.weights[3]->NumElements() != gate_width ||
            buffers.outputs[0]->NumElements() != rows * hidden) {
            status.reason =
                BackendFallbackReason::NvidiaProviderUnsupportedContract;
            status.detail = "gru_forward buffer contract violated";
            return status;
        }
        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            status.reason = BackendFallbackReason::NvidiaProviderUnavailable;
            status.detail = failure;
            return status;
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        DeviceBuffer d_x, d_wih, d_whh, d_bih, d_bhh, d_gih, d_ghh, d_h, d_y;
        if (!d_x.Allocate(rows * input * f) ||
            !d_wih.Allocate(gate_width * input * f) ||
            !d_whh.Allocate(gate_width * hidden * f) ||
            !d_bih.Allocate(gate_width * f) ||
            !d_bhh.Allocate(gate_width * f) ||
            !d_gih.Allocate(rows * gate_width * f) ||
            !d_ghh.Allocate(batch * gate_width * f) ||
            !d_h.Allocate(batch * hidden * f) ||
            !d_y.Allocate(rows * hidden * f)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderWorkspaceExhausted;
            status.detail = "device workspace allocation failed";
            return status;
        }
        const bool uploaded =
            cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       rows * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_wih.ptr, buffers.weights[0]->ReadData<float>(),
                       gate_width * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_whh.ptr, buffers.weights[1]->ReadData<float>(),
                       gate_width * hidden * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_bih.ptr, buffers.weights[2]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(d_bhh.ptr, buffers.weights[3]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemset(d_h.ptr, 0, batch * hidden * f) == cudaSuccess;
        if (!uploaded) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "host-to-device transfer failed";
            return status;
        }
        std::string detail;
        if (!RunGruForwardOnDevice(batch, seq, input, hidden, d_x, d_wih,
                                   d_whh, d_bih, d_bhh, d_gih, d_ghh, d_h,
                                   d_y, nullptr, detail)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = detail;
            return status;
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(buffers.outputs[0]->Data<float>(), d_y.ptr,
                       rows * hidden * f,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "device execution or readback failed";
            return status;
        }
        status.ok = true;
        status.reason = BackendFallbackReason::BackendInternalError;
        status.detail.clear();
        return status;
    }

    // gru_backward buffer contract (mirrors lstm_backward): inputs = {x,
    // dy}; weights = {W_ih, W_hh, b_ih, b_hh}; outputs = {dx}; gradients =
    // {dW_ih, dW_hh, db_ih, db_hh}. Self-contained recompute-forward +
    // BPTT; unlike LSTM, db_ih != db_hh (the n-gate bias gradients differ
    // by the reset gate), so both column sums are computed.
    NeuralOpStatus ExecuteGruBackward(const NeuralOpRequest& request,
                                      NeuralOpBuffers& buffers) {
        NeuralOpStatus status;
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t gate_width = 3 * hidden;
        const size_t rows = batch * seq;
        if (buffers.inputs.size() != 2 || buffers.weights.size() != 4 ||
            buffers.outputs.size() != 1 || buffers.gradients.size() != 4 ||
            !buffers.inputs[0] || !buffers.inputs[1] ||
            !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != rows * input ||
            buffers.inputs[1]->NumElements() != rows * hidden ||
            buffers.weights[0]->NumElements() != gate_width * input ||
            buffers.weights[1]->NumElements() != gate_width * hidden ||
            buffers.weights[2]->NumElements() != gate_width ||
            buffers.weights[3]->NumElements() != gate_width ||
            buffers.outputs[0]->NumElements() != rows * input ||
            !buffers.gradients[0] || !buffers.gradients[1] ||
            !buffers.gradients[2] || !buffers.gradients[3] ||
            buffers.gradients[0]->NumElements() != gate_width * input ||
            buffers.gradients[1]->NumElements() != gate_width * hidden ||
            buffers.gradients[2]->NumElements() != gate_width ||
            buffers.gradients[3]->NumElements() != gate_width) {
            status.reason =
                BackendFallbackReason::NvidiaProviderUnsupportedContract;
            status.detail = "gru_backward buffer contract violated";
            return status;
        }
        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            status.reason = BackendFallbackReason::NvidiaProviderUnavailable;
            status.detail = failure;
            return status;
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        DeviceBuffer d_x, d_wih, d_whh, d_bih, d_bhh, d_gih, d_ghh, d_h,
            d_y, d_gates, d_dy, d_dgx, d_dgh, d_dhrec, d_dx, d_dwih,
            d_dwhh, d_dbih, d_dbhh, d_ones;
        if (!d_x.Allocate(rows * input * f) ||
            !d_wih.Allocate(gate_width * input * f) ||
            !d_whh.Allocate(gate_width * hidden * f) ||
            !d_bih.Allocate(gate_width * f) ||
            !d_bhh.Allocate(gate_width * f) ||
            !d_gih.Allocate(rows * gate_width * f) ||
            !d_ghh.Allocate(batch * gate_width * f) ||
            !d_h.Allocate(batch * hidden * f) ||
            !d_y.Allocate(rows * hidden * f) ||
            !d_gates.Allocate(rows * 4 * hidden * f) ||
            !d_dy.Allocate(rows * hidden * f) ||
            !d_dgx.Allocate(rows * gate_width * f) ||
            !d_dgh.Allocate(rows * gate_width * f) ||
            !d_dhrec.Allocate(batch * hidden * f) ||
            !d_dx.Allocate(rows * input * f) ||
            !d_dwih.Allocate(gate_width * input * f) ||
            !d_dwhh.Allocate(gate_width * hidden * f) ||
            !d_dbih.Allocate(gate_width * f) ||
            !d_dbhh.Allocate(gate_width * f) ||
            !d_ones.Allocate(rows * f)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderWorkspaceExhausted;
            status.detail = "device workspace allocation failed";
            return status;
        }
        const bool uploaded =
            cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       rows * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_dy.ptr, buffers.inputs[1]->ReadData<float>(),
                       rows * hidden * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_wih.ptr, buffers.weights[0]->ReadData<float>(),
                       gate_width * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_whh.ptr, buffers.weights[1]->ReadData<float>(),
                       gate_width * hidden * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_bih.ptr, buffers.weights[2]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(d_bhh.ptr, buffers.weights[3]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemset(d_h.ptr, 0, batch * hidden * f) == cudaSuccess &&
            cudaMemset(d_dhrec.ptr, 0, batch * hidden * f) == cudaSuccess &&
            cudaMemset(d_dwih.ptr, 0, gate_width * input * f) ==
                cudaSuccess &&
            cudaMemset(d_dwhh.ptr, 0, gate_width * hidden * f) ==
                cudaSuccess;
        if (!uploaded) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "host-to-device transfer failed";
            return status;
        }

        // Recompute forward with the [r, z, n, hn_pre] cache.
        std::string detail;
        if (!RunGruForwardOnDevice(batch, seq, input, hidden, d_x, d_wih,
                                   d_whh, d_bih, d_bhh, d_gih, d_ghh, d_h,
                                   d_y, d_gates.ptr, detail)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "training forward: " + detail;
            return status;
        }

        const int total = static_cast<int>(batch * hidden);
        const int block = 256;
        const int grid = (total + block - 1) / block;
        int seq_arg = static_cast<int>(seq);
        int hidden_arg = static_cast<int>(hidden);
        int total_arg = total;
        const int ld_gates = static_cast<int>(seq * gate_width);
        for (size_t t = seq; t-- > 0;) {
            int t_arg = static_cast<int>(t);
            void* dy_ptr = d_dy.ptr;
            void* dhrec_ptr = d_dhrec.ptr;
            void* gates_ptr = d_gates.ptr;
            void* y_ptr = d_y.ptr;
            void* dgx_ptr = d_dgx.ptr;
            void* dgh_ptr = d_dgh.ptr;
            void* args[] = {&dy_ptr,  &dhrec_ptr, &gates_ptr, &y_ptr,
                            &dgx_ptr, &dgh_ptr,   &t_arg,     &seq_arg,
                            &hidden_arg, &total_arg};
            if (cuLaunchKernel(state_.GruBackwardKernel(), grid, 1, 1,
                               block, 1, 1, 0, nullptr, args,
                               nullptr) != CUDA_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "gru backward cell launch failed";
                return status;
            }
            const float* dgx_t = d_dgx.Float() + t * gate_width;
            const float* dgh_t = d_dgh.Float() + t * gate_width;
            // dh for t-1 = direct part (written by the cell) + dgh_t @ W_hh.
            if (!GemmRowMajorABAccum(state_.Cublas(), static_cast<int>(batch),
                                     static_cast<int>(hidden),
                                     static_cast<int>(gate_width), dgh_t,
                                     ld_gates, d_whh.Float(),
                                     static_cast<int>(hidden),
                                     d_dhrec.Float(),
                                     static_cast<int>(hidden))) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "recurrent gradient GEMM failed";
                return status;
            }
            // dW_ih += dgx_t^T x_t.
            if (!GemmRowMajorAtBAccum(
                    state_.Cublas(), static_cast<int>(batch),
                    static_cast<int>(gate_width), static_cast<int>(input),
                    dgx_t, ld_gates, d_x.Float() + t * input,
                    static_cast<int>(seq * input), d_dwih.Float(),
                    static_cast<int>(input))) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "input weight-gradient GEMM failed";
                return status;
            }
            // dW_hh += dgh_t^T h_{t-1} (h_0 = 0 contributes nothing).
            if (t > 0 &&
                !GemmRowMajorAtBAccum(
                    state_.Cublas(), static_cast<int>(batch),
                    static_cast<int>(gate_width), static_cast<int>(hidden),
                    dgh_t, ld_gates, d_y.Float() + (t - 1) * hidden,
                    static_cast<int>(seq * hidden), d_dwhh.Float(),
                    static_cast<int>(hidden))) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "hidden weight-gradient GEMM failed";
                return status;
            }
        }

        // dx = DGX * W_ih over the full stacked sequence.
        if (!GemmRowMajorAB(state_.Cublas(), static_cast<int>(rows),
                            static_cast<int>(input),
                            static_cast<int>(gate_width), d_dgx.Float(),
                            static_cast<int>(gate_width), d_wih.Float(),
                            static_cast<int>(input), d_dx.Float(),
                            static_cast<int>(input))) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "input gradient GEMM failed";
            return status;
        }
        // db_ih = colsum(DGX), db_hh = colsum(DGH) via ones-vector GEMVs.
        {
            void* ones_ptr = d_ones.ptr;
            float one_value = 1.0f;
            int count = static_cast<int>(rows);
            const int fill_grid = (count + block - 1) / block;
            void* args[] = {&ones_ptr, &one_value, &count};
            if (cuLaunchKernel(state_.FillKernel(), fill_grid, 1, 1, block,
                               1, 1, 0, nullptr, args,
                               nullptr) != CUDA_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "ones fill launch failed";
                return status;
            }
            const float alpha = 1.0f;
            const float beta = 0.0f;
            if (cublasSgemv(state_.Cublas(), CUBLAS_OP_N,
                            static_cast<int>(gate_width),
                            static_cast<int>(rows), &alpha, d_dgx.Float(),
                            static_cast<int>(gate_width), d_ones.Float(), 1,
                            &beta, d_dbih.Float(),
                            1) != CUBLAS_STATUS_SUCCESS ||
                cublasSgemv(state_.Cublas(), CUBLAS_OP_N,
                            static_cast<int>(gate_width),
                            static_cast<int>(rows), &alpha, d_dgh.Float(),
                            static_cast<int>(gate_width), d_ones.Float(), 1,
                            &beta, d_dbhh.Float(),
                            1) != CUBLAS_STATUS_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "bias gradient GEMV failed";
                return status;
            }
        }

        const bool downloaded =
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(buffers.outputs[0]->Data<float>(), d_dx.ptr,
                       rows * input * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(buffers.gradients[0]->Data<float>(), d_dwih.ptr,
                       gate_width * input * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(buffers.gradients[1]->Data<float>(), d_dwhh.ptr,
                       gate_width * hidden * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(buffers.gradients[2]->Data<float>(), d_dbih.ptr,
                       gate_width * f, cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(buffers.gradients[3]->Data<float>(), d_dbhh.ptr,
                       gate_width * f, cudaMemcpyDeviceToHost) ==
                cudaSuccess;
        if (!downloaded) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "device execution or readback failed";
            return status;
        }
        status.ok = true;
        status.reason = BackendFallbackReason::BackendInternalError;
        status.detail.clear();
        return status;
    }

    NeuralOpStatus ExecuteLstmBackward(const NeuralOpRequest& request,
                                       NeuralOpBuffers& buffers) {
        NeuralOpStatus status;
        const size_t batch = request.batch;
        const size_t seq = request.seq;
        const size_t input = request.input;
        const size_t hidden = request.hidden;
        const size_t gate_width = 4 * hidden;
        const size_t rows = batch * seq;
        if (buffers.inputs.size() != 2 || buffers.weights.size() != 4 ||
            buffers.outputs.size() != 1 || buffers.gradients.size() != 4 ||
            !buffers.inputs[0] || !buffers.inputs[1] ||
            !buffers.outputs[0] ||
            buffers.inputs[0]->NumElements() != rows * input ||
            buffers.inputs[1]->NumElements() != rows * hidden ||
            buffers.weights[0]->NumElements() != gate_width * input ||
            buffers.weights[1]->NumElements() != gate_width * hidden ||
            buffers.weights[2]->NumElements() != gate_width ||
            buffers.weights[3]->NumElements() != gate_width ||
            buffers.outputs[0]->NumElements() != rows * input ||
            !buffers.gradients[0] || !buffers.gradients[1] ||
            !buffers.gradients[2] || !buffers.gradients[3] ||
            buffers.gradients[0]->NumElements() != gate_width * input ||
            buffers.gradients[1]->NumElements() != gate_width * hidden ||
            buffers.gradients[2]->NumElements() != gate_width ||
            buffers.gradients[3]->NumElements() != gate_width) {
            status.reason =
                BackendFallbackReason::NvidiaProviderUnsupportedContract;
            status.detail = "lstm_backward buffer contract violated";
            return status;
        }

        std::string failure;
        if (!state_.Ensure(probe_.cc_major, probe_.cc_minor, failure)) {
            status.reason = BackendFallbackReason::NvidiaProviderUnavailable;
            status.detail = failure;
            return status;
        }
        std::lock_guard<std::mutex> lock(state_.ExecuteMutex());

        const size_t f = sizeof(float);
        DeviceBuffer d_x, d_wih, d_whh, d_bih, d_bhh, d_gih, d_ghh, d_h,
            d_c, d_y, d_gates, d_ccache, d_dy, d_da, d_dhrec, d_dx, d_dwih,
            d_dwhh, d_db, d_ones;
        if (!d_x.Allocate(rows * input * f) ||
            !d_wih.Allocate(gate_width * input * f) ||
            !d_whh.Allocate(gate_width * hidden * f) ||
            !d_bih.Allocate(gate_width * f) ||
            !d_bhh.Allocate(gate_width * f) ||
            !d_gih.Allocate(rows * gate_width * f) ||
            !d_ghh.Allocate(batch * gate_width * f) ||
            !d_h.Allocate(batch * hidden * f) ||
            !d_c.Allocate(batch * hidden * f) ||
            !d_y.Allocate(rows * hidden * f) ||
            !d_gates.Allocate(rows * gate_width * f) ||
            !d_ccache.Allocate(rows * hidden * f) ||
            !d_dy.Allocate(rows * hidden * f) ||
            !d_da.Allocate(rows * gate_width * f) ||
            !d_dhrec.Allocate(batch * hidden * f) ||
            !d_dx.Allocate(rows * input * f) ||
            !d_dwih.Allocate(gate_width * input * f) ||
            !d_dwhh.Allocate(gate_width * hidden * f) ||
            !d_db.Allocate(gate_width * f) ||
            !d_ones.Allocate(rows * f)) {
            status.reason =
                BackendFallbackReason::NvidiaProviderWorkspaceExhausted;
            status.detail = "device workspace allocation failed";
            return status;
        }
        const bool uploaded =
            cudaMemcpy(d_x.ptr, buffers.inputs[0]->ReadData<float>(),
                       rows * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_dy.ptr, buffers.inputs[1]->ReadData<float>(),
                       rows * hidden * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_wih.ptr, buffers.weights[0]->ReadData<float>(),
                       gate_width * input * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_whh.ptr, buffers.weights[1]->ReadData<float>(),
                       gate_width * hidden * f,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_bih.ptr, buffers.weights[2]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(d_bhh.ptr, buffers.weights[3]->ReadData<float>(),
                       gate_width * f, cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemset(d_h.ptr, 0, batch * hidden * f) == cudaSuccess &&
            cudaMemset(d_c.ptr, 0, batch * hidden * f) == cudaSuccess &&
            cudaMemset(d_dhrec.ptr, 0, batch * hidden * f) == cudaSuccess &&
            cudaMemset(d_dwih.ptr, 0, gate_width * input * f) ==
                cudaSuccess &&
            cudaMemset(d_dwhh.ptr, 0, gate_width * hidden * f) ==
                cudaSuccess;
        if (!uploaded) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "host-to-device transfer failed";
            return status;
        }
        // dc is reused in place across the backward loop; starts at zero.
        cudaMemset(d_c.ptr, 0, batch * hidden * f);

        const int total = static_cast<int>(batch * hidden);
        const int block = 256;
        const int grid = (total + block - 1) / block;
        int seq_arg = static_cast<int>(seq);
        int hidden_arg = static_cast<int>(hidden);
        int total_arg = total;

        // Recompute forward with activated-gate and cell caches.
        if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(rows),
                             static_cast<int>(gate_width),
                             static_cast<int>(input), d_x.Float(),
                             d_wih.Float(), d_gih.Float())) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "input-projection GEMM failed";
            return status;
        }
        for (size_t t = 0; t < seq; ++t) {
            if (!GemmRowMajorABt(state_.Cublas(), static_cast<int>(batch),
                                 static_cast<int>(gate_width),
                                 static_cast<int>(hidden), d_h.Float(),
                                 d_whh.Float(), d_ghh.Float())) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "recurrent-projection GEMM failed";
                return status;
            }
            int t_arg = static_cast<int>(t);
            void* gih_ptr = d_gih.ptr;
            void* ghh_ptr = d_ghh.ptr;
            void* bih_ptr = d_bih.ptr;
            void* bhh_ptr = d_bhh.ptr;
            void* c_ptr = d_c.ptr;
            void* h_ptr = d_h.ptr;
            void* y_ptr = d_y.ptr;
            void* gates_ptr = d_gates.ptr;
            void* ccache_ptr = d_ccache.ptr;
            void* args[] = {&gih_ptr, &ghh_ptr, &bih_ptr, &bhh_ptr,
                            &c_ptr,   &h_ptr,   &y_ptr,   &gates_ptr,
                            &ccache_ptr, &t_arg, &seq_arg, &hidden_arg,
                            &total_arg};
            if (cuLaunchKernel(state_.LstmKernel(), grid, 1, 1, block, 1, 1,
                               0, nullptr, args, nullptr) != CUDA_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "training forward cell launch failed";
                return status;
            }
        }

        // BPTT: dc reuses d_c (reset to zero above after the forward
        // consumed it — reset again since forward mutated it).
        cudaMemset(d_c.ptr, 0, batch * hidden * f);
        for (size_t t = seq; t-- > 0;) {
            int t_arg = static_cast<int>(t);
            void* dy_ptr = d_dy.ptr;
            void* dhrec_ptr = d_dhrec.ptr;
            void* dc_ptr = d_c.ptr;
            void* gates_ptr = d_gates.ptr;
            void* ccache_ptr = d_ccache.ptr;
            void* da_ptr = d_da.ptr;
            void* args[] = {&dy_ptr, &dhrec_ptr, &dc_ptr, &gates_ptr,
                            &ccache_ptr, &da_ptr, &t_arg, &seq_arg,
                            &hidden_arg, &total_arg};
            if (cuLaunchKernel(state_.LstmBackwardKernel(), grid, 1, 1,
                               block, 1, 1, 0, nullptr, args,
                               nullptr) != CUDA_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "backward cell launch failed";
                return status;
            }
            const float* da_t = d_da.Float() + t * gate_width;
            const int lda = static_cast<int>(seq * gate_width);
            // dh_recurrent for t-1.
            if (!GemmRowMajorAB(state_.Cublas(), static_cast<int>(batch),
                                static_cast<int>(hidden),
                                static_cast<int>(gate_width), da_t, lda,
                                d_whh.Float(), static_cast<int>(hidden),
                                d_dhrec.Float(),
                                static_cast<int>(hidden))) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "recurrent gradient GEMM failed";
                return status;
            }
            // dW_ih += da_t^T x_t.
            if (!GemmRowMajorAtBAccum(
                    state_.Cublas(), static_cast<int>(batch),
                    static_cast<int>(gate_width), static_cast<int>(input),
                    da_t, lda, d_x.Float() + t * input,
                    static_cast<int>(seq * input), d_dwih.Float(),
                    static_cast<int>(input))) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "input weight-gradient GEMM failed";
                return status;
            }
            // dW_hh += da_t^T h_{t-1} (h_0 = 0 contributes nothing).
            if (t > 0 &&
                !GemmRowMajorAtBAccum(
                    state_.Cublas(), static_cast<int>(batch),
                    static_cast<int>(gate_width), static_cast<int>(hidden),
                    da_t, lda, d_y.Float() + (t - 1) * hidden,
                    static_cast<int>(seq * hidden), d_dwhh.Float(),
                    static_cast<int>(hidden))) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "hidden weight-gradient GEMM failed";
                return status;
            }
        }

        // dx = DA * W_ih over the full stacked sequence.
        if (!GemmRowMajorAB(state_.Cublas(), static_cast<int>(rows),
                            static_cast<int>(input),
                            static_cast<int>(gate_width), d_da.Float(),
                            static_cast<int>(gate_width), d_wih.Float(),
                            static_cast<int>(input), d_dx.Float(),
                            static_cast<int>(input))) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "input gradient GEMM failed";
            return status;
        }
        // db = column sums of DA via ones-vector GEMV.
        {
            void* ones_ptr = d_ones.ptr;
            float one_value = 1.0f;
            int count = static_cast<int>(rows);
            const int fill_grid = (count + block - 1) / block;
            void* args[] = {&ones_ptr, &one_value, &count};
            if (cuLaunchKernel(state_.FillKernel(), fill_grid, 1, 1, block,
                               1, 1, 0, nullptr, args,
                               nullptr) != CUDA_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "ones fill launch failed";
                return status;
            }
            const float alpha = 1.0f;
            const float beta = 0.0f;
            if (cublasSgemv(state_.Cublas(), CUBLAS_OP_N,
                            static_cast<int>(gate_width),
                            static_cast<int>(rows), &alpha, d_da.Float(),
                            static_cast<int>(gate_width), d_ones.Float(), 1,
                            &beta, d_db.Float(),
                            1) != CUBLAS_STATUS_SUCCESS) {
                status.reason =
                    BackendFallbackReason::NvidiaProviderExecutionFailed;
                status.detail = "bias gradient GEMV failed";
                return status;
            }
        }

        const bool downloaded =
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(buffers.outputs[0]->Data<float>(), d_dx.ptr,
                       rows * input * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(buffers.gradients[0]->Data<float>(), d_dwih.ptr,
                       gate_width * input * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(buffers.gradients[1]->Data<float>(), d_dwhh.ptr,
                       gate_width * hidden * f,
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(buffers.gradients[2]->Data<float>(), d_db.ptr,
                       gate_width * f, cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(buffers.gradients[3]->Data<float>(), d_db.ptr,
                       gate_width * f, cudaMemcpyDeviceToHost) ==
                cudaSuccess;
        if (!downloaded) {
            status.reason =
                BackendFallbackReason::NvidiaProviderExecutionFailed;
            status.detail = "device execution or readback failed";
            return status;
        }
        status.ok = true;
        status.reason = BackendFallbackReason::BackendInternalError;
        status.detail.clear();
        return status;
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
