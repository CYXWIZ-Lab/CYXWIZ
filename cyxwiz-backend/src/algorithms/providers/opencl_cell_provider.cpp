// OpenCL neural-network provider (tofix68, device-keyed dispatch tenant #2).
//
// Second tenant of the INeuralNetworkProvider boundary, serving
// DeviceType::OPENCL. Same operation contract as the NVIDIA provider
// (contract 0.6.0: stacked LSTM/GRU forward + backward, single-layer RNN
// forward) so the layers, compiler placement and tests are shared; only
// the substrate differs: hand-written tiled OpenCL GEMM kernels and the
// same fused cell kernels ported from CUDA C to OpenCL C. No BLAS library
// dependency (CLBlast is the documented upgrade if the GEMMs become the
// bottleneck). Honours the tofix68 rules: capability answers per full
// tuple, typed failures, explicit host boundary copies (v1), no ArrayFire
// interop.
//
// Device selection follows the request's target.device_id over the
// enumeration of GPU devices across all OpenCL platforms (platform order,
// device order) — the same order ArrayFire's OpenCL backend exposes for a
// single-vendor machine; multi-vendor ordering is best-effort and recorded
// in Version()/detail strings.
#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER

#include "cyxwiz/neural_provider.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#ifdef CYXWIZ_HAS_OPENCL_CLBLAST
#include <clblast.h>
#endif

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

constexpr const char* kProviderId = "cyxwiz.opencl-cell";
constexpr const char* kProviderSemver = "0.4.0-retention-floor";
// Owner ruling 2026-09-23 (track68): below this hidden size the tenant is
// slower than native CPU (0.82x at hidden=8; 2.4x at hidden=16), so it
// declines and the layer keeps the portable path. Test-overridable only.
constexpr size_t kDefaultRetentionFloorHidden = 16;
std::atomic<size_t> g_retention_floor_hidden{kDefaultRetentionFloorHidden};

// Row-major float kernels. Layout conventions match the CUDA provider and
// the CPU references: x is [batch, seq, features], the row index of the
// pre-computed input projections for (b, t) is b*seq + t, recurrent
// projections and states are [batch, gates*hidden] contiguous. LSTM gate
// order i, f, g, o; GRU gate order r, z, n with hn_pre cached.
constexpr const char* kKernelSource = R"(
#define TILE 16

float cyx_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

__kernel void cyx_fill(__global float* p, const float v, const int n) {
  const int i = get_global_id(0);
  if (i < n) p[i] = v;
}

// C[m][n] = beta*C[m][n] + sum_k A[m*lda+k] * B[n*ldb+k]   (A: MxK, B: NxK)
__kernel void cyx_gemm_abt(const int M, const int N, const int K,
                           __global const float* A, const int a_off, const int lda,
                           __global const float* B, const int b_off, const int ldb,
                           __global float* C, const int c_off, const int ldc,
                           const float beta) {
  __local float As[TILE][TILE];
  __local float Bs[TILE][TILE];
  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int n = get_group_id(0) * TILE + tx;
  const int m = get_group_id(1) * TILE + ty;
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int ka = k0 + tx;
    As[ty][tx] = (m < M && ka < K) ? A[a_off + m * lda + ka] : 0.0f;
    const int kb = k0 + ty;
    Bs[ty][tx] = (n < N && kb < K) ? B[b_off + n * ldb + kb] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < TILE; ++kk) acc += As[ty][kk] * Bs[kk][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (m < M && n < N) {
    const int idx = c_off + m * ldc + n;
    C[idx] = (beta == 0.0f ? 0.0f : beta * C[idx]) + acc;
  }
}

// C[m][n] = beta*C[m][n] + sum_k A[m*lda+k] * B[k*ldb+n]   (A: MxK, B: KxN)
__kernel void cyx_gemm_ab(const int M, const int N, const int K,
                          __global const float* A, const int a_off, const int lda,
                          __global const float* B, const int b_off, const int ldb,
                          __global float* C, const int c_off, const int ldc,
                          const float beta) {
  __local float As[TILE][TILE];
  __local float Bs[TILE][TILE];
  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int n = get_group_id(0) * TILE + tx;
  const int m = get_group_id(1) * TILE + ty;
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int ka = k0 + tx;
    As[ty][tx] = (m < M && ka < K) ? A[a_off + m * lda + ka] : 0.0f;
    const int kb = k0 + ty;
    Bs[ty][tx] = (n < N && kb < K) ? B[b_off + kb * ldb + n] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < TILE; ++kk) acc += As[ty][kk] * Bs[kk][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (m < M && n < N) {
    const int idx = c_off + m * ldc + n;
    C[idx] = (beta == 0.0f ? 0.0f : beta * C[idx]) + acc;
  }
}

// C[k][n] += sum_m A[m*lda+k] * B[m*ldb+n]   (A: MxK, B: MxN, C: KxN)
__kernel void cyx_gemm_atb_accum(const int M, const int K, const int N,
                                 __global const float* A, const int a_off, const int lda,
                                 __global const float* B, const int b_off, const int ldb,
                                 __global float* C, const int c_off, const int ldc) {
  __local float As[TILE][TILE];
  __local float Bs[TILE][TILE];
  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int n = get_group_id(0) * TILE + tx;
  const int k = get_group_id(1) * TILE + ty;
  float acc = 0.0f;
  for (int m0 = 0; m0 < M; m0 += TILE) {
    const int ma = m0 + tx;
    As[ty][tx] = (k < K && ma < M) ? A[a_off + ma * lda + k] : 0.0f;
    const int mb = m0 + ty;
    Bs[ty][tx] = (n < N && mb < M) ? B[b_off + mb * ldb + n] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int mm = 0; mm < TILE; ++mm) acc += As[ty][mm] * Bs[mm][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (k < K && n < N) C[c_off + k * ldc + n] += acc;
}

// out[g] = sum_r A[r*lda+g]
__kernel void cyx_colsum(const int rows, const int cols,
                         __global const float* A, const int lda,
                         __global float* out) {
  const int g = get_global_id(0);
  if (g >= cols) return;
  float s = 0.0f;
  for (int r = 0; r < rows; ++r) s += A[r * lda + g];
  out[g] = s;
}

__kernel void cyx_rnn_cell(__global const float* gates_ih,
                           __global const float* gates_hh,
                           __global const float* bias_ih,
                           __global const float* bias_hh,
                           __global float* hidden_out,
                           __global float* sequence_out,
                           const int t, const int seq, const int hidden,
                           const int total, const int use_tanh) {
  const int idx = get_global_id(0);
  if (idx >= total) return;
  const int b = idx / hidden;
  const int j = idx - b * hidden;
  const float a = gates_ih[(b * seq + t) * hidden + j] + gates_hh[idx] +
                  bias_ih[j] + bias_hh[j];
  const float v = use_tanh ? tanh(a) : (a > 0.0f ? a : 0.0f);
  hidden_out[idx] = v;
  sequence_out[(b * seq + t) * hidden + j] = v;
}

__kernel void cyx_rnn_backward_cell(__global const float* grad_y,
                                    __global const float* dh_recurrent,
                                    __global const float* sequence_out,
                                    __global float* da,
                                    const int t, const int seq,
                                    const int hidden, const int total,
                                    const int use_tanh) {
  const int idx = get_global_id(0);
  if (idx >= total) return;
  const int b = idx / hidden;
  const int j = idx - b * hidden;
  const int pos = (b * seq + t) * hidden + j;
  const float h_t = sequence_out[pos];
  const float dh = grad_y[pos] + dh_recurrent[idx];
  const float deriv = use_tanh ? (1.0f - h_t * h_t) : (h_t > 0.0f ? 1.0f : 0.0f);
  da[pos] = dh * deriv;
}

__kernel void cyx_lstm_cell(__global const float* gates_ih,
                            __global const float* gates_hh,
                            __global const float* bias_ih,
                            __global const float* bias_hh,
                            __global float* cell_state,
                            __global float* hidden_out,
                            __global float* sequence_out,
                            __global float* act_gate_cache,
                            __global float* cell_cache,
                            const int t, const int seq, const int hidden,
                            const int total, const int has_cache) {
  const int idx = get_global_id(0);
  if (idx >= total) return;
  const int b = idx / hidden;
  const int j = idx - b * hidden;
  const int row_ih = (b * seq + t) * 4 * hidden;
  const int row_hh = b * 4 * hidden;
  const float a_i = gates_ih[row_ih + j] + gates_hh[row_hh + j] + bias_ih[j] + bias_hh[j];
  const float a_f = gates_ih[row_ih + hidden + j] + gates_hh[row_hh + hidden + j] +
                    bias_ih[hidden + j] + bias_hh[hidden + j];
  const float a_g = gates_ih[row_ih + 2 * hidden + j] + gates_hh[row_hh + 2 * hidden + j] +
                    bias_ih[2 * hidden + j] + bias_hh[2 * hidden + j];
  const float a_o = gates_ih[row_ih + 3 * hidden + j] + gates_hh[row_hh + 3 * hidden + j] +
                    bias_ih[3 * hidden + j] + bias_hh[3 * hidden + j];
  const float i_gate = cyx_sigmoid(a_i);
  const float f_gate = cyx_sigmoid(a_f);
  const float g_gate = tanh(a_g);
  const float o_gate = cyx_sigmoid(a_o);
  const float c = f_gate * cell_state[idx] + i_gate * g_gate;
  cell_state[idx] = c;
  const float h = o_gate * tanh(c);
  hidden_out[idx] = h;
  sequence_out[(b * seq + t) * hidden + j] = h;
  if (has_cache) {
    act_gate_cache[row_ih + j] = i_gate;
    act_gate_cache[row_ih + hidden + j] = f_gate;
    act_gate_cache[row_ih + 2 * hidden + j] = g_gate;
    act_gate_cache[row_ih + 3 * hidden + j] = o_gate;
    cell_cache[(b * seq + t) * hidden + j] = c;
  }
}

__kernel void cyx_lstm_backward_cell(__global const float* grad_y,
                                     __global const float* dh_recurrent,
                                     __global float* dc,
                                     __global const float* act_gate_cache,
                                     __global const float* cell_cache,
                                     __global float* da,
                                     const int t, const int seq,
                                     const int hidden, const int total) {
  const int idx = get_global_id(0);
  if (idx >= total) return;
  const int b = idx / hidden;
  const int j = idx - b * hidden;
  const int row = (b * seq + t) * 4 * hidden;
  const float i_gate = act_gate_cache[row + j];
  const float f_gate = act_gate_cache[row + hidden + j];
  const float g_gate = act_gate_cache[row + 2 * hidden + j];
  const float o_gate = act_gate_cache[row + 3 * hidden + j];
  const float c_t = cell_cache[(b * seq + t) * hidden + j];
  const float c_prev = t > 0 ? cell_cache[(b * seq + t - 1) * hidden + j] : 0.0f;
  const float tanh_c = tanh(c_t);
  const float dh = grad_y[(b * seq + t) * hidden + j] + dh_recurrent[idx];
  const float d_o = dh * tanh_c;
  const float d_c = dh * o_gate * (1.0f - tanh_c * tanh_c) + dc[idx];
  const float d_i = d_c * g_gate;
  const float d_f = d_c * c_prev;
  const float d_g = d_c * i_gate;
  da[row + j] = d_i * i_gate * (1.0f - i_gate);
  da[row + hidden + j] = d_f * f_gate * (1.0f - f_gate);
  da[row + 2 * hidden + j] = d_g * (1.0f - g_gate * g_gate);
  da[row + 3 * hidden + j] = d_o * o_gate * (1.0f - o_gate);
  dc[idx] = d_c * f_gate;
}

__kernel void cyx_gru_cell(__global const float* gates_ih,
                           __global const float* gates_hh,
                           __global const float* bias_ih,
                           __global const float* bias_hh,
                           __global float* hidden_state,
                           __global float* sequence_out,
                           __global float* gate_cache,
                           const int t, const int seq, const int hidden,
                           const int total, const int has_cache) {
  const int idx = get_global_id(0);
  if (idx >= total) return;
  const int b = idx / hidden;
  const int j = idx - b * hidden;
  const int row_ih = (b * seq + t) * 3 * hidden;
  const int row_hh = b * 3 * hidden;
  const float r = cyx_sigmoid(gates_ih[row_ih + j] + bias_ih[j] +
                              gates_hh[row_hh + j] + bias_hh[j]);
  const float z = cyx_sigmoid(gates_ih[row_ih + hidden + j] + bias_ih[hidden + j] +
                              gates_hh[row_hh + hidden + j] + bias_hh[hidden + j]);
  const float hn_pre = gates_hh[row_hh + 2 * hidden + j] + bias_hh[2 * hidden + j];
  const float n = tanh(gates_ih[row_ih + 2 * hidden + j] + bias_ih[2 * hidden + j] +
                       r * hn_pre);
  const float h_prev = hidden_state[idx];
  const float h = (1.0f - z) * n + z * h_prev;
  hidden_state[idx] = h;
  sequence_out[(b * seq + t) * hidden + j] = h;
  if (has_cache) {
    const int row_c = (b * seq + t) * 4 * hidden;
    gate_cache[row_c + j] = r;
    gate_cache[row_c + hidden + j] = z;
    gate_cache[row_c + 2 * hidden + j] = n;
    gate_cache[row_c + 3 * hidden + j] = hn_pre;
  }
}

__kernel void cyx_gru_backward_cell(__global const float* grad_y,
                                    __global float* dh_recurrent,
                                    __global const float* gate_cache,
                                    __global const float* sequence_out,
                                    __global float* dgates_x,
                                    __global float* dgates_h,
                                    const int t, const int seq,
                                    const int hidden, const int total) {
  const int idx = get_global_id(0);
  if (idx >= total) return;
  const int b = idx / hidden;
  const int j = idx - b * hidden;
  const int row_c = (b * seq + t) * 4 * hidden;
  const int row_g = (b * seq + t) * 3 * hidden;
  const float r = gate_cache[row_c + j];
  const float z = gate_cache[row_c + hidden + j];
  const float n = gate_cache[row_c + 2 * hidden + j];
  const float hn_pre = gate_cache[row_c + 3 * hidden + j];
  const float h_prev = t > 0 ? sequence_out[(b * seq + t - 1) * hidden + j] : 0.0f;
  const float dh = grad_y[(b * seq + t) * hidden + j] + dh_recurrent[idx];
  const float dn = dh * (1.0f - z);
  const float dz = dh * (h_prev - n);
  const float dn_pre = dn * (1.0f - n * n);
  const float dr = dn_pre * hn_pre;
  const float d_hn_pre = dn_pre * r;
  const float d_r_pre = dr * r * (1.0f - r);
  const float d_z_pre = dz * z * (1.0f - z);
  dgates_x[row_g + j] = d_r_pre;
  dgates_x[row_g + hidden + j] = d_z_pre;
  dgates_x[row_g + 2 * hidden + j] = dn_pre;
  dgates_h[row_g + j] = d_r_pre;
  dgates_h[row_g + hidden + j] = d_z_pre;
  dgates_h[row_g + 2 * hidden + j] = d_hn_pre;
  dh_recurrent[idx] = dh * z;
}
)";

struct OpenclDeviceInfo {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string name;
    std::string vendor;
    std::string version;
};

struct OpenclRuntimeProbe {
    bool ok = false;
    std::vector<OpenclDeviceInfo> gpu_devices;  // platform order, device order
    std::string failure;
};

std::string DeviceString(cl_device_id device, cl_device_info what) {
    size_t size = 0;
    if (clGetDeviceInfo(device, what, 0, nullptr, &size) != CL_SUCCESS ||
        size == 0) {
        return {};
    }
    std::string value(size, '\0');
    clGetDeviceInfo(device, what, size, value.data(), nullptr);
    while (!value.empty() && (value.back() == '\0' || value.back() == ' ')) {
        value.pop_back();
    }
    return value;
}

OpenclRuntimeProbe ProbeOpenclRuntime() {
    OpenclRuntimeProbe probe;
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS ||
        platform_count == 0) {
        probe.failure = "no OpenCL platform (ICD) available";
        return probe;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    for (cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                           &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count,
                       devices.data(), nullptr);
        for (cl_device_id device : devices) {
            OpenclDeviceInfo info;
            info.platform = platform;
            info.device = device;
            info.name = DeviceString(device, CL_DEVICE_NAME);
            info.vendor = DeviceString(device, CL_DEVICE_VENDOR);
            info.version = DeviceString(device, CL_DEVICE_VERSION);
            probe.gpu_devices.push_back(info);
        }
    }
    if (probe.gpu_devices.empty()) {
        probe.failure = "no OpenCL GPU device on any platform";
        return probe;
    }
    probe.ok = true;
    return probe;
}

// RAII device buffer.
struct ClBuffer {
    cl_mem mem = nullptr;
    size_t bytes = 0;
    ~ClBuffer() {
        if (mem) clReleaseMemObject(mem);
    }
    bool Allocate(cl_context context, size_t size) {
        if (mem) {
            clReleaseMemObject(mem);
            mem = nullptr;
        }
        bytes = std::max<size_t>(size, sizeof(float));  // OpenCL forbids 0
        cl_int err = CL_SUCCESS;
        mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            mem = nullptr;
            return false;
        }
        return true;
    }
};

// Device-resident (v2) kernels, built in ArrayFire's own context so they can
// read and write ArrayFire's cl_mem buffers on ArrayFire's queue
// (tofix112 phase 5b). Separate from the cell program.
constexpr const char* kDeviceKernelSource = R"CLC(
__kernel void cyx_device_probe(__global const float* x, __global float* y, int n) {
    const int i = get_global_id(0);
    if (i < n) y[i] = 2.0f * x[i];
}
)CLC";

constexpr const char* kAttentionKernelSource = R"CLC(
// Fused attention (tofix112 phase 5b), same algorithm and contract as the
// CUDA tenant: one work-item per query (or key) row, ATTN_BR rows per
// work-group, ATTN_BC-row K/V (or Q/dO) tiles in local memory, online
// softmax; HEAD_DIM is a build-time define.
#define ATTN_BR 64
#define ATTN_BC 32

// Counter-based attention-dropout mask, identical to the CUDA tenant's.
float cyx_keep(ulong seed, ulong index, float p) {
    if (p <= 0.0f) return 1.0f;
    ulong z = seed + index * 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z ^= z >> 31;
    const float u = (float)(z >> 40) * (1.0f / 16777216.0f);
    return u >= p ? 1.0f / (1.0f - p) : 0.0f;
}

float cyx_score(float dot, float scale, float softcap, float slope_term, float* tanh_out) {
    float x = dot * scale;
    float t = 0.0f;
    if (softcap > 0.0f) { t = tanh(x / softcap); x = softcap * t; }
    *tanh_out = t;
    return x + slope_term;
}

__kernel __attribute__((reqd_work_group_size(ATTN_BR, 1, 1)))
void cyx_attention_forward(__global const float* Q, __global const float* K, __global const float* V,
                           __global const float* slopes, int has_slopes, __global float* O, __global float* LSE,
                           int B, int H, int KVH, int Sq, int Sk, int q_offset,
                           int causal, int window, float softcap, float scale, float dropout, ulong seed)
{
    __local float Ks[ATTN_BC][HEAD_DIM];
    __local float Vs[ATTN_BC][HEAD_DIM];
    const int lid = get_local_id(0);
    const int h = get_group_id(1);
    const int b = get_group_id(2);
    const int kvh = h / (H / KVH);
    const int row = get_group_id(0) * ATTN_BR + lid;
    const bool active = row < Sq;
    const int qpos = row + q_offset;
    const size_t q_base = (((size_t)h * B + b) * Sq + row) * HEAD_DIM;
    const size_t kv_base = ((size_t)kvh * B + b) * Sk * HEAD_DIM;
    float q[HEAD_DIM];
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) { q[d] = active ? Q[q_base + d] : 0.0f; acc[d] = 0.0f; }
    const float slope = has_slopes ? slopes[h] : 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    const int first_q = get_group_id(0) * ATTN_BR + q_offset;
    const int last_q = min(Sq, (int)(get_group_id(0) + 1) * ATTN_BR) - 1 + q_offset;
    const int k_end = causal ? min(Sk, last_q + 1) : Sk;
    int k_begin = window > 0 ? max(0, first_q - window + 1) : 0;
    k_begin = (k_begin / ATTN_BC) * ATTN_BC;
    for (int k0 = k_begin; k0 < k_end; k0 += ATTN_BC) {
        for (int idx = lid; idx < ATTN_BC * HEAD_DIM; idx += ATTN_BR) {
            const int j = idx / HEAD_DIM;
            const int d = idx - j * HEAD_DIM;
            const int key = k0 + j;
            const bool in = key < Sk;
            Ks[j][d] = in ? K[kv_base + (size_t)key * HEAD_DIM + d] : 0.0f;
            Vs[j][d] = in ? V[kv_base + (size_t)key * HEAD_DIM + d] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (active) {
            float s[ATTN_BC];
            float tile_max = -INFINITY;
            for (int j = 0; j < ATTN_BC; ++j) {
                const int key = k0 + j;
                const bool valid = key < Sk && (!causal || key <= qpos) && (window <= 0 || qpos - key < window);
                if (!valid) { s[j] = -INFINITY; continue; }
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) dot += q[d] * Ks[j][d];
                float t;
                const float x = cyx_score(dot, scale, softcap, has_slopes ? slope * (float)(key - qpos) : 0.0f, &t);
                s[j] = x;
                tile_max = fmax(tile_max, x);
            }
            if (tile_max > -INFINITY) {
                const float m_new = fmax(m, tile_max);
                const float correction = (m == -INFINITY) ? 0.0f : exp(m - m_new);
                l *= correction;
                for (int d = 0; d < HEAD_DIM; ++d) acc[d] *= correction;
                const ulong drop_row = (((ulong)b * H + h) * Sq + row) * Sk;
                for (int j = 0; j < ATTN_BC; ++j) {
                    if (s[j] == -INFINITY) continue;
                    const float p = exp(s[j] - m_new);
                    l += p;
                    const float pk = p * cyx_keep(seed, drop_row + (ulong)(k0 + j), dropout);
                    for (int d = 0; d < HEAD_DIM; ++d) acc[d] += pk * Vs[j][d];
                }
                m = m_new;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) O[q_base + d] = acc[d] * inv;
        LSE[((size_t)h * B + b) * Sq + row] = l > 0.0f ? m + log(l) : -INFINITY;
    }
}

__kernel void cyx_attention_backward_delta(__global const float* O, __global const float* dO,
                                           __global float* delta, int rows)
{
    const int r = get_global_id(0);
    if (r >= rows) return;
    float sum = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) sum += O[(size_t)r * HEAD_DIM + d] * dO[(size_t)r * HEAD_DIM + d];
    delta[r] = sum;
}

__kernel __attribute__((reqd_work_group_size(ATTN_BR, 1, 1)))
void cyx_attention_backward_dq(__global const float* Q, __global const float* K, __global const float* V,
                               __global const float* dO, __global const float* LSE, __global const float* delta,
                               __global const float* slopes, int has_slopes, __global float* dQ,
                               int B, int H, int KVH, int Sq, int Sk, int q_offset,
                               int causal, int window, float softcap, float scale, float dropout, ulong seed)
{
    __local float Ks[ATTN_BC][HEAD_DIM];
    __local float Vs[ATTN_BC][HEAD_DIM];
    const int lid = get_local_id(0);
    const int h = get_group_id(1);
    const int b = get_group_id(2);
    const int kvh = h / (H / KVH);
    const int row = get_group_id(0) * ATTN_BR + lid;
    const bool active = row < Sq;
    const int qpos = row + q_offset;
    const size_t row_index = ((size_t)h * B + b) * Sq + row;
    const size_t q_base = row_index * HEAD_DIM;
    const size_t kv_base = ((size_t)kvh * B + b) * Sk * HEAD_DIM;
    float q[HEAD_DIM], go[HEAD_DIM], dq[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) {
        q[d] = active ? Q[q_base + d] : 0.0f;
        go[d] = active ? dO[q_base + d] : 0.0f;
        dq[d] = 0.0f;
    }
    const float lse = active ? LSE[row_index] : 0.0f;
    const float dl = active ? delta[row_index] : 0.0f;
    const float slope = has_slopes ? slopes[h] : 0.0f;
    const int first_q = get_group_id(0) * ATTN_BR + q_offset;
    const int last_q = min(Sq, (int)(get_group_id(0) + 1) * ATTN_BR) - 1 + q_offset;
    const int k_end = causal ? min(Sk, last_q + 1) : Sk;
    int k_begin = window > 0 ? max(0, first_q - window + 1) : 0;
    k_begin = (k_begin / ATTN_BC) * ATTN_BC;
    for (int k0 = k_begin; k0 < k_end; k0 += ATTN_BC) {
        for (int idx = lid; idx < ATTN_BC * HEAD_DIM; idx += ATTN_BR) {
            const int j = idx / HEAD_DIM;
            const int d = idx - j * HEAD_DIM;
            const int key = k0 + j;
            const bool in = key < Sk;
            Ks[j][d] = in ? K[kv_base + (size_t)key * HEAD_DIM + d] : 0.0f;
            Vs[j][d] = in ? V[kv_base + (size_t)key * HEAD_DIM + d] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (active && lse > -INFINITY) {
            for (int j = 0; j < ATTN_BC; ++j) {
                const int key = k0 + j;
                const bool valid = key < Sk && (!causal || key <= qpos) && (window <= 0 || qpos - key < window);
                if (!valid) continue;
                float dot = 0.0f, dp = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) { dot += q[d] * Ks[j][d]; dp += go[d] * Vs[j][d]; }
                float t;
                const float x = cyx_score(dot, scale, softcap, has_slopes ? slope * (float)(key - qpos) : 0.0f, &t);
                const float p = exp(x - lse);
                const float keep = cyx_keep(seed, (((ulong)b * H + h) * Sq + row) * Sk + key, dropout);
                float ds = p * (dp * keep - dl);
                if (softcap > 0.0f) ds *= (1.0f - t * t);
                ds *= scale;
                for (int d = 0; d < HEAD_DIM; ++d) dq[d] += ds * Ks[j][d];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        for (int d = 0; d < HEAD_DIM; ++d) dQ[q_base + d] = dq[d];
    }
}

__kernel __attribute__((reqd_work_group_size(ATTN_BR, 1, 1)))
void cyx_attention_backward_dkdv(__global const float* Q, __global const float* K, __global const float* V,
                                 __global const float* dO, __global const float* LSE, __global const float* delta,
                                 __global const float* slopes, int has_slopes, __global float* dK, __global float* dV,
                                 int B, int H, int KVH, int Sq, int Sk, int q_offset,
                                 int causal, int window, float softcap, float scale, float dropout, ulong seed)
{
    __local float Qs[ATTN_BC][HEAD_DIM];
    __local float Gs[ATTN_BC][HEAD_DIM];
    __local float Ls[ATTN_BC];
    __local float Ds[ATTN_BC];
    const int lid = get_local_id(0);
    const int kvh = get_group_id(1);
    const int b = get_group_id(2);
    const int key = get_group_id(0) * ATTN_BR + lid;
    const bool active = key < Sk;
    const size_t k_base = (((size_t)kvh * B + b) * Sk + key) * HEAD_DIM;
    float k[HEAD_DIM], v[HEAD_DIM], dk[HEAD_DIM], dv[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) {
        k[d] = active ? K[k_base + d] : 0.0f;
        v[d] = active ? V[k_base + d] : 0.0f;
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }
    const int groups = H / KVH;
    const int first_key = get_group_id(0) * ATTN_BR;
    const int last_key = min(Sk, (int)(get_group_id(0) + 1) * ATTN_BR) - 1;
    int q_begin = causal ? max(0, first_key - q_offset) : 0;
    q_begin = (q_begin / ATTN_BC) * ATTN_BC;
    const int q_end = window > 0 ? min(Sq, last_key + window - q_offset) : Sq;
    for (int g = 0; g < groups; ++g) {
        const int h = kvh * groups + g;
        const float slope = has_slopes ? slopes[h] : 0.0f;
        const size_t head_rows = ((size_t)h * B + b) * Sq;
        for (int i0 = q_begin; i0 < q_end; i0 += ATTN_BC) {
            for (int idx = lid; idx < ATTN_BC * HEAD_DIM; idx += ATTN_BR) {
                const int i = idx / HEAD_DIM;
                const int d = idx - i * HEAD_DIM;
                const int row = i0 + i;
                const bool in = row < Sq;
                Qs[i][d] = in ? Q[(head_rows + row) * HEAD_DIM + d] : 0.0f;
                Gs[i][d] = in ? dO[(head_rows + row) * HEAD_DIM + d] : 0.0f;
            }
            for (int i = lid; i < ATTN_BC; i += ATTN_BR) {
                const int row = i0 + i;
                Ls[i] = row < Sq ? LSE[head_rows + row] : -INFINITY;
                Ds[i] = row < Sq ? delta[head_rows + row] : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (active) {
                for (int i = 0; i < ATTN_BC; ++i) {
                    const int row = i0 + i;
                    const int qpos = row + q_offset;
                    const bool valid = row < Sq && Ls[i] > -INFINITY && (!causal || key <= qpos) &&
                                       (window <= 0 || qpos - key < window);
                    if (!valid) continue;
                    float dot = 0.0f, dp = 0.0f;
                    for (int d = 0; d < HEAD_DIM; ++d) { dot += Qs[i][d] * k[d]; dp += Gs[i][d] * v[d]; }
                    float t;
                    const float x = cyx_score(dot, scale, softcap, has_slopes ? slope * (float)(key - qpos) : 0.0f, &t);
                    const float p = exp(x - Ls[i]);
                    const float keep = cyx_keep(seed, (((ulong)b * H + h) * Sq + row) * Sk + key, dropout);
                    float ds = p * (dp * keep - Ds[i]);
                    if (softcap > 0.0f) ds *= (1.0f - t * t);
                    ds *= scale;
                    const float pk = p * keep;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dv[d] += pk * Gs[i][d];
                        dk[d] += ds * Qs[i][d];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    if (active) {
        for (int d = 0; d < HEAD_DIM; ++d) { dK[k_base + d] = dk[d]; dV[k_base + d] = dv[d]; }
    }
}
)CLC";

class DeviceResidentProgram {
public:
    ~DeviceResidentProgram() {
        ReleaseAttention();
        if (probe_) clReleaseKernel(probe_);
        if (program_) clReleaseProgram(program_);
    }

    bool Ensure(cl_context context, cl_device_id device, std::string& failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_ && context == context_ && device == device_) return true;
        if (probe_) clReleaseKernel(probe_);
        if (program_) clReleaseProgram(program_);
        probe_ = nullptr;
        program_ = nullptr;
        ready_ = false;
        cl_int err = CL_SUCCESS;
        const char* source = kDeviceKernelSource;
        const size_t length = std::strlen(kDeviceKernelSource);
        program_ = clCreateProgramWithSource(context, 1, &source, &length, &err);
        if (err != CL_SUCCESS) {
            failure = "OpenCL device program creation failed (" + std::to_string(err) + ")";
            return false;
        }
        err = clBuildProgram(program_, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            failure = "OpenCL device program build failed: " + log;
            return false;
        }
        probe_ = clCreateKernel(program_, "cyx_device_probe", &err);
        if (err != CL_SUCCESS) {
            failure = "OpenCL device kernel lookup failed: cyx_device_probe";
            return false;
        }
        context_ = context;
        device_ = device;
        ready_ = true;
        return true;
    }

    cl_kernel Probe() const { return probe_; }
    std::mutex& Mutex() { return mutex_; }

    struct AttentionKernels {
        cl_program program = nullptr;
        cl_kernel forward = nullptr;
        cl_kernel delta = nullptr;
        cl_kernel dq = nullptr;
        cl_kernel dkdv = nullptr;
    };
    // Attention kernels for ArrayFire's context/device and a head width;
    // built on first use. Caller holds Mutex().
    const AttentionKernels* Attention(cl_context context, cl_device_id device, int head_dim, std::string& failure) {
        if (context != attention_context_ || device != attention_device_) {
            ReleaseAttention();
            attention_context_ = context;
            attention_device_ = device;
        }
        auto found = attention_.find(head_dim);
        if (found != attention_.end()) return &found->second;
        cl_int err = CL_SUCCESS;
        const char* source = kAttentionKernelSource;
        const size_t length = std::strlen(kAttentionKernelSource);
        AttentionKernels kernels;
        kernels.program = clCreateProgramWithSource(context, 1, &source, &length, &err);
        if (err != CL_SUCCESS) {
            failure = "OpenCL attention program creation failed (" + std::to_string(err) + ")";
            return nullptr;
        }
        const std::string options = "-cl-std=CL1.2 -DHEAD_DIM=" + std::to_string(head_dim);
        err = clBuildProgram(kernels.program, 1, &device, options.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(kernels.program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(kernels.program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            clReleaseProgram(kernels.program);
            failure = "OpenCL attention program build failed: " + log;
            return nullptr;
        }
        kernels.forward = clCreateKernel(kernels.program, "cyx_attention_forward", &err);
        if (err == CL_SUCCESS) kernels.delta = clCreateKernel(kernels.program, "cyx_attention_backward_delta", &err);
        if (err == CL_SUCCESS) kernels.dq = clCreateKernel(kernels.program, "cyx_attention_backward_dq", &err);
        if (err == CL_SUCCESS) kernels.dkdv = clCreateKernel(kernels.program, "cyx_attention_backward_dkdv", &err);
        if (err != CL_SUCCESS) {
            for (cl_kernel k : {kernels.forward, kernels.delta, kernels.dq, kernels.dkdv}) if (k) clReleaseKernel(k);
            clReleaseProgram(kernels.program);
            failure = "OpenCL attention kernel lookup failed";
            return nullptr;
        }
        return &(attention_[head_dim] = kernels);
    }

private:
    void ReleaseAttention() {
        for (auto& [dim, k] : attention_) {
            for (cl_kernel kernel : {k.forward, k.delta, k.dq, k.dkdv}) if (kernel) clReleaseKernel(kernel);
            if (k.program) clReleaseProgram(k.program);
        }
        attention_.clear();
    }
    std::map<int, AttentionKernels> attention_;
    cl_context attention_context_ = nullptr;
    cl_device_id attention_device_ = nullptr;

    std::mutex mutex_;
    bool ready_ = false;
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel probe_ = nullptr;
};

// Per-device compiled state: context, in-order queue, program, kernels.
class OpenclExecutionState {
public:
    ~OpenclExecutionState() {
        for (cl_kernel k : {fill_, gemm_abt_, gemm_ab_, gemm_atb_accum_, colsum_,
                            rnn_cell_, rnn_backward_cell_, lstm_cell_,
                            lstm_backward_cell_, gru_cell_, gru_backward_cell_}) {
            if (k) clReleaseKernel(k);
        }
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    bool Ensure(const OpenclDeviceInfo& device, std::string& failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_) return true;
        cl_int err = CL_SUCCESS;
        const cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(device.platform), 0};
        context_ = clCreateContext(props, 1, &device.device, nullptr, nullptr,
                                   &err);
        if (err != CL_SUCCESS) {
            failure = "OpenCL context creation failed (" + std::to_string(err) + ")";
            return false;
        }
        queue_ = clCreateCommandQueue(context_, device.device, 0, &err);
        if (err != CL_SUCCESS) {
            failure = "OpenCL queue creation failed (" + std::to_string(err) + ")";
            return false;
        }
        const char* source = kKernelSource;
        const size_t length = std::strlen(kKernelSource);
        program_ = clCreateProgramWithSource(context_, 1, &source, &length, &err);
        if (err != CL_SUCCESS) {
            failure = "OpenCL program creation failed (" + std::to_string(err) + ")";
            return false;
        }
        err = clBuildProgram(program_, 1, &device.device,
                             "-cl-std=CL1.2 -cl-mad-enable", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device.device, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program_, device.device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);
            failure = "OpenCL program build failed: " + log;
            return false;
        }
        struct Slot {
            cl_kernel* target;
            const char* name;
        };
        const Slot slots[] = {
            {&fill_, "cyx_fill"},
            {&gemm_abt_, "cyx_gemm_abt"},
            {&gemm_ab_, "cyx_gemm_ab"},
            {&gemm_atb_accum_, "cyx_gemm_atb_accum"},
            {&colsum_, "cyx_colsum"},
            {&rnn_cell_, "cyx_rnn_cell"},
            {&rnn_backward_cell_, "cyx_rnn_backward_cell"},
            {&lstm_cell_, "cyx_lstm_cell"},
            {&lstm_backward_cell_, "cyx_lstm_backward_cell"},
            {&gru_cell_, "cyx_gru_cell"},
            {&gru_backward_cell_, "cyx_gru_backward_cell"},
        };
        for (const auto& slot : slots) {
            *slot.target = clCreateKernel(program_, slot.name, &err);
            if (err != CL_SUCCESS) {
                failure = std::string("OpenCL kernel lookup failed: ") + slot.name;
                return false;
            }
        }
        ready_ = true;
        return true;
    }

    cl_context Context() const { return context_; }
    cl_command_queue Queue() const { return queue_; }
    std::mutex& ExecuteMutex() { return mutex_; }
    cl_kernel Fill() const { return fill_; }
    cl_kernel GemmAbt() const { return gemm_abt_; }
    cl_kernel GemmAb() const { return gemm_ab_; }
    cl_kernel GemmAtbAccum() const { return gemm_atb_accum_; }
    cl_kernel Colsum() const { return colsum_; }
    cl_kernel RnnCell() const { return rnn_cell_; }
    cl_kernel RnnBackwardCell() const { return rnn_backward_cell_; }
    cl_kernel LstmCell() const { return lstm_cell_; }
    cl_kernel LstmBackwardCell() const { return lstm_backward_cell_; }
    cl_kernel GruCell() const { return gru_cell_; }
    cl_kernel GruBackwardCell() const { return gru_backward_cell_; }

private:
    std::mutex mutex_;
    bool ready_ = false;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel fill_ = nullptr;
    cl_kernel gemm_abt_ = nullptr;
    cl_kernel gemm_ab_ = nullptr;
    cl_kernel gemm_atb_accum_ = nullptr;
    cl_kernel colsum_ = nullptr;
    cl_kernel rnn_cell_ = nullptr;
    cl_kernel rnn_backward_cell_ = nullptr;
    cl_kernel lstm_cell_ = nullptr;
    cl_kernel lstm_backward_cell_ = nullptr;
    cl_kernel gru_cell_ = nullptr;
    cl_kernel gru_backward_cell_ = nullptr;
};

// ---- kernel-argument helpers -------------------------------------------
inline bool SetArg(cl_kernel k, cl_uint i, const ClBuffer& b) {
    return clSetKernelArg(k, i, sizeof(cl_mem), &b.mem) == CL_SUCCESS;
}
inline bool SetArg(cl_kernel k, cl_uint i, int v) {
    return clSetKernelArg(k, i, sizeof(int), &v) == CL_SUCCESS;
}
inline bool SetArg(cl_kernel k, cl_uint i, float v) {
    return clSetKernelArg(k, i, sizeof(float), &v) == CL_SUCCESS;
}
template <typename... Args>
bool SetArgs(cl_kernel k, Args&&... args) {
    cl_uint index = 0;
    bool ok = true;
    ((ok = ok && SetArg(k, index++, args)), ...);
    return ok;
}

constexpr size_t kTile = 16;
constexpr size_t kBlock = 256;

inline size_t RoundUp(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

inline bool Launch1D(cl_command_queue queue, cl_kernel kernel, size_t total) {
    const size_t global = RoundUp(std::max<size_t>(total, 1), kBlock);
    const size_t local = kBlock;
    return clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0,
                                  nullptr, nullptr) == CL_SUCCESS;
}

inline bool Launch2D(cl_command_queue queue, cl_kernel kernel, size_t rows,
                     size_t cols) {
    const size_t global[2] = {RoundUp(std::max<size_t>(cols, 1), kTile),
                              RoundUp(std::max<size_t>(rows, 1), kTile)};
    const size_t local[2] = {kTile, kTile};
    return clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0,
                                  nullptr, nullptr) == CL_SUCCESS;
}

inline bool Upload(cl_command_queue queue, const ClBuffer& dst, const float* src,
                   size_t bytes) {
    return clEnqueueWriteBuffer(queue, dst.mem, CL_TRUE, 0, bytes, src, 0,
                                nullptr, nullptr) == CL_SUCCESS;
}

inline bool Download(cl_command_queue queue, float* dst, const ClBuffer& src,
                     size_t bytes) {
    return clEnqueueReadBuffer(queue, src.mem, CL_TRUE, 0, bytes, dst, 0,
                               nullptr, nullptr) == CL_SUCCESS;
}

inline bool Zero(cl_command_queue queue, cl_kernel fill, const ClBuffer& b,
                 size_t count) {
    return SetArgs(fill, b, 0.0f, static_cast<int>(count)) &&
           Launch1D(queue, fill, count);
}

inline bool CopyDevice(cl_command_queue queue, const ClBuffer& dst,
                       size_t dst_offset_elems, const ClBuffer& src,
                       size_t count) {
    return clEnqueueCopyBuffer(queue, src.mem, dst.mem, 0,
                               dst_offset_elems * sizeof(float),
                               count * sizeof(float), 0, nullptr,
                               nullptr) == CL_SUCCESS;
}

class OpenclCellProvider final : public INeuralNetworkProvider {
public:
    explicit OpenclCellProvider(OpenclRuntimeProbe probe)
        : probe_(std::move(probe)),
          states_(probe_.gpu_devices.size()) {
        for (auto& state : states_) {
            state = std::make_unique<OpenclExecutionState>();
        }
    }

    const char* ProviderId() const override { return kProviderId; }
    DeviceType Platform() const override { return DeviceType::OPENCL; }

    std::string Version() const override {
        std::ostringstream out;
        out << kProviderId << " " << kProviderSemver << " / OpenCL GPU devices "
            << probe_.gpu_devices.size()
#ifdef CYXWIZ_HAS_OPENCL_CLBLAST
            << " / GEMM CLBlast"
#else
            << " / GEMM built-in tiled kernels"
#endif
            ;
        for (size_t i = 0; i < probe_.gpu_devices.size(); ++i) {
            const auto& d = probe_.gpu_devices[i];
            out << " / device" << i << " '" << d.name << "' (" << d.vendor
                << ", " << d.version << ")";
        }
        return out.str();
    }

    NeuralCapability QueryCapability(
        const NeuralOpRequest& request) const override {
        NeuralCapability capability;
        capability.reason =
            BackendFallbackReason::OpenclProviderUnsupportedContract;
        if (request.target.platform != DeviceType::OPENCL) {
            capability.detail =
                std::string("request targets ") +
                NeuralDevicePlatformName(request.target.platform) +
                "; this provider serves opencl only";
            return capability;
        }
        // v2 device-resident ops run in ArrayFire's context on its device
        // (tofix112 phase 5b), so the enumerated-device check does not apply.
        if ((request.op == NeuralOp::AttentionForward || request.op == NeuralOp::AttentionBackward) &&
            request.device_resident) {
            capability.detail = AttentionContractError(request);
            if (!capability.detail.empty()) return capability;
            capability.supported = true;
            capability.reason = BackendFallbackReason::BackendInternalError;
            return capability;
        }
        if (request.op == NeuralOp::DeviceProbe || request.device_resident) {
            if (request.op != NeuralOp::DeviceProbe || !request.device_resident) {
                capability.detail = "device-resident execution covers device_probe and attention";
                return capability;
            }
            if (request.dtype != DataType::Float32 || request.elements == 0 ||
                request.elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
                capability.detail = "device_probe needs Float32 and 1..INT_MAX elements";
                return capability;
            }
            capability.supported = true;
            capability.reason = BackendFallbackReason::BackendInternalError;
            capability.detail.clear();
            return capability;
        }
        if (request.target.device_id < 0 ||
            static_cast<size_t>(request.target.device_id) >=
                probe_.gpu_devices.size()) {
            capability.reason = BackendFallbackReason::OpenclProviderUnavailable;
            capability.detail =
                "target OpenCL device index " +
                std::to_string(request.target.device_id) +
                " is not an enumerated GPU device (" +
                std::to_string(probe_.gpu_devices.size()) + " available)";
            return capability;
        }
        const bool is_rnn = request.op == NeuralOp::RnnForward ||
                            request.op == NeuralOp::RnnBackward;
        const bool is_lstm = request.op == NeuralOp::LstmForward ||
                             request.op == NeuralOp::LstmBackward;
        const bool is_gru = request.op == NeuralOp::GruForward ||
                            request.op == NeuralOp::GruBackward;
        if (!is_rnn && !is_lstm && !is_gru) {
            capability.detail =
                std::string(NeuralOpName(request.op)) +
                " is not implemented yet (supported: rnn_forward, "
                "rnn_backward, lstm_forward, lstm_backward, gru_forward, "
                "gru_backward)";
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
        if (is_rnn && request.activation != NeuralActivation::Tanh &&
            request.activation != NeuralActivation::Relu) {
            capability.detail = "rnn_forward supports tanh or relu";
            return capability;
        }
        if ((is_lstm || is_gru) &&
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
        const size_t retention_floor = g_retention_floor_hidden.load();
        if (request.hidden < retention_floor) {
            capability.reason =
                BackendFallbackReason::OpenclProviderBelowRetentionFloor;
            capability.detail =
                "hidden=" + std::to_string(request.hidden) +
                " is below the OpenCL tenant retention floor (hidden >= " +
                std::to_string(retention_floor) +
                "): the tenant measured 0.82x of native CPU at hidden=8 and "
                "2.4x at hidden=16 (track68, 2026-09-23), so tiny layers stay "
                "on the portable path";
            return capability;
        }
        capability.supported = true;
        capability.reason = BackendFallbackReason::BackendInternalError;
        capability.detail.clear();
        return capability;
    }

    NeuralResourceEstimate EstimateResources(
        const NeuralOpRequest& request) const override {
        // Same buffer plan as the CUDA provider (contract 0.6.0).
        NeuralResourceEstimate estimate;
        const size_t f = sizeof(float);
        const bool is_lstm = request.op == NeuralOp::LstmForward ||
                             request.op == NeuralOp::LstmBackward;
        const bool is_gru = request.op == NeuralOp::GruForward ||
                            request.op == NeuralOp::GruBackward;
        const bool backward = request.op == NeuralOp::LstmBackward ||
                              request.op == NeuralOp::GruBackward ||
                              request.op == NeuralOp::RnnBackward;
        const size_t gates = is_lstm ? 4 : (is_gru ? 3 : 1);
        const size_t layers = std::max<size_t>(1, request.layers);
        const size_t rows = request.batch * request.seq;
        const size_t gate_width = gates * request.hidden;
        size_t weight_elems = 0;
        size_t input_elems = 0;
        for (size_t l = 0; l < layers; ++l) {
            const size_t in = l == 0 ? request.input : request.hidden;
            weight_elems += gate_width * in + gate_width * request.hidden +
                            2 * gate_width;
            input_elems += rows * in;
        }
        estimate.workspace_bytes =
            rows * request.input * f + weight_elems * f +
            rows * gate_width * f + request.batch * gate_width * f +
            2 * request.batch * request.hidden * f +
            layers * rows * request.hidden * f;
        if (backward) {
            estimate.workspace_bytes +=
                rows * request.hidden * f +
                ((is_lstm || is_gru) ? layers * rows * 4 * request.hidden * f : 0) +
                (is_lstm ? layers * rows * request.hidden * f : 0) +
                (is_gru ? 2 : 1) * rows * gate_width * f +
                request.batch * request.hidden * f + input_elems * f +
                weight_elems * f + rows * f;
        } else {
            estimate.workspace_bytes +=
                (is_lstm ? 2 : 1) * layers * request.batch * request.hidden * f;
        }
        return estimate;
    }

    NeuralOpStatus Execute(const NeuralOpRequest& request,
                           NeuralOpBuffers& buffers) override {
        if (request.device_resident) {
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        "device-resident requests run through ExecuteDevice");
        }
        const NeuralCapability capability = QueryCapability(request);
        if (!capability.supported) {
            return Fail(capability.reason, capability.detail);
        }
        switch (request.op) {
        case NeuralOp::RnnForward:
            return ExecuteStackedForward(request, buffers, CellKind::Rnn);
        case NeuralOp::RnnBackward:
            return ExecuteStackedBackward(request, buffers, CellKind::Rnn);
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
        return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                    "op is not executable by this provider");
    }

private:
    enum class CellKind { Rnn, Lstm, Gru };
    static size_t GatesFor(CellKind kind) {
        return kind == CellKind::Lstm ? 4 : (kind == CellKind::Gru ? 3 : 1);
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
    static NeuralOpStatus ExecFail(const std::string& detail) {
        return Fail(BackendFallbackReason::OpenclProviderExecutionFailed, detail);
    }
    static NeuralOpStatus AllocFail() {
        return Fail(BackendFallbackReason::OpenclProviderWorkspaceExhausted,
                    "device workspace allocation failed");
    }

    struct LayerDeviceWeights {
        ClBuffer w_ih, w_hh, b_ih, b_hh;
        size_t in_features = 0;
    };

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

    bool UploadStackedWeights(OpenclExecutionState& state,
                              const std::vector<const Tensor*>& weights,
                              size_t layers, size_t gate_width, size_t input,
                              size_t hidden,
                              std::vector<LayerDeviceWeights>& out,
                              NeuralOpStatus& failure) {
        const size_t f = sizeof(float);
        out.clear();
        out.resize(layers);
        for (size_t l = 0; l < layers; ++l) {
            auto& w = out[l];
            w.in_features = l == 0 ? input : hidden;
            if (!w.w_ih.Allocate(state.Context(), gate_width * w.in_features * f) ||
                !w.w_hh.Allocate(state.Context(), gate_width * hidden * f) ||
                !w.b_ih.Allocate(state.Context(), gate_width * f) ||
                !w.b_hh.Allocate(state.Context(), gate_width * f)) {
                failure = AllocFail();
                return false;
            }
            const bool uploaded =
                Upload(state.Queue(), w.w_ih, weights[4 * l]->ReadData<float>(),
                       gate_width * w.in_features * f) &&
                Upload(state.Queue(), w.w_hh, weights[4 * l + 1]->ReadData<float>(),
                       gate_width * hidden * f) &&
                Upload(state.Queue(), w.b_ih, weights[4 * l + 2]->ReadData<float>(),
                       gate_width * f) &&
                Upload(state.Queue(), w.b_hh, weights[4 * l + 3]->ReadData<float>(),
                       gate_width * f);
            if (!uploaded) {
                failure = ExecFail("host-to-device transfer failed (weights)");
                return false;
            }
        }
        return true;
    }

    // GEMM wrappers mirroring the CUDA provider's row-major helpers, with
    // element offsets in place of pointer arithmetic.
#ifdef CYXWIZ_HAS_OPENCL_CLBLAST
    // CLBlast row-major SGEMM with element offsets. Returns false on any
    // CLBlast status other than success so the caller can fall back.
    // Measured on the GTX 1050 Ti OpenCL platform (2026-09-23): CLBlast's
    // per-call overhead loses to the built-in tiled kernel on the small
    // per-timestep recurrent GEMMs, so it is used only above a work
    // threshold (m*n*k) where its tuned kernels win.
    static constexpr size_t kClblastMinWork = size_t{1} << 24;  // 16.8M MACs
    static bool ClblastGemm(OpenclExecutionState& s, bool transpose_a,
                            bool transpose_b, size_t m, size_t n, size_t k,
                            const ClBuffer& A, size_t a_off, size_t lda,
                            const ClBuffer& B, size_t b_off, size_t ldb,
                            ClBuffer& C, size_t c_off, size_t ldc, float beta) {
        if (m * n * k < kClblastMinWork) {
            return false;  // built-in kernel path
        }
        cl_command_queue queue = s.Queue();
        const auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo,
            transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo,
            m, n, k, 1.0f, A.mem, a_off, lda, B.mem, b_off, ldb, beta, C.mem,
            c_off, ldc, &queue, nullptr);
        return status == clblast::StatusCode::kSuccess;
    }
#endif

    bool GemmABt(OpenclExecutionState& s, size_t m, size_t n, size_t k,
                 const ClBuffer& A, size_t a_off, size_t lda, const ClBuffer& B,
                 size_t b_off, size_t ldb, ClBuffer& C, size_t c_off, size_t ldc,
                 float beta) {
#ifdef CYXWIZ_HAS_OPENCL_CLBLAST
        if (ClblastGemm(s, false, true, m, n, k, A, a_off, lda, B, b_off, ldb,
                        C, c_off, ldc, beta)) {
            return true;
        }
#endif
        cl_kernel kernel = s.GemmAbt();
        return SetArgs(kernel, static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), A, static_cast<int>(a_off),
                       static_cast<int>(lda), B, static_cast<int>(b_off),
                       static_cast<int>(ldb), C, static_cast<int>(c_off),
                       static_cast<int>(ldc), beta) &&
               Launch2D(s.Queue(), kernel, m, n);
    }
    bool GemmAB(OpenclExecutionState& s, size_t m, size_t n, size_t k,
                const ClBuffer& A, size_t a_off, size_t lda, const ClBuffer& B,
                size_t b_off, size_t ldb, ClBuffer& C, size_t c_off, size_t ldc,
                float beta) {
#ifdef CYXWIZ_HAS_OPENCL_CLBLAST
        if (ClblastGemm(s, false, false, m, n, k, A, a_off, lda, B, b_off, ldb,
                        C, c_off, ldc, beta)) {
            return true;
        }
#endif
        cl_kernel kernel = s.GemmAb();
        return SetArgs(kernel, static_cast<int>(m), static_cast<int>(n),
                       static_cast<int>(k), A, static_cast<int>(a_off),
                       static_cast<int>(lda), B, static_cast<int>(b_off),
                       static_cast<int>(ldb), C, static_cast<int>(c_off),
                       static_cast<int>(ldc), beta) &&
               Launch2D(s.Queue(), kernel, m, n);
    }
    // C[k x n] += A[m x k]^T * B[m x n]
    bool GemmAtBAccum(OpenclExecutionState& s, size_t m, size_t k, size_t n,
                      const ClBuffer& A, size_t a_off, size_t lda,
                      const ClBuffer& B, size_t b_off, size_t ldb, ClBuffer& C,
                      size_t c_off, size_t ldc) {
#ifdef CYXWIZ_HAS_OPENCL_CLBLAST
        // C[k x n] += A[m x k]^T * B[m x n]
        if (ClblastGemm(s, true, false, k, n, m, A, a_off, lda, B, b_off, ldb,
                        C, c_off, ldc, 1.0f)) {
            return true;
        }
#endif
        cl_kernel kernel = s.GemmAtbAccum();
        return SetArgs(kernel, static_cast<int>(m), static_cast<int>(k),
                       static_cast<int>(n), A, static_cast<int>(a_off),
                       static_cast<int>(lda), B, static_cast<int>(b_off),
                       static_cast<int>(ldb), C, static_cast<int>(c_off),
                       static_cast<int>(ldc)) &&
               Launch2D(s.Queue(), kernel, k, n);
    }
    bool Colsum(OpenclExecutionState& s, size_t rows, size_t cols,
                const ClBuffer& A, size_t lda, ClBuffer& out) {
        cl_kernel kernel = s.Colsum();
        return SetArgs(kernel, static_cast<int>(rows), static_cast<int>(cols), A,
                       static_cast<int>(lda), out) &&
               Launch1D(s.Queue(), kernel, cols);
    }

    OpenclExecutionState* EnsureState(const NeuralOpRequest& request,
                                      NeuralOpStatus& failure) {
        const size_t index = static_cast<size_t>(request.target.device_id);
        std::string message;
        if (!states_[index]->Ensure(probe_.gpu_devices[index], message)) {
            failure = Fail(BackendFallbackReason::OpenclProviderUnavailable, message);
            return nullptr;
        }
        return states_[index].get();
    }

    // One layer's forward from a ZERO initial state. Caches are written
    // when has_cache is set (training recompute); dummy buffers otherwise.
    bool RunStackedLayerForward(OpenclExecutionState& s, CellKind kind,
                                size_t batch, size_t seq, size_t hidden,
                                const ClBuffer& d_input,
                                const LayerDeviceWeights& w, ClBuffer& d_gih,
                                ClBuffer& d_ghh, ClBuffer& d_h, ClBuffer& d_c,
                                ClBuffer& d_y, ClBuffer& gate_cache,
                                ClBuffer& cell_cache, bool has_cache,
                                int use_tanh, std::string& detail) {
        const size_t gate_width = GatesFor(kind) * hidden;
        const size_t rows = batch * seq;
        if (!Zero(s.Queue(), s.Fill(), d_h, batch * hidden) ||
            (kind == CellKind::Lstm &&
             !Zero(s.Queue(), s.Fill(), d_c, batch * hidden))) {
            detail = "state reset failed";
            return false;
        }
        if (!GemmABt(s, rows, gate_width, w.in_features, d_input, 0,
                     w.in_features, w.w_ih, 0, w.in_features, d_gih, 0,
                     gate_width, 0.0f)) {
            detail = "input-projection GEMM failed";
            return false;
        }
        const int total = static_cast<int>(batch * hidden);
        for (size_t t = 0; t < seq; ++t) {
            if (!GemmABt(s, batch, gate_width, hidden, d_h, 0, hidden, w.w_hh, 0,
                         hidden, d_ghh, 0, gate_width, 0.0f)) {
                detail = "recurrent-projection GEMM failed";
                return false;
            }
            bool launched;
            if (kind == CellKind::Lstm) {
                launched = SetArgs(s.LstmCell(), d_gih, d_ghh, w.b_ih, w.b_hh, d_c,
                                   d_h, d_y, gate_cache, cell_cache,
                                   static_cast<int>(t), static_cast<int>(seq),
                                   static_cast<int>(hidden), total,
                                   has_cache ? 1 : 0) &&
                           Launch1D(s.Queue(), s.LstmCell(), batch * hidden);
            } else if (kind == CellKind::Gru) {
                launched = SetArgs(s.GruCell(), d_gih, d_ghh, w.b_ih, w.b_hh, d_h,
                                   d_y, gate_cache, static_cast<int>(t),
                                   static_cast<int>(seq),
                                   static_cast<int>(hidden), total,
                                   has_cache ? 1 : 0) &&
                           Launch1D(s.Queue(), s.GruCell(), batch * hidden);
            } else {
                launched = SetArgs(s.RnnCell(), d_gih, d_ghh, w.b_ih, w.b_hh, d_h,
                                   d_y, static_cast<int>(t),
                                   static_cast<int>(seq),
                                   static_cast<int>(hidden), total, use_tanh) &&
                           Launch1D(s.Queue(), s.RnnCell(), batch * hidden);
            }
            if (!launched) {
                detail = "cell kernel launch failed";
                return false;
            }
        }
        return true;
    }

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
        const int use_tanh = request.activation == NeuralActivation::Tanh ? 1 : 0;
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
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        std::string(NeuralOpName(request.op)) +
                            " buffer contract violated");
        }
        NeuralOpStatus failure;
        OpenclExecutionState* state = EnsureState(request, failure);
        if (!state) return failure;
        OpenclExecutionState& s = *state;
        std::lock_guard<std::mutex> lock(s.ExecuteMutex());
        const size_t f = sizeof(float);
        std::vector<LayerDeviceWeights> weights;
        if (!UploadStackedWeights(s, buffers.weights, layers, gate_width, input,
                                  hidden, weights, failure)) {
            return failure;
        }
        ClBuffer d_x, d_gih, d_ghh, d_h, d_c, d_hn, d_cn, dummy;
        std::vector<ClBuffer> d_y(layers);
        bool allocated = d_x.Allocate(s.Context(), rows * input * f) &&
                         d_gih.Allocate(s.Context(), rows * gate_width * f) &&
                         d_ghh.Allocate(s.Context(), batch * gate_width * f) &&
                         d_h.Allocate(s.Context(), batch * hidden * f) &&
                         d_c.Allocate(s.Context(), batch * hidden * f) &&
                         d_hn.Allocate(s.Context(), state_count * f) &&
                         d_cn.Allocate(s.Context(), state_count * f) &&
                         dummy.Allocate(s.Context(), f);
        for (size_t l = 0; allocated && l < layers; ++l) {
            allocated = d_y[l].Allocate(s.Context(), rows * hidden * f);
        }
        if (!allocated) return AllocFail();
        if (!Upload(s.Queue(), d_x, buffers.inputs[0]->ReadData<float>(),
                    rows * input * f)) {
            return ExecFail("host-to-device transfer failed");
        }
        for (size_t l = 0; l < layers; ++l) {
            const ClBuffer& d_input = l == 0 ? d_x : d_y[l - 1];
            std::string detail;
            if (!RunStackedLayerForward(s, kind, batch, seq, hidden, d_input,
                                        weights[l], d_gih, d_ghh, d_h, d_c,
                                        d_y[l], dummy, dummy, false, use_tanh,
                                        detail)) {
                return ExecFail("layer " + std::to_string(l) + ": " + detail);
            }
            if (!CopyDevice(s.Queue(), d_hn, l * batch * hidden, d_h,
                            batch * hidden) ||
                (kind == CellKind::Lstm &&
                 !CopyDevice(s.Queue(), d_cn, l * batch * hidden, d_c,
                             batch * hidden))) {
                return ExecFail("final state capture failed");
            }
        }
        bool downloaded =
            clFinish(s.Queue()) == CL_SUCCESS &&
            Download(s.Queue(), buffers.outputs[0]->Data<float>(),
                     d_y[layers - 1], rows * hidden * f);
        if (downloaded && wants_states) {
            downloaded =
                Download(s.Queue(), buffers.outputs[1]->Data<float>(), d_hn,
                         state_count * f) &&
                (kind != CellKind::Lstm ||
                 Download(s.Queue(), buffers.outputs[2]->Data<float>(), d_cn,
                          state_count * f));
        }
        if (!downloaded) return ExecFail("device execution or readback failed");
        return Ok();
    }

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
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        std::string(NeuralOpName(request.op)) +
                            " buffer contract violated");
        }
        NeuralOpStatus failure;
        OpenclExecutionState* state = EnsureState(request, failure);
        if (!state) return failure;
        OpenclExecutionState& s = *state;
        std::lock_guard<std::mutex> lock(s.ExecuteMutex());
        const size_t f = sizeof(float);
        const bool lstm = kind == CellKind::Lstm;
        const bool gated = kind != CellKind::Rnn;
        const bool shared_gates = kind != CellKind::Gru;
        const int use_tanh = request.activation == NeuralActivation::Tanh ? 1 : 0;
        std::vector<LayerDeviceWeights> weights;
        if (!UploadStackedWeights(s, buffers.weights, layers, gate_width, input,
                                  hidden, weights, failure)) {
            return failure;
        }
        ClBuffer d_x, d_dy, d_gih, d_ghh, d_h, d_c, d_dhrec, d_dgx, d_dgh, dummy;
        std::vector<ClBuffer> d_y(layers), d_gates(layers), d_ccache(layers),
            d_dx(layers), d_dwih(layers), d_dwhh(layers), d_dbih(layers),
            d_dbhh(layers);
        bool allocated = d_x.Allocate(s.Context(), rows * input * f) &&
                         d_dy.Allocate(s.Context(), rows * hidden * f) &&
                         d_gih.Allocate(s.Context(), rows * gate_width * f) &&
                         d_ghh.Allocate(s.Context(), batch * gate_width * f) &&
                         d_h.Allocate(s.Context(), batch * hidden * f) &&
                         d_c.Allocate(s.Context(), batch * hidden * f) &&
                         d_dhrec.Allocate(s.Context(), batch * hidden * f) &&
                         d_dgx.Allocate(s.Context(), rows * gate_width * f) &&
                         d_dgh.Allocate(s.Context(), shared_gates ? f : rows * gate_width * f) &&
                         dummy.Allocate(s.Context(), f);
        for (size_t l = 0; allocated && l < layers; ++l) {
            const size_t in = l == 0 ? input : hidden;
            allocated = d_y[l].Allocate(s.Context(), rows * hidden * f) &&
                        d_gates[l].Allocate(s.Context(), gated ? rows * 4 * hidden * f : f) &&
                        d_ccache[l].Allocate(s.Context(), lstm ? rows * hidden * f : f) &&
                        d_dx[l].Allocate(s.Context(), rows * in * f) &&
                        d_dwih[l].Allocate(s.Context(), gate_width * in * f) &&
                        d_dwhh[l].Allocate(s.Context(), gate_width * hidden * f) &&
                        d_dbih[l].Allocate(s.Context(), gate_width * f) &&
                        d_dbhh[l].Allocate(s.Context(), gate_width * f);
        }
        if (!allocated) return AllocFail();
        if (!Upload(s.Queue(), d_x, buffers.inputs[0]->ReadData<float>(),
                    rows * input * f) ||
            !Upload(s.Queue(), d_dy, buffers.inputs[1]->ReadData<float>(),
                    rows * hidden * f)) {
            return ExecFail("host-to-device transfer failed");
        }
        // Recompute every layer's forward with caches.
        for (size_t l = 0; l < layers; ++l) {
            const ClBuffer& d_input = l == 0 ? d_x : d_y[l - 1];
            std::string detail;
            if (!RunStackedLayerForward(s, kind, batch, seq, hidden, d_input,
                                        weights[l], d_gih, d_ghh, d_h, d_c,
                                        d_y[l], d_gates[l], d_ccache[l], gated,
                                        use_tanh, detail)) {
                return ExecFail("training forward layer " + std::to_string(l) +
                                ": " + detail);
            }
        }
        const int total = static_cast<int>(batch * hidden);
        const size_t ld_gates = seq * gate_width;
        // BPTT top-down; layer l's dx is layer l-1's dy.
        const ClBuffer* d_dy_cur = &d_dy;
        for (size_t li = layers; li-- > 0;) {
            const LayerDeviceWeights& w = weights[li];
            const size_t in = w.in_features;
            const ClBuffer& d_input = li == 0 ? d_x : d_y[li - 1];
            if (!Zero(s.Queue(), s.Fill(), d_dhrec, batch * hidden) ||
                !Zero(s.Queue(), s.Fill(), d_c, batch * hidden) ||
                !Zero(s.Queue(), s.Fill(), d_dwih[li], gate_width * in) ||
                !Zero(s.Queue(), s.Fill(), d_dwhh[li], gate_width * hidden)) {
                return ExecFail("gradient reset failed");
            }
            for (size_t t = seq; t-- > 0;) {
                bool launched;
                if (lstm) {
                    launched =
                        SetArgs(s.LstmBackwardCell(), *d_dy_cur, d_dhrec, d_c,
                                d_gates[li], d_ccache[li], d_dgx,
                                static_cast<int>(t), static_cast<int>(seq),
                                static_cast<int>(hidden), total) &&
                        Launch1D(s.Queue(), s.LstmBackwardCell(), batch * hidden);
                } else if (kind == CellKind::Gru) {
                    launched =
                        SetArgs(s.GruBackwardCell(), *d_dy_cur, d_dhrec,
                                d_gates[li], d_y[li], d_dgx, d_dgh,
                                static_cast<int>(t), static_cast<int>(seq),
                                static_cast<int>(hidden), total) &&
                        Launch1D(s.Queue(), s.GruBackwardCell(), batch * hidden);
                } else {
                    launched =
                        SetArgs(s.RnnBackwardCell(), *d_dy_cur, d_dhrec,
                                d_y[li], d_dgx, static_cast<int>(t),
                                static_cast<int>(seq), static_cast<int>(hidden),
                                total, use_tanh) &&
                        Launch1D(s.Queue(), s.RnnBackwardCell(), batch * hidden);
                }
                if (!launched) return ExecFail("backward cell launch failed");
                const size_t g_off = t * gate_width;
                const ClBuffer& dgh = shared_gates ? d_dgx : d_dgh;
                // dh for t-1: LSTM has no direct term (beta 0); GRU adds to
                // the direct part the cell wrote (beta 1).
                if (!GemmAB(s, batch, hidden, gate_width, dgh, g_off, ld_gates,
                            w.w_hh, 0, hidden, d_dhrec, 0, hidden,
                            shared_gates ? 0.0f : 1.0f)) {
                    return ExecFail("recurrent gradient GEMM failed");
                }
                if (!GemmAtBAccum(s, batch, gate_width, in, d_dgx, g_off, ld_gates,
                                  d_input, t * in, seq * in, d_dwih[li], 0, in)) {
                    return ExecFail("input weight-gradient GEMM failed");
                }
                if (t > 0 &&
                    !GemmAtBAccum(s, batch, gate_width, hidden, dgh, g_off,
                                  ld_gates, d_y[li], (t - 1) * hidden,
                                  seq * hidden, d_dwhh[li], 0, hidden)) {
                    return ExecFail("hidden weight-gradient GEMM failed");
                }
            }
            if (!GemmAB(s, rows, in, gate_width, d_dgx, 0, gate_width, w.w_ih, 0,
                        in, d_dx[li], 0, in, 0.0f)) {
                return ExecFail("input gradient GEMM failed");
            }
            if (!Colsum(s, rows, gate_width, d_dgx, gate_width, d_dbih[li])) {
                return ExecFail("bias gradient reduction failed");
            }
            if (shared_gates) {
                if (!CopyDevice(s.Queue(), d_dbhh[li], 0, d_dbih[li], gate_width)) {
                    return ExecFail("bias gradient copy failed");
                }
            } else if (!Colsum(s, rows, gate_width, d_dgh, gate_width,
                               d_dbhh[li])) {
                return ExecFail("bias gradient reduction failed");
            }
            d_dy_cur = &d_dx[li];
        }
        bool downloaded =
            clFinish(s.Queue()) == CL_SUCCESS &&
            Download(s.Queue(), buffers.outputs[0]->Data<float>(), d_dx[0],
                     rows * input * f);
        for (size_t l = 0; downloaded && l < layers; ++l) {
            const size_t in = l == 0 ? input : hidden;
            downloaded =
                Download(s.Queue(), buffers.gradients[4 * l]->Data<float>(),
                         d_dwih[l], gate_width * in * f) &&
                Download(s.Queue(), buffers.gradients[4 * l + 1]->Data<float>(),
                         d_dwhh[l], gate_width * hidden * f) &&
                Download(s.Queue(), buffers.gradients[4 * l + 2]->Data<float>(),
                         d_dbih[l], gate_width * f) &&
                Download(s.Queue(), buffers.gradients[4 * l + 3]->Data<float>(),
                         d_dbhh[l], gate_width * f);
        }
        if (!downloaded) return ExecFail("device execution or readback failed");
        return Ok();
    }

    static std::string AttentionContractError(const NeuralOpRequest& r) {
        if (r.dtype != DataType::Float32) return "attention contract is Float32 only";
        if (r.batch == 0 || r.heads == 0 || r.kv_heads == 0 || r.seq == 0 || r.kv_seq == 0) {
            return "attention dimensions must be positive";
        }
        if (r.heads % r.kv_heads != 0) return "heads must be a multiple of kv_heads";
        if (r.head_dim == 0 || r.head_dim > 128) return "attention head_dim must be 1..128";
        if (r.position_strategy != NeuralPositionStrategy::None &&
            r.position_strategy != NeuralPositionStrategy::Alibi) {
            return "attention kernels take positions as none or alibi (RoPE is applied before the call)";
        }
        if (!(r.softmax_scale > 0.0f) || !(r.logit_softcap >= 0.0f)) return "attention scale must be positive";
        if (!(r.attention_dropout >= 0.0f) || !(r.attention_dropout < 1.0f)) {
            return "attention dropout must be in [0, 1)";
        }
        const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
        if (r.seq > limit || r.kv_seq > limit || r.query_offset > limit || r.sliding_window > limit) {
            return "attention sizes exceed the int range";
        }
        return {};
    }

    NeuralOpStatus ExecuteAttention(const NeuralOpRequest& r, NeuralDeviceOpBuffers& buffers) {
        const bool backward = r.op == NeuralOp::AttentionBackward;
        const bool alibi = r.position_strategy == NeuralPositionStrategy::Alibi;
        const size_t q_elements = r.head_dim * r.seq * r.batch * r.heads;
        const size_t kv_elements = r.head_dim * r.kv_seq * r.batch * r.kv_heads;
        const size_t rows = r.seq * r.batch * r.heads;
        const auto& in = buffers.inputs;
        const auto& out = buffers.outputs;
        const bool shapes_ok = backward
            ? (in.size() == (alibi ? 7u : 6u) && out.size() == 4 && in[0].elements == q_elements &&
               in[1].elements == kv_elements && in[2].elements == kv_elements && in[3].elements == q_elements &&
               in[4].elements == q_elements && in[5].elements == rows && (!alibi || in[6].elements == r.heads) &&
               out[0].elements == q_elements && out[1].elements == kv_elements && out[2].elements == kv_elements &&
               out[3].elements == rows)
            : (in.size() == (alibi ? 4u : 3u) && out.size() == 2 && in[0].elements == q_elements &&
               in[1].elements == kv_elements && in[2].elements == kv_elements &&
               (!alibi || in[3].elements == r.heads) && out[0].elements == q_elements && out[1].elements == rows);
        if (!shapes_ok) {
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        "attention buffers do not match the request");
        }
        const auto context = static_cast<cl_context>(buffers.queue.cl_context);
        const auto device = static_cast<cl_device_id>(buffers.queue.cl_device);
        const auto queue = static_cast<cl_command_queue>(buffers.queue.cl_queue);
        std::lock_guard<std::mutex> lock(device_program_.Mutex());
        std::string failure;
        const auto* kernels = device_program_.Attention(context, device, static_cast<int>(r.head_dim), failure);
        if (!kernels) return ExecFail(failure);
        const auto mem = [](const NeuralDeviceBuffer& b) { return static_cast<cl_mem>(b.handle); };
        cl_mem slopes = alibi ? mem(in[alibi && backward ? 6 : 3]) : nullptr;
        const int has_slopes = alibi ? 1 : 0;
        const int batch = static_cast<int>(r.batch), heads = static_cast<int>(r.heads);
        const int kv_heads = static_cast<int>(r.kv_heads), sq = static_cast<int>(r.seq), sk = static_cast<int>(r.kv_seq);
        const int offset = static_cast<int>(r.query_offset), causal = r.causal ? 1 : 0;
        const int window = static_cast<int>(r.sliding_window);
        const float softcap = r.logit_softcap, scale = r.softmax_scale;
        cl_uint arg = 0;
        bool ok = true;
        const auto set_mem = [&](cl_kernel k, cl_mem m) {
            ok = ok && clSetKernelArg(k, arg++, sizeof(cl_mem), &m) == CL_SUCCESS;
        };
        const auto set_int = [&](cl_kernel k, int v) {
            ok = ok && clSetKernelArg(k, arg++, sizeof(int), &v) == CL_SUCCESS;
        };
        const auto set_float = [&](cl_kernel k, float v) {
            ok = ok && clSetKernelArg(k, arg++, sizeof(float), &v) == CL_SUCCESS;
        };
        const float dropout = r.training ? r.attention_dropout : 0.0f;
        const cl_ulong seed = static_cast<cl_ulong>(r.dropout_seed);
        const auto set_tail = [&](cl_kernel k) {
            for (int v : {batch, heads, kv_heads, sq, sk, offset, causal, window}) set_int(k, v);
            set_float(k, softcap);
            set_float(k, scale);
            set_float(k, dropout);
            ok = ok && clSetKernelArg(k, arg++, sizeof(cl_ulong), &seed) == CL_SUCCESS;
        };
        // Some OpenCL runtimes (e.g. Intel's CPU device) execute commands out
        // of order; explicit barriers order our kernels against ArrayFire's
        // work and against each other. They are free on in-order queues.
        const auto barrier = [&]() {
            ok = ok && clEnqueueBarrierWithWaitList(queue, 0, nullptr, nullptr) == CL_SUCCESS;
        };
        barrier();
        const size_t local[3] = {64, 1, 1};
        const auto launch3 = [&](cl_kernel k, size_t rows_to_cover, size_t dim1) {
            const size_t global[3] = {((rows_to_cover + 63) / 64) * 64, dim1, r.batch};
            ok = ok && clEnqueueNDRangeKernel(queue, k, 3, nullptr, global, local, 0, nullptr, nullptr) == CL_SUCCESS;
        };
        if (!backward) {
            cl_kernel k = kernels->forward;
            arg = 0;
            set_mem(k, mem(in[0])); set_mem(k, mem(in[1])); set_mem(k, mem(in[2]));
            set_mem(k, slopes); set_int(k, has_slopes);
            set_mem(k, mem(out[0])); set_mem(k, mem(out[1]));
            set_tail(k);
            launch3(k, r.seq, r.heads);
            barrier();
            return ok ? Ok() : ExecFail("attention_forward launch failed");
        }
        cl_kernel kd = kernels->delta;
        arg = 0;
        set_mem(kd, mem(in[3])); set_mem(kd, mem(in[4])); set_mem(kd, mem(out[3])); set_int(kd, static_cast<int>(rows));
        const size_t delta_global = ((rows + 63) / 64) * 64;
        const size_t delta_local = 64;
        ok = ok && clEnqueueNDRangeKernel(queue, kd, 1, nullptr, &delta_global, &delta_local, 0, nullptr, nullptr) ==
                       CL_SUCCESS;
        barrier();  // dQ and dK/dV read delta
        cl_kernel kq = kernels->dq;
        arg = 0;
        set_mem(kq, mem(in[0])); set_mem(kq, mem(in[1])); set_mem(kq, mem(in[2])); set_mem(kq, mem(in[4]));
        set_mem(kq, mem(in[5])); set_mem(kq, mem(out[3])); set_mem(kq, slopes); set_int(kq, has_slopes);
        set_mem(kq, mem(out[0]));
        set_tail(kq);
        launch3(kq, r.seq, r.heads);
        cl_kernel kkv = kernels->dkdv;
        arg = 0;
        set_mem(kkv, mem(in[0])); set_mem(kkv, mem(in[1])); set_mem(kkv, mem(in[2])); set_mem(kkv, mem(in[4]));
        set_mem(kkv, mem(in[5])); set_mem(kkv, mem(out[3])); set_mem(kkv, slopes); set_int(kkv, has_slopes);
        set_mem(kkv, mem(out[1])); set_mem(kkv, mem(out[2]));
        set_tail(kkv);
        launch3(kkv, r.kv_seq, r.kv_heads);
        barrier();
        return ok ? Ok() : ExecFail("attention_backward launch failed");
    }

    NeuralOpStatus ExecuteDevice(const NeuralOpRequest& request,
                                 NeuralDeviceOpBuffers& buffers) override {
        const NeuralCapability capability = QueryCapability(request);
        if (!capability.supported) return Fail(capability.reason, capability.detail);
        if (buffers.queue.platform != DeviceType::OPENCL || !buffers.queue.cl_context ||
            !buffers.queue.cl_queue || !buffers.queue.cl_device) {
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        "device-resident execution needs an OpenCL context, queue and device");
        }
        // This tenant serves GPU devices (like its v1 ops). ArrayFire's OpenCL
        // backend can also expose CPU devices; those keep the ArrayFire path.
        cl_device_type device_type = 0;
        if (clGetDeviceInfo(static_cast<cl_device_id>(buffers.queue.cl_device), CL_DEVICE_TYPE, sizeof(device_type),
                            &device_type, nullptr) != CL_SUCCESS ||
            (device_type & CL_DEVICE_TYPE_GPU) == 0) {
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        "the active OpenCL device is not a GPU; this tenant serves GPU devices only");
        }
        if (request.op == NeuralOp::AttentionForward || request.op == NeuralOp::AttentionBackward) {
            return ExecuteAttention(request, buffers);
        }
        if (buffers.inputs.size() != 1 || buffers.outputs.size() != 1 ||
            buffers.inputs[0].elements != request.elements ||
            buffers.outputs[0].elements != request.elements) {
            return Fail(BackendFallbackReason::OpenclProviderUnsupportedContract,
                        "device_probe needs one input and one output of request.elements");
        }
        const auto context = static_cast<cl_context>(buffers.queue.cl_context);
        const auto device = static_cast<cl_device_id>(buffers.queue.cl_device);
        std::string failure;
        if (!device_program_.Ensure(context, device, failure)) return ExecFail(failure);
        std::lock_guard<std::mutex> lock(device_program_.Mutex());
        cl_kernel kernel = device_program_.Probe();
        cl_mem x = static_cast<cl_mem>(buffers.inputs[0].handle);
        cl_mem y = static_cast<cl_mem>(buffers.outputs[0].handle);
        const int n = static_cast<int>(request.elements);
        if (clSetKernelArg(kernel, 0, sizeof(cl_mem), &x) != CL_SUCCESS ||
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &y) != CL_SUCCESS ||
            clSetKernelArg(kernel, 2, sizeof(int), &n) != CL_SUCCESS) {
            return ExecFail("device_probe argument binding failed");
        }
        const size_t local = 64;
        const size_t global = ((request.elements + local - 1) / local) * local;
        // Enqueue on ArrayFire's queue between barriers (ordered with its work
        // even on out-of-order queues), no finish.
        const auto probe_queue = static_cast<cl_command_queue>(buffers.queue.cl_queue);
        if (clEnqueueBarrierWithWaitList(probe_queue, 0, nullptr, nullptr) != CL_SUCCESS ||
            clEnqueueNDRangeKernel(probe_queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr) != CL_SUCCESS ||
            clEnqueueBarrierWithWaitList(probe_queue, 0, nullptr, nullptr) != CL_SUCCESS) {
            return ExecFail("device_probe launch failed");
        }
        return Ok();
    }

    OpenclRuntimeProbe probe_;
    std::vector<std::unique_ptr<OpenclExecutionState>> states_;
    DeviceResidentProgram device_program_;
};

} // namespace

size_t OpenclProviderRetentionFloorHidden() {
    return g_retention_floor_hidden.load();
}

void SetOpenclProviderRetentionFloorForTesting(size_t hidden_floor) {
    g_retention_floor_hidden.store(hidden_floor);
}

void RegisterOpenclCellNeuralProvider(NeuralProviderRegistry& registry) {
    const OpenclRuntimeProbe probe = ProbeOpenclRuntime();
    if (!probe.ok) {
        spdlog::info("OpenCL neural provider not registered (reason={}): {}",
                     BackendFallbackReasonName(
                         BackendFallbackReason::OpenclProviderUnavailable),
                     probe.failure);
        return;
    }
    auto provider = std::make_shared<OpenclCellProvider>(probe);
    spdlog::info("OpenCL neural provider registered: {}", provider->Version());
    registry.Register(std::move(provider));
}

} // namespace cyxwiz

#endif // CYXWIZ_HAS_OPENCL_DNN_PROVIDER
