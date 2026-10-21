// SYCL fused attention for the oneAPI neural provider (tofix112 phase 5b).
// Same algorithm and contract as the CUDA and OpenCL tenants: one work-item
// per query (or key) row, ATTN_BR rows per work-group, ATTN_BC-row tiles in
// local memory, online softmax, counter-based dropout. Built with the Intel
// DPC++ compiler into its own DLL (see cyxwiz_oneapi_kernels.h).
//
// Data is host-staged (ABI 2): each call wraps the caller's host arrays in
// buffers owned by this library, runs on a queue of its own for the device
// ArrayFire reports, and writes outputs back before returning.
#define CYXWIZ_ONEAPI_KERNELS_BUILD
#include "cyxwiz_oneapi_kernels.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <string>

namespace {

constexpr int ATTN_BR = 64;
constexpr int ATTN_BC = 32;
constexpr float CYX_INF = std::numeric_limits<float>::infinity();

using FloatBuffer = sycl::buffer<float, 1>;

void CopyError(const std::string& message, char* error, size_t error_size) {
    if (!error || error_size == 0) return;
    const size_t n = std::min(message.size(), error_size - 1);
    std::memcpy(error, message.data(), n);
    error[n] = '\0';
}

struct CallError {
    std::string message;
};

sycl::queue& QueueFor(const CyxOneapiDevice* selector) {
    static std::mutex mutex;
    static auto* queues = new std::map<std::string, sycl::queue>();  // process lifetime
    if (!selector || !selector->name) throw CallError{"no oneAPI device selector"};
    const std::string platform = selector->platform ? selector->platform : "";
    const std::string key = std::string(selector->name) + "|" + platform + "|" + std::to_string(selector->is_gpu);
    std::lock_guard<std::mutex> lock(mutex);
    auto found = queues->find(key);
    if (found != queues->end()) return found->second;
    const auto devices = sycl::device::get_devices();
    const sycl::device* chosen = nullptr;
    for (int pass = 0; pass < 2 && !chosen; ++pass) {
        for (const auto& device : devices) {
            if (selector->is_gpu >= 0 && device.is_gpu() != (selector->is_gpu != 0)) continue;
            if (device.get_info<sycl::info::device::name>() != selector->name) continue;
            if (pass == 0 && !platform.empty() &&
                device.get_platform().get_info<sycl::info::platform::name>() != platform) {
                continue;
            }
            chosen = &device;
            break;
        }
    }
    if (!chosen) throw CallError{std::string("no SYCL device named \"") + selector->name + "\""};
    auto inserted = queues->emplace(key, sycl::queue(*chosen, sycl::property::queue::in_order()));
    return inserted.first->second;
}

// Buffers for one call over the caller's host arrays: inputs are never
// written back; outputs are written back when the call's buffers go away.
class CallBuffers {
public:
    FloatBuffer& In(const CyxOneapiBuffer& b, const char* what) { return Make(b, what, false); }
    FloatBuffer& Out(const CyxOneapiBuffer& b, const char* what) { return Make(b, what, true); }

private:
    FloatBuffer& Make(const CyxOneapiBuffer& b, const char* what, bool write_back) {
        if (!b.handle || b.elements == 0) throw CallError{std::string(what) + " buffer is missing"};
        buffers_.emplace_back(static_cast<float*>(b.handle), sycl::range<1>(b.elements));
        if (!write_back) buffers_.back().set_final_data(nullptr);
        return buffers_.back();
    }
    std::deque<FloatBuffer> buffers_;  // stable addresses
};

// Placeholder for absent ALiBi slopes; kept for the process.
FloatBuffer& NoSlopes() {
    static auto* buffer = new FloatBuffer(sycl::range<1>(1));
    return *buffer;
}

inline float Keep(uint64_t seed, uint64_t index, float p) {
    if (p <= 0.0f) return 1.0f;
    uint64_t z = seed + index * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
    return u >= p ? 1.0f / (1.0f - p) : 0.0f;
}

inline float Score(float dot, float scale, float softcap, float slope_term, float& tanh_out) {
    float x = dot * scale;
    float t = 0.0f;
    if (softcap > 0.0f) {
        t = sycl::tanh(x / softcap);
        x = softcap * t;
    }
    tanh_out = t;
    return x + slope_term;
}

// Scalar kernel parameters shared by the attention kernels.
struct Params {
    int B, H, KVH, Sq, Sk, D, q_offset, causal, window, has_slopes;
    float softcap, scale, dropout;
    uint64_t seed;
};

Params MakeParams(const CyxOneapiAttentionArgs& a) {
    Params p{};
    p.B = a.batch;
    p.H = a.heads;
    p.KVH = a.kv_heads;
    p.Sq = a.seq;
    p.Sk = a.kv_seq;
    p.D = a.head_dim;
    p.q_offset = a.query_offset;
    p.causal = a.causal;
    p.window = a.window;
    p.has_slopes = a.slopes.handle ? 1 : 0;
    p.softcap = a.softcap;
    p.scale = a.scale;
    p.dropout = a.dropout;
    p.seed = a.seed;
    return p;
}

size_t RoundUp(size_t n) { return ((n + ATTN_BR - 1) / ATTN_BR) * ATTN_BR; }

// MAXD bounds head_dim at compile time (per-row arrays); p.D <= MAXD.
template <int MAXD>
void Forward(sycl::queue& queue, CallBuffers& call, const CyxOneapiAttentionArgs& a) {
    const Params p = MakeParams(a);
    FloatBuffer& bq = call.In(a.q, "Q");
    FloatBuffer& bk = call.In(a.k, "K");
    FloatBuffer& bv = call.In(a.v, "V");
    FloatBuffer& bo = call.Out(a.o, "O");
    FloatBuffer& blse = call.Out(a.lse, "LSE");
    FloatBuffer& bs = p.has_slopes ? call.In(a.slopes, "slopes") : NoSlopes();
    queue.submit([&](sycl::handler& h) {
        sycl::accessor Q(bq, h, sycl::read_only);
        sycl::accessor K(bk, h, sycl::read_only);
        sycl::accessor V(bv, h, sycl::read_only);
        sycl::accessor S(bs, h, sycl::read_only);
        sycl::accessor O(bo, h, sycl::write_only);
        sycl::accessor LSE(blse, h, sycl::write_only);
        sycl::local_accessor<float, 2> Ks(sycl::range<2>(ATTN_BC, MAXD), h);
        sycl::local_accessor<float, 2> Vs(sycl::range<2>(ATTN_BC, MAXD), h);
        const sycl::range<3> global(p.B, p.H, RoundUp(p.Sq));
        h.parallel_for(sycl::nd_range<3>(global, sycl::range<3>(1, 1, ATTN_BR)), [=](sycl::nd_item<3> it) {
            const int D = p.D;
            const int lid = static_cast<int>(it.get_local_id(2));
            const int block = static_cast<int>(it.get_group(2));
            const int hh = static_cast<int>(it.get_group(1));
            const int b = static_cast<int>(it.get_group(0));
            const int kvh = hh / (p.H / p.KVH);
            const int row = block * ATTN_BR + lid;
            const bool active = row < p.Sq;
            const int qpos = row + p.q_offset;
            const size_t q_base = ((static_cast<size_t>(hh) * p.B + b) * p.Sq + row) * D;
            const size_t kv_base = (static_cast<size_t>(kvh) * p.B + b) * p.Sk * D;
            float q[MAXD];
            float acc[MAXD];
            for (int d = 0; d < D; ++d) {
                q[d] = active ? Q[q_base + d] : 0.0f;
                acc[d] = 0.0f;
            }
            const float slope = p.has_slopes ? S[hh] : 0.0f;
            float m = -CYX_INF;
            float l = 0.0f;
            const int first_q = block * ATTN_BR + p.q_offset;
            const int last_q = sycl::min(p.Sq, (block + 1) * ATTN_BR) - 1 + p.q_offset;
            const int k_end = p.causal ? sycl::min(p.Sk, last_q + 1) : p.Sk;
            int k_begin = p.window > 0 ? sycl::max(0, first_q - p.window + 1) : 0;
            k_begin = (k_begin / ATTN_BC) * ATTN_BC;
            for (int k0 = k_begin; k0 < k_end; k0 += ATTN_BC) {
                for (int idx = lid; idx < ATTN_BC * D; idx += ATTN_BR) {
                    const int j = idx / D;
                    const int d = idx - j * D;
                    const int key = k0 + j;
                    const bool in = key < p.Sk;
                    Ks[j][d] = in ? K[kv_base + static_cast<size_t>(key) * D + d] : 0.0f;
                    Vs[j][d] = in ? V[kv_base + static_cast<size_t>(key) * D + d] : 0.0f;
                }
                sycl::group_barrier(it.get_group());
                if (active) {
                    float s[ATTN_BC];
                    float tile_max = -CYX_INF;
                    for (int j = 0; j < ATTN_BC; ++j) {
                        const int key = k0 + j;
                        const bool valid = key < p.Sk && (!p.causal || key <= qpos) &&
                                           (p.window <= 0 || qpos - key < p.window);
                        if (!valid) {
                            s[j] = -CYX_INF;
                            continue;
                        }
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) dot += q[d] * Ks[j][d];
                        float t;
                        const float x = Score(dot, p.scale, p.softcap,
                                              p.has_slopes ? slope * static_cast<float>(key - qpos) : 0.0f, t);
                        s[j] = x;
                        tile_max = sycl::fmax(tile_max, x);
                    }
                    if (tile_max > -CYX_INF) {
                        const float m_new = sycl::fmax(m, tile_max);
                        const float correction = (m == -CYX_INF) ? 0.0f : sycl::exp(m - m_new);
                        l *= correction;
                        for (int d = 0; d < D; ++d) acc[d] *= correction;
                        const uint64_t drop_row = ((static_cast<uint64_t>(b) * p.H + hh) * p.Sq + row) * p.Sk;
                        for (int j = 0; j < ATTN_BC; ++j) {
                            if (s[j] == -CYX_INF) continue;
                            const float pr = sycl::exp(s[j] - m_new);
                            l += pr;
                            const float pk = pr * Keep(p.seed, drop_row + static_cast<uint64_t>(k0 + j), p.dropout);
                            for (int d = 0; d < D; ++d) acc[d] += pk * Vs[j][d];
                        }
                        m = m_new;
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            if (active) {
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                for (int d = 0; d < D; ++d) O[q_base + d] = acc[d] * inv;
                LSE[(static_cast<size_t>(hh) * p.B + b) * p.Sq + row] = l > 0.0f ? m + sycl::log(l) : -CYX_INF;
            }
        });
    });
}

template <int MAXD>
void Backward(sycl::queue& queue, CallBuffers& call, const CyxOneapiAttentionArgs& a) {
    const Params p = MakeParams(a);
    FloatBuffer& bq = call.In(a.q, "Q");
    FloatBuffer& bk = call.In(a.k, "K");
    FloatBuffer& bv = call.In(a.v, "V");
    FloatBuffer& bo = call.In(a.o, "O");
    FloatBuffer& bgo = call.In(a.d_o, "dO");
    FloatBuffer& blse = call.In(a.lse, "LSE");
    FloatBuffer& bs = p.has_slopes ? call.In(a.slopes, "slopes") : NoSlopes();
    FloatBuffer& bdq = call.Out(a.d_q, "dQ");
    FloatBuffer& bdk = call.Out(a.d_k, "dK");
    FloatBuffer& bdv = call.Out(a.d_v, "dV");
    FloatBuffer& bdelta = call.Out(a.delta, "delta");
    const size_t rows = static_cast<size_t>(p.Sq) * p.B * p.H;

    // delta[r] = dot(O[r], dO[r])
    queue.submit([&](sycl::handler& h) {
        sycl::accessor O(bo, h, sycl::read_only);
        sycl::accessor GO(bgo, h, sycl::read_only);
        sycl::accessor DL(bdelta, h, sycl::write_only);
        const int D = p.D;
        h.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> r) {
            float sum = 0.0f;
            const size_t base = r[0] * D;
            for (int d = 0; d < D; ++d) sum += O[base + d] * GO[base + d];
            DL[r] = sum;
        });
    });

    // dQ: one work-item per query row.
    queue.submit([&](sycl::handler& h) {
        sycl::accessor Q(bq, h, sycl::read_only);
        sycl::accessor K(bk, h, sycl::read_only);
        sycl::accessor V(bv, h, sycl::read_only);
        sycl::accessor GO(bgo, h, sycl::read_only);
        sycl::accessor LSE(blse, h, sycl::read_only);
        sycl::accessor DL(bdelta, h, sycl::read_only);
        sycl::accessor S(bs, h, sycl::read_only);
        sycl::accessor DQ(bdq, h, sycl::write_only);
        sycl::local_accessor<float, 2> Ks(sycl::range<2>(ATTN_BC, MAXD), h);
        sycl::local_accessor<float, 2> Vs(sycl::range<2>(ATTN_BC, MAXD), h);
        const sycl::range<3> global(p.B, p.H, RoundUp(p.Sq));
        h.parallel_for(sycl::nd_range<3>(global, sycl::range<3>(1, 1, ATTN_BR)), [=](sycl::nd_item<3> it) {
            const int D = p.D;
            const int lid = static_cast<int>(it.get_local_id(2));
            const int block = static_cast<int>(it.get_group(2));
            const int hh = static_cast<int>(it.get_group(1));
            const int b = static_cast<int>(it.get_group(0));
            const int kvh = hh / (p.H / p.KVH);
            const int row = block * ATTN_BR + lid;
            const bool active = row < p.Sq;
            const int qpos = row + p.q_offset;
            const size_t row_index = (static_cast<size_t>(hh) * p.B + b) * p.Sq + row;
            const size_t q_base = row_index * D;
            const size_t kv_base = (static_cast<size_t>(kvh) * p.B + b) * p.Sk * D;
            float q[MAXD], go[MAXD], dq[MAXD];
            for (int d = 0; d < D; ++d) {
                q[d] = active ? Q[q_base + d] : 0.0f;
                go[d] = active ? GO[q_base + d] : 0.0f;
                dq[d] = 0.0f;
            }
            const float lse = active ? LSE[row_index] : 0.0f;
            const float dl = active ? DL[row_index] : 0.0f;
            const float slope = p.has_slopes ? S[hh] : 0.0f;
            const int first_q = block * ATTN_BR + p.q_offset;
            const int last_q = sycl::min(p.Sq, (block + 1) * ATTN_BR) - 1 + p.q_offset;
            const int k_end = p.causal ? sycl::min(p.Sk, last_q + 1) : p.Sk;
            int k_begin = p.window > 0 ? sycl::max(0, first_q - p.window + 1) : 0;
            k_begin = (k_begin / ATTN_BC) * ATTN_BC;
            for (int k0 = k_begin; k0 < k_end; k0 += ATTN_BC) {
                for (int idx = lid; idx < ATTN_BC * D; idx += ATTN_BR) {
                    const int j = idx / D;
                    const int d = idx - j * D;
                    const int key = k0 + j;
                    const bool in = key < p.Sk;
                    Ks[j][d] = in ? K[kv_base + static_cast<size_t>(key) * D + d] : 0.0f;
                    Vs[j][d] = in ? V[kv_base + static_cast<size_t>(key) * D + d] : 0.0f;
                }
                sycl::group_barrier(it.get_group());
                if (active && lse > -CYX_INF) {
                    for (int j = 0; j < ATTN_BC; ++j) {
                        const int key = k0 + j;
                        const bool valid = key < p.Sk && (!p.causal || key <= qpos) &&
                                           (p.window <= 0 || qpos - key < p.window);
                        if (!valid) continue;
                        float dot = 0.0f, dp = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            dot += q[d] * Ks[j][d];
                            dp += go[d] * Vs[j][d];
                        }
                        float t;
                        const float x = Score(dot, p.scale, p.softcap,
                                              p.has_slopes ? slope * static_cast<float>(key - qpos) : 0.0f, t);
                        const float pr = sycl::exp(x - lse);
                        const float keep =
                            Keep(p.seed, ((static_cast<uint64_t>(b) * p.H + hh) * p.Sq + row) * p.Sk + key, p.dropout);
                        float ds = pr * (dp * keep - dl);
                        if (p.softcap > 0.0f) ds *= (1.0f - t * t);
                        ds *= p.scale;
                        for (int d = 0; d < D; ++d) dq[d] += ds * Ks[j][d];
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            if (active) {
                for (int d = 0; d < D; ++d) DQ[q_base + d] = dq[d];
            }
        });
    });

    // dK/dV: one work-item per key row of a kv head; walks its query-head group.
    queue.submit([&](sycl::handler& h) {
        sycl::accessor Q(bq, h, sycl::read_only);
        sycl::accessor K(bk, h, sycl::read_only);
        sycl::accessor V(bv, h, sycl::read_only);
        sycl::accessor GO(bgo, h, sycl::read_only);
        sycl::accessor LSE(blse, h, sycl::read_only);
        sycl::accessor DL(bdelta, h, sycl::read_only);
        sycl::accessor S(bs, h, sycl::read_only);
        sycl::accessor DK(bdk, h, sycl::write_only);
        sycl::accessor DV(bdv, h, sycl::write_only);
        sycl::local_accessor<float, 2> Qs(sycl::range<2>(ATTN_BC, MAXD), h);
        sycl::local_accessor<float, 2> Gs(sycl::range<2>(ATTN_BC, MAXD), h);
        sycl::local_accessor<float, 1> Ls(sycl::range<1>(ATTN_BC), h);
        sycl::local_accessor<float, 1> Ds(sycl::range<1>(ATTN_BC), h);
        const sycl::range<3> global(p.B, p.KVH, RoundUp(p.Sk));
        h.parallel_for(sycl::nd_range<3>(global, sycl::range<3>(1, 1, ATTN_BR)), [=](sycl::nd_item<3> it) {
            const int D = p.D;
            const int lid = static_cast<int>(it.get_local_id(2));
            const int block = static_cast<int>(it.get_group(2));
            const int kvh = static_cast<int>(it.get_group(1));
            const int b = static_cast<int>(it.get_group(0));
            const int key = block * ATTN_BR + lid;
            const bool active = key < p.Sk;
            const size_t k_base = ((static_cast<size_t>(kvh) * p.B + b) * p.Sk + key) * D;
            float k[MAXD], v[MAXD], dk[MAXD], dv[MAXD];
            for (int d = 0; d < D; ++d) {
                k[d] = active ? K[k_base + d] : 0.0f;
                v[d] = active ? V[k_base + d] : 0.0f;
                dk[d] = 0.0f;
                dv[d] = 0.0f;
            }
            const int groups = p.H / p.KVH;
            const int first_key = block * ATTN_BR;
            const int last_key = sycl::min(p.Sk, (block + 1) * ATTN_BR) - 1;
            int q_begin = p.causal ? sycl::max(0, first_key - p.q_offset) : 0;
            q_begin = (q_begin / ATTN_BC) * ATTN_BC;
            const int q_end = p.window > 0 ? sycl::min(p.Sq, last_key + p.window - p.q_offset) : p.Sq;
            for (int g = 0; g < groups; ++g) {
                const int hh = kvh * groups + g;
                const float slope = p.has_slopes ? S[hh] : 0.0f;
                const size_t head_rows = (static_cast<size_t>(hh) * p.B + b) * p.Sq;
                for (int i0 = q_begin; i0 < q_end; i0 += ATTN_BC) {
                    for (int idx = lid; idx < ATTN_BC * D; idx += ATTN_BR) {
                        const int i = idx / D;
                        const int d = idx - i * D;
                        const int row = i0 + i;
                        const bool in = row < p.Sq;
                        Qs[i][d] = in ? Q[(head_rows + row) * D + d] : 0.0f;
                        Gs[i][d] = in ? GO[(head_rows + row) * D + d] : 0.0f;
                    }
                    for (int i = lid; i < ATTN_BC; i += ATTN_BR) {
                        const int row = i0 + i;
                        Ls[i] = row < p.Sq ? LSE[head_rows + row] : -CYX_INF;
                        Ds[i] = row < p.Sq ? DL[head_rows + row] : 0.0f;
                    }
                    sycl::group_barrier(it.get_group());
                    if (active) {
                        for (int i = 0; i < ATTN_BC; ++i) {
                            const int row = i0 + i;
                            const int qpos = row + p.q_offset;
                            const bool valid = row < p.Sq && Ls[i] > -CYX_INF && (!p.causal || key <= qpos) &&
                                               (p.window <= 0 || qpos - key < p.window);
                            if (!valid) continue;
                            float dot = 0.0f, dp = 0.0f;
                            for (int d = 0; d < D; ++d) {
                                dot += Qs[i][d] * k[d];
                                dp += Gs[i][d] * v[d];
                            }
                            float t;
                            const float x = Score(dot, p.scale, p.softcap,
                                                  p.has_slopes ? slope * static_cast<float>(key - qpos) : 0.0f, t);
                            const float pr = sycl::exp(x - Ls[i]);
                            const float keep = Keep(
                                p.seed, ((static_cast<uint64_t>(b) * p.H + hh) * p.Sq + row) * p.Sk + key, p.dropout);
                            float ds = pr * (dp * keep - Ds[i]);
                            if (p.softcap > 0.0f) ds *= (1.0f - t * t);
                            ds *= p.scale;
                            const float pk = pr * keep;
                            for (int d = 0; d < D; ++d) {
                                dv[d] += pk * Gs[i][d];
                                dk[d] += ds * Qs[i][d];
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            }
            if (active) {
                for (int d = 0; d < D; ++d) {
                    DK[k_base + d] = dk[d];
                    DV[k_base + d] = dv[d];
                }
            }
        });
    });
}

void CheckContract(const CyxOneapiAttentionArgs* a) {
    if (!a) throw CallError{"attention arguments are missing"};
    if (a->batch <= 0 || a->heads <= 0 || a->kv_heads <= 0 || a->seq <= 0 || a->kv_seq <= 0) {
        throw CallError{"attention dimensions must be positive"};
    }
    if (a->heads % a->kv_heads != 0) throw CallError{"heads must be a multiple of kv_heads"};
    if (a->head_dim <= 0 || a->head_dim > 128) throw CallError{"attention head_dim must be 1..128"};
}

// Head-dim buckets: per-row arrays sized for the bucket, loops bound by D.
template <bool BACKWARD>
void Dispatch(sycl::queue& queue, CallBuffers& call, const CyxOneapiAttentionArgs& a) {
    if (a.head_dim <= 16) BACKWARD ? Backward<16>(queue, call, a) : Forward<16>(queue, call, a);
    else if (a.head_dim <= 32) BACKWARD ? Backward<32>(queue, call, a) : Forward<32>(queue, call, a);
    else if (a.head_dim <= 64) BACKWARD ? Backward<64>(queue, call, a) : Forward<64>(queue, call, a);
    else BACKWARD ? Backward<128>(queue, call, a) : Forward<128>(queue, call, a);
}

template <typename Body>
int Guarded(Body&& body, char* error, size_t error_size) {
    try {
        body();
        return 0;
    } catch (const CallError& e) {
        CopyError(e.message, error, error_size);
    } catch (const sycl::exception& e) {
        CopyError(std::string("SYCL error: ") + e.what(), error, error_size);
    } catch (const std::exception& e) {
        CopyError(e.what(), error, error_size);
    } catch (...) {
        CopyError("unknown error in the oneAPI kernels", error, error_size);
    }
    return 1;
}

}  // namespace

extern "C" {

int cyxwiz_oneapi_kernels_abi(void) { return CYXWIZ_ONEAPI_KERNELS_ABI; }

int cyxwiz_oneapi_runtime_info(char* out, size_t out_size) {
    std::string info;
    const int status = Guarded(
        [&] {
            info = "SYCL " + std::to_string(__LIBSYCL_MAJOR_VERSION) + "." + std::to_string(__LIBSYCL_MINOR_VERSION);
            for (const auto& device : sycl::device::get_devices()) {
                info += " / '" + device.get_info<sycl::info::device::name>() + "' (" +
                        device.get_platform().get_info<sycl::info::platform::name>() +
                        (device.is_gpu() ? ", gpu)" : device.is_cpu() ? ", cpu)" : ", other)");
            }
        },
        out, out_size);
    if (status == 0) CopyError(info, out, out_size);
    return status;
}

int cyxwiz_oneapi_device_probe(const CyxOneapiDevice* device, CyxOneapiBuffer x, CyxOneapiBuffer y, char* error,
                               size_t error_size) {
    return Guarded(
        [&] {
            sycl::queue& queue = QueueFor(device);
            if (x.elements != y.elements || x.elements == 0) throw CallError{"device_probe needs matching sizes"};
            CallBuffers call;
            FloatBuffer& bx = call.In(x, "x");
            FloatBuffer& by = call.Out(y, "y");
            const size_t n = x.elements;
            queue.submit([&](sycl::handler& h) {
                sycl::accessor X(bx, h, sycl::read_only);
                sycl::accessor Y(by, h, sycl::write_only);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { Y[i] = 2.0f * X[i]; });
            });
            queue.wait_and_throw();
        },
        error, error_size);
}

int cyxwiz_oneapi_attention_forward(const CyxOneapiDevice* device, const CyxOneapiAttentionArgs* args, char* error,
                                    size_t error_size) {
    return Guarded(
        [&] {
            CheckContract(args);
            sycl::queue& queue = QueueFor(device);
            CallBuffers call;  // outputs are written back when this goes away
            Dispatch<false>(queue, call, *args);
            queue.wait_and_throw();
        },
        error, error_size);
}

int cyxwiz_oneapi_attention_backward(const CyxOneapiDevice* device, const CyxOneapiAttentionArgs* args, char* error,
                                     size_t error_size) {
    return Guarded(
        [&] {
            CheckContract(args);
            sycl::queue& queue = QueueFor(device);
            CallBuffers call;  // outputs are written back when this goes away
            Dispatch<true>(queue, call, *args);
            queue.wait_and_throw();
        },
        error, error_size);
}

}  // extern "C"
