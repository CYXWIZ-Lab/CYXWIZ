// C ABI of cyxwiz-oneapi-kernels, the SYCL half of the oneAPI neural provider
// (tofix112 phase 5b). The DLL is built with the Intel DPC++ compiler; the
// backend (MSVC) loads it at runtime, so only plain C types cross the
// boundary. Data is host-staged: handles are host float arrays; the library
// copies inputs into buffers it owns, runs on the device, and writes outputs
// back before returning (no SYCL object is shared with ArrayFire, whose
// queue and context are not exposed). The DLL binds the same SYCL runtime
// (sycl8.dll) as ArrayFire; the loader loads it only after ArrayFire's
// oneAPI backend.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef CYXWIZ_ONEAPI_KERNELS_BUILD
#define CYXWIZ_ONEAPI_KERNELS_API __declspec(dllexport)
#else
#define CYXWIZ_ONEAPI_KERNELS_API __declspec(dllimport)
#endif
#else
#define CYXWIZ_ONEAPI_KERNELS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Bumped on any change to the structs or entry points below.
#define CYXWIZ_ONEAPI_KERNELS_ABI 2

// Host float array of `elements` values (inputs read, outputs written).
typedef struct CyxOneapiBuffer {
    void* handle;
    size_t elements;
} CyxOneapiBuffer;

// Selects the SYCL device ArrayFire is running on (ArrayFire exposes no
// queue): matched by device name, and by type when is_gpu is 0 (CPU) or 1
// (GPU; -1 = any); the platform name (may be null) breaks ties between
// devices of the same name.
typedef struct CyxOneapiDevice {
    const char* name;
    const char* platform;
    int is_gpu;
} CyxOneapiDevice;

// Same contract as the CUDA/OpenCL tenants (neural_provider.h):
// Q [D,Sq,B,H], K/V [D,Sk,B,KVH], slopes [H] (ALiBi, optional),
// forward -> O [D,Sq,B,H], LSE [Sq,B,H];
// backward inputs Q,K,V,O,dO,LSE -> dQ, dK, dV, delta [Sq,B,H].
typedef struct CyxOneapiAttentionArgs {
    CyxOneapiBuffer q, k, v, o, d_o, lse, slopes;  // slopes.handle may be null
    CyxOneapiBuffer d_q, d_k, d_v, delta;          // backward outputs
    int batch, heads, kv_heads, seq, kv_seq, head_dim, query_offset;
    int causal, window;
    float softcap, scale, dropout;
    uint64_t seed;
} CyxOneapiAttentionArgs;

// All entry points return 0 on success, otherwise non-zero with a message
// in error (always NUL-terminated when error_size > 0). They block until the
// outputs are written.
CYXWIZ_ONEAPI_KERNELS_API int cyxwiz_oneapi_kernels_abi(void);
CYXWIZ_ONEAPI_KERNELS_API int cyxwiz_oneapi_runtime_info(char* out, size_t out_size);
CYXWIZ_ONEAPI_KERNELS_API int cyxwiz_oneapi_device_probe(const CyxOneapiDevice* device, CyxOneapiBuffer x,
                                                          CyxOneapiBuffer y, char* error, size_t error_size);
CYXWIZ_ONEAPI_KERNELS_API int cyxwiz_oneapi_attention_forward(const CyxOneapiDevice* device,
                                                               const CyxOneapiAttentionArgs* args, char* error,
                                                               size_t error_size);
CYXWIZ_ONEAPI_KERNELS_API int cyxwiz_oneapi_attention_backward(const CyxOneapiDevice* device,
                                                                const CyxOneapiAttentionArgs* args, char* error,
                                                                size_t error_size);

#ifdef __cplusplus
}
#endif
