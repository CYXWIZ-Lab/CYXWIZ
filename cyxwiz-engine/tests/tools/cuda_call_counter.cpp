// A CUDA call counter for any process (TOFIX206 3.2): CUDA loads this
// library at start when CUDA_INJECTION64_PATH names it, and it counts the
// driver and runtime API calls that launch kernels, wait for the device, or
// copy memory (CUPTI's callback API, which needs no administrator rights,
// unlike Nsight Systems on Windows). At exit it writes one summary to stderr
// and, when CYXWIZ_CUDA_COUNT_OUT names a file, appends it there:
//
//   cuda calls over 12.3 s: launches 1,234 (100.3/s), waits 56 (4.6/s),
//   copies 78 (6.3/s); top: cuLaunchKernel 1,000, cudaLaunchKernel 234, ...
//
// A development tool, not part of the Engine.

#include <cupti.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum class Kind : unsigned char { Unknown, Launch, Wait, Copy, Other };

constexpr int kMaxIds = 2048;
struct Domain {
    std::atomic<long long> calls[kMaxIds] = {};
    std::atomic<unsigned char> kind[kMaxIds] = {};  // 0 = not looked up yet
};

Domain driver, runtime;
CUpti_SubscriberHandle subscriber = nullptr;
const auto started = std::chrono::steady_clock::now();
std::atomic<bool> reported{false};

Kind KindOf(const char* name) {
    if (!name) return Kind::Other;
    const std::string n(name);
    if (n.find("Launch") != std::string::npos) return Kind::Launch;
    if (n.find("Synchronize") != std::string::npos) return Kind::Wait;
    if (n.find("Memcpy") != std::string::npos || n.find("memcpy") != std::string::npos) return Kind::Copy;
    return Kind::Other;
}

void CUPTIAPI OnCall(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (info->callbackSite != CUPTI_API_ENTER || id >= static_cast<CUpti_CallbackId>(kMaxIds)) return;
    Domain& d = domain == CUPTI_CB_DOMAIN_DRIVER_API ? driver : runtime;
    if (d.kind[id].load(std::memory_order_relaxed) == 0) {
        const char* name = nullptr;
        cuptiGetCallbackName(domain, id, &name);
        d.kind[id].store(static_cast<unsigned char>(KindOf(name)), std::memory_order_relaxed);
    }
    d.calls[id].fetch_add(1, std::memory_order_relaxed);
}

std::string Thousands(long long n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

void Report() {
    if (reported.exchange(true)) return;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    long long totals[5] = {};
    std::vector<std::pair<long long, std::string>> top;
    for (const auto [domain, d] : {std::pair{CUPTI_CB_DOMAIN_DRIVER_API, &driver},
                                   std::pair{CUPTI_CB_DOMAIN_RUNTIME_API, &runtime}}) {
        for (int id = 0; id < kMaxIds; ++id) {
            const long long n = d->calls[id].load();
            if (n == 0) continue;
            const auto kind = static_cast<Kind>(d->kind[id].load());
            totals[static_cast<int>(kind)] += n;
            if (kind == Kind::Launch || kind == Kind::Wait || kind == Kind::Copy) {
                const char* name = nullptr;
                cuptiGetCallbackName(domain, static_cast<CUpti_CallbackId>(id), &name);
                top.emplace_back(n, name ? name : "?");
            }
        }
    }
    std::sort(top.rbegin(), top.rend());
    const auto rate = [seconds](long long n) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.1f/s", seconds > 0.0 ? n / seconds : 0.0);
        return std::string(text);
    };
    char head[96];
    std::snprintf(head, sizeof(head), "cuda calls over %.1f s: ", seconds);
    std::string line = head;
    line += "launches " + Thousands(totals[1]) + " (" + rate(totals[1]) + "), waits " + Thousands(totals[2]) + " (" +
            rate(totals[2]) + "), copies " + Thousands(totals[3]) + " (" + rate(totals[3]) + "); top:";
    for (size_t i = 0; i < top.size() && i < 8; ++i)
        line += (i ? ", " : " ") + top[i].second + " " + Thousands(top[i].first);
    std::fprintf(stderr, "%s\n", line.c_str());
    if (const char* out = std::getenv("CYXWIZ_CUDA_COUNT_OUT")) {
        if (FILE* f = std::fopen(out, "a")) {
            std::fprintf(f, "%s\n", line.c_str());
            std::fclose(f);
        }
    }
}

}  // namespace

extern "C" __declspec(dllexport) int InitializeInjection(void) {
    if (cuptiSubscribe(&subscriber, OnCall, nullptr) != CUPTI_SUCCESS) return 0;
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    std::atexit(Report);
    return 1;
}
