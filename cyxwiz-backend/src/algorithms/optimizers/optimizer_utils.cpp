#include "optimizer_utils.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace optimizer_detail {

namespace {

bool s_use_gpu = false;
bool s_gpu_checked = false;

} // namespace

bool OptimizerGpuAvailable() {
    if (s_gpu_checked) {
        return s_use_gpu;
    }
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
    } catch (const af::exception&) {
        s_use_gpu = false;
    }
#endif

    return s_use_gpu;
}

} // namespace optimizer_detail
} // namespace cyxwiz
