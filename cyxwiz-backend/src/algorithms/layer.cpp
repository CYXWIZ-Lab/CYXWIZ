#include "cyxwiz/layer.h"
#include "cyxwiz/debug_hooks.h"
#include "cyxwiz/recurrent_cuda_placement.h"
#include "cyxwiz/tensor.h"
#include "layers/layer_utils.h"
#include "layers/layer_arrayfire_utils.h"
#include "layers/layer_recurrent_utils.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>
#include <atomic>
#include <limits>
#include <string>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with ArrayFire functions
// Must be AFTER all includes (Windows headers define these)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

} // namespace cyxwiz
