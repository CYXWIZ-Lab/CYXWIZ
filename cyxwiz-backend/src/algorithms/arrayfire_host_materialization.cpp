#include "arrayfire_host_materialization.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE

#include <arrayfire.h>

#include <stdexcept>
#include <utility>

namespace cyxwiz {
namespace {

std::string ArrayFireDtypeName(af::dtype type) {
    switch (type) {
        case f32: return "float32";
        case f64: return "float64";
        case c32: return "complex64";
        case c64: return "complex128";
        case b8: return "bool";
        case s32: return "int32";
        case u32: return "uint32";
        case u8: return "uint8";
        case s64: return "int64";
        case u64: return "uint64";
        case s16: return "int16";
        case u16: return "uint16";
        case f16: return "float16";
        case s8: return "int8";
        default: return "unknown";
    }
}

std::vector<size_t> ArrayFireShape(const af::array& source) {
    const af::dim4 dims = source.dims();
    const unsigned rank = source.numdims();
    std::vector<size_t> shape;
    shape.reserve(rank);
    for (unsigned axis = 0; axis < rank; ++axis) {
        shape.push_back(static_cast<size_t>(dims[axis]));
    }
    return shape;
}

} // namespace

void MaterializeArrayFireToHost(
    const af::array& source,
    void* destination,
    ArrayFireHostSyncCategory category,
    std::string operation_name,
    std::string layout,
    std::string reason_code,
    std::string context) {
    if (category == ArrayFireHostSyncCategory::Unknown) {
        throw std::invalid_argument(
            "ArrayFire host materialization requires a named category");
    }
    if (operation_name.empty() || layout.empty() || reason_code.empty()) {
        throw std::invalid_argument(
            "ArrayFire host materialization attribution is incomplete");
    }
    if (source.bytes() > 0 && destination == nullptr) {
        throw std::invalid_argument(
            "ArrayFire host materialization destination is null");
    }

    source.eval();
    source.host(destination);

    ArrayFireHostSyncEvent event;
    event.operation_name = "af::array::host";
    event.reason_code = std::move(reason_code);
    event.attribution_category = ArrayFireHostSyncCategoryName(category);
    event.attribution_operation = std::move(operation_name);
    event.tensor_shape = ArrayFireShape(source);
    event.tensor_dtype = ArrayFireDtypeName(source.type());
    event.tensor_layout = std::move(layout);
    event.context = std::move(context);
    event.bytes = static_cast<uint64_t>(source.bytes());
    NotifyArrayFireHostSync(std::move(event));
}

} // namespace cyxwiz

#endif
