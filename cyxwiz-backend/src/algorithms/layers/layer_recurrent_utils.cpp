#include "layer_recurrent_utils.h"

#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::recurrent_detail {

void ThrowWithoutArrayFire(const char* layer) {
    throw std::runtime_error(std::string(layer) +
                             " runs on ArrayFire, and this build has no ArrayFire");
}

namespace {

void RequireRank3(const Tensor& tensor, const char* operation) {
    if (tensor.Shape().size() != 3) {
        throw std::invalid_argument(std::string(operation) + " expects a rank-3 sequence tensor");
    }
}

}  // namespace

Tensor ReverseTime(const Tensor& sequence, bool batch_first) {
    RequireRank3(sequence, "ReverseTime");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return Tensor::FromSemanticArray(af::flip(sequence.GetSemanticArray(), batch_first ? 1 : 0),
                                         sequence.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent time reversal failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    (void)batch_first;
    ThrowWithoutArrayFire("Recurrent time reversal");
#endif
}

Tensor JoinFeatures(const Tensor& first, const Tensor& second) {
    RequireRank3(first, "JoinFeatures");
    RequireRank3(second, "JoinFeatures");
    if (first.Shape()[0] != second.Shape()[0] || first.Shape()[1] != second.Shape()[1]) {
        throw std::invalid_argument("JoinFeatures: sequences differ outside the feature axis");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return Tensor::FromSemanticArray(
            af::join(2, first.GetSemanticArray(), second.GetSemanticArray()),
            {first.Shape()[0], first.Shape()[1], first.Shape()[2] + second.Shape()[2]});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent feature join failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent feature join");
#endif
}

Tensor SliceFeatures(const Tensor& sequence, size_t offset, size_t width) {
    RequireRank3(sequence, "SliceFeatures");
    const auto& shape = sequence.Shape();
    if (width == 0 || offset + width > shape[2]) {
        throw std::invalid_argument("SliceFeatures: feature range is outside the sequence");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array sliced = sequence.GetSemanticArray()(
            af::span, af::span,
            af::seq(static_cast<double>(offset), static_cast<double>(offset + width - 1)));
        return Tensor::FromSemanticArray(af::moddims(sliced, af::dim4(static_cast<dim_t>(shape[0]),
                                                                     static_cast<dim_t>(shape[1]),
                                                                     static_cast<dim_t>(width))),
                                         {shape[0], shape[1], width});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent feature slice failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent feature slice");
#endif
}

Tensor LastTimeStep(const Tensor& sequence) {
    RequireRank3(sequence, "LastTimeStep");
    const auto& shape = sequence.Shape();
    if (shape[1] == 0) {
        throw std::invalid_argument("LastTimeStep: the sequence has no time steps");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array last =
            sequence.GetSemanticArray()(af::span, static_cast<int>(shape[1] - 1), af::span);
        return Tensor::FromSemanticArray(
            af::moddims(last, af::dim4(static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[2]))),
            {shape[0], shape[2]});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent last-step slice failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent last-step slice");
#endif
}

Tensor ExpandLastTimeStep(const Tensor& last_step_gradient, size_t seq_len) {
    const auto& shape = last_step_gradient.Shape();
    if (shape.size() != 2 || seq_len == 0) {
        throw std::invalid_argument("ExpandLastTimeStep expects a [batch, features] gradient");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t batch = static_cast<dim_t>(shape[0]);
        const dim_t features = static_cast<dim_t>(shape[1]);
        const af::array last = af::moddims(last_step_gradient.GetSemanticArray(),
                                           af::dim4(batch, 1, features));
        af::array expanded = last;
        if (seq_len > 1) {
            expanded = af::join(1, af::constant(0.0f, af::dim4(batch, static_cast<dim_t>(seq_len - 1),
                                                               features)),
                                last);
        }
        return Tensor::FromSemanticArray(expanded, {shape[0], seq_len, shape[1]});
    } catch (const af::exception& e) {
        throw std::runtime_error(
            std::string("Recurrent last-step gradient expansion failed on the ArrayFire device: ") +
            e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent last-step gradient expansion");
#endif
}

Tensor MakeDropoutMask(const std::vector<size_t>& shape, float dropout) {
    if (!(dropout > 0.0f && dropout < 1.0f) || shape.size() != 3) {
        throw std::invalid_argument("MakeDropoutMask expects 0 < dropout < 1 and a rank-3 shape");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array keep =
            (af::randu(af::dim4(static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[1]),
                                static_cast<dim_t>(shape[2])),
                       f32) >= dropout)
                .as(f32);
        return Tensor::FromSemanticArray(keep / (1.0f - dropout), shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent dropout mask failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent dropout");
#endif
}

Tensor StateAt(const Tensor& states, size_t index, const char* what) {
    const auto& shape = states.Shape();
    if (shape.size() != 3 || index >= shape[0]) {
        throw std::invalid_argument(std::string(what) +
                                    " must be [layers * directions, batch, hidden]");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array state = states.GetSemanticArray()(static_cast<int>(index), af::span, af::span);
        return Tensor::FromSemanticArray(
            af::moddims(state, af::dim4(1, static_cast<dim_t>(shape[1]), static_cast<dim_t>(shape[2]))),
            {1, shape[1], shape[2]});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent state slice failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent state slice");
#endif
}

Tensor StackStates(const std::vector<Tensor>& states) {
    if (states.empty()) {
        throw std::invalid_argument("StackStates needs at least one state");
    }
    const auto& shape = states.front().Shape();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        size_t count = 0;
        af::array stacked;
        for (const Tensor& state : states) {
            if (state.Shape().size() != 3 || state.Shape()[1] != shape[1] ||
                state.Shape()[2] != shape[2]) {
                throw std::invalid_argument("StackStates: states differ in batch or hidden size");
            }
            const af::array values = state.GetSemanticArray();
            stacked = count == 0 ? values : af::join(0, stacked, values);
            count += state.Shape()[0];
        }
        return Tensor::FromSemanticArray(stacked, {count, shape[1], shape[2]});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Recurrent state stacking failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    ThrowWithoutArrayFire("Recurrent state stacking");
#endif
}

}  // namespace cyxwiz::recurrent_detail
