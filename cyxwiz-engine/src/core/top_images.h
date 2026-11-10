#pragma once

// Top images (TOFIX178 6.3e, approved mockup boards Main and TopImages): the
// items from the data that excite a layer's channels most. Items run through
// the run's rebuilt model (DebugModelSession) in chunks on the device; the
// host gets one strength per channel and item.
//
// Items are in the model's input layout: images [H, W, C, N] (batch last),
// rows or tokens [N, F] (batch first). A layer's output is spatial
// [H, W, C, N] or flat [N, C].

#include "debug_model_session.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace cyxwiz::top_images {

// A spatial channel's strength for an item: its largest value over the map
// (it fires somewhere) or its mean (it fires all over). A flat output's
// value is its strength either way.
enum class Rank { Maximum, Mean };

struct Strengths {
    size_t channels = 0;
    size_t items = 0;
    std::vector<float> values;  // values[channel * items + item]
    bool cancelled = false;
    float At(size_t channel, size_t item) const { return values[channel * items + item]; }
};

// One layer output's strengths (spatial [H, W, C, N] or flat [N, C]).
Strengths ChannelStrengths(const Tensor& layer_output, Rank rank);

// The `count` strongest items of a channel, strongest first.
std::vector<size_t> TopItems(const Strengths& strengths, size_t channel, size_t count);

// Items `begin` .. `end` - 1 of a batch in the model's input layout.
Tensor ItemRange(const Tensor& items, size_t begin, size_t end);

// Every item through the model, `chunk` at a time; the layer's strengths.
// `progress` gets (items done, items); `cancelled` is polled between chunks.
Strengths Run(DebugModelSession& session, const Tensor& items, size_t module_index, Rank rank, size_t chunk,
              const std::function<void(size_t, size_t)>& progress = {},
              const std::function<bool()>& cancelled = {});

// The channels a Dense output head votes with most: its weights [outputs,
// channels] row-major; for each output, the `per_output` channels with the
// largest weight towards it. One output (a logit through a sigmoid) votes
// class 1 with its positive weights and class 0 with its negative ones.
struct Vote {
    size_t channel = 0;
    int towards = 0;  // the class the weight pushes towards
    float weight = 0.0f;
};
std::vector<Vote> StrongestVotes(const std::vector<float>& weights, size_t outputs, size_t channels, size_t per_output);

}  // namespace cyxwiz::top_images
