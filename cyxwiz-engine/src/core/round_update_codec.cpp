#include "round_update_codec.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace cyxwiz {

namespace {

// Float32 segments are copied as they are in memory.
static_assert(std::endian::native == std::endian::little, "the update wire format is little-endian");

constexpr double kInt8Sigmas = 6.0;

bool Exchanged(const StateSegment& segment) { return segment.role != ModelStateRole::Frozen; }

// The reference's quantize_int8: mean, standard deviation and bucket means
// are summed in double and stored as float; the codes are computed in float.
void QuantizeInt8(std::span<const float> values, uint8_t* codes, Int8Table& table) {
    const double count = static_cast<double>(values.size());
    double sum = 0.0;
    for (const float value : values) sum += value;
    const double mean = sum / count;
    double squares = 0.0;
    for (const float value : values) {
        const double distance = value - mean;
        squares += distance * distance;
    }
    const double deviation = std::sqrt(squares / count);
    const float low = static_cast<float>(mean - kInt8Sigmas * deviation);
    const float high = static_cast<float>(mean + kInt8Sigmas * deviation);
    if (!(high > low)) {
        // No spread: every code is 0 and decodes to the value itself.
        std::fill(codes, codes + values.size(), uint8_t{0});
        table.fill(static_cast<float>(mean));
        return;
    }

    const float range = high - low;
    const float top = static_cast<float>(kInt8Levels - 1);
    std::array<double, kInt8Levels> sums{};
    std::array<double, kInt8Levels> counts{};
    for (size_t i = 0; i < values.size(); ++i) {
        const float clipped = std::clamp(values[i], low, high);
        const float scaled = (clipped - low) / range * top;
        // Round half to even, as the reference does.
        const size_t code = static_cast<size_t>(std::clamp(std::nearbyint(scaled), 0.0f, top));
        codes[i] = static_cast<uint8_t>(code);
        sums[code] += clipped;
        counts[code] += 1.0;
    }
    for (size_t code = 0; code < kInt8Levels; ++code) {
        // An empty bucket decodes to its centre.
        table[code] = counts[code] > 0.0 ? static_cast<float>(sums[code] / counts[code])
                                         : low + range * static_cast<float>(code) / top;
    }
}

}  // namespace

bool SegmentTravelsAsInt8(const StateSegment& segment, UpdateWire wire) {
    return wire == UpdateWire::Int8 && segment.role == ModelStateRole::Trainable && segment.count >= kInt8MinValues;
}

size_t EncodedUpdateBytes(const ModelStateLayout& layout, UpdateWire wire) {
    size_t bytes = 0;
    for (const StateSegment& segment : layout.segments) {
        if (!Exchanged(segment)) continue;
        bytes += SegmentTravelsAsInt8(segment, wire) ? segment.count + sizeof(Int8Table) : segment.count * sizeof(float);
    }
    return bytes;
}

bool EncodeUpdate(std::span<const float> delta, const ModelStateLayout& layout, UpdateWire wire,
                  EncodedUpdate& encoded, std::string& error) {
    encoded = {};
    error.clear();
    if (delta.size() != layout.exchanged_count) {
        error = "the update has " + std::to_string(delta.size()) + " values, the layout needs " +
                std::to_string(layout.exchanged_count);
        return false;
    }
    size_t table_count = 0;
    for (const StateSegment& segment : layout.segments) {
        if (!Exchanged(segment)) continue;
        const float* first = delta.data() + segment.exchanged_offset;
        if (!std::all_of(first, first + segment.count, [](float value) { return std::isfinite(value); })) {
            error = "the update of " + segment.name + " holds a value that is not a finite number";
            return false;
        }
        if (SegmentTravelsAsInt8(segment, wire)) ++table_count;
    }

    encoded.bytes.resize(EncodedUpdateBytes(layout, wire) - table_count * sizeof(Int8Table));
    encoded.tables.reserve(table_count);
    uint8_t* out = encoded.bytes.data();
    for (const StateSegment& segment : layout.segments) {
        if (!Exchanged(segment)) continue;
        const std::span<const float> values = delta.subspan(segment.exchanged_offset, segment.count);
        if (SegmentTravelsAsInt8(segment, wire)) {
            QuantizeInt8(values, out, encoded.tables.emplace_back());
            out += segment.count;
        } else if (segment.count > 0) {
            std::memcpy(out, values.data(), values.size_bytes());
            out += values.size_bytes();
        }
    }
    return true;
}

bool DecodeUpdate(const EncodedUpdate& encoded, const ModelStateLayout& layout, UpdateWire wire,
                  std::vector<float>& delta, std::string& error) {
    delta.clear();
    error.clear();
    size_t table_count = 0;
    for (const StateSegment& segment : layout.segments) {
        if (Exchanged(segment) && SegmentTravelsAsInt8(segment, wire)) ++table_count;
    }
    const size_t payload_bytes = EncodedUpdateBytes(layout, wire) - table_count * sizeof(Int8Table);
    if (encoded.bytes.size() != payload_bytes || encoded.tables.size() != table_count) {
        error = "the update has " + std::to_string(encoded.bytes.size()) + " bytes and " +
                std::to_string(encoded.tables.size()) + " tables, the layout needs " + std::to_string(payload_bytes) +
                " and " + std::to_string(table_count);
        return false;
    }

    std::vector<float> values(layout.exchanged_count);
    const uint8_t* in = encoded.bytes.data();
    size_t table_index = 0;
    for (const StateSegment& segment : layout.segments) {
        if (!Exchanged(segment)) continue;
        float* out = values.data() + segment.exchanged_offset;
        if (SegmentTravelsAsInt8(segment, wire)) {
            const Int8Table& table = encoded.tables[table_index++];
            if (!std::all_of(table.begin(), table.end(), [](float value) { return std::isfinite(value); })) {
                error = "the table of " + segment.name + " holds a value that is not a finite number";
                return false;
            }
            for (size_t i = 0; i < segment.count; ++i) out[i] = table[in[i]];
            in += segment.count;
        } else {
            if (segment.count > 0) std::memcpy(out, in, segment.count * sizeof(float));
            in += segment.count * sizeof(float);
            if (!std::all_of(out, out + segment.count, [](float value) { return std::isfinite(value); })) {
                error = "the update of " + segment.name + " holds a value that is not a finite number";
                return false;
            }
        }
    }
    delta = std::move(values);
    return true;
}

}  // namespace cyxwiz
