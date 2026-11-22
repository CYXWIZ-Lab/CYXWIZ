#pragma once

// A worker's update of one training round on the wire (group training).
// The update covers the exchanged state of a ModelStateLayout: every segment
// that is not frozen, in layout order. Pure host memory.
//
// The format is the PyTorch reference's
// (tests/computation_truth/reference/round_training_reference.py):
// - Float32 wire: every segment as little-endian float32.
// - Int8 wire: a trainable segment of at least kInt8MinValues values travels
//   as one code per value plus a 256-entry decode table (values clipped to
//   mean +- 6 standard deviations, the range cut into 256 equal buckets, a
//   code decoded as the mean of its bucket). Smaller segments and statistics
//   stay float32: a table costs 1 KiB, and a rounded variance can turn
//   negative.

#include "model_state_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cyxwiz {

enum class UpdateWire { Float32, Int8 };

inline constexpr size_t kInt8Levels = 256;
inline constexpr size_t kInt8MinValues = 1024;

using Int8Table = std::array<float, kInt8Levels>;

struct EncodedUpdate {
    std::vector<uint8_t> bytes;     // the segments' payloads, in layout order
    std::vector<Int8Table> tables;  // one per int8 segment, in layout order
};

// True when `segment` travels as codes on this wire.
bool SegmentTravelsAsInt8(const StateSegment& segment, UpdateWire wire);

// Bytes an update of this layout takes, tables included.
size_t EncodedUpdateBytes(const ModelStateLayout& layout, UpdateWire wire);

// `delta` holds layout.exchanged_count values. An update with a value that is
// not a finite number is refused: the worker reports a failure instead.
bool EncodeUpdate(std::span<const float> delta, const ModelStateLayout& layout, UpdateWire wire,
                  EncodedUpdate& encoded, std::string& error);

// The receiver's side. Refused: a size that does not match the layout, and a
// value or a table entry that is not a finite number.
bool DecodeUpdate(const EncodedUpdate& encoded, const ModelStateLayout& layout, UpdateWire wire,
                  std::vector<float>& delta, std::string& error);

}  // namespace cyxwiz
