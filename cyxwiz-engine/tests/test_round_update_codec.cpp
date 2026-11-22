// The update codec against the PyTorch reference's fixtures
// (tests/computation_truth/reference/generate_round_training_fixtures.py).

#include "../src/core/round_update_codec.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using cyxwiz::EncodedUpdate;
using cyxwiz::ModelStateLayout;
using cyxwiz::ModelStateRole;
using cyxwiz::StateSegment;
using cyxwiz::UpdateWire;
using nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::vector<float> Floats(const json& tensor) { return tensor.at("values").get<std::vector<float>>(); }

bool Near(float actual, float expected) {
    return std::abs(actual - expected) <= 1e-6f * std::max(1.0f, std::abs(expected));
}

void Add(ModelStateLayout& layout, const std::string& name, size_t count, ModelStateRole role) {
    StateSegment segment;
    segment.name = name;
    segment.shape = {count};
    segment.role = role;
    segment.count = count;
    segment.offset = layout.total_count;
    layout.total_count += count;
    if (role != ModelStateRole::Frozen) {
        segment.exchanged_offset = layout.exchanged_count;
        layout.exchanged_count += count;
    }
    layout.segments.push_back(std::move(segment));
}

// The fixture's layout, with a frozen segment put in the middle: it is not
// part of an update and must not move anything.
ModelStateLayout LayoutOf(const json& segments) {
    ModelStateLayout layout;
    for (const json& segment : segments) {
        const bool statistic = segment.at("kind").get<std::string>() == "statistic";
        Add(layout, segment.at("name").get<std::string>(), segment.at("count").get<size_t>(),
            statistic ? ModelStateRole::Statistic : ModelStateRole::Trainable);
        if (layout.segments.size() == 2) Add(layout, "layer1.weight", 4096, ModelStateRole::Frozen);
    }
    return layout;
}

}  // namespace

int main() {
    std::ifstream file(CYXWIZ_ROUND_TRAINING_FIXTURE);
    Check(file.good(), std::string("the fixture should open: ") + CYXWIZ_ROUND_TRAINING_FIXTURE);
    const json fixture = json::parse(file);
    Check(fixture.at("schema_version").get<int>() == 2, "the fixture should be schema 2; regenerate it");
    const json& update = fixture.at("update");
    Check(update.at("int8_min_values").get<size_t>() == cyxwiz::kInt8MinValues,
          "the reference and the codec should agree on the smallest int8 segment");

    const ModelStateLayout layout = LayoutOf(update.at("segments"));
    const std::vector<float> delta = Floats(update.at("delta"));
    const std::vector<float> expected = Floats(update.at("decoded"));
    std::string error;

    // Int8 wire: codes exactly, tables and decoded values as the reference.
    {
        Check(cyxwiz::EncodedUpdateBytes(layout, UpdateWire::Int8) == update.at("wire_bytes").get<size_t>(),
              "the int8 update should take the reference's bytes");
        EncodedUpdate encoded;
        Check(cyxwiz::EncodeUpdate(delta, layout, UpdateWire::Int8, encoded, error), "the update should encode: " + error);
        Check(encoded.bytes.size() + encoded.tables.size() * sizeof(cyxwiz::Int8Table) ==
                  update.at("wire_bytes").get<size_t>(),
              "the encoded update should have the size EncodedUpdateBytes says");

        const uint8_t* bytes = encoded.bytes.data();
        size_t table_index = 0;
        size_t segment_index = 0;
        for (const json& segment : update.at("segments")) {
            const std::string name = segment.at("name").get<std::string>();
            const size_t count = segment.at("count").get<size_t>();
            const StateSegment& described = layout.segments[segment_index < 2 ? segment_index : segment_index + 1];
            ++segment_index;
            const bool int8 = segment.at("encoding").get<std::string>() == "int8";
            Check(cyxwiz::SegmentTravelsAsInt8(described, UpdateWire::Int8) == int8, name + " should travel as the reference says");
            if (int8) {
                const std::vector<int> codes = segment.at("codes").get<std::vector<int>>();
                size_t different = 0;
                for (size_t i = 0; i < count; ++i) different += bytes[i] != codes[i] ? 1 : 0;
                Check(different == 0, name + ": " + std::to_string(different) + " codes differ from the reference");
                const std::vector<float> table = Floats(segment.at("table"));
                Check(table_index < encoded.tables.size(), name + " should have a table");
                for (size_t i = 0; i < table.size(); ++i) {
                    Check(Near(encoded.tables[table_index][i], table[i]), name + ": table entry " + std::to_string(i) + " differs");
                }
                ++table_index;
                bytes += count;
            } else {
                Check(std::memcmp(bytes, delta.data() + described.exchanged_offset, count * sizeof(float)) == 0,
                      name + " should travel as its float32 values");
                bytes += count * sizeof(float);
            }
        }
        Check(table_index == encoded.tables.size(), "there should be one table per int8 segment");

        std::vector<float> decoded;
        Check(cyxwiz::DecodeUpdate(encoded, layout, UpdateWire::Int8, decoded, error), "the update should decode: " + error);
        Check(decoded.size() == expected.size(), "the decoded update should have the layout's length");
        for (size_t i = 0; i < decoded.size(); ++i) {
            Check(Near(decoded[i], expected[i]), "decoded value " + std::to_string(i) + " differs from the reference");
        }
        // The segment with no spread comes back exactly.
        const StateSegment& flat = layout.segments.back();
        for (size_t i = 0; i < flat.count; ++i) {
            Check(decoded[flat.exchanged_offset + i] == delta[flat.exchanged_offset + i], "a segment with no spread should decode exactly");
        }

        // What the receiver must refuse.
        EncodedUpdate broken = encoded;
        broken.bytes.pop_back();
        Check(!cyxwiz::DecodeUpdate(broken, layout, UpdateWire::Int8, decoded, error) && decoded.empty(),
              "an update of the wrong size should be refused");
        broken = encoded;
        broken.tables.pop_back();
        Check(!cyxwiz::DecodeUpdate(broken, layout, UpdateWire::Int8, decoded, error), "a missing table should be refused");
        broken = encoded;
        broken.tables[0][17] = std::numeric_limits<float>::quiet_NaN();
        Check(!cyxwiz::DecodeUpdate(broken, layout, UpdateWire::Int8, decoded, error) &&
                  error.find(layout.segments[0].name) != std::string::npos,
              "a table that is not finite should be refused and name its segment: " + error);
        broken = encoded;
        const float infinity = std::numeric_limits<float>::infinity();
        std::memcpy(broken.bytes.data() + layout.segments[0].count, &infinity, sizeof(float));  // first value of the float32 segment
        Check(!cyxwiz::DecodeUpdate(broken, layout, UpdateWire::Int8, decoded, error) &&
                  error.find(layout.segments[1].name) != std::string::npos,
              "a float32 value that is not finite should be refused and name its segment: " + error);
        Check(!cyxwiz::DecodeUpdate(encoded, layout, UpdateWire::Float32, decoded, error),
              "an int8 update should not decode as float32");
    }

    // Float32 wire: the update itself.
    {
        EncodedUpdate encoded;
        Check(cyxwiz::EncodeUpdate(delta, layout, UpdateWire::Float32, encoded, error), "the float32 update should encode: " + error);
        Check(encoded.tables.empty() && encoded.bytes.size() == update.at("wire_bytes_float32").get<size_t>(),
              "the float32 update should take 4 bytes a value and no table");
        std::vector<float> decoded;
        Check(cyxwiz::DecodeUpdate(encoded, layout, UpdateWire::Float32, decoded, error) && decoded == delta,
              "a float32 update should decode to itself");
    }

    // What the sender must refuse.
    {
        EncodedUpdate encoded;
        std::vector<float> short_delta(delta.begin(), delta.end() - 1);
        Check(!cyxwiz::EncodeUpdate(short_delta, layout, UpdateWire::Int8, encoded, error), "an update of the wrong length should be refused");
        std::vector<float> not_finite = delta;
        not_finite[layout.segments.back().exchanged_offset + 3] = std::numeric_limits<float>::quiet_NaN();
        Check(!cyxwiz::EncodeUpdate(not_finite, layout, UpdateWire::Int8, encoded, error) &&
                  error.find(layout.segments.back().name) != std::string::npos && encoded.bytes.empty(),
              "an update that is not finite should be refused and name its segment: " + error);
    }

    std::cout << "round update codec: all checks passed\n";
    return 0;
}
