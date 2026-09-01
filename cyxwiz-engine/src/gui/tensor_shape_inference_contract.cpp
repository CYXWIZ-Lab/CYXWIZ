#include "tensor_shape_inference_contract.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace gui::tensor_shape_inference {
namespace {

bool CheckedMultiply(size_t left, size_t right, size_t& product) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

std::optional<std::vector<int64_t>> ParseRequestedShape(
    std::string_view requested_shape) {
    std::string compact;
    compact.reserve(requested_shape.size());
    for (const char character : requested_shape) {
        if (!std::isspace(static_cast<unsigned char>(character))) {
            compact.push_back(character);
        }
    }

    if (compact.size() >= 2 && compact.front() == '[' &&
        compact.back() == ']') {
        compact = compact.substr(1, compact.size() - 2);
    }
    if (compact.empty() || compact.find_first_of("[]") != std::string::npos) {
        return std::nullopt;
    }

    std::vector<int64_t> dimensions;
    size_t token_start = 0;
    while (token_start <= compact.size()) {
        const size_t separator = compact.find(',', token_start);
        const size_t token_end = separator == std::string::npos
            ? compact.size()
            : separator;
        if (token_end == token_start) {
            return std::nullopt;
        }

        int64_t dimension = 0;
        const char* begin = compact.data() + token_start;
        const char* end = compact.data() + token_end;
        const auto parsed = std::from_chars(begin, end, dimension);
        if (parsed.ec != std::errc{} || parsed.ptr != end ||
            dimension == 0 || dimension < -1) {
            return std::nullopt;
        }
        dimensions.push_back(dimension);

        if (separator == std::string::npos) {
            break;
        }
        token_start = separator + 1;
    }
    return dimensions;
}

}  // namespace

std::optional<std::vector<size_t>> InferPool2DOutputShape(
    const std::vector<size_t>& input_shape,
    int pool_size,
    int stride) {
    if (input_shape.size() < 3 || pool_size <= 0 || stride <= 0) {
        return std::nullopt;
    }

    const size_t kernel = static_cast<size_t>(pool_size);
    const size_t step = static_cast<size_t>(stride);
    if (input_shape[0] < kernel || input_shape[1] < kernel) {
        return std::nullopt;
    }

    return std::vector<size_t>{
        (input_shape[0] - kernel) / step + 1,
        (input_shape[1] - kernel) / step + 1,
        input_shape[2]};
}

std::optional<std::vector<size_t>> ResolveReshapeOutputShape(
    const std::vector<size_t>& input_shape,
    std::string_view requested_shape) {
    if (input_shape.empty()) {
        return std::nullopt;
    }

    size_t input_elements = 1;
    for (const size_t dimension : input_shape) {
        if (!CheckedMultiply(input_elements, dimension, input_elements)) {
            return std::nullopt;
        }
    }

    const auto parsed_dimensions = ParseRequestedShape(requested_shape);
    if (!parsed_dimensions) {
        return std::nullopt;
    }

    size_t known_elements = 1;
    size_t inferred_index = 0;
    bool has_inferred_dimension = false;
    std::vector<size_t> output_shape;
    output_shape.reserve(parsed_dimensions->size());

    for (const int64_t dimension : *parsed_dimensions) {
        if (dimension == -1) {
            if (has_inferred_dimension) {
                return std::nullopt;
            }
            has_inferred_dimension = true;
            inferred_index = output_shape.size();
            output_shape.push_back(0);
            continue;
        }

        const uint64_t unsigned_dimension = static_cast<uint64_t>(dimension);
        if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
            if (unsigned_dimension > std::numeric_limits<size_t>::max()) {
                return std::nullopt;
            }
        }
        const size_t target_dimension = static_cast<size_t>(unsigned_dimension);
        if (!CheckedMultiply(
                known_elements, target_dimension, known_elements)) {
            return std::nullopt;
        }
        output_shape.push_back(target_dimension);
    }

    if (has_inferred_dimension) {
        if (known_elements == 0 || input_elements % known_elements != 0) {
            return std::nullopt;
        }
        output_shape[inferred_index] = input_elements / known_elements;
    } else if (known_elements != input_elements) {
        return std::nullopt;
    }

    return output_shape;
}

}  // namespace gui::tensor_shape_inference
