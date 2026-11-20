// Feature values do not depend on how a table is chunked (TOFIX223 0.1, audit
// finding TAB-01 in TOFIX222): for every numeric column type the batcher
// accepts, a table in one chunk and the same table in seven uneven chunks (one
// of them empty) give batches equal to the source values cast to float. The
// expected values come from the source vectors, not from the batcher.
//
// A CSV over 64 MB loads as one chunk per 64 MB block, and integer compaction
// turns columns into int8 / int16 / uint16 / uint32; the multi-chunk read once
// handled five types only and returned 0 for the others.
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "core/arrow_dataset.h"
#include "core/dataset_batcher.h"
#include "computation_truth/test_device_selection.h"

#include <arrow/api.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

constexpr int64_t kRows = 301;

template <typename Builder, typename T>
std::shared_ptr<arrow::Array> MakeArray(const std::vector<T>& values, int64_t begin, int64_t end) {
    Builder builder;
    for (int64_t i = begin; i < end; ++i) (void)builder.Append(values[static_cast<size_t>(i)]);
    std::shared_ptr<arrow::Array> out;
    (void)builder.Finish(&out);
    return out;
}

// Chunk boundaries: one chunk, or seven uneven chunks with an empty one.
std::vector<int64_t> Bounds(bool many) {
    if (!many) return {0, kRows};
    return {0, 1, 50, 50, 51, 170, 299, kRows};
}

template <typename Builder, typename T>
std::shared_ptr<arrow::ChunkedArray> MakeColumn(const std::vector<T>& values, bool many,
                                                const std::shared_ptr<arrow::DataType>& type) {
    const auto bounds = Bounds(many);
    std::vector<std::shared_ptr<arrow::Array>> chunks;
    for (size_t c = 0; c + 1 < bounds.size(); ++c)
        chunks.push_back(MakeArray<Builder>(values, bounds[c], bounds[c + 1]));
    return std::make_shared<arrow::ChunkedArray>(chunks, type);
}

// Two feature columns of type T (so the worker threads split them) and a label.
template <typename Builder, typename T>
void CheckType(const std::string& type_name, const std::shared_ptr<arrow::DataType>& type,
               const std::vector<T>& first, const std::vector<T>& second) {
    std::vector<int64_t> label(kRows);
    for (int64_t r = 0; r < kRows; ++r) label[static_cast<size_t>(r)] = r % 3;
    for (const bool many : {false, true}) {
        auto schema = arrow::schema({arrow::field("f0", type), arrow::field("f1", type),
                                     arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {MakeColumn<Builder>(first, many, type),
                                                 MakeColumn<Builder>(second, many, type),
                                                 MakeColumn<arrow::Int64Builder>(label, many, arrow::int64())});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "chunking_" + type_name);
        for (const bool resident : {false, true}) {
            for (const int workers : {0, 4}) {
                const std::string name = type_name + (many ? ", 7 chunks" : ", 1 chunk") +
                                         (resident ? ", resident" : ", host") +
                                         ", workers " + std::to_string(workers);
                cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 37, false, 1.0f, true, "", 0, workers);
                batcher.SetDeviceResidentFeatures(resident);
                batcher.Reset();
                int64_t seen = 0;
                size_t wrong = 0;
                while (!batcher.IsEpochComplete()) {
                    const cyxwiz::Batch batch = batcher.GetNextBatch();
                    Check(batch.size > 0 && batch.data.Shape() == std::vector<size_t>({batch.size, 2}),
                          name + ": batch shape");
                    if (batch.size == 0 || batch.data.Shape().size() != 2) break;
                    const float* data = batch.data.ReadData<float>();
                    for (size_t b = 0; b < batch.size; ++b) {
                        const auto row = static_cast<size_t>(batch.row_ids[b]);
                        if (data[b * 2] != static_cast<float>(first[row])) ++wrong;
                        if (data[b * 2 + 1] != static_cast<float>(second[row])) ++wrong;
                    }
                    seen += static_cast<int64_t>(batch.size);
                }
                Check(seen == kRows, name + ": every row once (saw " + std::to_string(seen) + ")");
                Check(wrong == 0, name + ": " + std::to_string(wrong) + " values differ from the source");
            }
        }
    }
}

template <typename T>
std::vector<T> Values(int64_t low, int64_t high, int64_t step) {
    std::vector<T> out(kRows);
    const int64_t span = high - low + 1;
    for (int64_t r = 0; r < kRows; ++r) out[static_cast<size_t>(r)] = static_cast<T>(low + (r * step) % span);
    return out;
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) {
        std::cerr << "the requested device could not be activated\n";
        return 1;
    }
    // Ranges reach each type's limits; every value is exact in float32 except
    // the wide ones, which are compared after the same cast.
    CheckType<arrow::Int8Builder, int8_t>("int8", arrow::int8(), Values<int8_t>(-128, 127, 7), Values<int8_t>(-50, 50, 3));
    CheckType<arrow::Int16Builder, int16_t>("int16", arrow::int16(), Values<int16_t>(-32768, 32767, 977),
                                            Values<int16_t>(-300, 300, 13));
    CheckType<arrow::Int32Builder, int32_t>("int32", arrow::int32(), Values<int32_t>(-2000000, 2000000, 104729),
                                            Values<int32_t>(-7, 7, 1));
    CheckType<arrow::Int64Builder, int64_t>("int64", arrow::int64(), Values<int64_t>(-(int64_t{1} << 40), int64_t{1} << 40, 7919),
                                            Values<int64_t>(0, 40000, 131));
    CheckType<arrow::UInt8Builder, uint8_t>("uint8", arrow::uint8(), Values<uint8_t>(0, 255, 11), Values<uint8_t>(0, 9, 1));
    CheckType<arrow::UInt16Builder, uint16_t>("uint16", arrow::uint16(), Values<uint16_t>(0, 65535, 2311),
                                              Values<uint16_t>(256, 40000, 97));
    CheckType<arrow::UInt32Builder, uint32_t>("uint32", arrow::uint32(), Values<uint32_t>(0, 4294967295LL, 15485863),
                                              Values<uint32_t>(70000, 90000, 17));
    CheckType<arrow::UInt64Builder, uint64_t>("uint64", arrow::uint64(), Values<uint64_t>(0, int64_t{1} << 50, 1000003),
                                              Values<uint64_t>(5, 500, 3));
    CheckType<arrow::FloatBuilder, float>("float32", arrow::float32(), Values<float>(-1000, 1000, 37),
                                          Values<float>(0, 3, 1));
    CheckType<arrow::DoubleBuilder, double>("float64", arrow::float64(), Values<double>(-123456, 123456, 811),
                                            Values<double>(-1, 1, 1));

    if (failures == 0) std::cout << "test_arrow_batcher_chunking: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
