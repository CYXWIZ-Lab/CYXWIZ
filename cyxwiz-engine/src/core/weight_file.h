#pragma once

// Reads the tensors of a published weight file without PyTorch:
//   - a PyTorch state-dict (.pth / .pt) as torch.save writes it since 1.6: a
//     ZIP with one pickle and one raw file per tensor;
//   - a .safetensors file.
// The pickle is read as data only. The reader knows the few opcodes and the
// few constructors a state-dict uses (ordered dict, tensor rebuild, storage
// reference); it never imports or calls anything the file names, and any
// other object in the file is refused with the reason.

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::weight_file {

struct Entry {
    std::string name;            // the key in the file, e.g. "layer1.0.conv1.weight"
    std::vector<size_t> shape;   // as stored (PyTorch layouts: Conv [out, in, kh, kw], Linear [out, in])
    std::string dtype;           // F32, F64, F16, BF16, I64, I32, I16, I8, U8, BOOL
    // Row-major values as float32 for the float kinds (F32, F64, F16); empty
    // for the others, which no layer weight uses (num_batches_tracked is I64).
    std::vector<float> values;
    bool is_float = false;
};

struct WeightFile {
    std::string format;          // "PyTorch state-dict" or "safetensors"
    std::vector<Entry> entries;  // in file order
    std::map<std::string, std::string> metadata;  // safetensors "__metadata__" text values
};

// False with the reason in `error` when the file cannot be used; `out` is
// then left empty.
bool Read(const std::filesystem::path& path, WeightFile& out, std::string& error);

}  // namespace cyxwiz::weight_file
