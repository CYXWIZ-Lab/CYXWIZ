#include "weight_file.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cyxwiz::weight_file {
namespace {

// Limits that keep a damaged or hostile file from asking for absurd memory.
constexpr uint64_t kMaxPickleBytes = 256ull << 20;       // the pickle of a state-dict is a few hundred KB
constexpr uint64_t kMaxSafetensorsHeader = 100ull << 20; // the format's own limit
constexpr uint64_t kMaxTensorBytes = 16ull << 30;
constexpr size_t kMaxZipEntries = 1u << 20;
constexpr size_t kMaxPickleObjects = 1u << 24;
constexpr size_t kMaxRank = 16;

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The bytes ran out inside a pickle: either the file is damaged or (old
// format) more of the file has to be read.
struct EndOfData : Failure {
    EndOfData() : Failure("the pickle ends early (the file is damaged)") {}
};

[[noreturn]] void Fail(const std::string& why) { throw Failure(why); }

uint64_t CheckedMul(uint64_t a, uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) Fail(std::string(what) + " is too large");
    return a * b;
}

// ---- element kinds ---------------------------------------------------------

struct Kind {
    const char* name;
    size_t bytes;
    bool is_float;  // a kind this reader converts to float32
};

const Kind* KindByName(const std::string& name) {
    static const Kind kinds[] = {
        {"F32", 4, true}, {"F64", 8, true}, {"F16", 2, true}, {"BF16", 2, false}, {"I64", 8, false},
        {"I32", 4, false}, {"I16", 2, false}, {"I8", 1, false}, {"U8", 1, false}, {"BOOL", 1, false},
        {"U16", 2, false}, {"U32", 4, false}, {"U64", 8, false},
    };
    for (const auto& kind : kinds) {
        if (name == kind.name) return &kind;
    }
    return nullptr;
}

// torch's storage class names (and the dtype names newer files use).
const Kind* KindByTorchName(const std::string& name) {
    static const std::pair<const char*, const char*> names[] = {
        {"FloatStorage", "F32"}, {"DoubleStorage", "F64"}, {"HalfStorage", "F16"}, {"BFloat16Storage", "BF16"},
        {"LongStorage", "I64"}, {"IntStorage", "I32"}, {"ShortStorage", "I16"}, {"CharStorage", "I8"},
        {"ByteStorage", "U8"}, {"BoolStorage", "BOOL"},
    };
    for (const auto& [torch_name, kind] : names) {
        if (name == torch_name) return KindByName(kind);
    }
    return nullptr;
}

float HalfToFloat(uint16_t half) {
    const uint32_t sign = (half >> 15) & 1u;
    const uint32_t exponent = (half >> 10) & 0x1Fu;
    const uint32_t fraction = half & 0x3FFu;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent == 31) {
        value = fraction == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(static_cast<float>(fraction | 0x400u), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

// One element at `bytes` (little-endian, as both formats store it) as float32.
float ElementAsFloat(const Kind& kind, const uint8_t* bytes) {
    if (kind.bytes == 4) {
        float value;
        std::memcpy(&value, bytes, 4);
        return value;
    }
    if (kind.bytes == 8) {
        double value;
        std::memcpy(&value, bytes, 8);
        return static_cast<float>(value);
    }
    uint16_t half;
    std::memcpy(&half, bytes, 2);
    return HalfToFloat(half);
}

// ---- file access -----------------------------------------------------------

class File {
public:
    explicit File(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        if (!stream_) Fail("the file cannot be opened");
        stream_.seekg(0, std::ios::end);
        size_ = static_cast<uint64_t>(stream_.tellg());
    }
    uint64_t Size() const { return size_; }
    void Read(uint64_t offset, void* out, uint64_t count) {
        if (offset > size_ || count > size_ - offset) Fail("the file ends before its contents do (truncated or damaged)");
        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(offset));
        stream_.read(static_cast<char*>(out), static_cast<std::streamsize>(count));
        if (static_cast<uint64_t>(stream_.gcount()) != count) Fail("the file could not be read to the end");
    }
    std::vector<uint8_t> ReadBytes(uint64_t offset, uint64_t count) {
        if (offset > size_ || count > size_ - offset) Fail("the file ends before its contents do (truncated or damaged)");
        std::vector<uint8_t> bytes(static_cast<size_t>(count));
        if (count) Read(offset, bytes.data(), count);
        return bytes;
    }

private:
    std::ifstream stream_;
    uint64_t size_ = 0;
};

template <typename T>
T Little(const uint8_t* bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;  // every supported build target is little-endian
}

// ---- ZIP (stored entries only; torch.save does not compress) ---------------

struct ZipEntry {
    uint16_t method = 0;
    uint64_t compressed = 0;
    uint64_t size = 0;
    uint64_t header_offset = 0;
};

std::map<std::string, ZipEntry> ReadZipDirectory(File& file) {
    // End of central directory: the last 22 bytes plus an optional comment.
    const uint64_t tail = std::min<uint64_t>(file.Size(), 22 + 65535);
    const auto end = file.ReadBytes(file.Size() - tail, tail);
    size_t at = end.size();
    for (size_t i = end.size() >= 22 ? end.size() - 22 + 1 : 0; i-- > 0;) {
        if (Little<uint32_t>(&end[i]) == 0x06054b50u) {
            at = i;
            break;
        }
    }
    if (at == end.size()) Fail("the ZIP directory was not found (the file is truncated or is not a weight file)");
    uint64_t count = Little<uint16_t>(&end[at + 10]);
    uint64_t directory_size = Little<uint32_t>(&end[at + 12]);
    uint64_t directory_offset = Little<uint32_t>(&end[at + 16]);
    if (count == 0xFFFFu || directory_size == 0xFFFFFFFFu || directory_offset == 0xFFFFFFFFu) {
        // ZIP64: the locator sits just before the end record.
        const uint64_t end_at = file.Size() - tail + at;
        if (end_at < 20) Fail("the ZIP64 directory locator is missing");
        const auto locator = file.ReadBytes(end_at - 20, 20);
        if (Little<uint32_t>(locator.data()) != 0x07064b50u) Fail("the ZIP64 directory locator is missing");
        const auto record = file.ReadBytes(Little<uint64_t>(&locator[8]), 56);
        if (Little<uint32_t>(record.data()) != 0x06064b50u) Fail("the ZIP64 directory record is damaged");
        count = Little<uint64_t>(&record[32]);
        directory_size = Little<uint64_t>(&record[40]);
        directory_offset = Little<uint64_t>(&record[48]);
    }
    if (count > kMaxZipEntries) Fail("the ZIP lists too many entries");
    const auto directory = file.ReadBytes(directory_offset, directory_size);
    std::map<std::string, ZipEntry> entries;
    size_t p = 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (p + 46 > directory.size() || Little<uint32_t>(&directory[p]) != 0x02014b50u) Fail("the ZIP directory is damaged");
        ZipEntry entry;
        entry.method = Little<uint16_t>(&directory[p + 10]);
        entry.compressed = Little<uint32_t>(&directory[p + 20]);
        entry.size = Little<uint32_t>(&directory[p + 24]);
        const size_t name_length = Little<uint16_t>(&directory[p + 28]);
        const size_t extra_length = Little<uint16_t>(&directory[p + 30]);
        const size_t comment_length = Little<uint16_t>(&directory[p + 32]);
        entry.header_offset = Little<uint32_t>(&directory[p + 42]);
        if (p + 46 + name_length + extra_length + comment_length > directory.size()) Fail("the ZIP directory is damaged");
        std::string name(reinterpret_cast<const char*>(&directory[p + 46]), name_length);
        // ZIP64 extra field: the values that did not fit, in this order.
        size_t e = p + 46 + name_length;
        const size_t extra_end = e + extra_length;
        while (e + 4 <= extra_end) {
            const uint16_t id = Little<uint16_t>(&directory[e]);
            const size_t length = Little<uint16_t>(&directory[e + 2]);
            size_t v = e + 4;
            if (v + length > extra_end) break;
            if (id == 0x0001u) {
                const auto take = [&](uint64_t& field) {
                    if (v + 8 > e + 4 + length) Fail("the ZIP64 entry sizes are damaged");
                    field = Little<uint64_t>(&directory[v]);
                    v += 8;
                };
                if (entry.size == 0xFFFFFFFFu) take(entry.size);
                if (entry.compressed == 0xFFFFFFFFu) take(entry.compressed);
                if (entry.header_offset == 0xFFFFFFFFu) take(entry.header_offset);
            }
            e += 4 + length;
        }
        std::replace(name.begin(), name.end(), '\\', '/');
        entries.emplace(std::move(name), entry);
        p += 46 + name_length + extra_length + comment_length;
    }
    return entries;
}

// Where a stored entry's bytes start.
uint64_t ZipDataOffset(File& file, const ZipEntry& entry, const std::string& name) {
    if (entry.method != 0)
        Fail("'" + name + "' is compressed inside the file; torch.save writes it uncompressed, so save the "
             "state-dict again with torch.save");
    uint8_t header[30];
    file.Read(entry.header_offset, header, 30);
    if (Little<uint32_t>(header) != 0x04034b50u) Fail("the ZIP entry '" + name + "' is damaged");
    const uint64_t data = entry.header_offset + 30 + Little<uint16_t>(&header[26]) + Little<uint16_t>(&header[28]);
    if (data > file.Size() || entry.size > file.Size() - data) Fail("the ZIP entry '" + name + "' runs past the end of the file");
    return data;
}

// ---- the pickle, read as data ----------------------------------------------

// Values are owned by the Unpickler's arena and refer to each other by plain
// pointer, so a deeply nested or self-referencing pickle is destroyed in one
// flat pass (an owning chain would recurse once per level).
struct Value;
using ValuePtr = Value*;

struct Value {
    enum class Type { None, Bool, Int, Text, Sequence, Dict, Global, Storage, Tensor, Mark, Other } type = Type::None;
    int64_t integer = 0;                                   // Bool, Int
    std::string text;                                      // Text; Global "module.name"; Storage key
    std::vector<ValuePtr> items;                           // Sequence (tuple or list)
    std::vector<std::pair<ValuePtr, ValuePtr>> pairs;      // Dict, in file order
    const Kind* kind = nullptr;                            // Storage, Tensor
    uint64_t storage_elements = 0;                         // Storage
    uint64_t storage_offset = 0;                           // Storage: elements into the stored key (old views)
    ValuePtr storage;                                      // Tensor
    uint64_t offset = 0;                                   // Tensor, in elements
    std::vector<uint64_t> shape, stride;                   // Tensor
};

class Unpickler {
public:
    explicit Unpickler(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

    // Bytes consumed by Run (where the next pickle in the file starts).
    size_t Consumed() const { return at_; }
    // Storage key -> element type, for every storage the pickle referred to.
    const std::map<std::string, const Kind*>& StorageKinds() const { return storage_kinds_; }

    ValuePtr Run() {
        while (true) {
            if (++steps_ > kMaxPickleObjects) Fail("the pickle is too large to be a state-dict");
            const uint8_t op = Byte();
            switch (op) {
                case 0x80: Byte(); break;                                    // PROTO
                case 0x95: Skip(8); break;                                   // FRAME
                case '.': return Pop();                                      // STOP
                case 'N': Push(Make(Value::Type::None)); break;
                case 0x88: PushInt(1, Value::Type::Bool); break;             // NEWTRUE
                case 0x89: PushInt(0, Value::Type::Bool); break;             // NEWFALSE
                case 'K': PushInt(Byte()); break;                            // BININT1
                case 'M': PushInt(Number<uint16_t>()); break;                // BININT2
                case 'J': PushInt(Number<int32_t>()); break;                 // BININT
                case 0x8a: {                                                 // LONG1
                    const size_t count = Byte();
                    if (count <= 8) {
                        PushInt(Long(count));
                    } else {
                        Skip(count);  // wider than any size (the old format's magic number)
                        Push(Make(Value::Type::Other));
                    }
                    break;
                }
                case 'G': Skip(8); Push(Make(Value::Type::Other)); break;    // BINFLOAT (a checkpoint's own numbers)
                case 'X': PushText(Number<uint32_t>()); break;               // BINUNICODE
                case 0x8c: PushText(Byte()); break;                          // SHORT_BINUNICODE
                case 0x8d: PushText(Number<uint64_t>()); break;              // BINUNICODE8
                case '}': Push(Make(Value::Type::Dict)); break;              // EMPTY_DICT
                case ']':                                                    // EMPTY_LIST
                case ')': Push(Make(Value::Type::Sequence)); break;          // EMPTY_TUPLE
                case '(': Push(Make(Value::Type::Mark)); break;              // MARK
                case 't': {                                                  // TUPLE
                    auto tuple = Make(Value::Type::Sequence);
                    tuple->items = PopToMark();
                    Push(std::move(tuple));
                    break;
                }
                case 0x85: Tuple(1); break;
                case 0x86: Tuple(2); break;
                case 0x87: Tuple(3); break;
                case 'a': {                                                  // APPEND
                    auto item = Pop();
                    Top(Value::Type::Sequence)->items.push_back(std::move(item));
                    break;
                }
                case 'e': {                                                  // APPENDS
                    auto items = PopToMark();
                    auto& list = Top(Value::Type::Sequence)->items;
                    list.insert(list.end(), items.begin(), items.end());
                    break;
                }
                case 's': {                                                  // SETITEM
                    auto value = Pop();
                    auto key = Pop();
                    Top(Value::Type::Dict)->pairs.emplace_back(std::move(key), std::move(value));
                    break;
                }
                case 'u': {                                                  // SETITEMS
                    auto items = PopToMark();
                    if (items.size() % 2) Fail("the pickle is damaged (a dictionary entry has no value)");
                    auto& pairs = Top(Value::Type::Dict)->pairs;
                    for (size_t i = 0; i < items.size(); i += 2) pairs.emplace_back(items[i], items[i + 1]);
                    break;
                }
                case 'q': Memo(Byte()); break;                               // BINPUT
                case 'r': Memo(Number<uint32_t>()); break;                   // LONG_BINPUT
                case 0x94: Memo(memo_.size()); break;                        // MEMOIZE
                case 'h': Push(Recall(Byte())); break;                       // BINGET
                case 'j': Push(Recall(Number<uint32_t>())); break;           // LONG_BINGET
                case 'c': {                                                  // GLOBAL: a name only, never imported
                    auto global = Make(Value::Type::Global);
                    const std::string module = Line();
                    global->text = module + "." + Line();
                    Push(std::move(global));
                    break;
                }
                case 0x93: {                                                 // STACK_GLOBAL
                    auto name = Pop();
                    auto module = Pop();
                    if (name->type != Value::Type::Text || module->type != Value::Type::Text)
                        Fail("the pickle is damaged (a name is not text)");
                    auto global = Make(Value::Type::Global);
                    global->text = module->text + "." + name->text;
                    Push(std::move(global));
                    break;
                }
                case 'Q': Push(PersistentStorage(Pop())); break;             // BINPERSID
                case 'R': {                                                  // REDUCE: three known constructors
                    auto arguments = Pop();
                    auto callable = Pop();
                    Push(Construct(*callable, *arguments));
                    break;
                }
                case 'b': {                                                  // BUILD: a dict's attributes (_metadata)
                    Pop();
                    if (stack_.empty()) Fail("the pickle is damaged");
                    break;
                }
                default:
                    Fail("the file holds something other than a plain state-dict (pickle code " + std::to_string(op) +
                         "); save model.state_dict() with torch.save");
            }
        }
    }

private:
    ValuePtr Make(Value::Type type) {
        if (arena_.size() >= kMaxPickleObjects) Fail("the pickle is too large to be a state-dict");
        arena_.push_back(std::make_unique<Value>());
        arena_.back()->type = type;
        return arena_.back().get();
    }
    uint8_t Byte() {
        if (at_ >= bytes_.size()) throw EndOfData();
        return bytes_[at_++];
    }
    void Skip(size_t count) {
        if (count > bytes_.size() - at_) throw EndOfData();
        at_ += count;
    }
    template <typename T>
    T Number() {
        if (sizeof(T) > bytes_.size() - at_) throw EndOfData();
        const T value = Little<T>(&bytes_[at_]);
        at_ += sizeof(T);
        return value;
    }
    int64_t Long(size_t count) {
        if (count > 8) Fail("the pickle holds a number too large for a tensor size");
        if (count > bytes_.size() - at_) throw EndOfData();
        uint64_t value = 0;
        for (size_t i = 0; i < count; ++i) value |= static_cast<uint64_t>(bytes_[at_ + i]) << (8 * i);
        if (count && count < 8 && (bytes_[at_ + count - 1] & 0x80u)) value |= ~uint64_t{0} << (8 * count);
        at_ += count;
        return static_cast<int64_t>(value);
    }
    std::string Line() {
        const size_t start = at_;
        while (at_ < bytes_.size() && bytes_[at_] != '\n') ++at_;
        if (at_ >= bytes_.size()) throw EndOfData();
        std::string line(reinterpret_cast<const char*>(&bytes_[start]), at_ - start);
        ++at_;
        return line;
    }
    void Push(ValuePtr value) { stack_.push_back(std::move(value)); }
    void PushInt(int64_t integer, Value::Type type = Value::Type::Int) {
        auto value = Make(type);
        value->integer = integer;
        Push(std::move(value));
    }
    void PushText(uint64_t length) {
        if (length > bytes_.size() - at_) throw EndOfData();
        auto value = Make(Value::Type::Text);
        value->text.assign(reinterpret_cast<const char*>(&bytes_[at_]), static_cast<size_t>(length));
        at_ += static_cast<size_t>(length);
        Push(std::move(value));
    }
    ValuePtr Pop() {
        if (stack_.empty() || stack_.back()->type == Value::Type::Mark) Fail("the pickle is damaged");
        auto value = stack_.back();
        stack_.pop_back();
        return value;
    }
    ValuePtr& Top(Value::Type type) {
        if (stack_.empty() || stack_.back()->type != type) Fail("the pickle is damaged");
        return stack_.back();
    }
    std::vector<ValuePtr> PopToMark() {
        size_t mark = stack_.size();
        while (mark > 0 && stack_[mark - 1]->type != Value::Type::Mark) --mark;
        if (mark == 0) Fail("the pickle is damaged");
        std::vector<ValuePtr> items(stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
        stack_.resize(mark - 1);
        return items;
    }
    void Tuple(size_t count) {
        auto tuple = Make(Value::Type::Sequence);
        tuple->items.resize(count);
        for (size_t i = count; i-- > 0;) tuple->items[i] = Pop();
        Push(std::move(tuple));
    }
    void Memo(size_t index) {
        if (stack_.empty()) Fail("the pickle is damaged");
        if (index > kMaxPickleObjects) Fail("the pickle is too large to be a state-dict");
        if (index >= memo_.size()) memo_.resize(index + 1);
        memo_[index] = stack_.back();
    }
    ValuePtr Recall(size_t index) {
        if (index >= memo_.size() || !memo_[index]) Fail("the pickle is damaged");
        return memo_[index];
    }

    static uint64_t Count(const Value& value, const char* what) {
        if (value.type != Value::Type::Int || value.integer < 0) Fail(std::string("the pickle is damaged (") + what + ")");
        return static_cast<uint64_t>(value.integer);
    }

    // ('storage', <storage class>, key, location, element count[, view metadata])
    ValuePtr PersistentStorage(const ValuePtr& id) {
        if (id->type != Value::Type::Sequence || id->items.size() < 5 || id->items[0]->type != Value::Type::Text ||
            id->items[0]->text != "storage" || id->items[1]->type != Value::Type::Global ||
            id->items[2]->type != Value::Type::Text)
            Fail("the file refers to something other than tensor storage");
        const std::string& global = id->items[1]->text;
        const std::string class_name = global.substr(global.rfind('.') + 1);
        auto storage = Make(Value::Type::Storage);
        storage->kind = global.rfind("torch.", 0) == 0 ? KindByTorchName(class_name) : nullptr;
        if (!storage->kind) Fail("the file uses a tensor type this reader does not know (" + global + ")");
        storage->text = id->items[2]->text;
        storage->storage_elements = Count(*id->items[4], "a storage size");
        // Old format: (view key, offset, size) when the storage is a part of a larger one.
        if (id->items.size() >= 6 && id->items[5]->type == Value::Type::Sequence) {
            const auto& view = id->items[5]->items;
            if (view.size() != 3 || view[0]->type != Value::Type::Text) Fail("the pickle is damaged (a storage view)");
            storage->storage_offset = Count(*view[1], "a storage view offset");
            storage->storage_elements = Count(*view[2], "a storage view size");
        }
        const auto [known, inserted] = storage_kinds_.emplace(storage->text, storage->kind);
        if (!inserted && known->second != storage->kind) Fail("the pickle is damaged (a storage with two types)");
        return storage;
    }

    static std::vector<uint64_t> Sizes(const Value& value, const char* what) {
        if (value.type != Value::Type::Sequence || value.items.size() > kMaxRank)
            Fail(std::string("the pickle is damaged (") + what + ")");
        std::vector<uint64_t> sizes;
        for (const auto& item : value.items) sizes.push_back(Count(*item, what));
        return sizes;
    }

    // The constructors a state-dict uses. Nothing is called: each is rebuilt here.
    ValuePtr Construct(const Value& callable, const Value& arguments) {
        if (callable.type != Value::Type::Global || arguments.type != Value::Type::Sequence)
            Fail("the pickle is damaged");
        const std::string& name = callable.text;
        const auto& args = arguments.items;
        if (name == "collections.OrderedDict") {
            if (!args.empty()) Fail("the pickle is damaged (a dictionary with arguments)");
            return Make(Value::Type::Dict);
        }
        if (name == "torch._utils._rebuild_tensor_v2" || name == "torch._utils._rebuild_tensor") {
            if (args.size() < 4 || args[0]->type != Value::Type::Storage) Fail("the pickle is damaged (a tensor)");
            auto tensor = Make(Value::Type::Tensor);
            tensor->storage = args[0];
            tensor->kind = args[0]->kind;
            tensor->offset = Count(*args[1], "a tensor offset");
            tensor->shape = Sizes(*args[2], "a tensor shape");
            tensor->stride = Sizes(*args[3], "a tensor stride");
            if (tensor->shape.size() != tensor->stride.size()) Fail("the pickle is damaged (a tensor's stride)");
            return tensor;
        }
        if (name == "torch._utils._rebuild_parameter" || name == "torch._utils._rebuild_parameter_with_state") {
            if (args.empty() || args[0]->type != Value::Type::Tensor) Fail("the pickle is damaged (a parameter)");
            return args[0];
        }
        Fail("the file holds a '" + name + "' object, not a plain state-dict. A whole model or an optimizer was "
             "saved; save model.state_dict() with torch.save instead");
    }

    std::vector<uint8_t> bytes_;
    size_t at_ = 0;
    size_t steps_ = 0;
    std::vector<ValuePtr> stack_;
    std::vector<ValuePtr> memo_;
    std::vector<std::unique_ptr<Value>> arena_;
    std::map<std::string, const Kind*> storage_kinds_;
};

bool HoldsTensors(const Value& dict) {
    if (dict.type != Value::Type::Dict || dict.pairs.empty()) return false;
    return std::all_of(dict.pairs.begin(), dict.pairs.end(), [](const auto& pair) {
        return pair.first->type == Value::Type::Text && pair.second->type == Value::Type::Tensor;
    });
}

// The state-dict itself, or the one a training checkpoint keeps under a usual key.
const Value& FindStateDict(const Value& root) {
    if (HoldsTensors(root)) return root;
    if (root.type == Value::Type::Dict) {
        for (const char* key : {"state_dict", "model_state_dict", "model", "net", "params"}) {
            for (const auto& [name, value] : root.pairs) {
                if (name->type == Value::Type::Text && name->text == key && HoldsTensors(*value)) return *value;
            }
        }
    }
    Fail("the file holds no state-dict: it must be a dictionary of layer names to tensors "
         "(torch.save(model.state_dict(), path))");
}

// A tensor's values in row-major order, whatever its strides.
// Where a storage key's raw values are in the file, and how many bytes there are.
struct Stored {
    uint64_t offset = 0;
    uint64_t size = 0;
};
using LocateStorage = std::function<Stored(const std::string& key, const std::string& tensor_name)>;

Entry MaterializeTensor(File& file, const LocateStorage& locate, const std::string& name, const Value& tensor,
                        bool load_values, uint64_t& budget) {
    Entry entry;
    entry.name = name;
    entry.dtype = tensor.kind->name;
    entry.is_float = tensor.kind->is_float;
    uint64_t elements = 1;
    for (const uint64_t dim : tensor.shape) {
        elements = CheckedMul(elements, dim, "a tensor");
        entry.shape.push_back(static_cast<size_t>(dim));
    }
    const uint64_t item = tensor.kind->bytes;
    if (CheckedMul(elements, item, "a tensor") > kMaxTensorBytes) Fail("the tensor '" + name + "' is too large");
    // The furthest element the strides reach must be inside the storage.
    uint64_t last = tensor.offset;
    if (elements) {
        for (size_t d = 0; d < tensor.shape.size(); ++d) {
            const uint64_t reach = CheckedMul(tensor.shape[d] - 1, tensor.stride[d], "a tensor");
            if (reach > std::numeric_limits<uint64_t>::max() - last) Fail("the tensor '" + name + "' is too large");
            last += reach;
        }
        if (last >= tensor.storage->storage_elements)
            Fail("the tensor '" + name + "' points outside its stored values (the file is damaged)");
        // A view that repeats its values (stride 0) could ask for far more than is stored.
        if (elements > tensor.storage->storage_elements)
            Fail("the tensor '" + name + "' repeats its stored values (an expanded view); save a plain copy of it");
    }
    const Stored stored = locate(tensor.storage->text, name);
    const uint64_t base = tensor.storage->storage_offset;
    if (base > std::numeric_limits<uint64_t>::max() - tensor.storage->storage_elements ||
        CheckedMul(base + tensor.storage->storage_elements, item, "a tensor") > stored.size)
        Fail("the values of '" + name + "' are shorter than its shape needs (the file is damaged)");
    if (!entry.is_float || elements == 0 || !load_values) return entry;
    // Several names may share one storage, but together they cannot need more than the file holds twice over.
    const uint64_t needed = elements * item;
    if (needed > budget) Fail("the file's tensors ask for more values than the file holds (it is damaged)");
    budget -= needed;

    const uint64_t data = stored.offset + base * item;
    const uint64_t first = tensor.offset;
    const auto bytes = file.ReadBytes(data + first * item, (last - first + 1) * item);
    entry.values.resize(static_cast<size_t>(elements));
    std::vector<uint64_t> index(tensor.shape.size(), 0);
    uint64_t position = 0;  // in elements from `first`
    for (uint64_t n = 0; n < elements; ++n) {
        entry.values[static_cast<size_t>(n)] = ElementAsFloat(*tensor.kind, &bytes[static_cast<size_t>(position * item)]);
        for (size_t d = index.size(); d-- > 0;) {
            if (++index[d] < tensor.shape[d]) {
                position += tensor.stride[d];
                break;
            }
            position -= (tensor.shape[d] - 1) * tensor.stride[d];
            index[d] = 0;
        }
    }
    return entry;
}

void ReadPyTorch(File& file, WeightFile& out, bool load_values) {
    const auto zip = ReadZipDirectory(file);
    std::string pickle_path;
    for (const auto& [name, entry] : zip) {
        if (name == "data.pkl" || (name.size() > 9 && name.compare(name.size() - 9, 9, "/data.pkl") == 0)) {
            pickle_path = name;
            break;
        }
    }
    if (pickle_path.empty())
        Fail("the ZIP holds no data.pkl, so it is not a PyTorch state-dict (a TorchScript or other archive?)");
    // The folder data.pkl sits in ("archive/"), where byteorder and data/<key> are too.
    const std::string folder = pickle_path.substr(0, pickle_path.size() - 8);
    if (const auto order = zip.find(folder + "byteorder"); order != zip.end()) {
        const auto text = file.ReadBytes(ZipDataOffset(file, order->second, order->first),
                                         std::min<uint64_t>(order->second.size, 16));
        if (std::string(text.begin(), text.end()).rfind("little", 0) != 0)
            Fail("the file was saved on a big-endian machine, which this reader does not convert");
    }
    const ZipEntry& pickle = zip.at(pickle_path);
    if (pickle.size > kMaxPickleBytes) Fail("the file's pickle is too large to be a state-dict");
    Unpickler unpickler(file.ReadBytes(ZipDataOffset(file, pickle, pickle_path), pickle.size));
    const Value& state = FindStateDict(*unpickler.Run());  // the values live as long as the unpickler
    out.format = "PyTorch state-dict";
    uint64_t budget = CheckedMul(file.Size(), 2, "the file");
    const LocateStorage locate = [&](const std::string& key, const std::string& tensor_name) {
        const std::string path = folder + "data/" + key;
        const auto found = zip.find(path);
        if (found == zip.end()) Fail("the values of '" + tensor_name + "' are missing from the file (" + path + ")");
        return Stored{ZipDataOffset(file, found->second, path), found->second.size};
    };
    for (const auto& [name, tensor] : state.pairs) {
        out.entries.push_back(MaterializeTensor(file, locate, name->text, *tensor, load_values, budget));
    }
}

// One pickle starting at `offset`: read from a growing prefix of the file, so
// a small pickle in front of large stored values costs a small read.
std::unique_ptr<Unpickler> PickleAt(File& file, uint64_t offset, ValuePtr& value) {
    if (offset >= file.Size()) Fail("the file ends before its contents do (truncated or damaged)");
    uint64_t length = std::min<uint64_t>(file.Size() - offset, 1u << 20);
    while (true) {
        auto unpickler = std::make_unique<Unpickler>(file.ReadBytes(offset, length));
        try {
            value = unpickler->Run();
            return unpickler;
        } catch (const EndOfData&) {
            if (length == file.Size() - offset || length >= kMaxPickleBytes) throw;
            length = std::min<uint64_t>({file.Size() - offset, length * 4, kMaxPickleBytes});
        }
    }
}

// The format torch.save wrote before PyTorch 1.6: five pickles in a row (a
// magic number, the protocol, the machine's sizes, the object, the storage
// keys), then each storage's element count and raw values in key order.
void ReadPyTorchOld(File& file, WeightFile& out, bool load_values) {
    static const uint8_t kMagic[] = {0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, 0x2e};
    uint8_t head[sizeof(kMagic)] = {};
    if (file.Size() < sizeof(kMagic)) Fail("the file is too short to be a PyTorch file");
    file.Read(0, head, sizeof(head));
    if (std::memcmp(head, kMagic, sizeof(kMagic)) != 0)
        Fail("the file starts as a pickle but is not one torch.save wrote (no PyTorch magic number)");
    uint64_t at = sizeof(kMagic);
    ValuePtr value = nullptr;
    auto protocol = PickleAt(file, at, value);
    if (value->type != Value::Type::Int || value->integer != 1001)
        Fail("the file is in an old PyTorch format this reader does not know (protocol " +
             (value->type == Value::Type::Int ? std::to_string(value->integer) : std::string("?")) + ")");
    at += protocol->Consumed();
    auto system = PickleAt(file, at, value);
    if (value->type == Value::Type::Dict) {
        for (const auto& [key, flag] : value->pairs) {
            if (key->type == Value::Type::Text && key->text == "little_endian" && flag->type == Value::Type::Bool &&
                flag->integer == 0)
                Fail("the file was saved on a big-endian machine, which this reader does not convert");
        }
    }
    at += system->Consumed();
    ValuePtr root = nullptr;
    auto object = PickleAt(file, at, root);
    const Value& state = FindStateDict(*root);
    at += object->Consumed();
    ValuePtr key_list = nullptr;
    auto keys = PickleAt(file, at, key_list);
    if (key_list->type != Value::Type::Sequence) Fail("the file is damaged (its storage keys are missing)");
    at += keys->Consumed();
    // Each storage: its element count (8 bytes), then its values.
    std::map<std::string, Stored> storages;
    const auto& kinds = object->StorageKinds();
    for (const ValuePtr key : key_list->items) {
        if (key->type != Value::Type::Text) Fail("the file is damaged (a storage key is not text)");
        const auto kind = kinds.find(key->text);
        if (kind == kinds.end()) Fail("the file is damaged (storage " + key->text + " has no type)");
        uint8_t count_bytes[8];
        file.Read(at, count_bytes, 8);
        const uint64_t count = Little<uint64_t>(count_bytes);
        const uint64_t bytes = CheckedMul(count, kind->second->bytes, "a storage");
        if (at + 8 > file.Size() || bytes > file.Size() - at - 8)
            Fail("the file ends before its stored values do (truncated or damaged)");
        storages[key->text] = Stored{at + 8, bytes};
        at += 8 + bytes;
    }
    out.format = "PyTorch state-dict (format before PyTorch 1.6)";
    uint64_t budget = CheckedMul(file.Size(), 2, "the file");
    const LocateStorage locate = [&](const std::string& key, const std::string& tensor_name) {
        const auto found = storages.find(key);
        if (found == storages.end()) Fail("the values of '" + tensor_name + "' are missing from the file");
        return found->second;
    };
    for (const auto& [name, tensor] : state.pairs) {
        out.entries.push_back(MaterializeTensor(file, locate, name->text, *tensor, load_values, budget));
    }
}

void ReadSafetensors(File& file, WeightFile& out, bool load_values) {
    if (file.Size() < 8) Fail("the file is too short to be a safetensors file");
    uint8_t size_bytes[8];
    file.Read(0, size_bytes, 8);
    const uint64_t header_size = Little<uint64_t>(size_bytes);
    if (header_size > kMaxSafetensorsHeader || header_size > file.Size() - 8)
        Fail("the safetensors header is larger than the file (damaged, or not a safetensors file)");
    const auto header_bytes = file.ReadBytes(8, header_size);
    const auto header = nlohmann::ordered_json::parse(header_bytes.begin(), header_bytes.end(), nullptr, false);
    if (!header.is_object()) Fail("the safetensors header is not valid JSON");
    const uint64_t data_start = 8 + header_size;
    const uint64_t data_size = file.Size() - data_start;
    uint64_t budget = data_size;  // the format does not allow tensors to share bytes
    // File order is the order of the data, not of the header's keys.
    std::vector<std::pair<uint64_t, Entry>> ordered;
    for (const auto& [name, info] : header.items()) {
        if (name == "__metadata__") {
            if (info.is_object()) {
                for (const auto& [key, value] : info.items()) {
                    if (value.is_string()) out.metadata[key] = value.get<std::string>();
                }
            }
            continue;
        }
        if (!info.is_object() || !info.contains("dtype") || !info.contains("shape") || !info.contains("data_offsets") ||
            !info["dtype"].is_string() || !info["shape"].is_array() || !info["data_offsets"].is_array() ||
            info["data_offsets"].size() != 2)
            Fail("the safetensors entry '" + name + "' is incomplete");
        Entry entry;
        entry.name = name;
        entry.dtype = info["dtype"].get<std::string>();
        const Kind* kind = KindByName(entry.dtype);
        if (!kind) Fail("the tensor '" + name + "' has a type this reader does not know (" + entry.dtype + ")");
        entry.is_float = kind->is_float;
        if (info["shape"].size() > kMaxRank) Fail("the tensor '" + name + "' has too many dimensions");
        uint64_t elements = 1;
        for (const auto& dim : info["shape"]) {
            if (!dim.is_number_unsigned()) Fail("the tensor '" + name + "' has an invalid shape");
            elements = CheckedMul(elements, dim.get<uint64_t>(), "a tensor");
            entry.shape.push_back(static_cast<size_t>(dim.get<uint64_t>()));
        }
        if (!info["data_offsets"][0].is_number_unsigned() || !info["data_offsets"][1].is_number_unsigned())
            Fail("the tensor '" + name + "' has invalid offsets");
        const uint64_t begin = info["data_offsets"][0].get<uint64_t>();
        const uint64_t end = info["data_offsets"][1].get<uint64_t>();
        const uint64_t bytes = CheckedMul(elements, kind->bytes, "a tensor");
        if (begin > end || end > data_size || end - begin != bytes)
            Fail("the values of '" + name + "' do not fit its shape (the file is damaged)");
        if (bytes > kMaxTensorBytes) Fail("the tensor '" + name + "' is too large");
        if (bytes > budget) Fail("the tensors overlap in the file, which the format does not allow (it is damaged)");
        budget -= bytes;
        if (entry.is_float && elements && load_values) {
            const auto raw = file.ReadBytes(data_start + begin, bytes);
            entry.values.resize(static_cast<size_t>(elements));
            for (uint64_t n = 0; n < elements; ++n)
                entry.values[static_cast<size_t>(n)] = ElementAsFloat(*kind, &raw[static_cast<size_t>(n * kind->bytes)]);
        }
        ordered.emplace_back(begin, std::move(entry));
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    out.format = "safetensors";
    for (auto& [offset, entry] : ordered) out.entries.push_back(std::move(entry));
}

}  // namespace

bool Read(const std::filesystem::path& path, WeightFile& out, std::string& error, bool load_values) {
    out = WeightFile{};
    error.clear();
    try {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) Fail("the file was not found");
        File file(path);
        uint8_t magic[4] = {0, 0, 0, 0};
        if (file.Size() >= 4) file.Read(0, magic, 4);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (magic[0] == 'P' && magic[1] == 'K') {
            ReadPyTorch(file, out, load_values);
        } else if (extension == ".safetensors") {
            ReadSafetensors(file, out, load_values);
        } else if (magic[0] == 0x80) {
            ReadPyTorchOld(file, out, load_values);
        } else {
            Fail("the file is neither a PyTorch state-dict (.pth, .pt) nor a .safetensors file");
        }
        if (out.entries.empty()) Fail("the file holds no tensors");
        return true;
    } catch (const Failure& failure) {
        error = failure.what();
    } catch (const std::exception& failure) {
        error = std::string("the file could not be read: ") + failure.what();
    }
    out = WeightFile{};
    return false;
}

}  // namespace cyxwiz::weight_file
