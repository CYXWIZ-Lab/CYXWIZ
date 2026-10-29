// Example plugin with a dataset format and a dataset analytic (TOFIX181,
// plugin API 1.3), using only the SDK's Arrow C stream header: no Arrow
// library.
//
//   ARFF (Weka's attribute-relation file format, .arff): IDataProvider.
//     Numeric attributes become float64 columns, nominal, string and date
//     attributes text columns, '?' an empty (null) value. The file is read
//     as it is streamed, 50,000 rows per batch, so its size is not limited by
//     memory.
//   Coefficient of variation (standard deviation / |mean| of each numeric
//     column): IAnalyticsProvider. It reads float64, float32, int64 and int32
//     columns of the stream the Engine hands it.

#include "plugin/plugin_types.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_analytics_provider.h"
#include "plugin/interfaces/i_data_provider.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::plugin::examples {

namespace {

constexpr int64_t kRowsPerBatch = 50000;
constexpr const char* kArffLoader = "com.cyxwiz.examples.data-formats.arff";
constexpr const char* kVariationAnalytic = "com.cyxwiz.examples.data-formats.variation";

// ---------------------------------------------------------------------------
// ARFF reading
// ---------------------------------------------------------------------------

std::string Trim(const std::string& text) {
    size_t begin = 0, end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// Splits on `separator` outside single or double quotes and removes the quotes.
std::vector<std::string> SplitQuoted(const std::string& line, char separator) {
    std::vector<std::string> fields;
    std::string field;
    char quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote) {
            if (c == '\\' && i + 1 < line.size()) {
                field += line[++i];
            } else if (c == quote) {
                quote = 0;
            } else {
                field += c;
            }
        } else if (c == '\'' || c == '"') {
            quote = c;
        } else if (c == separator) {
            fields.push_back(Trim(field));
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(Trim(field));
    return fields;
}

struct Attribute {
    std::string name;
    bool numeric = false;
};

// Reads the header up to @data, counting its lines. Returns false with
// `error` on a malformed one.
bool ReadHeader(std::ifstream& in, std::vector<Attribute>& attributes, int64_t& line_number,
                std::string& error) {
    std::string line;
    while (std::getline(in, line)) {
        ++line_number;
        const std::string text = Trim(line);
        if (text.empty() || text[0] == '%') continue;
        const std::string lower = Lower(text);
        if (lower.rfind("@relation", 0) == 0) continue;
        if (lower.rfind("@data", 0) == 0) {
            if (attributes.empty()) error = "the file declares no @attribute before @data";
            return attributes.empty() ? false : true;
        }
        if (lower.rfind("@attribute", 0) != 0) {
            error = "unexpected header line: " + text;
            return false;
        }
        // @attribute <name> <type>; the name may be quoted.
        std::string rest = Trim(text.substr(10));
        Attribute attribute;
        if (!rest.empty() && (rest[0] == '\'' || rest[0] == '"')) {
            const size_t close = rest.find(rest[0], 1);
            if (close == std::string::npos) {
                error = "unclosed quote in: " + text;
                return false;
            }
            attribute.name = rest.substr(1, close - 1);
            rest = Trim(rest.substr(close + 1));
        } else {
            const size_t space = rest.find_first_of(" \t");
            if (space == std::string::npos) {
                error = "attribute without a type: " + text;
                return false;
            }
            attribute.name = rest.substr(0, space);
            rest = Trim(rest.substr(space));
        }
        const std::string type = Lower(rest);
        if (type.rfind("relational", 0) == 0) {
            error = "relational attributes are not supported (" + attribute.name + ")";
            return false;
        }
        attribute.numeric = type == "numeric" || type == "real" || type == "integer";
        attributes.push_back(attribute);
    }
    error = "the file has no @data section";
    return false;
}

// ---------------------------------------------------------------------------
// Producing Arrow C data by hand. Each array and schema owns its memory
// through private_data and frees it in its release callback, as the
// specification asks; the parent's release releases its children.
// ---------------------------------------------------------------------------

struct OwnedSchema {
    std::string name;
    std::vector<ArrowSchema*> children;
};

void ReleaseSchema(ArrowSchema* schema) {
    auto* owned = static_cast<OwnedSchema*>(schema->private_data);
    for (ArrowSchema* child : owned->children) {
        if (child->release) child->release(child);
        delete child;
    }
    delete owned;
    schema->release = nullptr;
}

void MakeSchema(ArrowSchema* out, const char* format, const std::string& name, int64_t flags) {
    auto* owned = new OwnedSchema{name, {}};
    *out = ArrowSchema{};
    out->format = format;
    out->name = owned->name.c_str();
    out->flags = flags;
    out->release = ReleaseSchema;
    out->private_data = owned;
}

// One column of a batch: float64 ("g") or utf8 ("u"), nullable.
struct OwnedColumn {
    std::vector<uint8_t> validity;
    std::vector<double> numbers;
    std::vector<int32_t> offsets{0};
    std::string characters;
    const void* buffers[3] = {nullptr, nullptr, nullptr};
};

void ReleaseColumn(ArrowArray* array) {
    delete static_cast<OwnedColumn*>(array->private_data);
    array->release = nullptr;
}

struct OwnedBatch {
    std::vector<ArrowArray*> children;
    const void* buffers[1] = {nullptr};
};

void ReleaseBatch(ArrowArray* array) {
    auto* owned = static_cast<OwnedBatch*>(array->private_data);
    for (ArrowArray* child : owned->children) {
        if (child->release) child->release(child);
        delete child;
    }
    delete owned;
    array->release = nullptr;
}

struct ColumnBuilder {
    bool numeric = false;
    std::unique_ptr<OwnedColumn> column = std::make_unique<OwnedColumn>();
    int64_t length = 0;
    int64_t nulls = 0;

    void Append(const std::string& value) {
        const bool missing = value == "?" || (value.empty() && numeric);
        if (length % 8 == 0) column->validity.push_back(0);
        if (!missing) column->validity.back() |= static_cast<uint8_t>(1u << (length % 8));
        else ++nulls;
        if (numeric) {
            column->numbers.push_back(missing ? 0.0 : std::strtod(value.c_str(), nullptr));
        } else {
            if (!missing) column->characters += value;
            column->offsets.push_back(static_cast<int32_t>(column->characters.size()));
        }
        ++length;
    }

    ArrowArray* Finish() {
        auto* array = new ArrowArray{};
        OwnedColumn* owned = column.release();
        owned->buffers[0] = nulls ? owned->validity.data() : nullptr;
        if (numeric) {
            owned->buffers[1] = owned->numbers.data();
            array->n_buffers = 2;
        } else {
            owned->buffers[1] = owned->offsets.data();
            owned->buffers[2] = owned->characters.data();
            array->n_buffers = 3;
        }
        array->length = length;
        array->null_count = nulls;
        array->buffers = owned->buffers;
        array->release = ReleaseColumn;
        array->private_data = owned;
        return array;
    }
};

// The stream: the open file and its attributes; get_next reads the next rows.
struct ArffStream {
    std::ifstream in;
    std::vector<Attribute> attributes;
    std::string path;
    std::string last_error;
    int64_t line_number = 0;
};

int ArffGetSchema(ArrowArrayStream* stream, ArrowSchema* out) {
    auto* state = static_cast<ArffStream*>(stream->private_data);
    MakeSchema(out, "+s", "", 0);
    auto* owned = static_cast<OwnedSchema*>(out->private_data);
    for (const auto& attribute : state->attributes) {
        auto* child = new ArrowSchema{};
        MakeSchema(child, attribute.numeric ? "g" : "u", attribute.name, ARROW_FLAG_NULLABLE);
        owned->children.push_back(child);
    }
    out->n_children = static_cast<int64_t>(owned->children.size());
    out->children = owned->children.data();
    return 0;
}

int ArffGetNext(ArrowArrayStream* stream, ArrowArray* out) {
    auto* state = static_cast<ArffStream*>(stream->private_data);
    std::vector<ColumnBuilder> builders(state->attributes.size());
    for (size_t i = 0; i < builders.size(); ++i) builders[i].numeric = state->attributes[i].numeric;

    int64_t rows = 0;
    std::string line;
    while (rows < kRowsPerBatch && std::getline(state->in, line)) {
        ++state->line_number;
        const std::string text = Trim(line);
        if (text.empty() || text[0] == '%') continue;
        if (text[0] == '{') {
            state->last_error = "line " + std::to_string(state->line_number) +
                                ": sparse ARFF rows are not supported";
            return EINVAL;
        }
        const auto fields = SplitQuoted(text, ',');
        if (fields.size() != builders.size()) {
            state->last_error = "line " + std::to_string(state->line_number) + ": " +
                                std::to_string(fields.size()) + " values, the header declares " +
                                std::to_string(builders.size());
            return EINVAL;
        }
        for (size_t i = 0; i < fields.size(); ++i) builders[i].Append(fields[i]);
        ++rows;
    }

    *out = ArrowArray{};
    if (rows == 0) return 0;  // end of stream: out->release stays null

    auto* owned = new OwnedBatch;
    for (auto& builder : builders) owned->children.push_back(builder.Finish());
    out->length = rows;
    out->n_buffers = 1;
    out->buffers = owned->buffers;
    out->n_children = static_cast<int64_t>(owned->children.size());
    out->children = owned->children.data();
    out->release = ReleaseBatch;
    out->private_data = owned;
    return 0;
}

const char* ArffLastError(ArrowArrayStream* stream) {
    auto* state = static_cast<ArffStream*>(stream->private_data);
    return state->last_error.empty() ? nullptr : state->last_error.c_str();
}

void ArffRelease(ArrowArrayStream* stream) {
    delete static_cast<ArffStream*>(stream->private_data);
    stream->release = nullptr;
}

// ---------------------------------------------------------------------------
// Reading Arrow C data by hand (the analytic's input)
// ---------------------------------------------------------------------------

bool IsValid(const ArrowArray* column, int64_t row) {
    const auto* validity = static_cast<const uint8_t*>(column->buffers[0]);
    if (!validity || column->null_count == 0) return true;
    const int64_t bit = column->offset + row;
    return (validity[bit / 8] >> (bit % 8)) & 1;
}

// Calls add(value) for each valid value of a numeric column; false for
// another type.
template <typename Add>
bool ForEachNumber(const char* format, const ArrowArray* column, Add add) {
    const int64_t start = column->offset;
    const auto read = [&](auto* values) {
        for (int64_t row = 0; row < column->length; ++row)
            if (IsValid(column, row)) add(static_cast<double>(values[start + row]));
    };
    if (!std::strcmp(format, "g")) read(static_cast<const double*>(column->buffers[1]));
    else if (!std::strcmp(format, "f")) read(static_cast<const float*>(column->buffers[1]));
    else if (!std::strcmp(format, "l")) read(static_cast<const int64_t*>(column->buffers[1]));
    else if (!std::strcmp(format, "i")) read(static_cast<const int32_t*>(column->buffers[1]));
    else return false;
    return true;
}

}  // namespace

class DataFormatsPlugin : public IPlugin, public IDataProvider, public IAnalyticsProvider {
public:
    bool OnLoad(PluginContext& ctx) override {
        ctx.LogInfo("Example Data Formats: OnLoad");
        state_ = PluginState::Loaded;
        return true;
    }
    bool OnInitialize(PluginContext&) override {
        state_ = PluginState::Active;
        return true;
    }
    void OnShutdown(PluginContext&) override { state_ = PluginState::Loaded; }
    void OnUnload(PluginContext&) override { state_ = PluginState::Unloaded; }

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::DataRegistry);
    }
    PluginState GetState() const override { return state_; }
    void* QueryInterface(const char* name) override {
        const std::string interface_name = name;
        if (interface_name == "IDataProvider") return static_cast<IDataProvider*>(this);
        if (interface_name == "IAnalyticsProvider") return static_cast<IAnalyticsProvider*>(this);
        return nullptr;
    }

    // --- IDataProvider ---
    std::vector<PluginDataLoaderInfo> GetLoaders() override {
        return {{kArffLoader, "ARFF (Weka)", {"arff"},
                 "Weka's attribute-relation file format: numeric attributes as numbers, the others as text"}};
    }

    bool CanLoad(const std::string&, const std::string& file_path) override {
        std::ifstream in(file_path);
        std::string line;
        while (std::getline(in, line)) {
            const std::string text = Lower(Trim(line));
            if (text.empty() || text[0] == '%') continue;
            return text.rfind("@relation", 0) == 0 || text.rfind("@attribute", 0) == 0;
        }
        return false;
    }

    bool OpenDataset(const std::string&, const std::string& file_path, ArrowArrayStream* out,
                     std::string& error) override {
        auto state = std::make_unique<ArffStream>();
        state->path = file_path;
        state->in.open(file_path);
        if (!state->in) {
            error = "cannot open the file";
            return false;
        }
        if (!ReadHeader(state->in, state->attributes, state->line_number, error)) return false;
        *out = ArrowArrayStream{};
        out->get_schema = ArffGetSchema;
        out->get_next = ArffGetNext;
        out->get_last_error = ArffLastError;
        out->release = ArffRelease;
        out->private_data = state.release();
        return true;
    }

    // --- IAnalyticsProvider ---
    std::vector<PluginAnalyticsInfo> GetAnalytics() override {
        PluginAnalyticsInfo variation;
        variation.analytics_id = kVariationAnalytic;
        variation.name = "Coefficient of variation";
        variation.description = "Standard deviation divided by the absolute mean, per column: "
                                "how spread out a column is relative to its size";
        variation.category = "Distribution";
        variation.columns = AnalyticsColumns::Numeric;
        return {variation};
    }

    AnalyticsResult Compute(const std::string&, ArrowArrayStream* data) override {
        AnalyticsResult result;
        ArrowSchema schema{};
        if (data->get_schema(data, &schema) != 0) return Failed(result, data, "cannot read the columns");
        struct Sums {
            double count = 0, sum = 0, squares = 0;
        };
        std::vector<Sums> sums(static_cast<size_t>(schema.n_children));
        std::string error;

        while (error.empty()) {
            ArrowArray batch{};
            if (data->get_next(data, &batch) != 0) {
                error = "cannot read the rows";
                break;
            }
            if (!batch.release) break;  // end of stream
            for (int64_t c = 0; c < schema.n_children && error.empty(); ++c) {
                Sums& s = sums[static_cast<size_t>(c)];
                if (!ForEachNumber(schema.children[c]->format, batch.children[c], [&](double v) {
                        s.count += 1;
                        s.sum += v;
                        s.squares += v * v;
                    })) {
                    error = std::string("column ") + schema.children[c]->name + " is not numeric";
                }
            }
            batch.release(&batch);
        }

        if (error.empty()) {
            for (int64_t c = 0; c < schema.n_children; ++c) {
                const Sums& s = sums[static_cast<size_t>(c)];
                const double mean = s.count > 0 ? s.sum / s.count : 0.0;
                const double variance = s.count > 1 ? (s.squares - s.count * mean * mean) / (s.count - 1) : 0.0;
                const double cv = mean != 0.0 ? std::sqrt(std::max(variance, 0.0)) / std::abs(mean) : 0.0;
                result.vector_labels.push_back(schema.children[c]->name);
                result.vector_value.push_back(static_cast<float>(cv));
            }
            result.text_value = "0 means every value is the same; above 1, the spread exceeds the mean. "
                                "A column whose mean is 0 shows 0.";
        }
        schema.release(&schema);
        if (!error.empty()) {
            result.success = false;
            result.error_message = error;
        }
        return result;
    }

private:
    static AnalyticsResult Failed(AnalyticsResult& result, ArrowArrayStream* data, const std::string& what) {
        result.success = false;
        const char* reason = data->get_last_error ? data->get_last_error(data) : nullptr;
        result.error_message = what + (reason ? std::string(": ") + reason : std::string());
        return result;
    }

    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
};

}  // namespace cyxwiz::plugin::examples

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::DataFormatsPlugin)
