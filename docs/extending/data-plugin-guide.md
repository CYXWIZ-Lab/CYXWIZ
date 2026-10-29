# Data plugins: your own dataset formats and analytics

A C++ plugin can give CyxWiz two things for datasets:

- **A dataset format.** Data Input lists it beside CSV and Parquet and opens
  files of that format; from then on the table is like any other: Preview,
  Data Studio, SQL, training.
- **A dataset analytic.** Data Studio's **Explore > Plugins** tab lists it;
  the user picks columns and presses Run, and your code computes the answer.

This guide follows one working plugin from start to end: **Example Data
Formats**, which reads ARFF files (Weka's format) and computes the
coefficient of variation of numeric columns. You try it ready-built, build
it yourself, follow what the Engine calls in which order, read its code,
and then write your own format.

> Checked against plugin API 1.3.0 and pycyxwiz 0.2.0.
> Read the [quickstart](plugin-quickstart.md) first for the SDK folder, the
> compiler and the check tool; this guide uses its shell variables (`$sdk`,
> `$cyxwiz`).

## 1. The example in the SDK

Every installation carries the example in the SDK, as source and built:

| In `<sdk>/examples/plugins/data-formats/` | What it is |
|---|---|
| `data_formats_plugin.cpp` | The whole plugin, one file. Its first comment lists the order the Engine calls it in |
| `plugin.json` | The manifest |
| `CMakeLists.txt` | The build |
| `weather.arff` | A sample dataset: 14 rows, 5 attributes |
| `README.md` | The short version of this guide |
| `built/` | The same plugin, built by CyxWiz with the Engine: `plugin.json` and `bin/data_formats.dll` (`bin/libdata_formats.so`, `bin/libdata_formats.dylib`) |

In the CyxWiz sources it is `plugins/examples/data_formats`.

## 2. Try it without building

1. Open the **Plugin Manager** and press **Install...**. Enter the folder
   `<sdk>/examples/plugins/data-formats/built` (PowerShell prints it with
   `"$sdk\examples\plugins\data-formats\built"`), or pick it with
   **Browse**, and press **Install**. The card **Example Data Formats**
   appears with its description and the *Dataset Access* permission. The
   folder is inside the installation, which CyxWiz replaces when it updates:
   after an update, Install it again.
2. Open **Data Studio**, and under **Bring in** press **Add Data Input...**
   (or put a **Data Input** node on the canvas and double-click it). In
   **File or folder** enter
   `<sdk>/examples/plugins/data-formats/weather.arff`, or pick it with
   **Browse file...**, whose file types list *ARFF (Weka) (plugin)*.
   **Detected** shows *ARFF (Weka) (plugin) table*: your `CanLoad` claimed
   the file. Press **Apply**: *weather.arff Loaded as weather (table)*, 14
   rows, 5 columns. **Preview** shows the rows; `outlook`, `windy` and
   `play` are text, `temperature` and `humidity` numbers. Press **OK**.
3. In Data Studio pick **Data Input** under *Data Inputs*, then
   **Explore > Plugins**.
   **Analytic** shows *Coefficient of variation (Distribution)*, *from
   Example Data Formats 1.0.0*; **Columns** holds `temperature` and
   `humidity` (the text columns are listed dimmed: *text · this analytic
   reads numbers*). Press **Run**:

   | column | coefficient of variation |
   |---|---|
   | temperature | 0.0893 |
   | humidity | 0.1260 |

   with the note and *Computed by Example Data Formats 1.0.0 on 14 rows*.

The example is a teaching plugin: when you are done, **Unload** it in the
Plugin Manager, which also stops it loading at the next start.

## 3. Build it yourself

Copy the example out of the installation without `built/`, and build it
like the template ([quickstart, section 2](plugin-quickstart.md#2-build-the-template)).

PowerShell:

```powershell
Copy-Item -Recurse "$sdk\examples\plugins\data-formats" data_formats
Remove-Item -Recurse data_formats\built
cmake -S data_formats -B data_formats\build -DCMAKE_PREFIX_PATH="$sdk"
cmake --build data_formats\build --config Release
& "$cyxwiz\cyxwiz-cli.exe" plugin-check data_formats\build\data_formats
```

Linux and macOS:

```bash
cp -r "$sdk/examples/plugins/data-formats" data_formats
rm -rf data_formats/built
cmake -S data_formats -B data_formats/build -DCMAKE_PREFIX_PATH="$sdk" -DCMAKE_BUILD_TYPE=Release
cmake --build data_formats/build
"$cyxwiz/cyxwiz-cli" plugin-check data_formats/build/data_formats
```

The check loads the folder the way the Engine does and lists what the
plugin adds:

```
D:\work\data_formats\build\data_formats
  plugin com.cyxwiz.examples.data-formats 1.0.0 (API 1.3.0)
  ok   dataset format ARFF (Weka) (.arff)
  ok   analytic Coefficient of variation (Distribution)
cyxwiz-plugin-check: all plugins pass
```

Add `--open <file>` to read a file through your format the way Apply does,
and run your analytics on it, without starting the Engine:

PowerShell:

```powershell
& "$cyxwiz\cyxwiz-cli.exe" plugin-check --open data_formats\weather.arff data_formats\build\data_formats
```

Linux and macOS:

```bash
"$cyxwiz/cyxwiz-cli" plugin-check --open data_formats/weather.arff data_formats/build/data_formats
```

```
  ok   open weather.arff as ARFF (Weka): 14 rows, 5 columns in 1 batch
         columns: outlook (string), temperature (double), humidity (double), windy (string), play (string)
  ok   analytic Coefficient of variation on weather.arff (temperature, humidity): temperature 0.0893, humidity 0.1260
```

[Section 7](#7-test-and-debug) lists what `--open` checks and what its
failures mean.

The build file is the template's, with the example's names:

<!-- excerpt: plugins/examples/data_formats/CMakeLists.txt -->
```cmake
cmake_minimum_required(VERSION 3.20)
project(data_formats_plugin LANGUAGES CXX)

if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()

add_library(data_formats SHARED
    data_formats_plugin.cpp
)
target_link_libraries(data_formats PRIVATE CyxWiz::PluginSDK)

if(TARGET cyxwiz-backend)
    cyxwiz_plugin_folder(data_formats FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    cyxwiz_plugin_folder(data_formats)
endif()
```

The `if(TARGET cyxwiz-backend)` branch only matters inside the CyxWiz
sources, where the Engine's tests load the plugin from its source folder.
Against the SDK the `else` branch runs: the plugin folder is
`build/data_formats/`. Nothing else is linked: the plugin needs no Arrow
library (section 5).

To load your build, **Unload** the built example first if you installed it
(both have the same id), then **Install...** `data_formats/build/data_formats`.

## 4. What the Engine calls, in order

Your plugin is a class the Engine calls; it never calls the Engine back for
data. Every call below runs inside the Engine's crash guard: if your code
throws or crashes, that one action fails with your message and the Engine
keeps running.

| When | The Engine | Calls in your plugin | Engine source (CyxWiz repository) |
|---|---|---|---|
| Install, or the Engine starts | Loads the library, checks `plugin.json` (API 1.3 or newer) and the build fingerprint | `CreatePlugin` (from `CYXWIZ_PLUGIN_ENTRY`), `OnLoad`, `OnInitialize` | `plugin/plugin_loader.cpp`, `plugin/plugin_manager.cpp` (`LoadPlugin`, `InitializePlugin`) |
| Right after | Asks for each interface; needs the `DataRegistry` permission | `QueryInterface("IDataProvider")`, `QueryInterface("IAnalyticsProvider")` | `plugin/plugin_manager.cpp`, `plugin/engine_plugin_context.cpp` (`CheckDataInterfaces`) |
| Right after, once | Registers your formats and analytics | `GetLoaders()`, `GetAnalytics()` | `plugin/registries/plugin_data_loader_registry.cpp`, `plugin/registries/plugin_analytics_registry.cpp` (`Register`) |
| Data Input, Format *auto*, a file with your extension | Asks whether you read it (only when no built-in format reads that extension) | `CanLoad(loader_id, path)` | `PluginDataLoaderRegistry::LoaderForFile`, `core/data_input_external_formats.h` |
| Data Input, **Apply** (worker thread) | Opens the file through you, unless its Parquet copy is current | `OpenDataset(loader_id, path, out, error)` | `plugin/plugin_data_formats.cpp`, `core/data_registry.cpp` (`LoadTabularExternal`) |
| Still in Apply | Reads your stream to the end and writes each batch into the Parquet copy in the project's cache, then loads that copy (in memory, or on disk when large) | `get_schema`, `get_next` until no batch, `release` | `PluginStreamReader` in `plugin_data_loader_registry.cpp` |
| Explore > Plugins, **Run** (worker thread) | Streams the chosen columns of the table, from memory or from disk one row group at a time | `Compute(analytics_id, data)`, which reads `data` | `core/dataset_column_stream.cpp`, `PluginAnalyticsRegistry::Compute`, `gui/data_studio/plugins_view.cpp` |
| Unload, or the Engine closes | Removes your formats and analytics, waits for open streams and running `Compute` calls, frees the library | `OnShutdown`, `OnUnload` | `plugin/plugin_manager.cpp` (`UnloadPlugin`), `RemoveByPlugin` |

Two things follow from this order:

- **Apply reads your file once.** The Engine keeps a Parquet copy keyed by
  the file and your plugin's `version`, and reads that copy until the file
  changes or the version does. While you develop, raise the `version` in
  `plugin.json` (or touch the file) when your reader's output changes, or
  Apply keeps returning the old copy.
- **Training never calls your plugin.** It reads the table the Engine made.
  A graph that uses your format needs the plugin to Apply, not to train;
  remote training receives the loaded table.

## 5. The code, step by step

The plugin speaks the [Arrow C data and C stream
interface](https://arrow.apache.org/docs/format/CStreamInterface.html): a
handful of C structs (`ArrowSchema`, `ArrowArray`, `ArrowArrayStream`) that
`plugin/interfaces/arrow_c_stream.h` in the SDK declares exactly as the
Arrow specification does. Any Arrow library can fill and read them (Arrow
C++ `ExportRecordBatchReader`, nanoarrow, arrow-rs); the example fills and
reads them by hand, so you see every byte.

### 5.1 The manifest

```json
{
    "id": "com.cyxwiz.examples.data-formats",
    "name": "Example Data Formats",
    "version": "1.0.0",
    "api_version": "1.3.0",
    "capabilities": ["ProvidesData", "ProvidesAnalytics"],
    "permissions": ["DataRegistry"],
    "platforms": {
        "windows": { "library": "bin/data_formats.dll" },
        "linux": { "library": "bin/libdata_formats.so" },
        "macos": { "library": "bin/libdata_formats.dylib" }
    }
}
```

| Field | Why |
|---|---|
| `api_version` `1.3.0` | Formats and analytics took this shape in 1.3. A plugin declaring 1.2 loads, but the Engine does not ask it for them (*IDataProvider: built for plugin API 1.2.0; it needs 1.3.0 or newer*) |
| `permissions` `DataRegistry` | Without it the Engine does not register formats or analytics. The user sees it as *Dataset Access* on the card |
| `version` | Part of the cache key of every file read through you (section 4) |

The full field list is in the [API reference](plugin-api-reference.md#3-pluginjson).

### 5.2 The plugin class

One class is the plugin and both providers. `QueryInterface` hands the
Engine the interface it asks for:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
#include "plugin/plugin_types.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_analytics_provider.h"
#include "plugin/interfaces/i_data_provider.h"
// ...
class DataFormatsPlugin : public IPlugin, public IDataProvider, public IAnalyticsProvider {
public:
    bool OnLoad(PluginContext& ctx) override {
        ctx.LogInfo("Example Data Formats: OnLoad");
        state_ = PluginState::Loaded;
        return true;
    }
    // ...
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
```

and the last line of the file exports it:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::DataFormatsPlugin)
```

### 5.3 Declaring the format: `GetLoaders`

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
constexpr const char* kArffLoader = "com.cyxwiz.examples.data-formats.arff";
// ...
    std::vector<PluginDataLoaderInfo> GetLoaders() override {
        return {{kArffLoader, "ARFF (Weka)", {"arff"},
                 "Weka's attribute-relation file format: numeric attributes as numbers, the others as text"}};
    }
```

| Member | Becomes |
|---|---|
| `loader_id` | The format's identity. A Data Input saves it as `file_type` `plugin:com.cyxwiz.examples.data-formats.arff` in the graph: choose it once |
| `format_name` | *ARFF (Weka) (plugin)* in the Format list and Browse's file types |
| `extensions` | Matched in any case, with or without the dot |
| `description` | Shown with the format |

One plugin may return several loaders; the Engine passes the `loader_id`
back to every call, so one class can read several formats.

### 5.4 Claiming a file: `CanLoad`

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
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
```

Read only the beginning of the file and answer quickly: Data Input may ask
while the user is choosing. Return `false` for a file of another format
with the same extension, so the user gets a clear "unknown format" rather
than your parse error.

### 5.5 Opening the file: `OpenDataset`

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
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
```

`OpenDataset` does not read the rows. It checks the file can be read
(here: the header), and fills `*out` with four function pointers and a
pointer to your own state. The rows are read later, batch by batch, when the
Engine calls `get_next`.

| Outcome | Do | The user sees |
|---|---|---|
| The file cannot be read | Set `error`, return `false`, leave `*out` untouched | *ARFF (Weka) could not open &lt;file&gt;: cannot open the file* |
| It can | Fill `*out`, return `true` | Nothing yet: the rows come next |

The state here is the open file, the column list and a line counter for
error messages:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
struct ArffStream {
    std::ifstream in;
    std::vector<Attribute> attributes;
    std::string path;
    std::string last_error;
    int64_t line_number = 0;
};
```

### 5.6 The stream: schema, batches, errors, release

The Engine calls `get_schema` once: describe the columns. The schema of a
table is a struct (`"+s"`) whose children are the columns:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
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
```

Then `get_next` again and again. Each call returns the next rows as one
batch (a struct array with one child array per column), and a batch with
`release == nullptr` when there are no more:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
int ArffGetNext(ArrowArrayStream* stream, ArrowArray* out) {
    auto* state = static_cast<ArffStream*>(stream->private_data);
    std::vector<ColumnBuilder> builders(state->attributes.size());
    // ...
    int64_t rows = 0;
    std::string line;
    while (rows < kRowsPerBatch && std::getline(state->in, line)) {
        // ...
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
```

- **Bounded batches.** `kRowsPerBatch` is 50,000: the file is read as it
  is streamed, so a file larger than memory still loads (the Engine writes
  each batch to disk before asking for the next).
- **Errors while streaming.** Return a non-zero `errno` value and keep the
  message for `get_last_error`; the user sees it, with the line number here.

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
const char* ArffLastError(ArrowArrayStream* stream) {
    auto* state = static_cast<ArffStream*>(stream->private_data);
    return state->last_error.empty() ? nullptr : state->last_error.c_str();
}

void ArffRelease(ArrowArrayStream* stream) {
    delete static_cast<ArffStream*>(stream->private_data);
    stream->release = nullptr;
}
```

`release` frees your state and closes the file; the Engine calls it once,
when it has read the stream (or given up).

**Who owns what.** Every struct you hand over carries its own `release` and
`private_data`. The Engine copies each batch into its own memory as soon as
it has it and calls the batch's `release`; so your buffers only need to
live until then, and data read from your plugin stays readable after the
plugin is unloaded. The example keeps a column's buffers in a C++ object
that `private_data` points to, and deletes it in `release`:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
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
```

### 5.7 Columns in the Arrow layout

A column is a format string in the schema and buffers in each batch:

| Format | Type | Buffers of each batch's column |
|---|---|---|
| `g` | float64 | validity, values (`double[length]`) |
| `f` | float32 | validity, values (`float[length]`) |
| `l` | int64 | validity, values (`int64_t[length]`) |
| `i` | int32 | validity, values (`int32_t[length]`) |
| `b` | boolean | validity, values (one bit per row) |
| `u` | UTF-8 text | validity, offsets (`int32_t[length + 1]`), characters |
| `+s` | struct (the table, and each batch) | validity only; the columns are its children |

- **Validity** is one bit per row, lowest bit first, 1 for a value. It may
  be `nullptr` when `null_count` is 0.
- **Text**: row `r` is the characters from `offsets[r]` to
  `offsets[r + 1]`.
- Give numbers a number type: Explore and training work on number and text
  columns. Other Arrow types (dates, lists) load, but CyxWiz treats them as
  it treats the same type read from Parquet.

`ColumnBuilder` in the example fills one float64 or UTF-8 column; `?` (the
ARFF missing value) becomes a null:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
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
```

### 5.8 Declaring the analytic: `GetAnalytics`

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
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
```

| Member | Becomes in Explore > Plugins |
|---|---|
| `name`, `category` | *Coefficient of variation (Distribution)* in **Analytic**, grouped by category |
| `description` | The line under it |
| `columns` | Which columns the user may pick: `Numeric` dims the text columns with the reason |
| `min_columns`, `max_columns` | Run stays off outside this range (default: at least 1, no maximum) |

### 5.9 Computing: `Compute`

The Engine hands you the chosen columns, in the order picked, as a stream it
owns. Read it to the end during the call:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
    AnalyticsResult Compute(const std::string&, ArrowArrayStream* data) override {
        AnalyticsResult result;
        ArrowSchema schema{};
        if (data->get_schema(data, &schema) != 0) return Failed(result, data, "cannot read the columns");
        // ...
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
```

- **Every row, in batches.** A table kept on disk arrives one row group at a
  time, so keep running sums, not the whole column.
- **Release what you get.** Each batch and the schema you read are yours to
  `release`; the stream itself is the Engine's.
- **Cancel.** When the user presses Cancel, `get_next` fails: stop and
  return; the Engine drops the answer.
- **Respect `offset`.** A column may start `offset` rows into its buffers;
  the example's `IsValid` and `ForEachNumber` add it.

The answer is an `AnalyticsResult`:

<!-- excerpt: plugins/examples/data_formats/data_formats_plugin.cpp -->
```cpp
                result.vector_labels.push_back(schema.children[c]->name);
                result.vector_value.push_back(static_cast<float>(cv));
            }
            result.text_value = "0 means every value is the same; above 1, the spread exceeds the mean. "
                                "A column whose mean is 0 shows 0.";
```

| Field | Shown as |
|---|---|
| `scalar_value` with `has_scalar = true` | One large number at the top |
| `vector_value` with `vector_labels` | Bars with the value printed, and the **Values** table (Copy, Export CSV...) |
| `text_value` | A note under them |
| `success = false`, `error_message` | The failure, with your message |

## 6. Your own format

Start from the example, not from an empty file: it already has the
ownership code (5.6) and the column builders (5.7) right.

1. **Copy and rename.** Copy the example folder (without `built/`). In
   `plugin.json` set your own `id`, `name`, `version`, `author`, `license`;
   to rename the library, change `data_formats` in `CMakeLists.txt` and the
   three library paths in `plugin.json` (as in [quickstart section
   7](plugin-quickstart.md#7-make-it-yours)).
2. **Declare it.** In `GetLoaders`, a new `loader_id` (start it with your
   plugin id), the name users see, and the extensions.
3. **Recognise it.** In `CanLoad`, read the file's first bytes and answer
   whether it is yours.
4. **Open it.** In `OpenDataset`, open the file, read what you need before
   the rows (a header, a column list, a row count), keep it in your stream
   state, and fill `*out`. Fail here, with a message, when the file is not
   readable.
5. **Describe the columns.** In `get_schema`, one child per column, with a
   format from the table in 5.7. The schema must be the same for every
   batch.
6. **Read the rows.** In `get_next`, read up to a bounded number of rows,
   build one array per column, return them as one batch; at the end return
   a batch with `release` left null. On a bad row, keep the message for
   `get_last_error` and return `EINVAL`.
7. **Remove what you do not need.** If your plugin has no analytic, delete
   `IAnalyticsProvider` from the class, `QueryInterface` and the analytic's
   code, and `ProvidesAnalytics` from `plugin.json`.
8. **Build, check, Install, Apply** a real file of your format, then look
   at it in Preview and Data Studio. Raise `version` when you change what
   your reader returns (section 4).

A second format in the same plugin is one more `GetLoaders` entry and a
branch on `loader_id` in `CanLoad` and `OpenDataset`, for example (a sketch,
not part of the example):

<!-- not compiled: a sketch of a second loader in the same plugin -->
```cpp
std::vector<PluginDataLoaderInfo> GetLoaders() override {
    return {{kArffLoader, "ARFF (Weka)", {"arff"}, "..."},
            {kMyLoader, "My lab format", {"lab"}, "Readings exported by the lab's logger"}};
}

bool OpenDataset(const std::string& loader_id, const std::string& file_path, ArrowArrayStream* out,
                 std::string& error) override {
    if (loader_id == kMyLoader) return OpenLabFile(file_path, out, error);  // its own state and callbacks
    // ... the ARFF reader as above
}
```

**With an Arrow library.** If your plugin already links Arrow C++ (or uses
nanoarrow), build an `arrow::RecordBatchReader` and hand it over with
`arrow::ExportRecordBatchReader(reader, out)`; read the analytic's input
with `arrow::ImportRecordBatchReader(data)`. Any Arrow version works: the C
structs are the stable boundary. The SDK does not ship Arrow, so you add it
to your own build.

## 7. Test and debug

- **The check** (`cyxwiz-cli plugin-check <folder>`) proves the plugin
  loads and its formats and analytics register (`ok   dataset format ...`,
  `ok   analytic ...`, or `FAIL not added: ...` with the reason).
- **The check with files** (`plugin-check --open <file> <folder>`, as many
  `--open` as you like) reads each file as Apply does: the format its
  extension and your `CanLoad` pick, every batch, then the whole table
  validated against the Arrow layout (offsets, buffer sizes, null counts).
  Then each analytic runs on the columns it reads (up to its maximum), as
  Explore > Plugins does with its default columns. The answers to fix:

  | Line | Meaning |
  |---|---|
  | `ok   open <file> as <format>: R rows, C columns in B batches`, then the column types | The file reads; check the types are the ones you meant |
  | `FAIL open <file>: no dataset format of this plugin reads .<ext> files` | No loader of yours lists that extension |
  | `FAIL open <file>: <format> does not take the file (its CanLoad returned false)` | Your `CanLoad` refused it |
  | `FAIL open <file>: <format> could not open <file>: <error>` | `OpenDataset` returned false with that error |
  | `FAIL open <file>: batch N: <message>` | `get_next` failed on that batch; the message is your `get_last_error` |
  | `FAIL open <file>: the data is not valid Arrow: ...` | A batch's buffers do not match its schema: a wrong offset, length or null count |
  | `ok   analytic <name> on <file> (<columns>): <values>` or `FAIL analytic ...: <error>` | Your `Compute` answer, or its `error_message` |
  | `Plugin '<id>' crashed during <call>` in a FAIL line | Your code crashed there |

  A file that fails makes the exit code 1, like a node that fails.
- **Apply** in a Data Input reads it in the Engine. The Engine's log
  (`engine_log.txt` in the base folder, or the Console panel) has a line per
  load: *LoadTabularExternal: '&lt;file&gt;' (&lt;format&gt;) streamed 14 rows into
  &lt;cache file&gt;*, or the error.
- **Errors the user sees:** your `error` from `OpenDataset`
  (*&lt;format&gt; could not open &lt;file&gt;: ...*), your `get_last_error`
  message, your `error_message` from `Compute`. Write them for the user:
  what is wrong, and where (the example names the line).
- **A crash** in any of your calls fails that action with *Plugin '&lt;id&gt;'
  crashed during OpenDataset* (or `get_next`, `CanLoad`, `Compute`) *...;
  the Engine kept running*. Fix it before you ship: after a crash the
  plugin's state is unknown until the Engine restarts.
- **Debugging in Visual Studio:** attach to `cyxwiz-engine.exe` and set a
  breakpoint in `OpenDataset` or `Compute`; both run on a worker thread. The
  Engine loads a copy of your library ([developer guide, section
  11](plugin-developer-guide.md#11-loading-unloading-and-debugging)).
- **A new build:** Unload, rebuild, Install again, then raise `version` or
  touch the data file so Apply reads it through the new code.

The Engine's own tests load this example the same way:
`cyxwiz-engine/tests/test_plugin_data_providers.cpp` (registration,
batches, nulls, errors, unloading, crashes) and
`test_plugin_data_formats_input.cpp` (Data Input) and
`test_explore_plugins.cpp` (Explore > Plugins, in memory and on disk) in the
CyxWiz repository.

## 8. Limits today

| Works | Not yet |
|---|---|
| Table formats: rows and typed columns | Images, audio or text-document formats from a plugin |
| Files on disk, any size (streamed in batches) | Databases or network sources (Data Studio's SQL Workbench covers databases) |
| Analytics answering with a number, labelled values and a note | Analytics that draw their own charts, or add columns to the table |
| One plugin, several formats and analytics | Choosing the analytic's columns with `--open` (it uses the default columns) |
