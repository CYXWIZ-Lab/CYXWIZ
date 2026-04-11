# Multi-Format Data Pipeline — Design Doc

**Status:** Draft v2 — revised after feedback on single-responsibility
nodes. Needs approval before implementation starts.
**Owner:** Data pipeline workstream (follow-up to tabular Arrow/Parquet work).
**Companion docs:** `CLAUDE.md` data path section, `data_pipeline.md` user guide.

## Guiding principle (non-negotiable)

**One concern per node.** This is the architectural direction for the
v2 CyxWiz engine and it applies to every part of this plan:

- `DataInput` = read raw bytes from disk. That is the entire job. No
  resize, no normalize, no tokenize, no feature extraction, no
  augmentation. The dialog asks "what file, what category, what
  format" and nothing else.
- Every transform is a dedicated node: `Resize`, `Normalize`, `MFCC`,
  `Tokenizer`, `DataSplit`, `DataLoader`, `Filter`, and so on.
- The graph is the single source of truth for the data pipeline. You
  see the pipeline at a glance by looking at the canvas. No hidden
  state in dialog checkboxes.
- Nodes compose. Reusable building blocks. If you need normalization
  for images and normalization for tabular, you use the same
  `Normalize` node configured differently — not two different dialog
  toggles hidden inside two different loader types.

Any design decision that contradicts this principle is wrong by
definition. If you see "we should put this knob on the DataInput
dialog to save a node," push back.

## Context

The tabular data path (CSV / Arrow / Parquet) shipped in v0.2.0 and is fully
wired end-to-end: `DataInput` dialog → `LoadTabularCSV` → in-memory Arrow or
disk-backed Parquet → `ArrowDatasetBatcher` → `TrainingExecutor`. Compile
gate, registry cleanup, async load, cache hygiene, properties panel — all
done.

Non-tabular formats are a different story. The recon done for this design
turned up a mix of "works in isolation but unreachable" and "UI renders but
nothing happens." This doc lays out what's there, what's broken, and a
phased plan to get to the target: **one DataInput node that handles
tabular, image, audio, text, and time-series with their natural
representations — reusing what already exists wherever possible.**

## Current state inventory

### What already works in isolation (reusable)

| Component | File | What it does |
|---|---|---|
| **`ImageUtils`** | `src/core/image_utils.cpp` | OpenCV-backed load / resize (5 interpolations) / color-space convert / save |
| **`ImageFolderDataset`** | `src/core/datasets/image_folder_dataset.h` | Lazy LRU-cached image loader from class-subdir layout (ImageNet-style). Returns real float pixels |
| **`ImageCSVDataset`** | `src/core/datasets/image_csv_dataset.cpp` | Image folder + CSV labels, dual-mode, same LRU lazy loader |
| **`MNISTDataset` / `CIFAR10Dataset`** | `src/core/datasets/` | Binary format preset loaders |
| **`Conv2DLayer` / `MaxPool2DLayer`** | `cyxwiz-backend/include/cyxwiz/layer.h` | Trainable, ArrayFire-backed, hooked into `TrainingExecutor` |
| **`AudioProcessing`** | `cyxwiz-backend/include/cyxwiz/audio_processing.h` | libsndfile (wav/flac/ogg/aiff) + FFTW3 (Spectrogram / MelSpectrogram / MFCC / augmentation) |
| **`AudioDataset`** | `src/core/formats/audio_dataset.cpp` | Lazy per-sample feature extraction, labeled-subdir scan |
| **`Tokenizer`** | `cyxwiz-backend/include/cyxwiz/tokenizer.h` | Whitespace / Word / Character modes, padding, truncation, special tokens |
| **`Vocabulary`** | same | Build from corpus, save/load, min-freq + max-vocab |
| **`TextDataset`** | `src/core/formats/text_dataset.cpp` | Loads txt / csv / json / labeled-subdir corpus with lazy tokenization |
| **`TimeSeriesDataset`** | `src/core/formats/timeseries_dataset.cpp` | Sliding windows, lag features, rolling stats, differencing |

**This is a lot of existing code that works.** The dataset layer is largely
complete across all four non-tabular domains. The gap is integration, not
implementation.

### What's broken or stubbed (needs fixing)

| Symptom | Root cause | Severity |
|---|---|---|
| **Image folder Apply + Train crashes at first batch** | Dialog calls `LoadImageFolderToArrow` (metadata-only: stores `image_path` as string, `label_id` as int32). Training routes through `ArrowDatasetBatcher` which skips string columns. Model expects e.g. 150528 features for 224×224×3, gets 1. Tensor shape mismatch. | **Ship-blocker for v0.2.0** |
| Audio file picked from DataInput → silent failure | Apply's audio branch only saves params. Falls through to `LoadArrowTable(file_path_)` which tries to parse MP3 as Arrow IPC | High |
| Video file picked from DataInput → silent failure | Same fall-through, no video loader exists anywhere in the codebase | Medium (no one uses it) |
| Text has no UI category | `FileCategory::Tabular / Image / Audio / Video` — text is missing. `TextDataset` exists but has no entry point | Medium |
| Augmentation / Spectrogram / MFCC / Tokenizer NodeTypes are recognized by graph editor but never extracted | `graph_compiler.cpp:ExtractPreprocessing` only handles Normalize / Reshape / OneHot. All other preprocessing nodes fall to the default no-op case | Medium (for Phase 4) |

## Target architecture

### Single entry point, domain-specific backends

```
                    DataInput Node (single UI entry)
                              │
             ┌────────────────┼────────────────┬────────────────┐
             │                │                │                │
         Tabular            Image            Audio            Text
             │                │                │                │
     LoadTabularCSV    LoadImageFolder  LoadAudioDataset  LoadTextDataset
             │                │                │                │
   ┌─────────┴────────┐       │                │                │
  Arrow           Parquet     │                │                │
    │                │        │                │                │
    └────────┬───────┘        │                │                │
             │                │                │                │
     arrow_datasets_   image_datasets_  audio_datasets_  text_datasets_
     parquet_backed_         │                │                │
             │                │                │                │
             └────────────────┴────────────────┴────────────────┘
                                    │
                         StartTrainingFromGraph
                                    │
                  Dispatch by domain → select batcher
                                    │
             ┌──────────────┬────────┴────────┬──────────────┐
             │              │                 │              │
   ArrowDatasetBatcher  ImageBatcher   AudioBatcher   TextBatcher
   (+ Parquet flavor)        │                │              │
             │              │                 │              │
             └──────────────┴────────┬────────┴──────────────┘
                                     │
                             TrainingExecutor
                                     │
                                 (IBatcher)
```

### Key design principles

1. **DataInput remains the single user entry point for loading bytes.**
   The dialog already has `SourceType::File → FileCategory::{Tabular,
   Image, Audio, Video}`. We add Text as a fifth category (and cut Video
   until a loader exists). The DataInput dialog asks only: category,
   file / folder path, format hint. No transform knobs.

2. **Each domain gets its own DataRegistry map.** We already have
   `arrow_datasets_` and `parquet_backed_datasets_` living side by side.
   Adding `image_datasets_`, `audio_datasets_`, `text_datasets_` follows
   the exact same pattern — no architectural departure.

3. **`IBatcher` is the polymorphic seam.** We already introduced it for
   `ArrowDatasetBatcher` / `ParquetArrowBatcher`. New batchers implement
   the same interface. `TrainingExecutor::RunTrainingEpoch(IBatcher&)`
   doesn't care which backend it's iterating.

4. **Reuse wherever possible, rebuild only what's missing.** The existing
   dataset classes (`ImageFolderDataset`, `AudioDataset`, `TextDataset`)
   already provide the lazy-load + transform loop. We wrap them in thin
   `IBatcher` implementations and feed them a declaratively-built
   transform pipeline from the graph.

5. **Preprocessing lives in dedicated graph nodes, always.** No knobs
   on the DataInput dialog for normalization, resize, augmentation,
   feature extraction, or tokenization. Every transform is a first-class
   node the user drops onto the canvas. The graph is declarative and
   visible; the dialog is dumb. See the "Preprocessing pipeline" section
   below for how this works architecturally.

6. **Compile gate learns about data↔model compatibility.** New checks:
   "first layer is Dense but data is 3D image with no Flatten upstream"
   → error; "first layer is Conv2D but data is 1D tabular" → error;
   "image dataset has no Resize node in the preprocessing chain" →
   error (because we can't batch variable-size images); "first layer is
   Embedding but data isn't text" → error. The gate already has the
   hooks for this, it just needs domain awareness + pipeline awareness.

## Preprocessing pipeline — the architectural question

**The tension:** DataInput should produce raw bytes. But batching
requires uniform tensor shape. Images come in different sizes; a
folder of 640x480 JPEGs and 1024x768 PNGs can't be stacked into one
tensor. Something has to resize them before batching.

The wrong answer is to put a "target size" knob on the DataInput
dialog and resize inside the loader. That hides the transform in
opaque dialog state, couples `DataInput` to image-specific concerns,
and means "image preprocessing" lives in two places (dialog + nodes).
It's what PyTorch and TF both deliberately avoid.

Three right answers to pick from. Rationale follows each.

### Option A: The loader inspects the graph (recommended for v0.2.x)

At training time, the batcher walks the preprocessing chain from
`DataInput` forward through the graph, collects every preprocessing
node's parameters into a `TransformPipeline` struct, and passes that
pipeline to the dataset class. The dataset applies the pipeline inside
`GetItem`.

```
DataInput → Resize(224, 224) → Normalize(mean, std) → Augmentation(...)
                                       │
                                       ▼
                       TransformPipeline = [
                         ResizeStep(224, 224),
                         NormalizeStep(mean, std),
                         AugmentStep(...)
                       ]
                                       │
                                       ▼
               ImageFolderDataset(path, pipeline)
                                       │
                              GetItem(i):
                                 raw = load_and_decode(paths[i])
                                 return pipeline.apply(raw)
```

**Why this is right for now:**

- DataInput stays dumb — zero knowledge of downstream transforms.
- The graph is declarative — the pipeline is visible on the canvas.
- The dataset class stays efficient — it can fuse decode + resize in
  one pass via OpenCV imread, avoiding a full-res intermediate.
- The existing `ImageFolderDataset` / `AudioDataset` / `TextDataset`
  already accept transform-style parameters; the refactor is small.
- Matches PyTorch's `ImageFolder(transform=Compose([...]))` pattern
  exactly, which is the reference ergonomics for this space.

**Cost:** the batcher has to be graph-aware. Not huge — the graph is
already known at batch construction time (`StartTrainingFromGraph`
passes nodes + links to the executor). We just add a pass that
extracts the pipeline.

### Option B: Variable-shape Sample type

Introduce a dynamic `Sample` type. Every preprocessing node operates
on `Sample → Sample`. Shapes are resolved at batch time; the
`DataLoader` node errors if samples aren't uniform.

**Why we're not doing this now:** it's a ground-up rewrite of the
tensor plumbing. The existing `IBatcher` interface, `TrainingExecutor`,
and backend layers all assume uniform-shape tensors. Pushing variable
shape through end-to-end is a v3 pivot, not a v0.2.x enhancement.

### Option C: Explicit decode node

`DataInput` produces file references (paths + metadata). A dedicated
`ImageDecode` / `AudioDecode` / `TextDecode` node reads the bytes and
produces the raw tensor. Preprocessing nodes follow.

**Why we're not doing this now:** adds one more required node in every
graph (users will forget it), and there's no existing `Decode` node to
build on. It's the cleanest long-term story but the cost is higher than
Option A.

### Decision: Option A

Phases 1-3 implement Option A. The `TransformPipeline` type lives in
`src/core/preprocessing/transform_pipeline.h`. Each domain has a
concrete pipeline type (`ImageTransformPipeline`, `AudioTransformPipeline`,
`TextTransformPipeline`) built from graph nodes by a domain-specific
extractor.

We keep a note in the "Open questions" section about revisiting
Option B/C in v3 if the need arises. For v0.2.x, Option A gives us
single-responsibility nodes without a ground-up rewrite.

## ExtractPreprocessing refactor (prerequisite)

Before any phase, we refactor `graph_compiler::ExtractPreprocessing`
from its current 3-case switch into a table-driven design. Each
preprocessing node type registers:

- A domain tag (`Image` / `Audio` / `Text` / `Tabular` / `TimeSeries`)
- A parameter extraction function that pulls fields from
  `MLNode::parameters` into a typed struct
- A target position in the appropriate domain pipeline

```cpp
// graph_compiler_preprocessing.cpp
struct PreprocessingNodeSpec {
    NodeType type;
    PreprocessingDomain domain;
    std::function<void(const MLNode&, PreprocessingPipeline&)> extractor;
};

static const std::vector<PreprocessingNodeSpec> kPreprocessingNodes = {
    // Tabular (existing)
    {NodeType::Normalize,     PreprocessingDomain::Tabular,     ExtractNormalize},
    {NodeType::OneHotEncode,  PreprocessingDomain::Tabular,     ExtractOneHot},
    {NodeType::TensorReshape, PreprocessingDomain::Tabular,     ExtractReshape},
    // Image (new)
    {NodeType::Resize,        PreprocessingDomain::Image,       ExtractResize},
    {NodeType::Augmentation,  PreprocessingDomain::Image,       ExtractAugmentation},
    // Audio (new)
    {NodeType::Spectrogram,   PreprocessingDomain::Audio,       ExtractSpectrogram},
    {NodeType::MelSpectrogram,PreprocessingDomain::Audio,       ExtractMelSpec},
    {NodeType::MFCC,          PreprocessingDomain::Audio,       ExtractMFCC},
    // Text (new)
    {NodeType::TextTokenizer, PreprocessingDomain::Text,        ExtractTokenizer},
    {NodeType::TextPadding,   PreprocessingDomain::Text,        ExtractPadding},
    // ...
};
```

This is ~0.5 day up front but saves us from a 500-line switch by
phase 3. New preprocessing nodes register as one-line entries. Domain
classification means we can ask "does this graph have at least one
image-domain preprocessing node?" without walking the whole tree.

## Phased rollout

### Phase 0 — Ship-blocker (this sprint, 1 day)

**Goal: stop the image-folder crash.**

1. In `DataInputDialog::Apply` folder branch, call `LoadImageFolder`
   (legacy, real pixels) instead of `LoadImageFolderToArrow`
   (metadata-only).
2. Delete `LoadImageFolderToArrow` entirely — it's misleading dead code.
3. Training dispatch already falls through to the legacy `StartTraining`
   for non-Arrow datasets, so routing works. Verify on a small image
   folder (e.g. 100-image subset of MNIST-as-images).
4. Audio/Video category Apply → show a clear modal: "Audio/video data is
   not yet supported in v0.2.0. Coming in v0.3." Don't fail silently.
5. Test: image folder training runs to completion without crash.

**Deliverable:** v0.2.0 can ship without the crash, images work via the
legacy path. Everything else is either supported (tabular) or explicitly
gated with a "coming soon" message.

### Phase 1 — Image first-class (next sprint, 4-5 days)

**Goal: images as a proper peer to tabular, composed entirely from
single-responsibility nodes.**

1. **ExtractPreprocessing refactor** (the prerequisite section above).
   Table-driven, domain-tagged. ~0.5 day.
2. New `ImageTransformPipeline` type. Holds an ordered list of
   `ImageTransformStep` (Resize, Normalize, Flip, Crop, etc.). Provides
   `apply(cv::Mat) → cv::Mat`.
3. **Core image transforms (built-in nodes, added in Phase 1):**
   - `Resize` — target_h / target_w / interpolation mode (nearest,
     bilinear, bicubic, lanczos, area — already available from
     `ImageUtils`)
   - `CenterCrop`, `RandomCrop` — crop_h / crop_w
   - `HorizontalFlip`, `VerticalFlip` — probability param
   - `Rotate` — angle / probability params
   - `Normalize` — **domain-aware per Decision 1.** The same
     `Normalize` node already used for tabular. Dialog renders
     image-mode controls (pixel scale, optional per-channel mean/std
     with ImageNet preset) when upstream is an image DataInput.
     Extractor dispatches on upstream category and produces an
     `ImageNormalizeStep`.
   - `Augmentation` — already exists as a NodeType, currently
     unwired. Keep the name for back-compat but treat it as a
     compound node that applies a probability-weighted mix of
     flips/rotates/color-jitter. Add extractor.
   - `ColorJitter`, `Brightness`, `Contrast`
   - `GaussianBlur`, `Grayscale`
   - `Flatten` — already exists, ensure 3D→1D works when an image
     feeds a Dense head
4. **Plugin image transforms (optional):** the existing
   `plugins/examples/image_nodes/` stays as a plugin example. Mark
   its `GaussianBlur` deprecated (superseded by the core node), keep
   `EdgeDetect` as-is. Third-party plugins can register additional
   transforms (Canny, Sobel, Albumentations wrappers) via the
   existing plugin-node registry; they show up in the node editor
   under "Plugin Transforms" once Phase 1 lands the core nodes.
5. `ExtractImagePreprocessing(nodes, links, start_node) → ImageTransformPipeline`
   walks the graph forward from the DataInput node, collects every
   image-domain preprocessing node it encounters until it hits DataLoader
   or a model layer, in order, into a pipeline.
6. `ImageFolderDataset` / `ImageCSVDataset` refactored to accept an
   `ImageTransformPipeline` in their constructor (or a setter), applied
   inside `GetItem`. The existing resize-in-loader logic becomes the
   `ResizeStep::apply` implementation — no behavioral change, just
   relocation.
7. New `ImageDatasetBatcher` implementing `IBatcher`. Thin wrapper over
   the existing dataset classes. Handles shuffling, batching, label
   one-hot. Uses the LRU cache the datasets already provide.
8. New `DataRegistry::image_datasets_` map + `GetImageDataset` /
   `IsImageDataset` / `RegisterImage` / `LoadImageFolder` /
   `LoadImageCSV`. The load functions take **no transform params** —
   the pipeline is built from the graph at training start, not at
   load time.
9. `MainWindow::StartTrainingFromGraph` learns a fourth dispatch
   branch: `IsArrow → IsParquet → IsImage → legacy`.
10. `TrainingManager::StartTrainingImage` mirrors the Arrow/Parquet
    methods. Takes the graph nodes + links so it can build the
    transform pipeline and pass it to the batcher.
11. **Compile gate: image-specific checks (errors unless noted).**
    - Image dataset + no `Resize` node upstream of DataLoader → error
      ("images need a Resize node to have a uniform batchable shape")
    - Image dataset + first model layer is Dense + no `Flatten` → error
    - Image dataset + first model layer is Conv2D → OK
    - Image dataset + Conv2D + input channel mismatch with image color
      mode → error (3 channels expected, dataset is grayscale)
    - `Normalize` before `Resize` → **warning** per Decision 2
    - `Flatten` before `Augmentation` → **warning** per Decision 2
12. **DataInputDialog image branch: SHRINKS.** Removes target_width,
    target_height, normalize, rgb, labels_csv. Only keeps: folder/file
    path, category=image. All the removed knobs become graph nodes.
13. Properties panel: still resolves image shape from the graph's
    Resize node (not from the dialog). `GetInputShapeFromDataset`
    walks the preprocessing chain forward from DataInput looking for
    a Resize node to get the target shape.
14. User guide: expand `data_pipeline.md` with an image section
    explaining the node-based pipeline.
15. Example project: `examples/image_classification/mnist_folder.cyxgraph`
    with a full node chain: DataInput → Resize(28,28) → Normalize →
    Flatten → Dense → Dense → CrossEntropyLoss + Adam.

**Deliverable:** image training works end-to-end via a fully node-
based preprocessing pipeline. The DataInput dialog has no image-
specific knobs. Users compose the pipeline visually. Compile gate
catches missing Resize / Flatten / shape mismatches before Train.

### Phase 2 — Audio (4-5 days)

**Goal: audio as a peer, same node-based pattern.**

1. New `AudioTransformPipeline` type. Ordered list of steps:
   `ResampleStep`, `TrimSilenceStep`, `SpectrogramStep`,
   `MelSpectrogramStep`, `MFCCStep`, `AudioAugmentationStep`.
2. **Wire existing preprocessing nodes for audio domain:**
   - `Spectrogram`, `MelSpectrogram`, `MFCC` (already exist as
     NodeTypes, currently unwired — add extractors)
   - `AudioAugmentation` (already exists — add extractor)
   - `Resample` (add new node for explicit resampling)
3. `ExtractAudioPreprocessing` walks the graph, builds the pipeline.
4. `AudioDataset` refactored to accept an `AudioTransformPipeline`
   instead of baking config into its constructor. The existing feature
   extraction moves into the `SpectrogramStep::apply` implementations.
5. New `AudioDatasetBatcher` implementing `IBatcher`.
6. New `DataRegistry::audio_datasets_` map + `LoadAudioDataset` that
   takes no transform params.
7. New `StartTrainingFromGraph` dispatch branch for audio.
8. `TrainingManager::StartTrainingAudio`.
9. **Compile gate: audio-specific checks.**
   - Audio dataset + no feature-extraction node (Spectrogram / MelSpec
     / MFCC) upstream of the model → error ("audio needs feature
     extraction; raw waveform is rarely useful as model input")
   - Audio dataset + first layer is Conv2D + 1-channel expected → OK
     for spectrogram; error if channels mismatch
10. **DataInputDialog audio branch: SHRINKS.** Removes sample_rate,
    mono, duration — those become `Resample` node params, `TrimSilence`
    node params, `MaxDuration` node params. Only keeps: folder/file
    path, category=audio.
11. User guide: audio section.
12. Example: `examples/audio_classification/esc10_mfcc.cyxgraph`.

**Deliverable:** audio training works via composed nodes. Load folder
of labeled wav files, drop Resample → MFCC → Normalize → Conv2D → ...
onto canvas, click Train.

### Phase 3 — Text (5-6 days)

**Goal: text as a peer, same node-based pattern.**

1. Add `FileCategory::Text` to the dialog.
2. New `TextTransformPipeline`. Steps: `LowercaseStep`,
   `TokenizeStep`, `BuildVocabStep`, `PadStep`, `TruncateStep`.
3. **Wire existing preprocessing nodes for text domain:**
   - `TextTokenizer` (already exists — add extractor with mode
     param: Whitespace / Word / Character)
   - `TextVocabulary` (already exists — add extractor with
     min_freq / max_size params, optional vocab file path)
   - `TextPadding` (already exists — add extractor with max_length)
   - `Lowercase` (add new node)
4. `ExtractTextPreprocessing` walks the graph.
5. `TextDataset` refactored to accept a `TextTransformPipeline`
   instead of baking tokenizer config into its constructor.
6. New `TextDatasetBatcher` implementing `IBatcher`.
7. New `DataRegistry::text_datasets_` map + `LoadTextDataset`.
8. Dispatch branch + `StartTrainingText`.
9. **Compile gate: text-specific checks.**
   - Text dataset + no `Tokenizer` node → error
   - Text dataset + no `Padding` node → error (can't batch
     variable-length sequences)
   - Text dataset + first layer is Dense directly on token IDs → warn
     (usually you want Embedding first)
   - Text dataset + `Embedding` as first layer + vocab_size mismatch
     with Tokenizer's vocab → error
10. **DataInputDialog text branch.** New category. Asks: folder / file,
    text column (for CSV / JSON), label column. No tokenizer knobs —
    those are all on the TextTokenizer node.
11. User guide: text section.
12. Example: `examples/text_classification/sst2_tokenized.cyxgraph`
    with DataInput → Lowercase → Tokenizer → Padding → Embedding →
    LSTM → Dense → CrossEntropyLoss.

**Deliverable:** text classification works end-to-end via composed
nodes. The Tokenizer config is on a node, visible and editable.

### Phase 2 — Audio (3-4 days)

**Goal: audio as a peer to image and tabular.**

1. New `DataRegistry::LoadAudioDataset(path, name, config)` wrapping
   `AudioDataset`. The config carries feature type (Spectrogram /
   MelSpectrogram / MFCC), sample rate, n_fft, hop_length, n_mels /
   n_mfcc, max_duration.
2. New `audio_datasets_` map + accessors.
3. New `AudioDatasetBatcher` implementing `IBatcher`. The existing
   `AudioDataset::GetItem` returns a flat float vector per sample; the
   batcher stacks them. Label vector from directory structure.
4. `StartTrainingFromGraph` fifth dispatch branch.
5. `TrainingManager::StartTrainingAudio`.
6. `DataInputDialog` audio branch rewritten: feature type dropdown,
   sample rate, window params, max duration. Apply calls
   `LoadAudioDataset`. No more silent fall-through.
7. Compile gate: "first layer is 1D Conv → input is 1D audio features";
   "first layer is Dense on raw waveform → warn about input size".
8. User guide: audio section.
9. Example: `examples/audio_classification/esc50_mfcc.cyxgraph`.

**Deliverable:** audio training works. Users can load a folder of labeled
wav/flac/mp3 files and train a CNN on MFCCs.

### Phase 3 — Text (3-5 days)

**Goal: text as a peer.**

1. Add `FileCategory::Text` to the dialog. File types: .txt, .csv with a
   text column, .json with a text field, .jsonl, or a labeled-subdir
   corpus.
2. New `DataRegistry::LoadTextDataset(path, name, config)` wrapping
   `TextDataset`. Config: tokenizer mode, max_length, lowercase,
   min_freq, max_vocab_size, optional vocab file path.
3. New `text_datasets_` map + accessors.
4. New `TextDatasetBatcher`. Returns `int[max_length]` per sample
   (padded). Label handling same as other domains.
5. `StartTrainingFromGraph` sixth dispatch branch.
6. `TrainingManager::StartTrainingText`.
7. `DataInputDialog` text branch: tokenizer mode, vocab config, text
   column selector (for CSV/JSON), lowercase toggle, max length. Apply
   calls `LoadTextDataset` which builds or loads the vocabulary.
8. Embedding layer integration: graph_compiler picks up `Embedding` as
   the expected first layer for text; compile gate enforces it.
9. User guide: text section.
10. Example: `examples/text_classification/imdb_tokenized.cyxgraph`.

**Deliverable:** text classification end-to-end. Load a labeled text
corpus, tokenize, embed, train.

### Phase 4 — Time-series (3-4 days)

**Goal: time-series forecasting, same node-based pattern.**

Note: the old "Phase 4 — preprocessing graph nodes" bolt-on is
**absorbed into phases 1-3**. Each domain phase wires its own
preprocessing nodes as it goes. There is no separate catch-up phase.

1. New `TimeSeriesTransformPipeline`. Steps: `WindowStep`,
   `LagFeaturesStep`, `RollingStatsStep`, `DifferenceStep`.
2. **Wire existing preprocessing nodes for time-series domain:**
   - `TimeSeriesWindow` (already exists — add extractor with
     window_size / stride / horizon params)
   - `TimeSeriesFeatures` (already exists — add extractor for
     lag / rolling / diff)
   - `TimeSeriesSplit` (already exists — add extractor; note this
     overrides the generic `DataSplit` behavior for time-series to
     enforce chronological split)
3. `ExtractTimeSeriesPreprocessing` walks the graph.
4. `TimeSeriesDataset` refactored to accept a pipeline rather than
   a config struct.
5. New `TimeSeriesDatasetBatcher` implementing `IBatcher`.
6. New `DataRegistry::timeseries_datasets_` map. Loading is triggered
   when the graph has a `TimeSeriesWindow` node downstream of a
   tabular DataInput — the loader notices this and registers the
   dataset under `timeseries_datasets_` instead of `arrow_datasets_`.
   **Important:** users don't pick "time-series" in the dialog — they
   pick a CSV and add a `TimeSeriesWindow` node. The graph drives the
   semantics.
7. Dispatch + `StartTrainingTimeSeries` + compile gate.
8. **Compile gate: time-series checks.**
   - `TimeSeriesWindow` node present + `DataSplit` node present
     (instead of `TimeSeriesSplit`) → warn (random split breaks
     temporal structure)
9. User guide: time-series section.
10. Example: `examples/timeseries/stock_forecast.cyxgraph`.

**Deliverable:** time-series workflow works via node composition.
Same DataInput, different downstream node chain.

### Phase 5 — Video (deferred, optional)

No existing code, low priority, no user asking for it. Cut from the
current plan. Revisit when someone files a feature request. Whenever
it lands, it follows the same pattern: `DataInput(category=video)` →
`VideoDecode` node → `VideoResize` node → `TemporalSample` node →
`Normalize` → model. The `VideoDataset` class doesn't exist yet; it
would wrap FFmpeg or similar.

## What gets deleted

- `DataRegistry::LoadImageFolderToArrow` — metadata-only, misleading,
  crashes training. Replaced by `LoadImageFolder` (Phase 0).
- `FileCategory::Video` in the dialog — no backing code, remove from
  the UI until Phase 5 (if ever).
- **All preprocessing knobs on the DataInput dialog for non-tabular
  categories.** Image: target_width, target_height, normalize, rgb,
  labels_csv. Audio: sample_rate, mono, duration. These move to
  dedicated nodes and the dialog shrinks. (Phase 1 for image, Phase 2
  for audio.)
- Any leftover "Memory Policy" / LRU / prefetch UI fragments still
  lurking in non-tabular panels (same as the tabular cleanup pass).
- `AudioDatasetConfig` struct fields that duplicate what preprocessing
  nodes will carry. The struct becomes a thin "loader path + optional
  pipeline" wrapper.

## Phase 0 action items (concrete)

To fix the ship-blocker NOW, even before the rest of the plan is
approved:

1. `data_input_dialog.cpp` folder branch — change the one line that
   calls `LoadImageFolderToArrow` to call `LoadImageFolder` (legacy).
2. Delete `DataRegistry::LoadImageFolderToArrow` (declaration + impl).
3. `StartTrainingFromGraph` — verify the legacy fallback branch actually
   picks up legacy image datasets (I believe it does since the existing
   dispatch goes Arrow → Parquet → legacy, but confirm by trace).
4. Audio/Video Apply branches — show a modal explaining they're not
   supported yet, return without triggering the tabular fall-through.
5. Test path: drop in a small image folder (e.g. 3 classes, 30 images
   each), click Apply, click Compile, click Train. Verify training runs
   for 1-2 epochs without crash.

## Resolved decisions

These were open in v2 and have been decided. Recording them here so
the rationale survives into future sessions.

### Decision 1: `Normalize` is a single domain-aware node

Just as `DataInput` is one node that supports tabular / image / audio /
text / time-series, `Normalize` is one node that dispatches by domain.
There is no `NormalizeTabular` / `NormalizeImage` split.

Implementation:

- The `Normalize` node's dialog inspects the upstream `DataInput`'s
  `file_category` and renders domain-specific controls:
  - **Tabular:** per-column z-score (stored mean/std), or min/max
  - **Image:** pixel scale ([0,1] or [-1,1]), optional per-channel
    mean/std subtraction (ImageNet defaults as a preset)
  - **Audio:** amplitude normalization (peak / RMS), or log-scale for
    spectrograms
  - **Text:** rarely useful, show "not applicable for text" hint
  - **Time-series:** z-score per feature, or rolling normalization
- `ExtractNormalize` in `graph_compiler_preprocessing.cpp` dispatches
  on the same upstream category and produces a domain-specific
  `NormalizeStep` inside the appropriate `TransformPipeline`.
- The `Normalize` node's pin types adapt to the upstream data type.

This is the same single-responsibility principle applied to
preprocessing nodes: one node, one concern ("normalize my data"),
implementation varies by data type.

### Decision 2: Compile gate warns on bad pipeline ordering

The compile gate emits Warning-level issues for pipeline ordering
that's almost certainly a mistake. It does **not** block Train —
users can ship what they want, they just see the flag.

Initial warning rules (extensible as we discover more):

- `Normalize` before `Resize` on an image → warn. Normalization stats
  are scale-dependent, so resizing after normalization gives wrong
  per-pixel values.
- `Flatten` before `Augmentation` on an image → warn. Augmentation
  needs spatial structure (rotate, flip, crop) that Flatten destroys.
- `DataSplit` before any data-dependent transform (like `Normalize`
  using dataset statistics) → warn if the transform uses statistics
  computed over the training set. This is more subtle; skip for
  phase 1 and revisit when we have domain-specific normalizers.
- Image dataset + no `Resize` before `DataLoader` → **error** (this
  one stays blocking because batching can't proceed without uniform
  shape). Already in the Phase 1 compile-gate list.

Each rule is a one-liner in the compile pass. Table-driven registration
so adding new rules is trivial.

### Decision 3: Core ships common transforms; plugins extend

The core engine ships a curated set of image / audio / text transforms
as built-in nodes. The plugin system is the extension point for more
specialized or third-party transforms.

**Core image transforms** (Phase 1 deliverable):
- `Resize`, `CenterCrop`, `RandomCrop`
- `HorizontalFlip`, `VerticalFlip`, `Rotate`
- `Normalize` (domain-aware per Decision 1)
- `ColorJitter`, `Brightness`, `Contrast`
- `GaussianBlur`, `Grayscale`
- `Flatten` (already exists)

**Plugin image transforms** (existing + future):
- `plugins/examples/image_nodes/` keeps `GaussianBlur` (marked
  deprecated — the core node supersedes it) and `EdgeDetect` as
  example plugins demonstrating how to register a new image
  transform from a DLL.
- Third-party plugins can add Canny / Sobel / Laplacian / advanced
  OpenCV ops, PyTorch-style augmentation via bindings, Albumentations
  wrappers, or user-specific transforms.
- Plugin-registered transforms surface in the node editor under a
  "Plugin Transforms" submenu, same way plugin ML nodes already do.

**Core audio transforms** (Phase 2 deliverable):
- `Resample`, `TrimSilence`, `NormalizeVolume`
- `Spectrogram`, `MelSpectrogram`, `MFCC`
- `AudioAugmentation` (compound node for TimeStretch / PitchShift /
  AddNoise; or break into three nodes — to be decided in Phase 2
  design)

**Core text transforms** (Phase 3 deliverable):
- `Lowercase`, `RemovePunctuation`
- `Tokenizer` (Whitespace / Word / Character)
- `Vocabulary`, `Padding`, `Truncation`
- Stop word removal, stemming, etc. — via plugin

### Decision 4: Phase 0 is the legacy-route hotfix

Phase 0 does NOT try to do node-based preprocessing. It uses the
legacy `LoadImageFolder` path with its baked-in resize. This is
acceptable because:
- It's a one-day hotfix to unblock a crash.
- Phase 1 immediately replaces it with the proper node-based pipeline.
- The legacy path is already written and tested; we're just routing
  to it instead of the broken `LoadImageFolderToArrow`.

Phase 0 ships with v0.2.0. Phase 1 lands in v0.2.1.

## Remaining open questions

1. **Text without labels** (language modeling vs classification):
   Phase 3 assumes classification. LM training needs a different
   "label = next token" wiring. Follow-up after Phase 3.
2. **ML dataset shortcuts** (MNIST, CIFAR10 via `MLDatasetType` enum
   in the dialog): currently dead because `LoadMLDatasetToArrow`
   returns null with a TODO. Fix as part of Phase 1 — call the
   existing `MNISTDataset` / `CIFAR10Dataset` loaders and register
   under `image_datasets_`, same compile-gate treatment as a normal
   image folder.
3. **Transform fusion:** should `ResizeStep` and `NormalizeStep` be
   fused into a single OpenCV call for performance? Optimization,
   not correctness — defer until after Phase 1 ships and we have
   benchmarks.
4. **AudioAugmentation as one node or three:** Phase 2 design
   question. Lean toward three separate nodes (`TimeStretch`,
   `PitchShift`, `AddNoise`) for composability, but one compound
   node is simpler to drop onto the canvas. Revisit when Phase 2
   starts.
5. **Pipeline cycle detection:** if a user wires preprocessing nodes
   into a loop, what happens? Existing `HasCycle` check in
   graph_compiler already catches this; verify it still fires for
   preprocessing chains in Phase 1.

## Effort estimate

| Phase | Days | Cumulative |
|---|---|---|
| Phase 0 (ship-blocker) | 1 | 1 |
| ExtractPreprocessing refactor (prerequisite for phase 1+) | 0.5 | 1.5 |
| Phase 1 (image first-class, node-based preprocessing) | 4-5 | 5.5-6.5 |
| Phase 2 (audio, node-based preprocessing) | 4-5 | 9.5-11.5 |
| Phase 3 (text, node-based preprocessing) | 5-6 | 14.5-17.5 |
| Phase 4 (time-series) | 3-4 | 17.5-21.5 |
| Phase 5 (video) | deferred | - |

Roughly 3-4 sprints of focused work for the full plan (phases 0-4).
Phase 0 is 1 day, immediate. Phase 0 + 1 together unblock the image
workflow with proper node-based preprocessing.

**Note on the total vs the previous draft:** this is slightly more
work (~17-21 days vs 14-20) because the old Phase 4 "wire the
preprocessing nodes" catchup is now spread across phases 1-3 where
it belongs. Net effort is roughly the same; architectural quality is
much higher because we ship each domain correctly from day one
instead of shipping a half-built version first and patching later.

## What this doc does NOT cover

- **Solana / payments integration** — unrelated.
- **Plugin API for custom datasets** — future work. For now everything is
  in-tree.
- **Distributed training via P2P with multi-format datasets** — assumes
  the Phase 2+ work has landed; cross-machine serialization for image /
  audio / text datasets is a separate design.
- **ONNX / PyTorch model import with multi-format data** — import
  pipeline is a separate track.

## Decision needed

Approve the phased plan (with or without edits), or push back on the
architecture. Specifically:

- **Yes to Phase 0 now?** Fixes the crash today, minimal risk. Uses
  the legacy path's baked-in resize. No node-based preprocessing in
  Phase 0 — that's Phase 1. Acceptable as a ship-blocker hotfix.
- **Yes to Option A (loader inspects the graph) for the architectural
  seam?** The `TransformPipeline` pattern. Alternative is Option B
  (variable-shape Sample type) which is a ground-up rewrite deferred
  to v3.
- **Yes to the ExtractPreprocessing refactor as a prerequisite?**
  Half-day up front, saves us from a 500-line switch by Phase 3.
- **Phases 1-4 order and scope ok?** Image → Audio → Text →
  Time-series. Or re-prioritize (e.g. text before audio)?
- **Are any of the open questions blockers for the plan itself?**
  In particular question 6 (one `Normalize` node or two).

Once approved, Phase 0 is ~1 day of work and can start immediately.
The ExtractPreprocessing refactor is a clean ~0.5 day that can also
start in parallel without blocking anything.

## Changelog

- **v2.1 (2026-04-11):** Resolved four open questions after feedback.
  (1) `Normalize` is a single domain-aware node, mirroring how
  `DataInput` handles multiple formats in one node. (2) Bad pipeline
  ordering is a warning, not a block — users can ship what they want
  but see the flag. (3) Core ships common transforms as built-in
  nodes; the plugin system is the extension point for specialized
  or third-party ops. GaussianBlur lives in core; the existing
  plugin example is deprecated. (4) Phase 0 stays as the legacy-route
  hotfix; node-based preprocessing lands in Phase 1. Added concrete
  list of core image/audio/text transforms to ship.
- **v2 (2026-04-11):** Revised after feedback. Removed "preprocessing
  on DataInput dialog" shortcut — contradicts single-responsibility
  principle. All preprocessing moves to dedicated graph nodes. Added
  Option A vs B vs C architectural discussion. Added
  `TransformPipeline` pattern. Added ExtractPreprocessing refactor as
  prerequisite. Absorbed the old Phase 4 (preprocessing node wiring)
  into phases 1-3 where it belongs. Effort estimate updated upward
  by ~3 days; architectural quality is much higher.
- **v1 (2026-04-11):** Initial draft with per-dialog preprocessing.
