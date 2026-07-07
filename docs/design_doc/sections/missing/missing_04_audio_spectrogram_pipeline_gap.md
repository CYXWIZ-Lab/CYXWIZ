# M-04) Advanced audio feature nodes end-to-end execution gap

## M-04.1 Question
Do `Spectrogram`, `MelSpectrogram`, and `MFCC` nodes run end-to-end in native training runtime today?

## M-04.2 Design facts present in code

```text
Graph compile
  -> detect audio preprocessing nodes
  -> TrainingConfiguration.audio_preprocessing populated
  -> PipelineMaterializer::Materialize reads source kind and batcher config
  -> TrainingExecutor receives dataset mode / batcher interface
```

## M-04.3 Observed boundary
- Audio preprocessing configuration exists in compiler and materializer data structures.
- Runtime capability tables still contain fail-closed support points for audio feature transforms in `PipelineRuntimeCapabilities`.
- Native execution path coverage is therefore partial: configuration is carried, but feature-node tensor transform coverage is incomplete for full E2E guarantee.

## M-04.4 Clarification of scope
- Audio preprocessing config extraction is **not** equivalent to guaranteed execution.
- The existing gap is specifically the **operator/runtime coverage** side: whether those preprocessing operators are implemented in execution stage for all input formats and dataset modes.

## M-04.5 Required validation next
- End-to-end launch test using:
  - graph with source -> Spectrogram/MelSpectrogram/MFCC -> model nodes,
  - dataset materialization path under audio dataset mode,
  - batcher + training loop with success/failure metric capture.

## M-04.6 Evidence anchors
- `cyxwiz-engine/src/core/graph_compiler.cpp:3681` (audio-domain preprocessing inference)
- `cyxwiz-engine/src/core/graph_compiler.cpp:4317` (audio extractor phase comment and behavior)
- `cyxwiz-engine/src/core/graph_compiler.cpp:4328` (ExtractSpectrogram sets audio preprocessing config)
- `cyxwiz-engine/src/core/graph_compiler.cpp:4344` (ExtractMelSpectrogram sets audio preprocessing config)
- `cyxwiz-engine/src/core/graph_compiler.cpp:4364` (ExtractMFCC sets audio preprocessing config)
- `cyxwiz-engine/src/core/graph_compiler.cpp:4464` (audio extractor mapping in domain table)
- `cyxwiz-engine/src/core/audio_dataset_batcher.h:35` (preprocess_config source contract)
- `cyxwiz-engine/src/core/audio_dataset_batcher.cpp:22` (feature type materialization policy)
- `cyxwiz-engine/src/core/audio_dataset_batcher.cpp:48` (graph/entry feature precedence comments)
- `cyxwiz-engine/src/core/audio_dataset_batcher.cpp:55` (Spectrogram override logic)
- `cyxwiz-engine/src/core/audio_dataset_batcher.cpp:71` (graph feature config overrides dialog)
- `cyxwiz-engine/src/core/audio_dataset_batcher.cpp:89` (AudioDataset construction from merged config)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:84` (Spectrogram declared not implemented in PipelineExecutor)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:85` (MelSpectrogram declared not implemented)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:88` (MFCC declared not implemented)
- `cyxwiz-engine/src/core/training_executor.cpp:306` (external IBatcher path used for audio/text/image)
- `cyxwiz-engine/src/core/training_executor.cpp:567` (comment path note for image/audio/text batchers)
