---
title: Error messages and diagnostic codes
description: Every CW diagnostic code the Engine and plugins use, with a typical message, its cause and the fix.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - ../troubleshooting/index.md
  - ../troubleshooting/compile-and-training.md
  - ../troubleshooting/data-loading.md
  - ../user-guide/graph-editor/compile-and-run.md
---

Many Engine messages start with a diagnostic code in brackets, for example
`[CW-C-0201] Data is not loaded - open the node and click Apply`. The code
names the kind of problem; the text names the node and the fix. A message
can end with ". Detail: ..." (what was seen) and ". Hint: ..." (what to
try).

## How a code is built

`CW-<family>-<number>`: the letter is the family, the first two digits the
kind of problem, the last two the case.

| Family | Letter | Covers |
| --- | --- | --- |
| Compiler | `C` | Graph checks in the compile report (F7) |
| Runtime | `R` | Running a data pipeline or a graph |
| GPU back end | `G` | GPU routes |
| CPU back end | `P` | The CPU route |
| Data | `D` | Columns, labels, splits, preparing data |
| File | `F` | Reading and writing files |
| Memory | `M` | Host and GPU memory, batch size |
| UI workflow | `U` | An action that cannot run in the current state |
| Serialization | `S` | Saving, exporting and loading models and checkpoints |
| Training | `T` | Building and running a training |
| External | `X` | Plugins, Python, remote services |

The middle digits repeat across families: `01` missing or unavailable,
`02` unsupported, `03` mismatch, `04` invalid parameter, `05` failed while
running, `06` a dependency failed, `07` resources exhausted, `08` state
could not be saved, `09` an internal invariant broke (a defect: report it).

The codes are defined in the SDK header `cyxwiz/error_codes.h`, so a plugin
can raise them too. Codes marked "not raised by the Engine" are in the
catalog for plugins and for later use.

## Find a code in the Console

In **Console > Commands** type `show code CW-C-0201`: it prints the family,
the code's name and the last 20 log events with that code. `show logs
codes CW-G-*` lists events of a whole family. See
[Console and logs](../main-window/console.md).

## Compiler (CW-C)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-C-0001` | GenericIssue | A finding with no code of its own (the report shows none) | Read the message. |
| `CW-C-0101` | MissingTrainingPathNode | "Graph must have a DataInput node" (or a loss, an optimizer, a model layer); "Graph is empty - add nodes to create a model" | Add the missing node and connect it. |
| `CW-C-0102` | UnsupportedTrainingNode | "Sparse feature training supports exactly one reachable CountVectorizer or TFIDFVectorizer on the selected path" | Change the path as the message says; sparse features need a Dense first layer and one vectorizer. |
| `CW-C-0103` | InvalidConnectivity | "Scheduler '...' is not connected to the training optimizer"; "Regularization '...' must sit between the training loss and the optimizer" | Fix the link the message names. |
| `CW-C-0201` | DataNotLoaded | "Data is not loaded - open the node and click Apply" | **Open its dialog**, then **Apply**. |
| `CW-C-0202` | DataCheckSuggested | "Check the data before a long training: Data Studio's Explore and Validate show missing values, class balance and column types" | A note. **Open Data Studio** to check the data. |
| `CW-C-0203` | TextNotTokenized | "'...' gives text, and nothing turns it into numbers before the model" | **Add a Text Tokenizer** (or a TF-IDF or Count vectorizer). |
| `CW-C-0204` | ImageNotResized | "'...' gives images, and no Resize sets their size" | **Add a Resize** after the Data Loader. |
| `CW-C-0205` | AudioFeaturesMissing | "'...' gives audio, and no node turns it into features" | **Add Mel Spectrogram**, **Add Spectrogram** or **Add MFCC**. |
| `CW-C-0206` | LearnsFromEveryRow | "'...' learns from every row before the Data Split, the Validation and Test rows too" | Move the step after the Data Split; see [Steps learn from the Train rows](../concepts/steps-learn-from-train-rows.md). |
| `CW-C-0207` | ColumnRolesKeepOut | "Column roles keep ... column(s) out of the features: ..." | A note on the Data Input's column roles; change the roles if a column should be a feature. |
| `CW-C-0301` | TensorShapeMismatch | "Cross Attention Key and Value must have the same length"; "Extension node '...' cannot take input ..." | Fix the shapes of the layers the message names. |
| `CW-C-0302` | LabelOutputShapeMismatch | "TimeSeriesWindow target width (...) does not match the model output width (...)"; "CrossEntropy class_weights size (...) does not match class/output count" | Make the last layer's width match the labels or the forecast horizon. |
| `CW-C-0401` | InvalidParameter | "label_smoothing must be a finite float in [0, 1)"; "The optimizer's lr_schedule (...) and the scheduler '...' both set the learning rate" | Correct the setting in Properties. |
| `CW-C-0901` | InvariantViolation | "Compilation threw an exception: ..." | A defect. Report it with the log. |

## Runtime (CW-R)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-R-0101` | PipelineMalformed | "Link references missing start node id: ..."; "Source node '...' must not have input connections" | Remove the link into the source node; a link to a missing node means a damaged graph file: recreate the link. |
| `CW-R-0102` | InvalidState | "Pipeline is already executing" | Wait for the running pipeline. |
| `CW-R-0201` | UnsupportedNode | "Node '...' of type '...' is not supported by PipelineExecutor" | The node runs only in training; Execute Pipeline leaves it to Train. |
| `CW-R-0202` | OperatorMissing | Not raised by the Engine | - |
| `CW-R-0301` | InputDatasetMissing | "Node '...' requires an input connection"; "No dataset is selected or loaded." | Connect the node's input; load the Data Input. |
| `CW-R-0401` | InvalidParameter | "Node '...' of type '...' is missing required parameter '...'" | Set the parameter in Properties. |
| `CW-R-0501` | ExecutionFailed | A run failed while executing; also the code of a Windows crash report the Studio Debugger imports | Read the message; for a crash see [Crash dumps and logs](../troubleshooting/crash-dumps-and-logs.md). |
| `CW-R-0901` | InvariantViolation | Not raised by the Engine | - |

## GPU and CPU back ends (CW-G, CW-P)

None of these is raised by the Engine today; GPU route problems are shown in
**Preferences > Devices** (see [GPU and drivers](../troubleshooting/gpu-and-drivers.md)).

| Code | Name |
| --- | --- |
| `CW-G-0101` | BackendUnavailable |
| `CW-G-0201` | PathDisabledByPolicy |
| `CW-G-0301` | UnsupportedShape |
| `CW-G-0501` | KernelExecutionFailed |
| `CW-G-0601` | DependencyFailure |
| `CW-G-0701` | MemoryExhausted |
| `CW-P-0101` | BackendUnavailable |
| `CW-P-0301` | UnsupportedShape |
| `CW-P-0501` | OperationFailed |
| `CW-P-0701` | MemoryExhausted |

## Data (CW-D)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-D-0101` | RequiredColumnMissing | Not raised by the Engine | - |
| `CW-D-0102` | RequiredLabelColumnMissing | "Selected label column '...' is not present in registered dataset '...'"; "No label column selected - apply the Data Input node so the compiler can inspect its schema" | Choose an existing label column. |
| `CW-D-0301` | ColumnTypeMismatch | "Registered dataset '...' has no numeric feature columns after excluding label column '...'"; "Preprocessing node '...' is for ... data, but the selected DataInput is ..." | Use numeric features (encode text columns), or a step that matches the data kind. |
| `CW-D-0302` | RowCountMismatch | "Text dataset has no training samples." | Check the file and the split ratios. |
| `CW-D-0303` | ClassLabelMismatch | Not raised by the Engine | - |
| `CW-D-0304` | VocabularyCoverageWarning | Not raised by the Engine | - |
| `CW-D-0401` | InvalidSplit | "Validation split is 0 - training will run without validation metrics"; "Class '...' has ... image(s): its Test part is empty" | Set a validation ratio; give small classes more images or a larger ratio. |
| `CW-D-0501` | MaterializationFailed | "Smoke Run stopped while preparing data." | Run **Execute Pipeline** to see which step fails. |

## File (CW-F)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-F-0101` | PathMissing | "Input file is empty. Select a supported data file first."; "Output path is empty. Choose where to write the converted file." | Choose the file or the output path. |
| `CW-F-0102` | NotFound | "Input file does not exist: ..."; "File not found: ..."; "TextVocabulary vocab_file does not exist: ..." | Check the path. |
| `CW-F-0201` | UnsupportedFormat | "Input format is not supported. Choose ..."; "Output format is not supported. Choose ..."; "XLSX export is not supported. Choose a supported table output format." | Pick a format from the list in the message. |
| `CW-F-0401` | InvalidOption | "Output path extension does not match output_format '...'. Expected ..." | Give the output file the extension of its format. |
| `CW-F-0501` | ReadFailed | The reader's reason | Check the file and its format options. |
| `CW-F-0502` | WriteFailed | The writer's reason | Check free space and that the folder is writable. |
| `CW-F-0601` | PermissionDenied | "Could not create output folder '...': ..." | Choose a folder you can write to. |

## Memory (CW-M)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-M-0101` | ResourceCheckFailed | Not raised by the Engine | - |
| `CW-M-0701` | HostExhausted | Not raised by the Engine (memory problems while preparing data are explained in [Data loading problems](../troubleshooting/data-loading.md)) | - |
| `CW-M-0702` | GpuExhausted | Not raised by the Engine | - |
| `CW-M-0703` | BatchTooLarge | "batch_size (...) is larger than the train split (... rows)"; a warning above half the train rows | Lower **Batch size** on the Data Loader. |

## UI workflow (CW-U)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-U-0101` | InvalidWorkflowState | "Graph changed since the last Local Debug run. Consider F6 before F5."; "The last Local Debug (F6) failed on this graph. Consider fixing it before Train."; "Local Debug runs one batch of a model trained on data. An RL graph has no batch" | Follow the message. |
| `CW-U-0102` | MissingSelection | Not raised by the Engine | - |
| `CW-U-0201` | RuntimeOnlyNode | Not raised by the Engine | - |
| `CW-U-0401` | InvalidParameter | Not raised by the Engine | - |
| `CW-U-0901` | StateInvariantViolation | Not raised by the Engine | - |

## Serialization (CW-S)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-S-0101` | ArtifactPathMissing | Not raised by the Engine | - |
| `CW-S-0201` | ExportFormatUnavailable | "A graph with branches exports to a CyxWiz package (.cyxmodel) or Safetensors, not this format yet"; "...; ONNX cannot describe its computation" | Export as `.cyxmodel` or Safetensors. |
| `CW-S-0202` | ExportFormatNotCompiled | "ONNX export support not compiled. ..." | This build has no ONNX export; use another format, or build with `CYXWIZ_ENABLE_ONNX=ON`. |
| `CW-S-0501` | ModelSaveFailed | "A model with extension nodes needs its graph in the package (Include: Graph)" | Tick **Graph** under **Include** when you export. |
| `CW-S-0502` | ModelLoadFailed | "Invalid binary .cyxmodel" | The file is damaged or not a model; export it again. |
| `CW-S-0801` | CheckpointSerializationFailed | Not raised by the Engine | - |

## Training (CW-T)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-T-0101` | InvalidTrainingSetup | "The data steps before the Data Split failed: ..."; "... could not be started (a data pipeline run is in progress, or nothing feeds the Split)." | See [Compile and training problems](../troubleshooting/compile-and-training.md). |
| `CW-T-0102` | ModelBuildFailed | "Model build failed (no layers produced or invalid config)" | Compile (F7) and fix the layer findings. |
| `CW-T-0401` | LossSetupFailed | "ModelBuilder failed to create loss" | Check the loss node's settings. |
| `CW-T-0402` | OptimizerSetupFailed | "ModelBuilder failed to create optimizer" | Check the optimizer node's settings. |
| `CW-T-0501` | TrainingExecutionFailed | "Local Debug loss was not finite."; "Local Debug forward output contained non-finite values." | Scale the inputs, check the loss, lower the learning rate; see [The Studio Debugger](../user-guide/training/studio-debugger.md). |

## External (CW-X)

| Code | Name | Typical message | What to do |
| --- | --- | --- | --- |
| `CW-X-0201` | OptionalIntegrationUnavailable | Not raised by the Engine | - |
| `CW-X-0501` | ThirdPartyCallFailed | Not raised by the Engine | - |
| `CW-X-0601` | PluginDependencyFailure | "Extension node '...' is not installed. Install and approve the extension that provides it, or remove the node." | Install and approve the plugin, or remove the node. |
| `CW-X-0602` | PythonBridgeFailure | Not raised by the Engine | - |
| `CW-X-0701` | RemoteServiceUnavailable | Not raised by the Engine | - |

## Messages without a code

Most dialogs show plain messages with the fix in the text. They are listed
by symptom in [Troubleshooting](../troubleshooting/index.md):

- Train greyed out, "Training did not start: ...", Local Debug:
  [Compile and training problems](../troubleshooting/compile-and-training.md)
- Data Input loads, the memory guard, the data budget:
  [Data loading problems](../troubleshooting/data-loading.md)
- Device verification and GPU routes:
  [GPU and driver problems](../troubleshooting/gpu-and-drivers.md)
- Plugin loading: [Plugin and Python problems](../troubleshooting/plugins-and-python.md)
- Python for scripting, the installer:
  [Install and first launch problems](../troubleshooting/install-and-first-launch.md)

## Related

- [Troubleshooting](../troubleshooting/index.md)
- [Compile and run a graph](../user-guide/graph-editor/compile-and-run.md)
- [Console and logs](../main-window/console.md)
