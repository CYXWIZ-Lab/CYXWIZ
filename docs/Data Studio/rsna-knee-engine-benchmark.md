# RSNA Knee Challenge Engine Benchmark

## Purpose

Use the RSNA Knee MRI abnormality detection challenge as a practical benchmark
for CyxWiz engine capability. The goal is not only leaderboard performance. The
goal is to expose where the engine handles real-world multimodal medical ML
workflows well, where it fails, and which missing pieces should be added first.

## Challenge Shape

Expected workload:

- Study-level MRI abnormality prediction.
- Multiple imaging series per exam.
- DICOM or DICOM-derived volume input.
- Radiology report text paired with imaging data.
- Multi-label or finding-level classification.
- Class imbalance and clinically meaningful validation metrics.
- Kaggle-style offline inference and submission packaging.

The exact schema, labels, metric, and test-time availability of reports must be
confirmed from the Kaggle Data and Evaluation tabs before implementation.

## Current Engine Fit

Usable now:

- Python scripting for exploratory pipelines and external PyTorch/MONAI runs.
- Training dashboard and custom metric plotting.
- CSV, Arrow, Parquet, HDF5, NPY, image-folder, image-CSV, text, audio, and
  time-series data surfaces.
- Image preprocessing and augmentation for 2D image classification workflows.
- BCE/BCEWithLogits, focal, Dice-family losses, and class imbalance hooks.
- Transformer, embedding, LSTM/GRU, attention, and basic sequence assets.
- ONNX/model import-export surfaces for downstream inference.
- CUDA/device reporting and backend placement diagnostics.

## Limitations Exposed

### Data Ingestion

Current limitation:

- No first-class DICOM study loader.
- No study/series hierarchy contract.
- No metadata-aware grouping by patient, study, series, plane, or sequence.
- Existing Kaggle loader is generic CSV/image-folder oriented and does not
  download or parse Kaggle competition assets.
- HDF5 can lazily load 3D/4D numeric arrays, but this assumes DICOM has already
  been converted elsewhere.

Smallest fix:

- Add a `MedicalVolume` dataset contract that can represent preconverted
  volumes first, before adding direct DICOM parsing.
- Add a manifest format mapping `study_id`, `series_id`, `path`, `modality`,
  `plane`, `shape`, and label columns.

### Volume Modeling

Current limitation:

- The graph compiler exposes Conv2D and has a Conv3D enum, but the active
  training path is still centered on flattened tensors, 2D image batchers, and
  sequential modules.
- No confirmed native 3D CNN training path for `[D, H, W, C]` or
  `[C, D, H, W]` volumes.
- No native multi-series study aggregation primitive.

Smallest fix:

- Support an external PyTorch training adapter as the first volume-modeling
  route.
- Then add graph/runtime contracts for `VolumeInput`, `SeriesEncoder`,
  `StudyPooling`, and `MultiLabelHead`.

### Multimodal Fusion

Current limitation:

- Text and image workflows exist separately, but there is no explicit
  image-plus-report fusion graph contract.
- There is no rule-aware control for whether report text is allowed at test
  time.

Smallest fix:

- Add a challenge config flag for `report_usage = train_only | train_and_test |
  disabled`.
- Add a manifest field for report text and keep image-only baseline runnable.

### Metrics And Validation

Current limitation:

- Classification metrics and ROC curve tools exist, but medical multi-label
  validation needs per-label AUROC/AUPRC, macro metrics, threshold sweeps,
  calibration checks, and grouped splits.
- Random row splits are unsafe for medical datasets if patient/study leakage is
  possible.

Smallest fix:

- Add grouped split support keyed by `patient_id` or `study_id`.
- Add a multi-label metric report artifact with per-label AUROC, AUPRC, F1,
  sensitivity, specificity, and threshold.

### Kaggle Packaging

Current limitation:

- The engine has a Kaggle dataset class, but it is not a competition workflow:
  it does not handle authenticated download, rule acceptance, competition file
  layout, submission generation, or Kaggle notebook constraints.

Smallest fix:

- Add a Kaggle competition manifest/import wizard that works from an already
  downloaded competition folder.
- Generate a lean Python inference script and `submission.csv` outside the C++
  engine runtime.

## Recommended Work Tracks

### Track 1: External Baseline Harness

Build the first competitive baseline in Python and use CyxWiz to orchestrate and
monitor it.

Deliverables:

- `experiments/rsna_knee/` Python project.
- Dataset scan command.
- Train command.
- Validation metric command.
- Inference/submission command.
- Training dashboard callback or log importer.

This gives immediate feedback while avoiding premature C++ medical-imaging
implementation.

### Track 2: Medical Volume Dataset Contract

Add a small engine-native dataset surface for preconverted medical volumes.

Deliverables:

- `DatasetType::MedicalVolume`.
- Manifest schema and validator.
- Lazy loading from `.npy` or `.h5`.
- Group-aware split metadata.
- Shape/profile preview in Data Studio.

This is the first engine change that directly addresses MRI limitations without
requiring full DICOM support.

### Track 3: Medical Metrics

Add metrics that match real medical competition workflows.

Deliverables:

- Multi-label metric report.
- Per-label AUROC/AUPRC.
- Threshold sweep and selected threshold export.
- Grouped validation split support.
- Dashboard import for metric CSVs.

### Track 4: Native Or Imported Inference

Use the engine for deployed inference after training in PyTorch.

Deliverables:

- Import ONNX or TorchScript model.
- Run a single preconverted study through the model.
- Validate output shape and label mapping.
- Generate local predictions table.

### Track 5: Direct DICOM Support

Only add this after the preconverted-volume path is working.

Deliverables:

- DICOM directory scanner.
- Series sorting by slice position/instance number.
- Metadata extraction.
- Windowing/normalization policy.
- Volume cache generation to `.h5` or `.npy`.

## First Implementation Slice

Start with Track 1 plus the smallest part of Track 2:

1. Create an `experiments/rsna_knee/` baseline harness once competition files
   are available locally.
2. Define a manifest format for preconverted studies.
3. Add a validator that can read the manifest and report:
   sample count, labels, study grouping, missing paths, duplicate studies, and
   volume shapes.

This gives a concrete engine limitation test:

- If CyxWiz cannot ingest and profile a manifest of medical volumes, it is not
  ready for this class of Kaggle challenge.
- If it can ingest/profile but not train volumes natively, the limitation is
  isolated to model/runtime support rather than data management.

## Data Needed Next

Before writing challenge-specific code, collect:

- `train.csv`
- `sample_submission.csv`
- file tree under the training images/reports folder
- one or two example studies
- Data tab description
- Evaluation tab description
- rules around report text at test time
