# Data Registry Refactoring Plan

## Current State
- **File:** `cyxwiz-engine/src/core/data_registry.cpp`
- **Size:** 5,497 lines
- **Problem:** Monolithic file with 13 dataset implementations embedded

## Target Architecture

### New Directory Structure
```
cyxwiz-engine/src/core/
├── data_registry.{h,cpp}           (~800 lines - coordinator only)
├── dataset_base.h                   (~200 lines - Dataset interface)
└── datasets/
    ├── mnist_dataset.{h,cpp}        (~250 lines)
    ├── cifar_dataset.{h,cpp}        (~200 lines)
    ├── csv_dataset.{h,cpp}          (~300 lines)
    ├── hdf5_dataset.{h,cpp}         (~600 lines with ifdef guards)
    ├── tsv_dataset.{h,cpp}          (~250 lines)
    ├── json_dataset.{h,cpp}         (~300 lines)
    ├── txt_dataset.{h,cpp}          (~250 lines)
    ├── image_folder_dataset.{h,cpp} (~350 lines)
    ├── image_csv_dataset.{h,cpp}    (~750 lines - largest individual)
    ├── huggingface_dataset.{h,cpp}  (~450 lines)
    ├── streaming_dataset.{h,cpp}    (~500 lines)
    ├── kaggle_dataset.{h,cpp}       (~400 lines)
    └── custom_dataset.{h,cpp}       (~600 lines)
```

## Benefits
1. **Maintainability:** Each dataset is self-contained and easy to understand
2. **Build times:** Parallel compilation of dataset modules
3. **Testing:** Individual dataset unit tests
4. **Modularity:** Can swap implementations without affecting others
5. **Clarity:** 1k line limit enforces single responsibility

## Implementation Phases

### Phase 1: Extract Base Interface
- [x] Create `dataset_base.h` with Dataset abstract class
- [x] Move Dataset, DatasetHandle, DatasetInfo to base file

### Phase 2: Extract Dataset Implementations (13 files)
1. MNIST Dataset (~117 lines) - simple binary format loader
2. CIFAR10 Dataset (~83 lines) - binary batch format loader
3. CSV Dataset (~129 lines) - CSV parser with column mapping
4. HDF5 Dataset (~682 lines) - HDF5 file format with HighFive
5. TSV Dataset (~113 lines) - tab-separated values
6. JSON Dataset (~130 lines) - JSON array/object parsing
7. TXT Dataset (~122 lines) - plain text line-by-line
8. ImageFolder Dataset (~162 lines) - filesystem folder structure
9. ImageCSV Dataset (~504 lines) - images + CSV labels (LARGEST)
10. HuggingFace Dataset (~255 lines) - HF datasets library integration
11. Streaming Dataset (~296 lines) - chunked lazy loading
12. Kaggle Dataset (~894 lines) - Kaggle API integration
13. Custom Dataset (~142 lines) - user-defined loader

### Phase 3: Refactor DataRegistry
- Keep as factory/coordinator pattern
- LoadDataset() switches on type and instantiates appropriate class
- Maintains same public API (no breaking changes)

### Phase 4: Update Build System
- Add all new .cpp files to CMakeLists.txt
- Ensure proper include paths
- Verify no circular dependencies

### Phase 5: Testing & Validation
- Run existing tests (should pass with no changes)
- Verify no regressions
- Check memory usage (should be unchanged)

## Estimated Effort
- **Lines to move:** ~4,700 lines (dataset implementations)
- **Files to create:** 26 files (13 .h + 13 .cpp)
- **Files to modify:** 3 files (data_registry.{h,cpp}, CMakeLists.txt)
- **Time estimate:** 2-3 hours for careful extraction and testing

## Breaking Change Risk
**LOW** - Public API in data_registry.h remains unchanged. Only internal implementation split.

## Next Steps
1. User approval of plan
2. Create dataset_base.h
3. Extract datasets one by one (start with smallest: CIFAR10)
4. Test after each extraction
5. Update CMakeLists.txt incrementally
6. Final integration test

## Alternative: Phased Approach
If full refactoring is too aggressive, we can do in 2 phases:
- **Phase A:** Extract just the 5 largest datasets (saves 2,500 lines)
- **Phase B:** Extract remaining 8 datasets later

**Recommendation:** Full refactoring in one go to avoid merge conflicts
