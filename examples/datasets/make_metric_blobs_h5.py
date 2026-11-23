"""Write metric_blobs.h5: the metric_blobs.csv table as an HDF5 file.

Data Input loads one numeric HDF5 dataset as a table (rank 1 or 2), with an
optional rank-1 label dataset of the same row count. This file keeps the 8
features in /blobs/features (480 x 8, float32) and the class in /blobs/labels
(480, int32), so Data Input reads it as columns col_0..col_7 plus "label".

    pip install h5py numpy
    python make_metric_blobs_h5.py
"""
import csv
from pathlib import Path

import h5py
import numpy as np

here = Path(__file__).resolve().parent
with open(here / "metric_blobs.csv", newline="") as source:
    rows = list(csv.DictReader(source))
features = np.array([[float(row[f"f{i}"]) for i in range(8)] for row in rows], dtype=np.float32)
labels = np.array([int(row["class"]) for row in rows], dtype=np.int32)

with h5py.File(here / "metric_blobs.h5", "w") as out:
    group = out.create_group("blobs")
    group.create_dataset("features", data=features)
    group.create_dataset("labels", data=labels)
    group.attrs["source"] = "metric_blobs.csv (480 rows, 8 features, 4 classes)"
print(f"wrote {here / 'metric_blobs.h5'}: features {features.shape}, labels {labels.shape}")
