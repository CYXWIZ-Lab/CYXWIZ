#!/usr/bin/env python3
"""
Generate test HDF5 file for TableViewer testing
"""

import numpy as np

try:
    import h5py

    # Create sample data
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ])

    # Save to HDF5
    with h5py.File('test_data.h5', 'w') as f:
        f.create_dataset('data', data=data)
        print(f"Created test_data.h5 with shape {data.shape}")

except ImportError:
    print("h5py not installed. Install with: pip install h5py")
    print("Alternatively, create HDF5 file manually.")
