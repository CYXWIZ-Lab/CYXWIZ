#!/usr/bin/env python3
"""Smoke tests for tensor parity bindings."""

import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUILD_CONFIG = "Release" if (ROOT / "build" / "lib" / "Release").exists() else "Debug"
BUILD_LIB = ROOT / "build" / "lib" / BUILD_CONFIG
BUILD_BIN = ROOT / "build" / "bin" / BUILD_CONFIG
VCPKG_BIN = ROOT / "build" / "vcpkg_installed" / "x64-windows" / "bin"
ARRAYFIRE_BIN = Path(r"C:\Program Files\ArrayFire\v3\lib")

if os.name == "nt":
    dll_dirs = [BUILD_BIN, BUILD_LIB, VCPKG_BIN]
    if ARRAYFIRE_BIN.exists():
        dll_dirs.append(ARRAYFIRE_BIN)
    for dll_dir in dll_dirs:
        os.add_dll_directory(str(dll_dir))
    os.environ["PATH"] = os.pathsep.join(str(p) for p in dll_dirs) + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(BUILD_LIB))

import pycyxwiz as cx  # noqa: E402


def assert_array(actual, expected):
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=actual.dtype))


def main() -> None:
    base = cx.Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))

    assert base.reshape([3, 2]).shape() == [3, 2]
    assert base.unsqueeze(0).shape() == [1, 2, 3]
    assert base.flatten().shape() == [6]
    assert base.transpose().shape() == [3, 2]
    assert base.permute([1, 0]).shape() == [3, 2]

    assert_array(base.sum(0).to_numpy(), [3, 5, 7])
    assert_array(base.mean(1, True).to_numpy(), [[1], [4]])
    assert_array(base.max(1).to_numpy(), [2, 5])
    assert_array(base.var(1).to_numpy(), [2 / 3, 2 / 3])

    base.set(1, 2, 42.0)
    assert base.at(1, 2) == 42.0
    assert_array(base.slice(1, 0, -1, 2).to_numpy(), [[0, 2], [3, 42]])
    assert_array(base.index_select(0, [1, 0]).to_numpy(), [[3, 4, 42], [0, 1, 2]])

    left = cx.Tensor.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    right = cx.Tensor.from_numpy(np.array([4, 5, 6], dtype=np.int32))
    assert_array(left.dot(right).to_numpy(), [32])

    a = cx.Tensor.from_numpy(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = cx.Tensor.from_numpy(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    assert a.batch_matmul(b).shape() == [2, 2, 2]

    assert cx.Tensor.is_broadcastable([2, 1], [2, 3])
    assert cx.Tensor.broadcast_shape([2, 1], [2, 3]) == [2, 3]
    assert_array(cx.Tensor.from_numpy(np.array([[1], [2]], dtype=np.int32)).expand([2, 3]).to_numpy(),
                 [[1, 1, 1], [2, 2, 2]])

    cat = cx.Tensor.cat([cx.Tensor.ones([1, 2]), cx.Tensor.zeros([1, 2])], 0)
    assert cat.shape() == [2, 2]
    assert len(cat.split(1, 0)) == 2
    assert len(cat.chunk(2, 0)) == 2

    mask = base > 1.0
    assert mask.get_data_type() == cx.DataType.UInt8
    assert_array((mask & mask.logical_not()).to_numpy(), np.zeros((2, 3), dtype=np.uint8))


if __name__ == "__main__":
    main()
    print("tensor parity binding smoke test passed")
