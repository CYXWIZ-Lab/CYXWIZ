# oneAPI: indexed min / max along dim 1 return wrong results

A standalone reproducer for an ArrayFire oneAPI backend bug, filed upstream on 2026-10-14 as
https://github.com/arrayfire/arrayfire/issues/3732. It depends on ArrayFire only. When a release fixes
it, run the probe with `oneapi` before removing the workaround.

## Summary

On the oneAPI backend, `af::max(value, index, in, 1)` and `af::min(value, index, in, 1)` (the
reductions that also return the position) reduce only part of each row when the array has few rows
and more columns. Both the values and the indices are wrong. The plain reductions (`af::max(in, 1)`),
the same call along dim 0, and every other backend on the same machine are correct.

## Environment

- ArrayFire v3.10.0 (build 492718b5a), Windows binary installer.
- Windows 11 Pro Education, build 26300.
- Intel Iris Xe Graphics (11th Gen Core i5-1135G7), driver 32.0.101.7088 (2026-06-17).
- Wrong on: `AF_BACKEND_ONEAPI`, device 0 (Iris Xe).
- Right on the same machine: `AF_BACKEND_OPENCL` (Iris Xe) and `AF_BACKEND_CPU`.
- Right elsewhere: CUDA, OpenCL and CPU on an NVIDIA machine.

## Reproduce

```
cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
cmake --build build --config Release
build/Release/oneapi_indexed_ops_probe oneapi
```

A [3, 5] float32 matrix, column-major:

```
         col0    col1    col2    col3    col4
row 0  0.0555  0.8969  0.0944  0.2531  0.3456     max at col 1
row 1  0.5535  0.3002  0.0550  0.7665  0.1136     max at col 3
row 2  0.2385  0.3137  0.0022  0.0864  0.9150     max at col 4
```

`af::max(peak, index, x, 1)` returns indices `1 0 1` on oneAPI instead of `1 3 4`. Rows 1 and 2
look only at the first columns.

## Output on oneAPI

```
ArrayFire v3.10.0 (oneAPI, 64-bit Windows, build 492718b5a)
[0] Intel(R) OpenCL Graphics: Intel(R) Iris(R) Xe Graphics, 7129 MB (fp16 )
[3,5] argmax along dim 1: 1 0 1 (expected 1 3 4)
af::max with indices, dim 1, fixed [3,5]             WRONG
af::max with indices, dim 1, randn [3,5]             WRONG
af::min with indices, dim 1, randn [3,5]             WRONG
af::max without indices, dim 1, randn [3,5]          ok
af::max with indices, dim 1, randn [3,4]             ok
af::min with indices, dim 1, randn [3,4]             WRONG
af::max with indices, dim 1, randn [8,5]             ok
af::min with indices, dim 1, randn [8,5]             ok
af::max with indices, dim 1, randn [37,100]          WRONG
af::min with indices, dim 1, randn [37,100]          WRONG
af::max with indices, dim 1, randn [5,1000]          WRONG
af::min with indices, dim 1, randn [5,1000]          WRONG
af::max with indices, dim 1, randn [100,37]          ok
af::max with indices, dim 1, randn [2000,200]        ok
(every "without indices" control: ok)
some checks WRONG
```

The same binary prints `all checks ok` with `opencl` and `cpu` on that machine.

## Workaround

Reduce along dim 0 instead: swap the dimension with dim 0 (`af::reorder`), reduce, then swap
back. CyxWiz does this in `cyxwiz-backend/src/algorithms/arrayfire_indexed.h` (`MaxWithIndex`,
`MinWithIndex`).
