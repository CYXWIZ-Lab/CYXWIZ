# oneAPI: indexing a slice with an index array ignores the slice's offset

A standalone reproducer for an ArrayFire oneAPI backend bug, filed upstream on 2026-10-14 as
https://github.com/arrayfire/arrayfire/issues/3733. It depends on ArrayFire only. When a release fixes
it, run the probe with `oneapi`.

## Summary

On the oneAPI backend, indexing a 1-D array that is a slice with a non-zero offset (for example
`data(af::seq(10, 29))`) by an index array returns elements counted from the start of the
underlying array instead of from the slice. `af::lookup` on the same slice, indexing it with a
`seq`, and indexing a `copy()` of it are all correct. The OpenCL backend on the same GPU, CUDA
and CPU are correct.

## Environment

- ArrayFire v3.10.0 (build 492718b5a), Windows binary installer.
- Windows 11 Pro Education, build 26300.
- Intel Iris Xe Graphics (11th Gen Core i5-1135G7), driver 32.0.101.7088 (2026-06-17).
- Wrong on: `AF_BACKEND_ONEAPI`, device 0 (Iris Xe).
- Right: `AF_BACKEND_OPENCL` on that GPU, `AF_BACKEND_CPU`, and CUDA / OpenCL on an NVIDIA machine.

## Reproduce

```cpp
af::setBackend(AF_BACKEND_ONEAPI);
const af::array data = af::range(af::dim4(100), 0, f32);  // 0 .. 99
const af::array view = data(af::seq(10, 29));             // 10 .. 29
const unsigned i[4] = {0, 1, 2, 3};
const af::array idx(4, i);
af_print(view(idx));               // oneAPI: 0 1 2 3   expected: 10 11 12 13
af_print(af::lookup(view, idx));   // 10 11 12 13 on every backend
```

## Output on oneAPI

```
expected: 10 11 12 13
view(idx)                      WRONG got 0 1 2 3
flat(view)(idx)                WRONG got 0 1 2 3
lookup(view, idx)              ok    got 10 11 12 13
view(seq(0, 3))                ok    got 10 11 12 13
some checks WRONG
```

## Effect and workaround

In CyxWiz the word2vec trainer picked a batch's word ids this way from a slice of the epoch's
examples; on oneAPI it updated the wrong rows and the loss rose instead of falling. Using
`af::lookup(slice, idx)` (or indexing a `copy()`) gives the right elements.
