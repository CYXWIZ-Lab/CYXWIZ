#pragma once

// ArrayFire calls that also return indices, routed around two oneAPI backend
// bugs (Intel Iris Xe, ArrayFire 3.10, found 2026-10-14). Use these instead of
// the af:: calls.
//
// - Sort-with-index (af::sort(out, indices, in, ...)) never returns, while
//   sort-by-key works on every backend: SortWithIndex sorts by key with a
//   0..n-1 payload along the sorted dimension. Same results: both are stable.
// - Indexed min / max along a dimension other than 0 (af::max(value, index,
//   in, dim)) reduce only part of each row: on a [3, 5] matrix it returned
//   column 1 where column 4 holds the maximum. Along dim 0 they are right, so
//   MaxWithIndex / MinWithIndex swap `dim` with dim 0, reduce there and swap
//   back.

#include <arrayfire.h>

namespace cyxwiz {

// `sorted` = `values` sorted along `dim`; `index` (u32) = each sorted
// element's position along `dim` in `values`.
inline void SortWithIndex(af::array& sorted, af::array& index, const af::array& values, unsigned dim = 0,
                          bool ascending = true) {
    af::sort(sorted, index, values, af::range(values.dims(), static_cast<int>(dim), u32), dim, ascending);
}

// `values` with dimension `dim` and dimension 0 exchanged (its own inverse).
inline af::array SwapWithFirstDim(const af::array& values, unsigned dim) {
    switch (dim) {
        case 1: return af::reorder(values, 1, 0, 2, 3);
        case 2: return af::reorder(values, 2, 1, 0, 3);
        case 3: return af::reorder(values, 3, 1, 2, 0);
        default: return values;
    }
}

// The maximum along `dim` and its index (u32; the first one on ties).
inline void MaxWithIndex(af::array& value, af::array& index, const af::array& values, unsigned dim) {
    (af::max)(value, index, SwapWithFirstDim(values, dim), 0);  // (parenthesised: windows.h macros)
    value = SwapWithFirstDim(value, dim);
    index = SwapWithFirstDim(index, dim);
}

// The minimum along `dim` and its index (u32; the first one on ties).
inline void MinWithIndex(af::array& value, af::array& index, const af::array& values, unsigned dim) {
    (af::min)(value, index, SwapWithFirstDim(values, dim), 0);
    value = SwapWithFirstDim(value, dim);
    index = SwapWithFirstDim(index, dim);
}

}  // namespace cyxwiz
