#pragma once

// Sorting that also returns where each element came from. ArrayFire's
// sort-with-index (af::sort(out, indices, in, ...)) never returns on the oneAPI
// backend (Intel Iris Xe, ArrayFire 3.x, found 2026-10-14) while its
// sort-by-key works on every backend, so every argsort goes through sort-by-key
// with a 0..n-1 payload along the sorted dimension. Same results: both are
// stable sorts.

#include <arrayfire.h>

namespace cyxwiz {

// `sorted` = `values` sorted along `dim`; `index` (u32) = each sorted
// element's position along `dim` in `values`.
inline void SortWithIndex(af::array& sorted, af::array& index, const af::array& values, unsigned dim = 0,
                          bool ascending = true) {
    af::sort(sorted, index, values, af::range(values.dims(), static_cast<int>(dim), u32), dim, ascending);
}

}  // namespace cyxwiz
