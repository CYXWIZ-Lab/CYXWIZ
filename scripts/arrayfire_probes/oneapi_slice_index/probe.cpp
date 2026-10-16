// Minimal reproducer for an ArrayFire oneAPI backend bug (found 2026-10-14 on an
// Intel Iris Xe, ArrayFire 3.10.0). Depends on ArrayFire only.
//
// Indexing a 1-D slice that starts at a non-zero offset with an index array
// ignores the slice's offset: for data = 0..99 and view = data(seq(10, 29)),
// view(idx) with idx = {0, 1, 2, 3} returns 0 1 2 3 instead of 10 11 12 13.
// af::lookup(view, idx), view(seq(...)) and a copied slice are right, as are
// the OpenCL backend on the same GPU, CUDA and CPU.
//
//   oneapi_slice_index_probe [cpu|cuda|opencl|oneapi]
//
// Prints one line per check and exits 1 if any check is wrong.

#define NOMINMAX
#include <arrayfire.h>

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void Check(const char* what, const af::array& got, const af::array& want) {
    const bool same = got.elements() == want.elements() &&
                      af::allTrue<bool>(af::flat(got).as(f32) == af::flat(want).as(f32));
    float g[4] = {0, 0, 0, 0};
    af::flat(got)(af::seq(0, 3)).as(f32).host(g);
    std::printf("%-30s %-5s got %g %g %g %g\n", what, same ? "ok" : "WRONG", g[0], g[1], g[2], g[3]);
    if (!same) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            const char* name = argv[1];
            if (!std::strcmp(name, "cpu")) af::setBackend(AF_BACKEND_CPU);
            else if (!std::strcmp(name, "cuda")) af::setBackend(AF_BACKEND_CUDA);
            else if (!std::strcmp(name, "opencl")) af::setBackend(AF_BACKEND_OPENCL);
            else if (!std::strcmp(name, "oneapi")) af::setBackend(AF_BACKEND_ONEAPI);
            else {
                std::fprintf(stderr, "usage: %s [cpu|cuda|opencl|oneapi]\n", argv[0]);
                return 2;
            }
        }
        af::info();
        const af::array data = af::range(af::dim4(100), 0, f32);  // 0 .. 99
        const af::array view = data(af::seq(10, 29));             // 10 .. 29
        const af::array copy = view.copy();
        const unsigned i[4] = {0, 1, 2, 3};
        const af::array idx(4, i);
        std::printf("expected: 10 11 12 13\n");
        Check("view(idx)", view(idx), copy(idx));
        Check("flat(view)(idx)", af::flat(view)(idx), af::flat(copy)(idx));
        Check("lookup(view, idx)", af::lookup(view, idx), af::lookup(copy, idx));
        Check("view(seq(0, 3))", view(af::seq(0, 3)), copy(af::seq(0, 3)));
    } catch (const af::exception& e) {
        std::fprintf(stderr, "ArrayFire error: %s\n", e.what());
        return 2;
    }
    std::printf("%s\n", failures == 0 ? "all checks ok" : "some checks WRONG");
    return failures == 0 ? 0 : 1;
}
