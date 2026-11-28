# A one-column product reads its reused result buffer (OpenCL)

A standalone check that `af::matmul` does not read its result buffer. It allocates the inputs,
fills a buffer of the result's size with NaN and releases it (so ArrayFire's memory pool hands it
out next), forms the product of the finite inputs and counts non-finite values in the result. It
depends on ArrayFire only.

On the dev machine (GTX 1050 Ti, ArrayFire 3.10.0, 2026-11-28) CPU (MKL) and CUDA (cuBLAS) give 0
bad products. OpenCL gives NaN for every product whose result has one column (4x8 * 8x1, 16x8 *
8x1, 1x8 * 8x1, plain and with the right side transposed); single-row and wider results are right.
ArrayFire forms a one-column product with its BLAS's matrix-vector routine (CLBlast, built into
afopencl.dll): y = A x + beta y with beta 0, and that routine reads y although the BLAS rule says
y is not read when beta is 0. In the Engine it showed as a model whose weights went NaN making the
next model's Dense 1 output NaN (TOFIX242 2.2), and earlier as random NaN rows of a float64
regression (ORA-14). The Engine forms every product through `cyxwiz::DeviceMatMul`
(`cyxwiz-backend/src/algorithms/arrayfire_matmul.h`), which gives a one-column product a zero
second column on OpenCL; a source scan refuses a direct `af::matmul`. Run the probe after an
ArrayFire upgrade: when OpenCL passes, the widening can go.

```
cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
cmake --build build --config Release
build/Release/matmul_reused_buffer_probe <cpu|cuda|opencl|oneapi>
```
