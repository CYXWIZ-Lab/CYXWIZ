# sumByKey on sorted keys with runs

A standalone check that `af::sumByKey` merges runs of equal sorted `int32` keys (1 and 8 value
columns, runs of 1 to 30, 64 to 100,000 keys) and that a row-to-item mapping built from it
(inclusive scan of counts with zeros, `sumByKey`, scatter, scan) matches the host. It depends on
ArrayFire only.

word2vec, GloVe and fastText (TOFIX152) sum gradients of repeated rows with `sumByKey` on sorted keys.
During fastText P6 the CBOW trainer failed on OpenCL with a row mapping of that shape (the last
example of a batch lost its rows); this probe passes on OpenCL, CUDA and CPU on the dev machine
(GTX 1050 Ti, ArrayFire 3.10.0), so the cause was not `sumByKey` itself and was not isolated. The
trainer now maps rows without `sumByKey` (each item's index scattered to its first row, then a
running maximum). Run the probe after an ArrayFire upgrade.

```
cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
cmake --build build --config Release
build/Release/sum_by_key_runs_probe <cpu|cuda|opencl|oneapi>
```
