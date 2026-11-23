# Seeded ArrayFire draws across backends and operating systems

A standalone check of whether ArrayFire gives the same numbers for one seed
everywhere. For the Philox, Threefry and Mersenne engines (seed 42) it prints
the first values of `randu` and `randn`, the first values of a 65,536-value
`randu`, that draw's sum, and bit hashes of 65,536-value `randu` and `randn`
draws. It depends on ArrayFire only. Compare the output
line by line between machines and backends.

Result on 2026-11-10 (ArrayFire 3.10.0; dev PC GTX 1050 Ti on Windows, hp-linux
Arch with GCC 16):

- **The operating system doesn't change the numbers.** CPU on Windows and CPU
  on Linux print the same numbers (Mersenne `randn` differs in the 7th decimal).
- **CUDA and OpenCL agree with each other** (sums to float rounding).
- **The CPU backend differs from CUDA/OpenCL:**
  - Philox (the default engine): `randu` matches on all three backends, but
    `randn` differs after its first value.
  - Threefry: `randu` and `randn` differ after the first value.
  - Mersenne: the first 6 `randu` values match; larger draws and `randn` differ.

Bit-level check: the probe also hashes the bits of a 65,536-value draw.

- Philox `randu` is bit-identical on Windows CPU, CUDA, OpenCL and Linux CPU.
- Threefry and Mersenne `randu` are bit-identical within CPU (both OSes) and
  within CUDA/OpenCL, but not across the two.
- `randn` is never bit-identical, not even between Windows CPU and Linux CPU.

So the only seeded ArrayFire draw that is the same everywhere is Philox
`randu`. Normal values built from it (Box-Muller) agree to float rounding
(TOFIX205).

```
cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
cmake --build build --config Release
build/Release/random_engine_portability_probe <cpu|cuda|opencl|oneapi>
```
