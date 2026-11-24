---
title: Filter a signal or split it by scale
description: Keep or remove a frequency band, run a kernel over a column, or split it by scale with wavelets in Explore > Signal, and add the step.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - signal-processing.md
  - explore.md
---

The **Filter** and **Wavelet** looks of Data Studio's **Explore > Signal**
tab change a number column: a band kept or removed, a kernel run over it,
or the column split by scale and denoised. Each puts a step into the graph
that adds the new columns beside the others, so training reads them. The
signal, its sample rate and the display unit are read as described in
[Analyse a signal](signal-processing.md#the-signal-the-sample-rate-and-the-unit).

## Before you start

- A loaded dataset with a number column, shown in Data Studio's rail.

## Filter a column

1. In **Data Studio > Explore**, open the **Signal** tab and pick the
   number column.
2. Pick the **Filter** look (or **Tools > Signal Processing > Filter
   Designer**).
3. Choose **Design** (a frequency band) or **Kernel** (a list of taps), and
   set it (see the sections that follow).
4. Read the figures and plots: the column with the filtered column, and the
   response in dB with the −3 dB mark.
5. Select **Add as a step**. The Filter Designer node (Design) or the
   Convolution 1D node (Kernel) goes after the data in use and adds
   `<column>_filtered` beside the other columns.

The filter is applied **zero-phase**: it is centred on each row (no delay),
and the rows beyond the ends repeat the edge value, so the first and last
(taps − 1) / 2 rows lean on the edges (the sentence says so). The figures:
the half-power (−3 dB) frequency of a low- or high-pass, the RMS removed and
kept of the column's RMS, the taps and their span.

### Design

A windowed-sinc FIR (scipy's `firwin`):

- **Keep** below (low-pass), above (high-pass), between (band-pass) or
  outside (band-stop) the **Cut**, typed in the display unit (per hour, for
  example); 0 means a tenth of the Nyquist frequency; band filters take a
  second cut.
- **Taps**: 101 by default, odd. More taps cut sharper and lean on more edge
  rows.
- **Shape**: Hamming by default, Hann, Blackman.

### Kernel

The Convolution Calculator: a list of taps run over the column, "same"
length (a table column keeps its rows):

- **Moving average** of N rows (1 / N each);
- **Gaussian** of N rows (sigma = N / 6, normalised);
- **Difference** ([−1, 0, 1] / 2: the change per row);
- **Second difference** ([1, −2, 1]: the curvature);
- **Typed** (comma-separated; the sum is shown).

The kernel's taps are listed under the response.

## Split a column by scale

1. In the **Signal** tab, pick the **Wavelet** look (or **Tools > Signal
   Processing > Wavelet Transform**).
2. Pick the **Wavelet** (db4 by default; haar, db1 .. db8, sym2 .. sym8,
   coif1 .. coif5) and the **Levels** (0: 4, or the maximum for the rows,
   which is said).
3. Optional: set **Denoise** (see the next section).
4. Read the parts. Each part is brought back to the rows: the approximation
   A<L> (the cycles of 2^(L+1) steps and longer) and the details D<L> .. D1
   (each an octave: D1 holds the cycles of 2 to 4 steps, D2 4 to 8, ...),
   named in time units from the step ("D4: 2.7 min .. 5.3 min" at 10-second
   steps).
5. Select **Add as a step**. The Wavelet Transform node adds
   `<column>_A<L>`, `<column>_D<L>` .. `<column>_D1` and, with denoising on,
   `<column>_denoised` beside the other columns.

The transform is PyWavelets' `wavedec` in symmetric mode. The figures: each
part's share of the energy (the sum of its squared coefficients over the
total), the reconstruction error (the largest difference between the column
and the parts summed: 0 up to rounding), and the threshold when denoising.
The plots: the column with its approximation, then each detail; the table:
part, scale, coefficients, energy.

### Denoise

**Denoise** off, **soft** (every detail coefficient shrunk by the
threshold) or **hard** (those below it zeroed), at the **threshold** typed,
or 0: the universal threshold, the noise SD (the median |D1| / 0.6745) times
sqrt(2 ln n). The denoised column is drawn against the original, and the
sentence says the noise SD and by how much RMS the column changed.

## The nodes

The Filter Designer, Convolution 1D and Wavelet Transform nodes, their
parameters and their output columns are listed in
[Analyse a signal](signal-processing.md#the-nodes).

## Related

- [Analyse a signal](signal-processing.md)
- [Explore your data](explore.md)
