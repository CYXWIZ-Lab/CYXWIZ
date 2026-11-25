---
title: Analyse a signal
description: Find the cycles in a number column with the spectrum and spectrogram of Explore > Signal, and the signal nodes that do the same in a graph.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - filter-a-signal.md
  - time-series.md
  - explore.md
redirect_from:
  - ../../usage/signal_processing.md
---

Data Studio's **Explore > Signal** tab reads one number column as a signal
over time: which cycles carry its energy, how that changes along the rows, a
band kept or removed, and the column split by scale. This page covers the
**Spectrum** and **Spectrogram** looks; **Filter** and **Wavelet** are in
[Filter a signal or split it by scale](filter-a-signal.md). The nodes listed
at the end do the same inside a graph, so the answer runs with the rest of
the pipeline.

**Tools > Signal Processing** opens the tab on the right look: **FFT
(Fourier Transform)** on Spectrum, **Spectrogram (STFT)** on Spectrogram,
**Filter Designer** on Filter, **Convolution Calculator** on Filter in
Kernel mode, **Wavelet Transform** on Wavelet.

The figures are the same as numpy, scipy and PyWavelets give
(`numpy.fft.rfft` / `irfft`, `scipy.signal.periodogram`, `welch`,
`spectrogram`, `firwin`, `freqz`, `find_peaks`, `pywt.wavedec` / `waverec`);
the backend is checked against them on three signals. The device does the
work: the FFTs, the spectrogram as one batched FFT over the frames, the
convolutions, the windowed sincs, the wavelet filter passes. Only the filter
tables (PyWavelets' coefficients) and the peak search run on the host.

## Before you start

- A loaded dataset with a number column, shown in Data Studio's rail.

## Find the cycles in a column

1. In **Data Studio > Explore**, open the **Signal** tab.
2. Pick the number column. **Time** and the sample rate are read for you
   (see the next section); type over the rate when it is wrong.
3. Pick **Spectrum** and read the strongest cycles and the peaks.
4. Optional: pick **Spectrogram** to see how the cycles change along the
   rows.
5. Optional: **Keep as a node** adds the FFT node after the data in use as a
   branch.

## The signal, the sample rate and the unit

The tab reads **Time** as the Series tab does (see
[Analyse a time series](time-series.md#the-time-column-the-step-and-the-period)):
the Data Input's time column, else the one date column, else the row order.
The rows are sorted by it; rows missing the time or the value are left out
and counted.

The **sample rate** is 1 / the step between rows: 0.1 Hz for rows every
10 s, 12 per year for months, 1 per unit for a number time column, 1 per row
for the row order. The field shows 0 ("1 / the step") and can be typed over
in Hz (per unit, per row). Uneven spacing is said ("uneven: the rows are
taken in order") and the look still runs on the rows in order.

Frequencies are shown in the **unit that fits**: the smallest of Hz, per
minute, per hour, per day, per week, per month and per year in which the
Nyquist frequency (half the sample rate) reads as at least 10. A 10-second
step reads per hour (Nyquist 180 per hour); a monthly step reads per year.
Cycles are also worded as periods ("every 34 min", "every 12 months"), and
the Filter look's cut is typed in that same unit.

Every look ends its text with the rate line: "Sample rate 0.1 Hz = 360 per
hour (from the step: every 10 s, even); 1,440 rows, none missing."

## Spectrum

The one-sided amplitude spectrum of the column: a **Window** (Hann by
default; Hamming, Blackman, or None for a signal already periodic in its
length) and the **mean removed** first (else the 0 bin tops everything);
**the trend** removed too when asked (a least-squares line: a drifting
series otherwise puts its whole length on top). The amplitude of a bin is
2 |X| / the window's sum, so a pure tone reads as its own amplitude in the
column's units. **Scale** shows the amplitude, the power density (scipy's
`periodogram`, density) or that in dB.

**Average**: the whole series, or **Welch** segments of N rows (half
overlapping, the mean removed per segment, scipy's `welch`): smoother and
coarser bins, power only. Above 100,000 rows Welch is on by default
(1,024-row segments) until picked otherwise.

**Peaks in the spectrum**: the bins that rise a twentieth of the tallest
above their surroundings (scipy's `find_peaks` by prominence; the 0 bin is
not a cycle), the strongest first, at most ten, each as a period and a
frequency; a peak at an integer multiple of the strongest is named
"harmonic 2 of it". The strongest cycle is marked on the plot and worded in
the sentence ("Motor_current cycles every 34 min (1.75 per hour, amplitude
1.944), with its harmonic every 17 min. Resolution 0.25 per hour (4 h of
rows)."). The resolution is the sample rate over the rows (or over the
segment).

**Peaks in the rows**: the rows that stand out by **N SD** (typed; the SD
of the column is shown as a value) above the higher of the valleys around
them, at least **N rows** apart (typed; the span is shown), in time order
with their value and the gap to the one before; the count and the median gap
in the footer and the sentence. The rows plot marks them.

**Keep as a node** adds the FFT node as a branch: its table is the spectrum
(one row per bin), so training does not read it. The node has no Welch
averaging, so the button is off while Average is Welch (its reason on
hover). **Export CSV** writes the bins (frequency, period, amplitude, power
density); **Export the row peaks** writes them (row, time, value,
stands_out_by).

## Spectrogram

How the spectrum changes along the rows: frames of **Window** rows every
**Hop** rows (256 and 64 by default; their spans are said), the **Shape**
(Hann by default), the power density per frame and bin (scipy's
`spectrogram`, density, frame times at the window's centre), drawn as a
heatmap with the frames across and the frequencies up. **Scale** dB or
power; **Range** is the floor, that many dB below the loudest bin (60 by
default): cells below it take the background, and the frequency axis is
cropped to the highest bin that rises above it (the heading says so). The
figures: frames x bins, the resolution (the sample rate over the window,
and the hop's span), the loudest bin.

At most 4,000 frames are drawn: on a long series the hop is raised to stay
within them, and the sentence says from what.

## The nodes

Every node reads `signal_col` in table order and refuses a missing value
("fill or drop those rows first"): a node keeps every row, so it cannot
leave rows out the way the tab does. **Add as a step** and **Keep as a
node** fill the parameters from what the tab showed, the sample rate
resolved.

| Node | Parameters | Output |
| --- | --- | --- |
| FFT | `signal_col`, `sample_rate` (per second; the tab writes the step's), `window` hann / hamming / blackman / boxcar, `remove_mean`, `remove_trend` | a new table of n / 2 + 1 rows: `frequency`, `period` (null at 0), `amplitude` (2 \|X\| / the window's sum), `phase`, `power_density`, `real`, `imag` (numpy's `rfft` of the windowed, mean-removed signal). A branch: the row count changes, so training does not read it |
| IFFT | `samples` (0: 2 (bins − 1), the even length) | a new table of `samples` rows with `signal` from `real` and `imag` (numpy's `irfft`): the signal the FFT node transformed, windowed and mean-removed as it was |
| Filter Designer | `signal_col`, `filter_type` lowpass / highpass / bandpass / bandstop, `cutoff` (per second; 0.1 by default), `cutoff_high` (band filters), `sample_rate` (1 by default), `taps` (31; odd for highpass / bandstop), `window` hamming / hann / blackman / boxcar | the input plus `<signal_col>_filtered` (float64), zero-phase with the edges repeated |
| Convolution 1D | `signal_col`, `kernel` (comma-separated taps) | the input plus `<signal_col>_filtered`, "same" length, zero-phase |
| Wavelet Transform | `signal_col`, `wavelet` (21 names), `level` (3), `denoise` off / soft / hard, `threshold` (0: universal) | the input plus `<signal_col>_A<level>`, `<signal_col>_D<level>` .. `_D1` and `<signal_col>_denoised` when denoising is on |

Filter Designer graphs that carry `order` instead of `taps` still load: it
is read as 2 · order + 1 taps. The filtered column sits beside the
original; it does not replace it.

## Python

`pycyxwiz.signal` exposes the same backend: `spectrum`, `welch`,
`spectrogram`, `fft` / `ifft`, `lowpass` / `highpass` / `bandpass` /
`bandstop` (taps mean taps), `filter`, `conv` / `conv2`, `wavedec`,
`denoise`, `findpeaks` / `findpeaks_prominence`, and the generators `sine`,
`square`, `noise` (see the
[Python API reference](../../sdk/python-api/index.md#pycyxwizsignal)).

## Related

- [Filter a signal or split it by scale](filter-a-signal.md)
- [Analyse a time series](time-series.md)
- [Explore your data](explore.md)
