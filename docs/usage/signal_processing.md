# Signal processing

One number column read as a signal over time: which cycles carry its energy,
how that changes along the rows, a band kept or removed, and the column split
by scale. Data Studio's Explore tab **Signal** answers those on any table
([Data Studio](data_studio.md#signal)); the nodes below do the same inside a
graph, so the answer runs with the rest of the pipeline. Tools > Signal
Processing opens the tab on the right look (FFT: Spectrum; Spectrogram;
Filter Designer: Filter; Convolution Calculator: Filter in Kernel mode;
Wavelet Transform: Wavelet).

The figures are the same as numpy, scipy and PyWavelets give (`numpy.fft.rfft`
/ `irfft`, `scipy.signal.periodogram`, `welch`, `spectrogram`, `firwin`,
`freqz`, `find_peaks`, `pywt.wavedec` / `waverec`): the backend is checked
against them on three signals (two tones in noise at 1 kHz, an odd length, a
slow 0.1 Hz series) in `test_computation_truth_signal`, 80 cases. The device
does the work: the FFTs, the spectrogram as one batched FFT over the frames,
the convolutions, the windowed sincs, the wavelet filter passes. Only the
filter tables (PyWavelets' coefficients) and the peak search run on the host.

## The signal, the sample rate and the unit

The tab reads **Time** as the Series tab does ([time
series](time_series.md#the-time-column-the-step-and-the-period)): the Data
Input's time column, else the one date column, else the row order; a date
column, a text that reads as dates, or numbers. The rows are sorted by it;
rows missing the time or the value are left out and counted.

The **sample rate** is 1 / the step between rows: 0.1 Hz for rows every 10 s,
12 per year for months, 1 per unit for a number time column, 1 per row for the
row order. The field shows 0 ("1 / the step") and can be typed over in Hz
(per unit, per row). Uneven spacing is said ("uneven: the rows are taken in
order") and the look still runs on the rows in order.

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
length) and the **mean removed** first (else the 0 bin tops everything); **the
trend** removed too when asked (a least-squares line: a drifting series
otherwise puts its whole length on top). The amplitude of a bin is 2 |X| /
the window's sum, so a pure tone reads as its own amplitude in the column's
units. **Scale** shows the amplitude, the power density (scipy's
`periodogram`, density) or that in dB.

**Average**: the whole series, or **Welch** segments of N rows (half
overlapping, the mean removed per segment, scipy's `welch`): smoother and
coarser bins, power only. Above 100,000 rows Welch is on by default (1,024-row
segments) until picked otherwise.

**Peaks in the spectrum**: the bins that rise a twentieth of the tallest
above their surroundings (scipy's `find_peaks` by prominence; the 0 bin is
not a cycle), the strongest first, at most ten, each as a period and a
frequency; a peak at an integer multiple of the strongest is named "harmonic
2 of it". The strongest cycle is marked on the plot and worded in the
sentence ("Motor_current cycles every 34 min (1.75 per hour, amplitude
1.944), with its harmonic every 17 min. Resolution 0.25 per hour (4 h of
rows)."). The resolution is the sample rate over the rows (or over the
segment).

**Peaks in the rows**: the rows that stand out by **N SD** (typed; the SD of
the column is shown as a value) above the higher of the valleys around them,
at least **N rows** apart (typed; the span is shown), in time order with
their value and the gap to the one before; the count and the median gap in
the footer and the sentence ("7 peaks stand out in the rows (by 1 SD = 1.919,
at least 100 s apart), 34 min apart on the median."). The rows plot marks
them.

**Keep as a node** adds the FFT node after the data in use as a branch: its
table is the spectrum (one row per bin), so training does not read it. The
node has no Welch averaging, so the button is off while Average is Welch
(its reason on hover).
**Export CSV** writes the bins (frequency, period, amplitude, power density);
**Export the row peaks** writes them (row, time, value, stands_out_by).

## Spectrogram

How the spectrum changes along the rows: frames of **Window** rows every
**Hop** rows (256 and 64 by default; their spans are said), the **Shape**
(Hann by default), the power density per frame and bin (scipy's
`spectrogram`, density, frame times at the window's centre), drawn as a
heatmap with the frames across and the frequencies up. **Scale** dB or
power; **Range** is the floor, that many dB below the loudest bin (60 by
default): cells below it take the background, and the frequency axis is
cropped to the highest bin that rises above it (the heading says so). The
figures: frames x bins, the resolution (the sample rate over the window, and
the hop's span), the loudest bin.

At most 4,000 frames are drawn: on a long series the hop is raised to stay
within them, and the sentence says from what.

## Filter

A band kept or removed, applied **zero-phase**: the filter is centred on each
row (no delay), and the rows beyond the ends repeat the edge value, so the
first and last (taps − 1) / 2 rows lean on the edges (said in the sentence).
The figures: the half-power (−3 dB) frequency of a low- or high-pass, the RMS
removed and kept of the column's RMS, the taps and their span. The plots: the
column with the filtered column, and the response in dB over the display unit
with the −3 dB mark.

**Design**: a windowed-sinc FIR (scipy's `firwin`): **Keep** below (low-pass),
above (high-pass), between (band-pass) or outside (band-stop) the **Cut**
(typed in the display unit, per hour; 0: a tenth of the Nyquist frequency;
band filters take a second cut), **Taps** (101 by default, odd; more taps cut
sharper and lean on more edge rows) and the **Shape** (Hamming by default,
Hann, Blackman).

**Kernel** (the Convolution Calculator): a list of taps run over the column,
"same" length (a table column keeps its rows): **Moving average** of N rows
(1 / N each), **Gaussian** of N rows (sigma = N / 6, normalised),
**Difference** ([−1, 0, 1] / 2: the change per row), **Second difference**
([1, −2, 1]: the curvature) or **Typed** (comma-separated; the sum is shown).
The kernel's taps are listed under the response.

**Add as a step** adds the Filter Designer node (Design) or the Convolution 1D
node (Kernel) after the data in use: it adds `<column>_filtered` beside the
other columns, and training reads it.

## Wavelet

The column split by scale with the discrete wavelet transform (PyWavelets'
`wavedec`, symmetric mode): a **Wavelet** (db4 by default; haar, db1 .. db8,
sym2 .. sym8, coif1 .. coif5), **Levels** (0: 4, or the maximum for the rows,
which is said) and each part brought back to the rows: the approximation
A<L> (the cycles of 2^(L+1) steps and longer) and the details D<L> .. D1
(each an octave: D1 holds the cycles of 2 to 4 steps, D2 4 to 8, ...), named
in time units from the step ("D4: 2.7 min .. 5.3 min" at 10-second steps). The figures: each part's share of the energy
(the sum of its squared coefficients over the total), the reconstruction
error (the largest difference between the column and the parts summed: 0 up
to rounding), and the threshold when denoising. The plots: the column with
its approximation, then each detail; the table: part, scale, coefficients,
energy.

**Denoise** off, **soft** (every detail coefficient shrunk by the threshold)
or **hard** (those below it zeroed) at the **threshold** typed, or 0: the
universal threshold, the noise SD (the median |D1| / 0.6745) times
sqrt(2 ln n). The denoised column is drawn against the original and the
sentence says the noise SD and by how much RMS the column changed.

**Add as a step** adds the Wavelet Transform node after the data in use: it
adds `<column>_A<L>`, `<column>_D<L>` .. `<column>_D1` and, with denoising
on, `<column>_denoised` beside the other columns, and training reads them.

## The nodes

Every node reads `signal_col` in table order and refuses a missing value
("fill or drop those rows first"): a node keeps every row, so it cannot leave
rows out the way the tab does. The tab's "Add as a step" / "Keep as a node"
fill the parameters from what it showed, the sample rate resolved.

| Node | Parameters | Output |
| --- | --- | --- |
| FFT | `signal_col`, `sample_rate` (per second; the tab writes the step's), `window` hann / hamming / blackman / boxcar, `remove_mean`, `remove_trend` | a NEW table of n / 2 + 1 rows: `frequency`, `period` (null at 0), `amplitude` (2 \|X\| / the window's sum), `phase`, `power_density`, `real`, `imag` (numpy's `rfft` of the windowed, mean-removed signal). A branch: the row count changes, so training does not read it |
| IFFT | `samples` (0: 2 (bins − 1), the even length) | a NEW table of `samples` rows with `signal` from `real` and `imag` (numpy's `irfft`): the signal the FFT node transformed, windowed and mean-removed as it was |
| Filter Designer | `signal_col`, `filter_type` lowpass / highpass / bandpass / bandstop, `cutoff` (per second; 0.1 by default), `cutoff_high` (band filters), `sample_rate` (1 by default), `taps` (31; odd for highpass / bandstop), `window` hamming / hann / blackman / boxcar | the input plus `<signal_col>_filtered` (float64), zero-phase with the edges repeated |
| Convolution 1D | `signal_col`, `kernel` (comma-separated taps) | the input plus `<signal_col>_filtered`, "same" length, zero-phase |
| Wavelet Transform | `signal_col`, `wavelet` (21 names), `level` (3), `denoise` off / soft / hard, `threshold` (0: universal) | the input plus `<signal_col>_A<level>`, `<signal_col>_D<level>` .. `_D1` and `<signal_col>_denoised` when denoising is on |

The older Filter Designer graphs that carried `order` still load: it is read
as 2 · order + 1 taps when `taps` is not set. Nodes that replaced the column
in place no longer do: the filtered column sits beside the original.

## Python

`pycyxwiz.signal` exposes the same backend: `spectrum`, `welch`,
`spectrogram`, `fft` / `ifft`, `lowpass` / `highpass` / `bandpass` /
`bandstop` (taps mean taps), `filter`, `conv` / `conv2`, `wavedec`,
`denoise`, `findpeaks` / `findpeaks_prominence`, and the generators `sine`,
`square`, `noise` ([Python API reference](../extending/python-api-reference.md#pycyxwizsignal)).
