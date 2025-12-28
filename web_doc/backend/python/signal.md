# Signal Processing (`cx.signal`)

The `signal` submodule provides signal processing functions including FFT, filtering, convolution, and spectral analysis.

## Overview

```python
import pycyxwiz as cx
import numpy as np

# Generate signal
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)

# FFT analysis
result = cx.signal.fft(x.tolist(), sample_rate=1000)

# Design and apply filter
b, a = cx.signal.lowpass(cutoff=15, fs=1000, order=4)
filtered = cx.signal.filter(x.tolist(), b['b'], a['a'])
```

## Spectral Analysis

### `fft(x, sample_rate=1.0)`

Fast Fourier Transform.

```python
import numpy as np

# Generate test signal
fs = 1000  # Sample rate
t = np.linspace(0, 1, fs)
x = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

result = cx.signal.fft(x.tolist(), sample_rate=fs)

# Result contains:
print(result.keys())
# ['magnitude', 'phase', 'frequencies', 'complex']

magnitude = result['magnitude']    # Amplitude spectrum
phase = result['phase']           # Phase spectrum (radians)
frequencies = result['frequencies']  # Frequency bins (Hz)
complex_out = result['complex']   # Complex FFT coefficients
```

**Parameters:**
- `x` (list): Input signal (1D)
- `sample_rate` (float): Sampling frequency in Hz. Default: 1.0

**Returns:** dict with keys:
- `magnitude`: List of amplitude values
- `phase`: List of phase values (radians)
- `frequencies`: List of frequency bins (Hz)
- `complex`: List of complex coefficients

**Example - Frequency Detection:**
```python
result = cx.signal.fft(x, sample_rate=1000)

# Find dominant frequency
mag = result['magnitude']
freq = result['frequencies']
max_idx = mag.index(max(mag))
print(f"Dominant frequency: {freq[max_idx]} Hz")
```

---

### `ifft(X)`

Inverse Fast Fourier Transform.

```python
# Forward FFT
result = cx.signal.fft(x, sample_rate=1000)
X = result['complex']

# Inverse FFT
x_reconstructed = cx.signal.ifft(X)
```

**Parameters:**
- `X` (list): Complex FFT coefficients

**Returns:** list, reconstructed time-domain signal

---

### `spectrogram(x, window_size=256, hop_size=128, sample_rate=1.0, window="hann")`

Compute Short-Time Fourier Transform (STFT) spectrogram.

```python
result = cx.signal.spectrogram(
    x,
    window_size=256,
    hop_size=128,
    sample_rate=1000,
    window="hann"
)

S = result['S']              # 2D spectrogram (freq x time)
frequencies = result['frequencies']  # Frequency axis
times = result['times']      # Time axis
```

**Parameters:**
- `x` (list): Input signal
- `window_size` (int): FFT window size. Default: 256
- `hop_size` (int): Hop between windows. Default: 128
- `sample_rate` (float): Sampling frequency. Default: 1.0
- `window` (str): Window type. Options: "hann", "hamming", "blackman", "rect". Default: "hann"

**Returns:** dict with keys:
- `S`: 2D list, spectrogram (magnitude)
- `frequencies`: List, frequency axis values
- `times`: List, time axis values

**Example - Plot Spectrogram:**
```python
import matplotlib.pyplot as plt

result = cx.signal.spectrogram(x, window_size=256, hop_size=64, sample_rate=1000)

plt.figure(figsize=(10, 4))
plt.imshow(result['S'], aspect='auto', origin='lower',
           extent=[result['times'][0], result['times'][-1],
                  result['frequencies'][0], result['frequencies'][-1]])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude')
plt.title('Spectrogram')
plt.show()
```

## Filtering

### `lowpass(cutoff, fs, order=4)`

Design a lowpass Butterworth filter.

```python
coeffs = cx.signal.lowpass(cutoff=10, fs=100, order=4)
b = coeffs['b']  # Numerator coefficients
a = coeffs['a']  # Denominator coefficients
```

**Parameters:**
- `cutoff` (float): Cutoff frequency (Hz)
- `fs` (float): Sampling frequency (Hz)
- `order` (int): Filter order. Default: 4

**Returns:** dict with keys `b` and `a` (filter coefficients)

---

### `highpass(cutoff, fs, order=4)`

Design a highpass Butterworth filter.

```python
coeffs = cx.signal.highpass(cutoff=5, fs=100, order=4)
b = coeffs['b']
a = coeffs['a']
```

**Parameters:**
- `cutoff` (float): Cutoff frequency (Hz)
- `fs` (float): Sampling frequency (Hz)
- `order` (int): Filter order. Default: 4

**Returns:** dict with keys `b` and `a`

---

### `bandpass(low, high, fs, order=4)`

Design a bandpass Butterworth filter.

```python
coeffs = cx.signal.bandpass(low=5, high=15, fs=100, order=4)
b = coeffs['b']
a = coeffs['a']
```

**Parameters:**
- `low` (float): Lower cutoff frequency (Hz)
- `high` (float): Upper cutoff frequency (Hz)
- `fs` (float): Sampling frequency (Hz)
- `order` (int): Filter order. Default: 4

**Returns:** dict with keys `b` and `a`

---

### `filter(x, b, a)`

Apply IIR filter to signal.

```python
# Design lowpass filter
coeffs = cx.signal.lowpass(cutoff=10, fs=100, order=4)

# Apply filter
filtered = cx.signal.filter(x, coeffs['b'], coeffs['a'])
```

**Parameters:**
- `x` (list): Input signal
- `b` (list): Numerator coefficients
- `a` (list): Denominator coefficients

**Returns:** list, filtered signal

**Example - Complete Filtering Pipeline:**
```python
import numpy as np

# Generate noisy signal
fs = 1000
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
noise = 0.5 * np.random.randn(len(t))  # High-frequency noise
noisy = (signal + noise).tolist()

# Design and apply lowpass filter
coeffs = cx.signal.lowpass(cutoff=20, fs=fs, order=4)
filtered = cx.signal.filter(noisy, coeffs['b'], coeffs['a'])

# Plot results
import matplotlib.pyplot as plt
plt.plot(t, noisy, 'b-', alpha=0.5, label='Noisy')
plt.plot(t, filtered, 'r-', linewidth=2, label='Filtered')
plt.legend()
plt.show()
```

## Convolution

### `conv(x, h, mode="same")`

1D convolution.

```python
x = [1, 2, 3, 4, 5]
h = [1, 0, -1]  # Difference kernel

y = cx.signal.conv(x, h, mode='same')
# Output length = len(x)
```

**Parameters:**
- `x` (list): Input signal
- `h` (list): Convolution kernel
- `mode` (str): Output mode. Options:
  - `"full"`: Full convolution (len(x) + len(h) - 1)
  - `"same"`: Same length as input (default)
  - `"valid"`: Only where kernel fully overlaps

**Returns:** list, convolved signal

---

### `conv2(x, h, mode="same")`

2D convolution.

```python
import numpy as np

# Image (10x10)
image = np.random.randn(10, 10).tolist()

# Averaging kernel (3x3)
kernel = [[1/9]*3 for _ in range(3)]

# Convolve
smoothed = cx.signal.conv2(image, kernel, mode='same')
```

**Parameters:**
- `x` (2D list): Input image/matrix
- `h` (2D list): Convolution kernel
- `mode` (str): Output mode. Options: "full", "same", "valid"

**Returns:** 2D list, convolved output

**Example - Edge Detection:**
```python
# Sobel edge detection
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

sobel_y = [[-1, -2, -1],
           [0, 0, 0],
           [1, 2, 1]]

edges_x = cx.signal.conv2(image, sobel_x, mode='same')
edges_y = cx.signal.conv2(image, sobel_y, mode='same')
```

## Peak Detection

### `findpeaks(x, min_height=0.0, min_distance=1)`

Find local maxima in signal.

```python
x = [0, 1, 0, 2, 0, 3, 0, 1, 0]

result = cx.signal.findpeaks(x, min_height=0.5, min_distance=1)

print(result['indices'])  # [1, 3, 5, 7]
print(result['values'])   # [1, 2, 3, 1]
```

**Parameters:**
- `x` (list): Input signal
- `min_height` (float): Minimum peak height. Default: 0.0
- `min_distance` (int): Minimum distance between peaks. Default: 1

**Returns:** dict with keys:
- `indices`: List of peak indices
- `values`: List of peak values

**Example - Heartbeat Detection:**
```python
# ECG-like signal
import numpy as np
t = np.linspace(0, 10, 1000)
ecg = np.sin(2 * np.pi * 1.2 * t) * np.exp(-((t % 1) - 0.5)**2 / 0.01)

# Find R-peaks
peaks = cx.signal.findpeaks(ecg.tolist(), min_height=0.5, min_distance=50)

print(f"Heart rate: {len(peaks['indices']) / 10 * 60:.0f} BPM")
```

## Signal Generation

### `sine(freq, fs, n, amp=1.0, phase=0.0)`

Generate sine wave.

```python
# 10 Hz sine wave, 1 second at 1000 Hz sample rate
wave = cx.signal.sine(freq=10, fs=1000, n=1000, amp=1.0, phase=0.0)
```

**Parameters:**
- `freq` (float): Frequency in Hz
- `fs` (float): Sample rate in Hz
- `n` (int): Number of samples
- `amp` (float): Amplitude. Default: 1.0
- `phase` (float): Initial phase in radians. Default: 0.0

**Returns:** list, sine wave samples

---

### `square(freq, fs, n, amp=1.0)`

Generate square wave.

```python
wave = cx.signal.square(freq=5, fs=1000, n=1000, amp=1.0)
```

**Parameters:**
- `freq` (float): Frequency in Hz
- `fs` (float): Sample rate in Hz
- `n` (int): Number of samples
- `amp` (float): Amplitude. Default: 1.0

**Returns:** list, square wave samples

---

### `noise(n, amp=1.0)`

Generate white noise (Gaussian).

```python
noise_signal = cx.signal.noise(n=1000, amp=0.1)
```

**Parameters:**
- `n` (int): Number of samples
- `amp` (float): Standard deviation of noise. Default: 1.0

**Returns:** list, noise samples

## Complete Example

```python
import pycyxwiz as cx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sample rate
duration = 2  # seconds
n = fs * duration

# Generate composite signal
t = np.linspace(0, duration, n)
signal = (cx.signal.sine(freq=5, fs=fs, n=n, amp=1.0) +
          cx.signal.sine(freq=50, fs=fs, n=n, amp=0.3))

# Add noise
noise = cx.signal.noise(n=n, amp=0.1)
noisy_signal = [s + n for s, n in zip(signal, noise)]

# FFT analysis
fft_result = cx.signal.fft(noisy_signal, sample_rate=fs)

# Design lowpass filter (remove 50 Hz component)
filter_coeffs = cx.signal.lowpass(cutoff=20, fs=fs, order=4)

# Apply filter
filtered = cx.signal.filter(noisy_signal, filter_coeffs['b'], filter_coeffs['a'])

# Find peaks in filtered signal
peaks = cx.signal.findpeaks(filtered, min_height=0.5, min_distance=int(fs/10))

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Time domain
axes[0].plot(t, noisy_signal, 'b-', alpha=0.5, label='Noisy')
axes[0].plot(t, filtered, 'r-', linewidth=2, label='Filtered')
axes[0].scatter([t[i] for i in peaks['indices']], peaks['values'], c='g', s=50, label='Peaks')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].set_title('Time Domain')

# Frequency domain
freqs = fft_result['frequencies'][:len(fft_result['frequencies'])//2]
mags = fft_result['magnitude'][:len(fft_result['magnitude'])//2]
axes[1].plot(freqs, mags)
axes[1].axvline(x=20, color='r', linestyle='--', label='Filter cutoff')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Frequency Spectrum')
axes[1].legend()

# Spectrogram
spec = cx.signal.spectrogram(noisy_signal, window_size=128, hop_size=32, sample_rate=fs)
axes[2].imshow(spec['S'], aspect='auto', origin='lower',
               extent=[spec['times'][0], spec['times'][-1],
                      spec['frequencies'][0], spec['frequencies'][-1]],
               cmap='viridis')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_title('Spectrogram')

plt.tight_layout()
plt.show()
```

---

**Next**: [Statistics](stats.md) | [Back to Index](index.md)
