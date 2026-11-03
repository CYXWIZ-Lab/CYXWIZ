"""Truth for cyxwiz::SignalProcessing (TOFIX129 D7): numpy.fft, scipy.signal
and PyWavelets on seeded signals, for test_computation_truth_signal.

Signals: "tones" (fs 1000, 1024 samples: 0.5 + 0.8 sin 50 Hz + 0.3 sin 120 Hz
+ noise), "odd" (257 samples, a smoothed random walk; odd lengths exercise
every length rule), "slow" (fs 0.1, 5,000 samples with a 2,048 s cycle, the
MetroPT-3 shape, for Welch). For each function the fixture holds the inputs
and scipy's / numpy's / pywt's outputs:

  fft           numpy.fft.fft: two-sided magnitude, phase, fftfreq
  spectrum      one-sided amplitude (rfft of the mean-removed, periodic-Hann
                windowed signal, 2 |X| / sum(w), DC and Nyquist not doubled)
                and scipy.signal.periodogram (hann, detrend constant, density)
  welch         scipy.signal.welch (hann, nperseg, noverlap nperseg // 2,
                detrend constant, density)
  spectrogram   scipy.signal.spectrogram (hann, nperseg, noverlap, detrend
                False, density, psd): frequencies, times, Sxx[bin][frame]
  window        scipy.signal.get_window, periodic (fftbins True) and symmetric
  firwin        scipy.signal.firwin (hamming, scale True, fs) for low / high /
                band-pass / band-stop, freqz on the inclusive 0..Nyquist grid,
                and the zero-phase apply: numpy.convolve of the edge-padded
                signal (numpy.pad "edge", (taps - 1) // 2 left) with b, "valid"
  convolve      numpy.convolve full / same / valid, N > M and M > N
  dwt           pywt.wavedec(x, w, mode="symmetric", level) per wavelet and
                length, pywt.waverec back, energy shares, dwt_max_level
  denoise       the universal threshold (sigma = median |cD1| / 0.6745,
                thr = sigma sqrt(2 ln N)), soft and hard, pywt.waverec
  peaks         scipy.signal.find_peaks(x, prominence, distance): indices,
                prominences, left / right bases

It also writes cyxwiz-backend/src/algorithms/wavelet_filters.inc: pywt's
dec_lo / dec_hi / rec_lo / rec_hi for haar, db1..db8, sym2..sym8, coif1..coif5,
and asserts the index alignment the C++ DWT / IDWT use (symmetric extension by
F - 1 on each side, full convolution, samples F - 1 + 1 + 2 i for the analysis;
zero-interleaved upsampling, full convolution, slice [F - 2, F - 2 + 2 n - F + 2)
for the synthesis), so a wrong alignment fails here, not in C++.

Run with numpy 2.3, scipy 1.17, PyWavelets 1.8:
  python generate_signal_fixtures.py
Writes ../fixtures/signal_scipy.json and the filter table.
"""
import json
import os

import numpy as np
import pywt
from scipy import signal as sig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fixtures", "signal_scipy.json")
INC = os.path.join(HERE, "..", "..", "..", "..", "cyxwiz-backend", "src", "algorithms", "wavelet_filters.inc")

WAVELETS = ["haar"] + [f"db{k}" for k in range(1, 9)] + [f"sym{k}" for k in range(2, 9)] + [f"coif{k}" for k in range(1, 6)]


def vec(a):
    return [float(v) for v in np.asarray(a, dtype=float).ravel()]


def signals():
    rng = np.random.default_rng(1782)
    fs = 1000.0
    n = 1024
    t = np.arange(n) / fs
    tones = 0.5 + 0.8 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t + 0.7) + 0.1 * rng.standard_normal(n)
    odd = np.cumsum(rng.standard_normal(257))
    odd = np.convolve(odd, np.ones(5) / 5, "same")
    fs_slow = 0.1
    ts = np.arange(5000) / fs_slow
    slow = 4.0 + 1.9 * np.sin(2 * np.pi * ts / 2048.0) + 0.6 * np.sin(2 * np.pi * ts / 1024.0) + 0.4 * rng.standard_normal(5000)
    return {"tones": (tones, fs), "odd": (odd, 1.0), "slow": (slow, fs_slow)}


def amplitude_spectrum(x, fs):
    w = sig.get_window("hann", len(x), fftbins=True)
    y = (x - x.mean()) * w
    X = np.fft.rfft(y)
    amp = np.abs(X) * 2.0 / w.sum()
    amp[0] /= 2.0
    if len(x) % 2 == 0:
        amp[-1] /= 2.0
    return np.fft.rfftfreq(len(x), 1.0 / fs), amp


def dwt_alignment_checks():
    """The C++ analysis and synthesis use these index rules; assert them against pywt."""
    rng = np.random.default_rng(5)
    for name in ["haar", "db2", "db4", "sym5", "coif3"]:
        w = pywt.Wavelet(name)
        F = len(w.dec_lo)
        for n in [16, 17, 64, 101]:
            x = rng.standard_normal(n)
            cA, cD = pywt.dwt(x, w, mode="symmetric")
            ext = pywt.pad(x, (F - 1, F - 1), mode="symmetric")
            full_lo = np.convolve(ext, w.dec_lo, "full")
            full_hi = np.convolve(ext, w.dec_hi, "full")
            start = F  # F - 1 + 1
            out_len = (n + F - 1) // 2
            got_a = full_lo[start:start + 2 * out_len:2]
            got_d = full_hi[start:start + 2 * out_len:2]
            assert np.allclose(got_a, cA, atol=1e-12) and np.allclose(got_d, cD, atol=1e-12), (name, n, "analysis alignment")
            # Synthesis: upsample, full convolve with rec filters, add, slice.
            up_a = np.zeros(2 * len(cA)); up_a[::2] = cA
            up_d = np.zeros(2 * len(cD)); up_d[::2] = cD
            rec = np.convolve(up_a, w.rec_lo, "full") + np.convolve(up_d, w.rec_hi, "full")
            rec_len = 2 * len(cA) - F + 2
            got_x = rec[F - 2:F - 2 + rec_len]
            want = pywt.idwt(cA, cD, w, mode="symmetric")
            assert np.allclose(got_x, want, atol=1e-12), (name, n, "synthesis alignment")
            # waverec trims one sample when the length overshoots an odd original.
            assert len(want) == rec_len
    return True


def main():
    dwt_alignment_checks()
    cases = []

    def add(op, label, **kw):
        cases.append({"op": op, "label": label, **kw})

    S = signals()
    # 1. fft
    for label in ["tones", "odd"]:
        x, fs = S[label]
        X = np.fft.fft(x)
        add("fft", label, signal=label, fs=fs, magnitude=vec(np.abs(X)), phase=vec(np.angle(X)), frequencies=vec(np.fft.fftfreq(len(x), 1.0 / fs)))
        # inverse: back to the signal
        add("ifft", label, real=vec(X.real), imag=vec(X.imag), signal=label)
    # 2. spectrum
    for label in ["tones", "odd"]:
        x, fs = S[label]
        f, amp = amplitude_spectrum(x, fs)
        fp, pxx = sig.periodogram(x, fs=fs, window="hann", detrend="constant", scaling="density")
        assert np.allclose(f, fp)
        add("spectrum", label, signal=label, fs=fs, frequencies=vec(f), amplitude=vec(amp), psd=vec(pxx))
    # 3. welch
    for label, nperseg in [("slow", 1024), ("tones", 256), ("odd", 64)]:
        x, fs = S[label]
        f, p = sig.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant", scaling="density")
        add("welch", f"{label}_{nperseg}", signal=label, fs=fs, nperseg=nperseg, frequencies=vec(f), psd=vec(p))
    # 4. spectrogram
    for label, nperseg, hop in [("tones", 128, 32), ("slow", 256, 128), ("odd", 32, 8)]:
        x, fs = S[label]
        f, t, Sxx = sig.spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg - hop, detrend=False, scaling="density", mode="psd")
        add("spectrogram", f"{label}_{nperseg}_{hop}", signal=label, fs=fs, nperseg=nperseg, hop=hop, frequencies=vec(f), times=vec(t),
            sxx=[vec(Sxx[:, j]) for j in range(Sxx.shape[1])])  # [frame][bin]
    # 5. windows
    for name in ["hann", "hamming", "blackman", "boxcar"]:
        for n in [16, 17, 1]:
            add("window", f"{name}_{n}", name=name, size=n,
                periodic=vec(sig.get_window(name, n, fftbins=True)), symmetric=vec(sig.get_window(name, n, fftbins=False)))
    # 6. firwin
    x, fs = S["tones"]
    grid = np.linspace(0.0, np.pi, 512)
    for label, taps, cutoff, pass_zero in [("lowpass", 101, 80.0, True), ("highpass", 101, 80.0, False),
                                           ("bandpass", 65, [40.0, 100.0], False), ("bandstop", 65, [40.0, 100.0], True),
                                           ("lowpass_short", 11, 200.0, True)]:
        b = sig.firwin(taps, cutoff, window="hamming", pass_zero=pass_zero, scale=True, fs=fs)
        w, h = sig.freqz(b, worN=grid)
        left = (taps - 1) // 2
        y = np.convolve(np.pad(x, (left, taps - 1 - left), mode="edge"), b, "valid")  # zero-phase, edges repeated
        add("firwin", label, taps=taps, cutoff=cutoff if isinstance(cutoff, list) else [cutoff], pass_zero=pass_zero, fs=fs,
            b=vec(b), response_mag=vec(np.abs(h)), response_phase=vec(np.angle(h)), signal="tones", filtered=vec(y))
    # 7. convolve
    rng = np.random.default_rng(9)
    a = rng.standard_normal(20)
    k = np.array([0.25, 0.5, 0.25])
    big = rng.standard_normal(9)
    for label, s, h in [("n_gt_m", a, k), ("m_gt_n", k, big), ("even_kernel", a, np.array([1.0, -1.0, 2.0, 0.5]))]:
        for mode in ["full", "same", "valid"]:
            add("convolve", f"{label}_{mode}", x=vec(s), h=vec(h), mode=mode, y=vec(np.convolve(s, h, mode)))
    # 8. dwt
    for label in ["tones", "odd"]:
        x, fs = S[label]
        for name in ["haar", "db4", "sym4", "coif2", "db8"]:
            w = pywt.Wavelet(name)
            max_level = pywt.dwt_max_level(len(x), w.dec_len)
            for level in sorted({1, min(3, max_level), max_level}):
                coeffs = pywt.wavedec(x, w, mode="symmetric", level=level)
                rec = pywt.waverec(coeffs, w, mode="symmetric")[:len(x)]
                energies = [float(np.sum(c ** 2)) for c in coeffs]
                total = float(sum(energies))
                add("dwt", f"{label}_{name}_{level}", signal=label, wavelet=name, level=level, max_level=int(max_level),
                    approximation=vec(coeffs[0]), details=[vec(c) for c in coeffs[1:]],  # details[0] = the coarsest (cD_level)
                    reconstruction=vec(rec), energy_share=[e / total for e in energies])
    # 9. denoise
    for label, name, level in [("tones", "db4", 4), ("odd", "sym4", 3)]:
        x, fs = S[label]
        w = pywt.Wavelet(name)
        coeffs = pywt.wavedec(x, w, mode="symmetric", level=level)
        sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745)
        thr = sigma * float(np.sqrt(2.0 * np.log(len(x))))
        for kind in ["soft", "hard"]:
            thresholded = [coeffs[0]] + [pywt.threshold(c, thr, mode=kind) for c in coeffs[1:]]
            den = pywt.waverec(thresholded, w, mode="symmetric")[:len(x)]
            add("denoise", f"{label}_{name}_{level}_{kind}", signal=label, wavelet=name, level=level, kind=kind,
                sigma=sigma, threshold=thr, denoised=vec(den))
    # 10. peaks
    x, fs = S["tones"]
    for label, data, prominence, distance in [("tones_rows", x, 0.5, 5), ("slow_rows", S["slow"][0], 1.0, 60),
                                              ("spectrum", amplitude_spectrum(x, fs)[1], 0.05, 1),
                                              ("plateau", np.array([0, 1, 3, 3, 3, 1, 0, 2, 5, 2, 0, 4, 4, 1, 0.0]), 0.5, 1)]:
        idx, props = sig.find_peaks(data, prominence=prominence, distance=distance)
        add("peaks", label, x=vec(data), prominence=prominence, distance=distance, indices=[int(i) for i in idx],
            prominences=vec(props["prominences"]), left_bases=[int(i) for i in props["left_bases"]], right_bases=[int(i) for i in props["right_bases"]])

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "pywt": pywt.__version__,
                   "signals": {label: {"x": vec(x), "fs": fs} for label, (x, fs) in S.items()}, "cases": cases}, f)
    # The filter table the backend includes.
    lines = ["// Generated by tests/computation_truth/reference/generate_signal_fixtures.py from PyWavelets %s;" % pywt.__version__,
             "// do not edit. {name, dec_lo, dec_hi, rec_lo, rec_hi} as pywt.Wavelet(name) gives them.",
             "// clang-format off"]
    for name in WAVELETS:
        w = pywt.Wavelet(name)
        def arr(v):
            return "{" + ", ".join("%.17g" % float(c) for c in v) + "}"
        lines.append('{"%s", %s, %s, %s, %s},' % (name, arr(w.dec_lo), arr(w.dec_hi), arr(w.rec_lo), arr(w.rec_hi)))
    lines.append("// clang-format on")
    with open(INC, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, len(cases), "cases;", INC, len(WAVELETS), "wavelets")


if __name__ == "__main__":
    main()
