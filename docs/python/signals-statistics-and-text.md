---
title: Signals, statistics and text with pycyxwiz
description: Filter and transform signals, cluster and reduce data, analyse time series, split and score text, use the utilities, and read the device's memory from Python.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - reference/signal.md
  - reference/stats.md
  - reference/timeseries.md
  - reference/text.md
  - reference/utilities.md
---

Use this for the algorithm modules of pycyxwiz that are not about training:
`signal`, `stats`, `timeseries`, `text`, `utilities` and `system`. Each
takes plain Python lists (and NumPy arrays where a list is expected) and
returns lists or dicts, so it needs no tensors. They are the code behind
the matching Data Studio views, nodes and **Tools** windows.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script.

## Signals

```python
import pycyxwiz as cx
import pycyxwiz.signal as sig

cx.initialize()

x = sig.sine(freq=10, fs=200, n=2000)                 # 10 Hz, sampled at 200 Hz, 10 s
noisy = [a + b for a, b in zip(x, sig.noise(2000, amp=0.3))]
lp = sig.lowpass(cutoff=20, fs=200, taps=31)          # a FIR filter: {'b': [...], 'a': [...]}
smooth = sig.filter(noisy, lp["b"], lp["a"])

spectrum = sig.fft(x, sample_rate=200)
peak = max(range(len(spectrum["magnitude"]) // 2), key=lambda i: spectrum["magnitude"][i])
print(round(spectrum["frequencies"][peak], 1))        # 10.0
print(sorted(sig.spectrogram(x, window_size=128, hop_size=64, sample_rate=200)))   # ['S', 'frequencies', 'times']
```

Also: `highpass`, `bandpass`, `bandstop`, `conv`, `conv2`, `ifft`,
`spectrum`, `welch`, `findpeaks`, `findpeaks_prominence`, the wavelet
`wavedec` and `denoise`, and the generators `sine`, `square` and `noise`.
Each filter design matches `scipy.signal.firwin`; see
[pycyxwiz.signal](reference/signal.md).

## Clustering, PCA and metrics

```python
import pycyxwiz.stats as st

points = [[1.0, 2.0], [1.2, 2.1], [0.9, 1.8], [5.0, 5.1], [5.2, 4.9], [4.9, 5.2]]
km = st.kmeans(points, k=2)
print(len(set(km["labels"][:3])), len(set(km["labels"][3:])), km["labels"][0] != km["labels"][3])   # 1 1 True
print(round(st.silhouette(points, km["labels"]), 2))  # 0.94

pca = st.pca(points, n_components=1)
print(sorted(pca))            # ['components', 'explained_variance', 'transformed']
```

`dbscan`, `gmm` and `tsne` are there too; `confusion_matrix` and `roc` are
in [Evaluate a model](evaluation-and-metrics.md).

## Time series

```python
import pycyxwiz.timeseries as ts

series = [float(i % 12) + 0.1 * i for i in range(60)]     # a 12-step season and a trend
print(sorted(ts.stationarity(series))[:3])  # ['adf_pvalue', 'adf_statistic', 'is_stationary']
parts = ts.decompose(series, period=12)
forecast = ts.arima(series, horizon=3)
print(len(forecast["forecast"]))            # 3

config = ts.WindowConfig()
config.window_size = 12
config.forecast_horizon = 1
windows = ts.create_windows(series, config)
print(windows.success, windows.num_windows)   # True 48
```

`acf`, `pacf`, `diff`, `rolling_mean`, `rolling_std`, `add_features` and
`chronological_split` complete the module.

## Words and sentiment

The rules of Data Studio's **Explore > Text** and the text nodes:

```python
import pycyxwiz.text as tx

print(tx.words("I don't feel like myself."))      # ["don't", 'feel', 'like', 'myself']
print(tx.terms("not good at all", stop_words=None, ngram_range=(1, 2)))
# ['not', 'good', 'at', 'all', 'not good', 'good at', 'at all']
print(tx.stem("feeling"))                         # feel
print(round(tx.vader("The food was not good.")["compound"], 4))   # -0.3412
```

`english_stop_words()` lists the stop words `terms` leaves out. To turn
text into token ids for a model, use `cx.Tokenizer`
([Tokenizers](reference/tokenizers.md)).

## Utilities

The **Tools > Utilities** windows from Python:

```python
u = cx.utilities
print(u.evaluate("sqrt(x^2 + y^2)", {"x": 3, "y": 4}))   # 5.0
print(u.convert(100, "km/h", "knot"))                    # 53.99568034557235
print(u.hash("CyxWiz")[:16])                             # the start of the SHA-256 hex digest
print(u.json_check('{"a": 1,}')["valid"])                # False
print([m["text"] for m in u.regex_find(r"\d+", "3 cats, 12 dogs")])   # ['3', '12']
r = u.random("normal", 1000, 10, 2, seed=42)             # the same values on every platform
print(round(r["mean"], 4))                               # 9.9239
```

Units, hashes of files, JSON paths and UUIDs are on
[pycyxwiz.utilities](reference/utilities.md).

## Memory on the device

`pycyxwiz.system` gives the figures of **Tools > Monitoring**: what this
process holds on its compute device, what the device has free, and the
host tensors.

```python
s = cx.system
big = cx.Tensor.random([1024, 1024], seed=1)    # 4 MB of float32
held = s.device_memory_held()
print(sorted(held))     # ['allocated_bytes', 'buffers', 'buffers_in_use', 'in_use_bytes', 'known']
del big
freed = s.release_unused_device_memory()        # gives buffers kept for reuse back to the driver
print(freed["ran"])                             # True
print(sorted(s.device_memory_free()))           # ['free_bytes', 'known', 'source', 'total_bytes']
```

ArrayFire keeps freed buffers for reuse, so `allocated_bytes` stays up after
a tensor goes until you release them. `device_memory_free` reports `known:
False` on a device whose driver does not tell; nothing is guessed. For the
machine's CPU and main memory, use `psutil`.

## Related

- The reference pages: [signal](reference/signal.md),
  [stats](reference/stats.md), [timeseries](reference/timeseries.md),
  [text](reference/text.md), [utilities](reference/utilities.md),
  [system](reference/system.md).
- [Examples](examples.md): `text_words_and_sentiment.py`, `utilities.py`
  and `system_memory.py`.
