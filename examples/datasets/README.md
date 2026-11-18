# Example datasets

Small, redistributable samples that example graphs read by a repo-relative
path (`examples/datasets/...`). The Engine finds them from the graph's folder
or the folder it was started from, so an example trains right after it opens,
offline. When a starter opens, its Data Input is pointed at the sample it
found, so a copy saved anywhere keeps its data.

These files are in a source checkout; release packages do not ship
`examples/` yet.

## Starter samples

The empty canvas (**Open an example graph**) and the start page offer five
starter graphs. Each trains on one of these samples. Rebuild the downloaded
ones with `python examples/datasets/make_starter_samples.py` (needs
scikit-learn and Pillow).

| Sample | Starter graph | What it is | Source and licence | Full dataset |
| --- | --- | --- | --- | --- |
| `airline_passengers.csv` | Time-series forecasting (`cyxgraph/timeseries/airline_passengers_dense.cyxgraph`) | 144 monthly airline passenger totals, 1949-1960 (`Month`, `Passengers`) | Box & Jenkins, *Time Series Analysis* (series G; R's `AirPassengers`); a long-standing public teaching series, no licence stated by its source | this is the whole series |
| `digits_8x8.csv` | Multiclass image classification (`cyxgraph/mnist_mlp.cyxgraph`) | 1,797 handwritten digits, 8x8 grey levels 0..16 (`pixel_0` .. `pixel_63`, `class` 0-9) | UCI Optical Recognition of Handwritten Digits (E. Alpaydin, C. Kaynak, 1998; the scikit-learn copy); CC BY 4.0 | MNIST 28x28 (`mnist_784`, OpenML) |
| `review_sentences.csv` | Text classification (`cyxgraph/review_sentiment_classifier.cyxgraph`) | 3,000 review sentences from Amazon, IMDb and Yelp (`sentence`, `sentiment` positive / negative, `source`) | UCI Sentiment Labelled Sentences (Kotzias et al., 2015); CC BY 4.0 | this is the whole dataset |
| `spoken_digits/` | Audio classification (`cyxgraph/speech_command_classifier.cyxgraph`) | 400 clips of the digits 0-9 spoken by four speakers (ten takes each), 8 kHz WAV, one folder per digit | Free Spoken Digit Dataset (Z. Jackson et al., github.com/Jakobovski/free-spoken-digit-dataset); CC BY-SA 4.0 | FSDD (3,000 clips), or Speech Commands (CC BY 4.0) |
| `cats_dogs_sample/` | Binary image classification (`cyxgraph/cats_dogs_classifier.cyxgraph`) | 60 photos, 96x96, one folder per class (`cat/`, `dog/`) | Wikimedia Commons quality and featured pictures; each photo's author, licence (CC0, public domain, CC BY or CC BY-SA) and the change made (centre-cropped to a square, resized to 96x96) are in `cats_dogs_sample/ATTRIBUTION.csv` | Oxford-IIIT Pet (CC BY-SA 4.0) |

These samples are for trying a graph end to end. They are far too small to
measure a model: point the graph's Data Input at the full dataset for that.

## Other samples

- `metric_blobs.csv`, `metric_blobs.h5` (`make_metric_blobs_h5.py`): generated
  blobs for the metric-learning examples.
- `volume_shapes.csv` (`generate_volume_shapes.py`): generated 3D volumes for
  the Conv3D examples.
