#!/usr/bin/env python3
"""Download the MNIST test set and write it as mnist_test.csv for mnist_cnn.cyxgraph.

The file has a header (label, pixel0 ... pixel783) and one row per image:
the digit, then its 28 x 28 pixels row by row, 0 to 255. Standard library
only; about 2 MB to download, 17.4 MB written.

    python get_mnist.py              # writes mnist_test.csv next to this script
    python get_mnist.py --out D:/data/mnist_test.csv
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

# The mirror torchvision downloads from (the original yann.lecun.com site
# often refuses requests).
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def fetch(name: str) -> bytes:
    with urllib.request.urlopen(MIRROR + name, timeout=60) as response:
        data = response.read()
    digest = hashlib.md5(data).hexdigest()
    if digest != FILES[name]:
        raise SystemExit(f"{name}: checksum {digest}, expected {FILES[name]}")
    return gzip.decompress(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).with_name("mnist_test.csv")), help="CSV to write")
    args = parser.parse_args()

    images = fetch("t10k-images-idx3-ubyte.gz")
    labels = fetch("t10k-labels-idx1-ubyte.gz")
    magic, count, rows, cols = struct.unpack(">IIII", images[:16])
    label_magic, label_count = struct.unpack(">II", labels[:8])
    if magic != 2051 or label_magic != 2049 or count != label_count:
        raise SystemExit("unexpected IDX headers")
    size = rows * cols

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("label," + ",".join(f"pixel{i}" for i in range(size)) + "\n")
        for n in range(count):
            pixels = images[16 + n * size: 16 + (n + 1) * size]
            f.write(str(labels[8 + n]) + "," + ",".join(map(str, pixels)) + "\n")
    print(f"wrote {out} ({count} rows x {size + 1} columns)")


if __name__ == "__main__":
    main()
