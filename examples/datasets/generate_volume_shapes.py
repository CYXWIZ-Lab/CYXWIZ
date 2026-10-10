#!/usr/bin/env python3
"""Volume shapes: a small 3-D classification set for the Conv3D example.

Each row is one 8x8x8 single-channel volume (512 voxels, v0..v511 in depth,
height, width order: v[(d*8 + h)*8 + w]) holding one noisy shape:
    0 ball   a filled sphere, radius 1.5-2.5
    1 rod    a 1-voxel-thick line along x, y or z, length 5-8
    2 plate  a 1-voxel-thick square slab across two axes, side 4-6
at a random position, plus Gaussian noise (sigma 0.15). Label column: shape.
Use it with a Data Input whose shape is [8, 8, 8, 1].

    py -3.12 generate_volume_shapes.py [--output volume_shapes.csv] [--rows-per-class 160]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

SIZE = 8


def ball(rng: random.Random) -> list[float]:
    radius = rng.uniform(1.5, 2.5)
    cd, ch, cw = (rng.uniform(radius, SIZE - 1 - radius) for _ in range(3))
    return [1.0 if (d - cd) ** 2 + (h - ch) ** 2 + (w - cw) ** 2 <= radius ** 2 else 0.0
            for d in range(SIZE) for h in range(SIZE) for w in range(SIZE)]


def rod(rng: random.Random) -> list[float]:
    axis = rng.randrange(3)
    length = rng.randint(5, SIZE)
    start = rng.randint(0, SIZE - length)
    fixed = [rng.randrange(SIZE) for _ in range(3)]
    volume = [0.0] * SIZE ** 3
    for t in range(start, start + length):
        p = list(fixed)
        p[axis] = t
        volume[(p[0] * SIZE + p[1]) * SIZE + p[2]] = 1.0
    return volume


def plate(rng: random.Random) -> list[float]:
    normal = rng.randrange(3)
    side = rng.randint(4, 6)
    level = rng.randrange(SIZE)
    a0, b0 = rng.randint(0, SIZE - side), rng.randint(0, SIZE - side)
    volume = [0.0] * SIZE ** 3
    for a in range(a0, a0 + side):
        for b in range(b0, b0 + side):
            p = [0, 0, 0]
            p[normal] = level
            others = [axis for axis in range(3) if axis != normal]
            p[others[0]], p[others[1]] = a, b
            volume[(p[0] * SIZE + p[1]) * SIZE + p[2]] = 1.0
    return volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "volume_shapes.csv")
    parser.add_argument("--rows-per-class", type=int, default=160)
    args = parser.parse_args()
    rng = random.Random(140)
    rows = []
    for label, make in enumerate((ball, rod, plate)):
        for _ in range(args.rows_per_class):
            rows.append([round(v + rng.gauss(0.0, 0.15), 2) for v in make(rng)] + [label])
    rng.shuffle(rows)
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow([f"v{i}" for i in range(SIZE ** 3)] + ["shape"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} volumes to {args.output}")


if __name__ == "__main__":
    main()
