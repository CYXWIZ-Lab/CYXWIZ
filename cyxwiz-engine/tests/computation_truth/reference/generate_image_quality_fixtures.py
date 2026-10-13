"""Reference measurements for the Quality Analyzer (TOFIX140).

OpenCV is the oracle: luminance with the RGB2GRAY weights, blur as the
variance of cv2.Laplacian(ksize=1, reflect-101), brightness as the mean,
contrast as the standard deviation / 255, and the difference hash from
cv2.resize(INTER_AREA) to 9 x 8. Writes fixtures/image_quality_opencv.json.

    py -3.12 generate_image_quality_fixtures.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "image_quality_opencv.json"


def luminance(image):
    image = image.astype(np.float64) * 255.0
    if image.shape[2] == 1:
        return image[:, :, 0]
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def measure(image):
    luma = luminance(image)
    # cv2's own conversion agrees with the explicit weights.
    if image.shape[2] == 3:
        cv_luma = cv2.cvtColor((image * 255.0).astype(np.float32), cv2.COLOR_RGB2GRAY)
        assert np.abs(cv_luma - luma).max() < 1e-3
    lap = cv2.Laplacian(luma, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
    small = cv2.resize(luma, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] - small[:, :-1]
    bits = diff > 0
    value = 0
    for r in range(8):
        for c in range(8):
            if bits[r, c]:
                value |= 1 << (r * 8 + c)
    return {
        "blur": float(lap.var()),
        "brightness": float(luma.mean()),
        "contrast": float(luma.std() / 255.0),
        "hash": str(value),
        "margin": float(np.abs(diff).min()),
    }


def scene(rng, height, width, channels):
    """Gradients, blobs and noise: something with edges and texture."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.zeros((height, width, channels))
    for ch in range(channels):
        fx, fy = rng.uniform(0.5, 3.0, 2)
        image[:, :, ch] = 0.5 + 0.3 * np.sin(fx * x / width * 6.28 + ch) * np.cos(fy * y / height * 6.28)
        cx, cy, radius = rng.uniform(0, width), rng.uniform(0, height), rng.uniform(3, 10)
        image[:, :, ch] += 0.3 * (((x - cx) ** 2 + (y - cy) ** 2) < radius ** 2)
    image += rng.normal(0, 0.05, image.shape)
    return np.clip(image, 0.0, 1.0)


def make_case(rng, name, height, width, channels):
    images = []
    base = scene(rng, height, width, channels)
    images.append(base)
    images.append(cv2.GaussianBlur(base.astype(np.float32), (7, 7), 2.0).reshape(base.shape).astype(np.float64))
    images.append(np.clip(base * 0.2, 0, 1))                       # dark
    images.append(np.clip(0.5 + (base - 0.5) * 0.1, 0, 1))          # low contrast
    images.append(np.clip(base + rng.normal(0, 0.002, base.shape), 0, 1))  # near-duplicate of base
    images.append(scene(rng, height, width, channels))
    rows = np.stack(images).astype(np.float32)
    metrics = [measure(image) for image in rows]
    return {
        "name": name,
        "rows": {"shape": list(rows.shape), "values": rows.reshape(-1).tolist()},
        "metrics": metrics,
    }


def main():
    cases = []
    for seed, (name, h, w, c) in enumerate([("rgb_64x64", 64, 64, 3), ("rgb_37x50", 37, 50, 3), ("gray_16x23", 16, 23, 1)]):
        # Regenerate until no hash bit sits on a near-tie, so float32 device
        # arithmetic cannot flip one.
        for attempt in range(100):
            case = make_case(np.random.default_rng(seed * 100 + attempt), name, h, w, c)
            if min(m["margin"] for m in case["metrics"]) > 1e-2:
                break
        else:
            raise RuntimeError(f"{name}: no tie-free scene")
        cases.append(case)
    OUT.write_text(json.dumps({"opencv": cv2.__version__, "cases": cases}))
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
