#!/usr/bin/env python3
"""
Test a deployed CyxWiz embedded image classifier with a local image.

This matches the cats/dogs demo graph preprocessing:
- RGB
- resize to 64x64
- scale pixels to [0, 1]
- normalize with mean=0.5, std=0.5
- flatten in HWC order
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run embedded inference against a deployed CyxWiz model."
    )
    parser.add_argument("image", help="Path to the image to classify")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/v1/predict",
        help="Embedded inference endpoint",
    )
    parser.add_argument(
        "--health-url",
        default="http://localhost:8080/health",
        help="Health endpoint to verify the embedded server is running",
    )
    parser.add_argument("--width", type=int, default=64, help="Resize width")
    parser.add_argument("--height", type=int, default=64, help="Resize height")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["cat", "dog"],
        help="Class labels in output-index order",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many class scores to print",
    )
    return parser.parse_args()


def preprocess_image(image_path: Path, width: int, height: int) -> list[float]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.BILINEAR)

        pixels: list[float] = []
        for red, green, blue in image.getdata():
            for channel in (red, green, blue):
                value = channel / 255.0
                value = (value - 0.5) / 0.5
                pixels.append(value)
        return pixels


def http_json(url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers)

    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    if total == 0.0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


def label_for(index: int, labels: list[str]) -> str:
    if 0 <= index < len(labels):
        return labels[index]
    return f"class_{index}"


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        health = http_json(args.health_url)
    except urllib.error.URLError as exc:
        print(f"Failed to reach embedded server at {args.health_url}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Health endpoint returned invalid JSON: {exc}", file=sys.stderr)
        return 2

    if health.get("status") != "healthy":
        print(f"Server is not healthy: {health}", file=sys.stderr)
        return 3

    if not health.get("model_loaded", False):
        print("Embedded server is running but no model is loaded.", file=sys.stderr)
        return 4

    features = preprocess_image(image_path, args.width, args.height)
    payload = {"input": features}

    try:
        result = http_json(args.url, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Inference request failed: HTTP {exc.code}\n{body}", file=sys.stderr)
        return 5
    except urllib.error.URLError as exc:
        print(f"Failed to send inference request to {args.url}: {exc}", file=sys.stderr)
        return 5
    except json.JSONDecodeError as exc:
        print(f"Inference endpoint returned invalid JSON: {exc}", file=sys.stderr)
        return 5

    output = result.get("output", [])
    if not isinstance(output, list) or not output:
        print(f"Unexpected inference response: {result}", file=sys.stderr)
        return 6

    probabilities = softmax([float(v) for v in output])
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)

    print(f"Image: {image_path}")
    print(f"Model: {health.get('model_name', 'unknown')}")
    print(f"Predicted class: {label_for(best_index, args.labels)}")
    print(f"Confidence: {probabilities[best_index]:.4f}")
    print(f"Latency (ms): {result.get('latency_ms', 'n/a')}")
    print("Scores:")

    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    for index, probability in ranked[: max(1, args.top_k)]:
        raw_score = float(output[index])
        print(
            f"  {label_for(index, args.labels)}: "
            f"logit={raw_score:.6f}, prob={probability:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
