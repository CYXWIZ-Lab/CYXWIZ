#!/usr/bin/env python3
"""
MNIST Inference Test Script for CyxWiz Server Node

Tests the deployed MNIST model by sending real test images and verifying predictions.

Usage:
    python test_inference.py [options]

Options:
    --endpoint URL      Server endpoint (default: http://localhost:8080/v1/predict)
    --deployment-id ID  Deployment ID (auto-detected if not specified)
    --num-samples N     Number of samples to test (default: 100)
    --show-errors       Show details of incorrect predictions
    --download          Download MNIST data if not present
"""
# python test_inference.py --download --num-samples 100
import argparse
import json
import os
import sys
import struct
import gzip
import urllib.request
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request


# MNIST URLs (using PyTorch's S3 mirror which is reliable)
MNIST_URLS = {
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_mnist(data_dir):
    """Download MNIST test data if not present."""
    os.makedirs(data_dir, exist_ok=True)

    for name, url in MNIST_URLS.items():
        filename = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filename)
            print(f"  Saved to {filename}")
        else:
            print(f"Found {filename}")

    return data_dir


def load_mnist_images(filepath):
    """Load MNIST images from IDX file."""
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows * cols)

        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0

    return images


def load_mnist_labels(filepath):
    """Load MNIST labels from IDX file."""
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def get_deployment_id(endpoint_base):
    """Auto-detect deployment ID from deployments endpoint."""
    deployments_url = endpoint_base.replace('/v1/predict', '/v1/deployments')

    try:
        if HAS_REQUESTS:
            resp = requests.get(deployments_url, timeout=5)
            data = resp.json()
        else:
            with urllib.request.urlopen(deployments_url, timeout=5) as resp:
                data = json.loads(resp.read().decode())

        deployments = data.get('deployments', [])
        if not deployments:
            print("Error: No active deployments found")
            return None

        # Return first deployment ID
        deployment_id = deployments[0].get('deployment_id')
        print(f"Auto-detected deployment: {deployment_id}")
        return deployment_id

    except Exception as e:
        print(f"Error checking deployments: {e}")
        return None


def run_inference(endpoint, deployment_id, input_data):
    """Send inference request to server."""
    payload = {
        'deployment_id': deployment_id,
        'input': input_data.tolist()
    }

    headers = {'Content-Type': 'application/json'}

    try:
        if HAS_REQUESTS:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            result = resp.json()
        else:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(endpoint, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())

        if 'error' in result:
            return None, result['error']

        return result.get('output'), None

    except Exception as e:
        return None, str(e)


def print_digit(image, label, predicted):
    """Print ASCII art of digit."""
    pixels = image.reshape(28, 28)
    chars = ' .:-=+*#%@'

    print(f"\nLabel: {label}, Predicted: {predicted} {'OK' if label == predicted else 'X'}")
    print("+" + "-" * 28 + "+")
    for row in pixels:
        line = ""
        for pixel in row:
            idx = int(pixel * (len(chars) - 1))
            line += chars[idx]
        print("|" + line + "|")
    print("+" + "-" * 28 + "+")


def main():
    parser = argparse.ArgumentParser(description='Test MNIST model inference')
    parser.add_argument('--endpoint', default='http://localhost:8080/v1/predict',
                        help='Inference endpoint URL')
    parser.add_argument('--deployment-id', default=None,
                        help='Deployment ID (auto-detected if not specified)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to test')
    parser.add_argument('--show-errors', action='store_true',
                        help='Show ASCII art for incorrect predictions')
    parser.add_argument('--show-all', action='store_true',
                        help='Show ASCII art for all predictions')
    parser.add_argument('--data-dir', default='./mnist_data',
                        help='Directory for MNIST data')
    parser.add_argument('--download', action='store_true',
                        help='Download MNIST data if not present')
    args = parser.parse_args()

    # Auto-detect deployment ID if not provided
    if not args.deployment_id:
        args.deployment_id = get_deployment_id(args.endpoint)
        if not args.deployment_id:
            print("Error: Could not auto-detect deployment ID. Please specify with --deployment-id")
            sys.exit(1)

    # Download data if requested
    if args.download:
        download_mnist(args.data_dir)

    # Check for data files
    images_file = os.path.join(args.data_dir, 't10k-images-idx3-ubyte.gz')
    labels_file = os.path.join(args.data_dir, 't10k-labels-idx1-ubyte.gz')

    if not os.path.exists(images_file) or not os.path.exists(labels_file):
        print(f"Error: MNIST data not found in {args.data_dir}")
        print("Run with --download to download the data")
        sys.exit(1)

    # Load data
    print(f"Loading MNIST test data from {args.data_dir}...")
    images = load_mnist_images(images_file)
    labels = load_mnist_labels(labels_file)
    print(f"Loaded {len(images)} test images")

    # Run inference
    print(f"\nTesting {args.num_samples} samples...")
    print(f"Endpoint: {args.endpoint}")
    print(f"Deployment: {args.deployment_id}")
    print("-" * 50)

    correct = 0
    errors = []
    total_latency = 0

    for i in range(min(args.num_samples, len(images))):
        image = images[i]
        label = labels[i]

        output, error = run_inference(args.endpoint, args.deployment_id, image)

        if error:
            print(f"\rSample {i+1}: Error - {error}")
            continue

        predicted = np.argmax(output)
        confidence = output[predicted] * 100

        if predicted == label:
            correct += 1
            status = "OK"
        else:
            status = "X"
            errors.append((i, image, label, predicted, confidence))

        # Progress
        accuracy = correct / (i + 1) * 100
        print(f"\rProgress: {i+1}/{args.num_samples} | Accuracy: {accuracy:.1f}% | Last: {label}->{predicted} {status}  ", end='')

        if args.show_all:
            print()
            print_digit(image, label, predicted)

    print()
    print("=" * 50)
    print(f"Results: {correct}/{args.num_samples} correct ({correct/args.num_samples*100:.1f}% accuracy)")
    print(f"Errors: {len(errors)}")

    if args.show_errors and errors:
        print("\nIncorrect predictions:")
        for idx, image, label, predicted, conf in errors[:10]:  # Show first 10
            print_digit(image, label, predicted)
            print(f"  Confidence: {conf:.1f}%")

    return 0 if correct / args.num_samples > 0.8 else 1


if __name__ == '__main__':
    sys.exit(main())
