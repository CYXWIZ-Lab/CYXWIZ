#!/usr/bin/env python3
"""
ONNX MNIST Model Test Script

Tests the deployed MNIST model on the CyxWiz Server Node.
Supports both direct ONNX Runtime testing and REST API testing.

Usage:
    python onnx_test.py                    # Test via REST API (default)
    python onnx_test.py --direct           # Test directly with ONNX Runtime
    python onnx_test.py --samples 100      # Test with 100 samples
    python onnx_test.py --deployment-id ID # Specify deployment ID
"""

import argparse
import json
import struct
import gzip
import urllib.request
import os
import sys
import time
from pathlib import Path

import numpy as np

# URLs for MNIST dataset
MNIST_URLS = {
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}

CACHE_DIR = Path(__file__).parent / '.mnist_cache'


def download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't exist."""
    if dest.exists():
        return
    print(f"Downloading {url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def load_mnist_images(path: Path) -> np.ndarray:
    """Load MNIST images from IDX file."""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols)  # [N, 1, 28, 28]
        return images.astype(np.float32) / 255.0


def load_mnist_labels(path: Path) -> np.ndarray:
    """Load MNIST labels from IDX file."""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def get_mnist_test_data(num_samples: int = 100) -> tuple:
    """Download and load MNIST test data."""
    # Download files
    images_path = CACHE_DIR / 't10k-images-idx3-ubyte.gz'
    labels_path = CACHE_DIR / 't10k-labels-idx1-ubyte.gz'

    download_file(MNIST_URLS['test_images'], images_path)
    download_file(MNIST_URLS['test_labels'], labels_path)

    # Load data
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)

    # Return subset
    return images[:num_samples], labels[:num_samples]


def test_direct_onnx(model_path: str, images: np.ndarray, labels: np.ndarray) -> dict:
    """Test model directly using ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DIRECT ONNX RUNTIME TEST")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Samples: {len(images)}")

    # Create session
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape

    print(f"Input: {input_name} {input_shape}")
    print(f"Output: {output_name}")
    print()

    # Run inference
    correct = 0
    total = len(images)
    latencies = []

    for i, (image, label) in enumerate(zip(images, labels)):
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]  # [1, 1, 28, 28]

        start = time.perf_counter()
        outputs = session.run([output_name], {input_name: image})
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        # Get prediction
        logits = outputs[0][0]
        predicted = np.argmax(logits)

        if predicted == label:
            correct += 1

        # Progress
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  Progress: {i+1}/{total} | Accuracy: {correct/(i+1)*100:.1f}%")

    accuracy = correct / total * 100
    avg_latency = np.mean(latencies)

    print()
    print(f"Results:")
    print(f"  Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  Total Time: {sum(latencies):.0f} ms")

    return {
        'method': 'direct_onnx',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_latency_ms': avg_latency
    }


def test_rest_api(deployment_id: str, images: np.ndarray, labels: np.ndarray,
                  host: str = 'localhost', port: int = 8080) -> dict:
    """Test model via Server Node REST API."""
    import urllib.request
    import urllib.error

    print(f"\n{'='*60}")
    print("REST API TEST")
    print(f"{'='*60}")
    print(f"Endpoint: http://{host}:{port}/v1/predict")
    print(f"Deployment: {deployment_id}")
    print(f"Samples: {len(images)}")
    print()

    url = f"http://{host}:{port}/v1/predict"

    correct = 0
    total = len(images)
    latencies = []
    errors = 0

    for i, (image, label) in enumerate(zip(images, labels)):
        # Prepare input as nested array [batch, channel, height, width]
        # The image is already [1, 28, 28], add batch dimension
        input_data = image.tolist()  # [1, 28, 28]

        payload = {
            'deployment_id': deployment_id,
            'input': [input_data]  # [[1, 28, 28]] for batch of 1
        }

        try:
            start = time.perf_counter()

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))

            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            # Check for error
            if 'error' in result:
                errors += 1
                if errors <= 3:
                    print(f"  Sample {i}: Error - {result['error'].get('message', 'Unknown')}")
                continue

            # Get prediction from output
            if 'output' in result:
                logits = result['output']
                if isinstance(logits[0], list):
                    logits = logits[0]  # Unbatch if needed
                predicted = np.argmax(logits)

                if predicted == label:
                    correct += 1

        except urllib.error.URLError as e:
            errors += 1
            if errors <= 3:
                print(f"  Sample {i}: Connection error - {e}")
            continue
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Sample {i}: Error - {e}")
            continue

        # Progress
        if (i + 1) % 20 == 0 or i == total - 1:
            acc = correct / (i + 1 - errors) * 100 if (i + 1 - errors) > 0 else 0
            print(f"  Progress: {i+1}/{total} | Accuracy: {acc:.1f}% | Errors: {errors}")

    successful = total - errors
    accuracy = correct / successful * 100 if successful > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print()
    print(f"Results:")
    print(f"  Successful: {successful}/{total}")
    print(f"  Accuracy: {correct}/{successful} ({accuracy:.2f}%)")
    print(f"  Errors: {errors}")
    if latencies:
        print(f"  Avg Latency: {avg_latency:.2f} ms")
        print(f"  Total Time: {sum(latencies):.0f} ms")

    return {
        'method': 'rest_api',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'successful': successful,
        'errors': errors,
        'avg_latency_ms': avg_latency
    }


def get_deployment_id(host: str = 'localhost', port: int = 8080) -> str:
    """Get the first available deployment ID from the server."""
    try:
        url = f"http://{host}:{port}/v1/deployments"
        with urllib.request.urlopen(url, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            if result.get('deployments'):
                return result['deployments'][0]['deployment_id']
    except Exception as e:
        print(f"Warning: Could not fetch deployments: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Test ONNX MNIST model deployment')
    parser.add_argument('--direct', action='store_true',
                        help='Test directly with ONNX Runtime (skip REST API)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of test samples (default: 100)')
    parser.add_argument('--model', type=str, default='mnist.onnx',
                        help='Path to ONNX model (default: mnist.onnx)')
    parser.add_argument('--deployment-id', type=str, default=None,
                        help='Deployment ID for REST API test')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Server Node host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server Node HTTP port (default: 8080)')
    parser.add_argument('--both', action='store_true',
                        help='Run both direct and REST API tests')

    args = parser.parse_args()

    print("="*60)
    print("CYXWIZ ONNX MNIST TEST")
    print("="*60)

    # Load test data
    print(f"\nLoading MNIST test data ({args.samples} samples)...")
    images, labels = get_mnist_test_data(args.samples)
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")

    results = []

    # Direct ONNX Runtime test
    if args.direct or args.both:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)

        result = test_direct_onnx(str(model_path), images, labels)
        results.append(result)

    # REST API test
    if not args.direct or args.both:
        # Get deployment ID
        deployment_id = args.deployment_id
        if not deployment_id:
            print("\nFetching deployment ID from server...")
            deployment_id = get_deployment_id(args.host, args.port)
            if not deployment_id:
                print("ERROR: No deployment found. Deploy a model first.")
                if not args.both:
                    sys.exit(1)
            else:
                print(f"  Found: {deployment_id}")

        if deployment_id:
            result = test_rest_api(deployment_id, images, labels, args.host, args.port)
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"\n{r['method'].upper()}:")
        print(f"  Accuracy: {r['accuracy']:.2f}%")
        print(f"  Avg Latency: {r['avg_latency_ms']:.2f} ms")

    print("\nTest completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
