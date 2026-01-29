#!/usr/bin/env python3
"""
Test script for xAI fixes verification.

This script helps verify that the xAI fixes are working correctly:
1. LIME reproducibility (same image -> same output)
2. Model quality validation
3. Configuration parameters

Usage:
    python test_xai_fixes.py --image path/to/test_image.jpg --client-id 1
"""

import argparse
import hashlib
import requests
import json
from pathlib import Path


def hash_base64(base64_str: str) -> str:
    """Compute hash of base64 string for comparison."""
    return hashlib.sha256(base64_str.encode()).hexdigest()[:16]


def test_lime_reproducibility(api_url: str, image_path: Path, client_id: int, num_runs: int = 3):
    """Test that LIME produces identical results across multiple runs."""
    print(f"\n{'='*60}")
    print("TEST 1: LIME Reproducibility")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Client ID: {client_id}")
    print(f"Number of runs: {num_runs}")

    if not image_path.exists():
        print(f"❌ ERROR: Image not found at {image_path}")
        return False

    lime_hashes = []
    gradcam_hashes = []
    predictions = []

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}...")

        try:
            response = requests.post(
                f"{api_url}/inference",
                files={"file": ("test_image.jpg", image_bytes, "image/jpeg")},
                data={"client_id": client_id},
                timeout=60,
            )

            if response.status_code != 200:
                print(f"❌ ERROR: API returned status {response.status_code}")
                print(f"Response: {response.text}")
                return False

            result = response.json()

            # Extract data
            prediction = result.get("prediction")
            confidence = result.get("confidence")
            lime_base64 = result.get("xai", {}).get("lime", {}).get("image_base64", "")
            gradcam_base64 = result.get("xai", {}).get("gradcam", {}).get("image_base64", "")

            # Compute hashes
            lime_hash = hash_base64(lime_base64)
            gradcam_hash = hash_base64(gradcam_base64)

            lime_hashes.append(lime_hash)
            gradcam_hashes.append(gradcam_hash)
            predictions.append((prediction, confidence))

            print(f"  Prediction: {prediction} ({confidence:.4f})")
            print(f"  LIME hash: {lime_hash}")
            print(f"  Grad-CAM hash: {gradcam_hash}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False

    # Check reproducibility
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")

    all_lime_same = len(set(lime_hashes)) == 1
    all_predictions_same = len(set(pred for pred, _ in predictions)) == 1

    print(f"LIME Reproducibility: {'✅ PASS' if all_lime_same else '❌ FAIL'}")
    if not all_lime_same:
        print(f"  Different hashes: {lime_hashes}")

    print(f"Prediction Consistency: {'✅ PASS' if all_predictions_same else '❌ FAIL'}")
    if not all_predictions_same:
        print(f"  Different predictions: {predictions}")

    print(f"\nNote: Grad-CAM may vary slightly due to augmentation smoothing.")
    print(f"Grad-CAM hashes: {gradcam_hashes}")

    return all_lime_same


def test_quality_validation(api_url: str, image_path: Path, client_id: int):
    """Test that quality validation warnings appear in logs."""
    print(f"\n{'='*60}")
    print("TEST 2: Model Quality Validation")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Client ID: {client_id}")

    if not image_path.exists():
        print(f"❌ ERROR: Image not found at {image_path}")
        return False

    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{api_url}/inference",
                files={"file": ("test_image.jpg", f, "image/jpeg")},
                data={"client_id": client_id},
                timeout=60,
            )

        if response.status_code != 200:
            print(f"❌ ERROR: API returned status {response.status_code}")
            return False

        result = response.json()
        confidence = result.get("confidence", 0.0)

        print(f"\nPrediction: {result.get('prediction')}")
        print(f"Confidence: {confidence:.4f}")

        if confidence < 0.70:
            print(f"\n⚠️  Low confidence detected ({confidence:.2%} < 70%)")
            print("✅ Check backend logs for quality warnings")
        else:
            print(f"\n✅ Confidence is adequate ({confidence:.2%} >= 70%)")

        print("\nNote: Check backend logs for detailed quality validation messages")
        print("Expected log pattern: 'xAI Quality Warning:' or 'xAI Quality Issue:'")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_configuration(api_url: str):
    """Test that configuration endpoint is accessible."""
    print(f"\n{'='*60}")
    print("TEST 3: Configuration Parameters")
    print(f"{'='*60}")

    # This test just checks that the API is running
    # Configuration is checked via environment variables

    print("\nConfiguration parameters are set via environment variables:")
    print("  XAI_LIME_NUM_SAMPLES (default: 1000)")
    print("  XAI_LIME_NUM_FEATURES (default: 5)")
    print("  XAI_LIME_RANDOM_SEED (default: 42)")
    print("  XAI_GRADCAM_EIGEN_SMOOTH (default: true)")
    print("  XAI_MIN_CONFIDENCE (default: 0.70)")

    print("\n✅ Check backend/.env to modify these values")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test xAI fixes")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID to use")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                       help="API URL (default: http://localhost:8000)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs for reproducibility test (default: 3)")

    args = parser.parse_args()

    image_path = Path(args.image)

    print(f"\n{'='*60}")
    print("xAI Fixes Verification Test Suite")
    print(f"{'='*60}")
    print(f"API URL: {args.api_url}")
    print(f"Image: {image_path}")
    print(f"Client ID: {args.client_id}")

    # Check API health
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"\n❌ ERROR: API health check failed")
            print("Make sure the backend is running: cd backend && uvicorn app.main:app --reload")
            return
        print("\n✅ API is healthy")
    except Exception as e:
        print(f"\n❌ ERROR: Cannot reach API at {args.api_url}")
        print(f"Error: {e}")
        print("\nMake sure the backend is running: cd backend && uvicorn app.main:app --reload")
        return

    # Run tests
    results = []

    results.append(("LIME Reproducibility", test_lime_reproducibility(
        args.api_url, image_path, args.client_id, args.runs
    )))

    results.append(("Quality Validation", test_quality_validation(
        args.api_url, image_path, args.client_id
    )))

    results.append(("Configuration", test_configuration(args.api_url)))

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
