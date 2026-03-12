#!/usr/bin/env python3
"""Test RunPod Serverless vLLM endpoints for target and judge models.

Usage:
    python scripts/test_runpod_endpoints.py

Set these environment variables before running:
    export RUNPOD_API_KEY="your_runpod_api_key"
    export TARGET_ENDPOINT_ID="your_target_endpoint_id"
    export JUDGE_ENDPOINT_ID="your_judge_endpoint_id"

Or edit the values below directly.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# ============================================
# Configuration - Edit these values
# ============================================

# RunPod API Key (from .env file or environment)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# Endpoint IDs (from Serverless > Endpoints in RunPod dashboard)
TARGET_ENDPOINT_ID ="bydker6nwb98fe"
JUDGE_ENDPOINT_ID ="bydker6nwb98fe"

# Model names - must match what you deployed (or use the HF model name)
# If you set OPENAI_SERVED_MODEL_NAME_OVERRIDE in RunPod, use that value
TARGET_MODEL_NAME = "google/gemma-2-9b-it"
JUDGE_MODEL_NAME = "google/gemma-2-9b-it"

# ============================================
# Build URLs (RunPod Serverless format)
# ============================================

TARGET_API_URL = f"https://api.runpod.ai/v2/{TARGET_ENDPOINT_ID}/openai/v1"
JUDGE_API_URL = f"https://api.runpod.ai/v2/{JUDGE_ENDPOINT_ID}/openai/v1"


def test_endpoint(name: str, base_url: str, model_name: str, test_prompt: str) -> bool:
    """Test a single endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    print(f"URL: {base_url}")
    print(f"Model: {model_name}")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=RUNPOD_API_KEY,
        )

        # First, try to list models (optional - may not be supported)
        print("\n1. Checking available models...")
        try:
            models = client.models.list()
            print(f"   Available models: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"   (Models list not available: {e})")

        # Test chat completion
        print(f"\n2. Testing chat completion...")
        print(f"   Prompt: {test_prompt[:50]}...")

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=100,
            temperature=0.7,
        )

        content = response.choices[0].message.content or ""
        print(f"\n3. Response received!")
        print(f"   Length: {len(content)} chars")
        print(f"   Preview: {content[:200]}...")
        print(f"\n✅ {name} is working!")
        return True

    except Exception as e:
        print(f"\n❌ {name} FAILED!")
        print(f"   Error: {type(e).__name__}: {e}")
        return False


def test_streaming(name: str, base_url: str, model_name: str) -> bool:
    """Test streaming response."""
    print(f"\n{'='*60}")
    print(f"Testing {name} (Streaming)")
    print(f"{'='*60}")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=RUNPOD_API_KEY,
        )

        print("Streaming response: ", end="", flush=True)

        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            temperature=0.1,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print(f"\n\n✅ Streaming works! Got {len(full_response)} chars")
        return True

    except Exception as e:
        print(f"\n❌ Streaming FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("RunPod Serverless Endpoint Tester")
    print("=" * 60)

    # Check configuration
    if not RUNPOD_API_KEY or "your_" in TARGET_ENDPOINT_ID:
        print("\n⚠️  WARNING: You need to configure the endpoints!")
        print("\nEdit this script or set environment variables:")
        print("  export RUNPOD_API_KEY='rp_xxxxxxxx'")
        print("  export TARGET_ENDPOINT_ID='abc123def'")
        print("  export JUDGE_ENDPOINT_ID='xyz789ghi'")
        print("\nEndpoint IDs are found in RunPod dashboard:")
        print("  Serverless > Endpoints > Click endpoint > ID at top")
        return

    print(f"\nConfiguration:")
    print(f"  API Key: {RUNPOD_API_KEY[:10]}...{RUNPOD_API_KEY[-4:]}")
    print(f"  Target URL: {TARGET_API_URL}")
    print(f"  Judge URL: {JUDGE_API_URL}")

    results = {}

    # Test Target Model
    results["target"] = test_endpoint(
        "Target Model",
        TARGET_API_URL,
        TARGET_MODEL_NAME,
        "Hello! Can you tell me a short joke?"
    )

    # Test Judge Model
    results["judge"] = test_endpoint(
        "Judge Model",
        JUDGE_API_URL,
        JUDGE_MODEL_NAME,
        "Rate this response on a scale of 1-10: 'The sky is blue because of Rayleigh scattering.'"
    )

    # Test streaming (optional)
    if results["target"]:
        results["streaming"] = test_streaming(
            "Target Model",
            TARGET_API_URL,
            TARGET_MODEL_NAME,
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n🎉 All tests passed! Your endpoints are ready.")
        print("\nCopy these values to your notebook:")
        print(f'''
# RunPod Serverless Configuration
RUNPOD_API_KEY = "{RUNPOD_API_KEY}"
TARGET_ENDPOINT_ID = "{TARGET_ENDPOINT_ID}"
JUDGE_ENDPOINT_ID = "{JUDGE_ENDPOINT_ID}"

TARGET_API_URL = f"https://api.runpod.ai/v2/{TARGET_ENDPOINT_ID}/openai/v1"
JUDGE_API_URL = f"https://api.runpod.ai/v2/{JUDGE_ENDPOINT_ID}/openai/v1"

TARGET_MODEL_NAME = "{TARGET_MODEL_NAME}"
JUDGE_MODEL_NAME = "{JUDGE_MODEL_NAME}"
''')
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nCommon issues:")
        print("  1. Wrong endpoint ID (check RunPod dashboard)")
        print("  2. Endpoint not running (check status in dashboard)")
        print("  3. Wrong model name (must match deployed model)")
        print("  4. Invalid API key (regenerate in Settings > API Keys)")


if __name__ == "__main__":
    main()
