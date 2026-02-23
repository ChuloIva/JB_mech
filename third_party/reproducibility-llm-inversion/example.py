"""
Example usage of the SipIt algorithm with various configurations.

This script demonstrates different use cases of the SipIt algorithm:
1. Basic inversion with random policy
2. Inversion with gradient-guided policy
3. Testing on multiple prompts
4. Analyzing different layers
5. Examining the effect of epsilon tolerance
"""

import torch
from sipit import SipIt
import time


def basic_example():
    """
    Basic example: Invert a simple text prompt.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Inversion")
    print("=" * 80)

    sipit = SipIt(model_name="openai-community/gpt2", layer_idx=0, epsilon=1e-3)

    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original text: '{test_text}'")

    start_time = time.time()
    recovered_text, avg_distance = sipit.invert_from_text(
        test_text,
        policy="gradient",
        verbose=True
    )
    elapsed = time.time() - start_time

    print(f"\nRecovered text: '{recovered_text}'")
    print(f"Average distance: {avg_distance:.6f}")
    print(f"Exact match: {test_text == recovered_text}")
    print(f"Time elapsed: {elapsed:.2f} seconds")


def policy_comparison():
    """
    Compare random vs gradient-guided policies.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Policy Comparison")
    print("=" * 80)

    sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-5)

    test_text = "Hello world"
    print(f"Original text: '{test_text}'")

    # Test with random policy
    print("\n--- Random Policy ---")
    start_time = time.time()
    recovered_random, dist_random = sipit.invert_from_text(
        test_text,
        policy="random",
        verbose=True
    )
    time_random = time.time() - start_time

    print(f"Recovered: '{recovered_random}'")
    print(f"Distance: {dist_random:.6f}")
    print(f"Time: {time_random:.2f}s")

    # Test with gradient policy
    print("\n--- Gradient-Guided Policy ---")
    start_time = time.time()
    recovered_grad, dist_grad = sipit.invert_from_text(
        test_text,
        policy="gradient",
        verbose=True
    )
    time_grad = time.time() - start_time

    print(f"Recovered: '{recovered_grad}'")
    print(f"Distance: {dist_grad:.6f}")
    print(f"Time: {time_grad:.2f}s")


def multiple_prompts():
    """
    Test inversion on multiple prompts of varying length.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multiple Prompts")
    print("=" * 80)

    sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-4)

    test_prompts = [
        "Hi",
        "Hello, world!",
        "The cat sat on the mat.",
        "Machine learning is fascinating.",
        "Natural language processing enables computers to understand human language.",
    ]

    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Testing: '{prompt}'")

        start_time = time.time()
        recovered, avg_dist = sipit.invert_from_text(
            prompt,
            policy="random",
            verbose=True
        )
        elapsed = time.time() - start_time

        match = prompt == recovered
        print(f"  Recovered: '{recovered}'")
        print(f"  Match: {match}, Distance: {avg_dist:.6f}, Time: {elapsed:.2f}s")

        results.append({
            'original': prompt,
            'recovered': recovered,
            'match': match,
            'distance': avg_dist,
            'time': elapsed
        })

    print("\n" + "-" * 80)
    print("Summary:")
    matches = sum(1 for r in results if r['match'])
    print(f"  Total prompts: {len(results)}")
    print(f"  Exact matches: {matches}/{len(results)}")
    print(f"  Average distance: {sum(r['distance'] for r in results) / len(results):.6f}")
    print(f"  Average time: {sum(r['time'] for r in results) / len(results):.2f}s")


def layer_analysis():
    """
    Analyze inversion quality across different layers.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Layer Analysis")
    print("=" * 80)

    test_text = "Language models are powerful."
    print(f"Original text: '{test_text}'")

    # GPT-2 has 12 layers (0-11)
    layers_to_test = [0, 3, 6, 9, 11]

    results = []
    for layer_idx in layers_to_test:
        print(f"\n--- Testing Layer {layer_idx} ---")

        sipit = SipIt(model_name="gpt2", layer_idx=layer_idx, epsilon=1e-4)

        start_time = time.time()
        recovered, avg_dist = sipit.invert_from_text(
            test_text,
            policy="random",
            verbose=True
        )
        elapsed = time.time() - start_time

        match = test_text == recovered
        print(f"Recovered: '{recovered}'")
        print(f"Match: {match}, Distance: {avg_dist:.6f}, Time: {elapsed:.2f}s")

        results.append({
            'layer': layer_idx,
            'match': match,
            'distance': avg_dist,
            'time': elapsed
        })

    print("\n" + "-" * 80)
    print("Layer Summary:")
    for r in results:
        print(f"  Layer {r['layer']:2d}: Match={r['match']}, "
              f"Distance={r['distance']:.6f}, Time={r['time']:.2f}s")


def epsilon_analysis():
    """
    Examine the effect of epsilon tolerance on inversion.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Epsilon Tolerance Analysis")
    print("=" * 80)

    test_text = "Testing epsilon values."
    print(f"Original text: '{test_text}'")

    epsilon_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    results = []
    for eps in epsilon_values:
        print(f"\n--- Epsilon = {eps} ---")

        sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=eps)

        start_time = time.time()
        recovered, avg_dist = sipit.invert_from_text(
            test_text,
            policy="random",
            verbose=True
        )
        elapsed = time.time() - start_time

        match = test_text == recovered
        print(f"Recovered: '{recovered}'")
        print(f"Match: {match}, Distance: {avg_dist:.6f}, Time: {elapsed:.2f}s")

        results.append({
            'epsilon': eps,
            'match': match,
            'distance': avg_dist,
            'time': elapsed
        })

    print("\n" + "-" * 80)
    print("Epsilon Summary:")
    for r in results:
        print(f"  ε={r['epsilon']:.0e}: Match={r['match']}, "
              f"Distance={r['distance']:.6f}, Time={r['time']:.2f}s")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 80)
    print("SipIt Algorithm - Comprehensive Examples")
    print("Paper: Language Models are Injective and Hence Invertible")
    print("https://arxiv.org/abs/2510.15511")
    print("=" * 80)

    try:
        # Run examples
        basic_example()
        # policy_comparison()
        # multiple_prompts()
        # layer_analysis()
        # epsilon_analysis()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
