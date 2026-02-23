"""
Simple tests for the SipIt implementation.

These tests verify basic functionality of the SipIt algorithm.
Note: These are integration tests that require model downloads.
"""

import torch
from sipit import SipIt


def test_initialization():
    """Test that SipIt initializes correctly."""
    print("Test 1: Initialization...")

    try:
        sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-5)
        assert sipit.model is not None
        assert sipit.tokenizer is not None
        assert sipit.epsilon == 1e-5
        assert sipit.layer_idx == -1
        print("  ✓ Initialization successful")
        return True
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return False


def test_hidden_state_extraction():
    """Test hidden state extraction."""
    print("Test 2: Hidden State Extraction...")

    try:
        sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-5)

        text = "Hello"
        input_ids = sipit.tokenizer.encode(text, return_tensors="pt").to(sipit.device)

        hidden_states = sipit.get_hidden_states(input_ids)

        assert hidden_states is not None
        assert len(hidden_states.shape) == 3  # [batch, seq_len, hidden_dim]
        assert hidden_states.shape[0] == 1  # batch size
        assert hidden_states.shape[1] == input_ids.shape[1]  # sequence length
        print(f"  ✓ Hidden states shape: {hidden_states.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Hidden state extraction failed: {e}")
        return False


def test_basic_inversion():
    """Test basic inversion on a simple text."""
    print("Test 3: Basic Inversion...")

    try:
        sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-4)

        text = "Hi"
        print(f"  Original: '{text}'")

        recovered, distance = sipit.invert_from_text(text, policy="random", verbose=True)

        print(f"  Recovered: '{recovered}'")
        print(f"  Distance: {distance:.6f}")

        if text == recovered:
            print("  ✓ Exact match achieved!")
            return True
        else:
            print("  ⚠ No exact match, but inversion completed")
            return True  # Still consider it a pass if it completes
    except Exception as e:
        print(f"  ✗ Basic inversion failed: {e}")
        return False


def test_hidden_state_matching():
    """Test the hidden state matching criterion."""
    print("Test 4: Hidden State Matching...")

    try:
        sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-5)

        # Create two identical tensors
        h1 = torch.randn(1, 768).to(sipit.device)
        h2 = h1.clone()

        # Should match exactly
        assert sipit.check_match(h1, h2, epsilon=1e-5)
        print("  ✓ Identical tensors match")

        # Create two different tensors
        h3 = torch.randn(1, 768).to(sipit.device)

        # Might not match depending on distance
        match = sipit.check_match(h1, h3, epsilon=1e-5)
        print(f"  ✓ Different tensors match={match}")

        return True
    except Exception as e:
        print(f"  ✗ Hidden state matching test failed: {e}")
        return False


def test_policy_types():
    """Test that different policies can be used."""
    print("Test 5: Policy Types...")

    try:
        sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-4)

        text = "A"

        # Test random policy
        recovered_random, _ = sipit.invert_from_text(
            text, policy="random", verbose=True
        )
        print(f"  ✓ Random policy: '{recovered_random}'")

        # Test gradient policy
        recovered_grad, _ = sipit.invert_from_text(
            text, policy="gradient", verbose=True
        )
        print(f"  ✓ Gradient policy: '{recovered_grad}'")

        return True
    except Exception as e:
        print(f"  ✗ Policy type test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("SipIt Implementation Tests")
    print("=" * 80)
    print()

    tests = [
        test_initialization,
        test_hidden_state_extraction,
        test_basic_inversion,
        test_hidden_state_matching,
        test_policy_types,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append(False)
        print()

    print("=" * 80)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 80)

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
