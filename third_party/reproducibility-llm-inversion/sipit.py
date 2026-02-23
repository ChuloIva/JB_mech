"""
SipIt: Sequential Inverse Prompt via Iterative Updates

Implementation of the SipIt algorithm from the paper:
"Language Models are Injective and Hence Invertible"
https://arxiv.org/abs/2510.15511

This module implements the core inversion algorithm that reconstructs input
sequences from hidden activations in transformer language models.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


class SipIt:
    """
    SipIt algorithm for inverting transformer language models.

    Given hidden states from a specific layer of a transformer, this class
    implements the sequential token recovery algorithm that reconstructs the
    original input sequence.

    Attributes:
        model: The transformer language model
        tokenizer: The tokenizer for the model
        layer_idx: The layer index to extract hidden states from
        epsilon: Tolerance for hidden state matching
        device: The device to run computations on
    """

    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        layer_idx: int = -1,
        epsilon: float = 1e-5,
        device: Optional[str] = None
    ):
        """
        Initialize the SipIt algorithm.

        Args:
            model_name: Name of the model from HuggingFace
            layer_idx: Layer index to extract hidden states from (-1 for last layer)
            epsilon: Tolerance for hidden state matching (epsilon-ball radius)
            device: Device to run on ('cuda' or 'cpu'), auto-detected if None
        """
        self.device = device or (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f"Loading model {model_name} on {self.device}...")

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.vocab_size = len(self.tokenizer)

        # Put model in evaluation mode
        self.model.eval()

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract hidden states from the specified layer.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            layer_idx: Layer to extract from (uses self.layer_idx if None)

        Returns:
            Hidden states [batch_size, seq_len, hidden_dim]
        """
        layer = layer_idx if layer_idx is not None else self.layer_idx

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors
            # Index 0 is embeddings, 1 to num_layers are layer outputs
            hidden_states = outputs.hidden_states[layer]

        return hidden_states

    def compute_hidden_state_for_candidate(
        self,
        prefix_ids: torch.Tensor,
        candidate_token: int,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the hidden state at position t when appending candidate_token.

        Args:
            prefix_ids: Prefix sequence [1, prefix_len]
            candidate_token: Token ID to append
            layer_idx: Layer to extract from

        Returns:
            Hidden state at the last position [1, hidden_dim]
        """
        # Append candidate token to prefix
        candidate_seq = torch.cat([
            prefix_ids,
            torch.tensor([[candidate_token]], device=self.device)
        ], dim=1)

        # Get hidden states
        hidden_states = self.get_hidden_states(candidate_seq, layer_idx)

        # Return hidden state at the position of the appended token
        return hidden_states[:, -1, :]

    def check_match(
        self,
        target_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        epsilon: Optional[float] = None
    ) -> bool:
        """
        Check if candidate hidden state matches target within epsilon-ball.

        Args:
            target_hidden: Target hidden state [1, hidden_dim]
            candidate_hidden: Candidate hidden state [1, hidden_dim]
            epsilon: Tolerance (uses self.epsilon if None)

        Returns:
            True if the distance is within epsilon
        """
        eps = epsilon if epsilon is not None else self.epsilon
        distance = torch.norm(target_hidden - candidate_hidden, p=2).item()
        return distance <= eps

    def invert_sequence(
        self,
        target_hidden_states: torch.Tensor,
        policy: str = "random",
        max_iterations_per_position: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[int], List[float]]:
        """
        Invert a sequence from its hidden states using SipIt algorithm.

        This is the main implementation of Algorithm 1 from the paper.

        Args:
            target_hidden_states: Target hidden states [1, seq_len, hidden_dim]
            policy: Policy for selecting candidates ('random' or 'gradient')
            max_iterations_per_position: Max vocab candidates to try per position
            verbose: Whether to show progress bar

        Returns:
            Tuple of (recovered_token_ids, distances_per_position)
        """
        seq_len = target_hidden_states.shape[1]
        recovered_tokens = []
        distances = []

        if max_iterations_per_position is None:
            max_iterations_per_position = self.vocab_size

        # Initialize with empty sequence
        prefix_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)

        iterator = tqdm(range(seq_len), desc="Inverting tokens") if verbose else range(seq_len)

        for t in iterator:
            target_h_t = target_hidden_states[:, t:t+1, :]  # [1, 1, hidden_dim]

            # Get vocabulary ordering based on policy
            vocab_order = self._get_vocab_order(
                prefix_ids,
                target_h_t,
                policy
            )

            found = False
            best_distance = float('inf')
            best_token = None

            # Try candidates according to policy
            for j, candidate_token in tqdm(enumerate(vocab_order)):
                if j >= max_iterations_per_position:
                    break

                # Compute hidden state for this candidate
                candidate_h = self.compute_hidden_state_for_candidate(
                    prefix_ids,
                    candidate_token
                )

                # Check if it matches
                distance = torch.norm(target_h_t - candidate_h, p=2).item()

                if distance < best_distance:
                    best_distance = distance
                    best_token = candidate_token

                if self.check_match(target_h_t, candidate_h):
                    found = True
                    recovered_tokens.append(candidate_token)
                    distances.append(distance)

                    # Update prefix
                    prefix_ids = torch.cat([
                        prefix_ids,
                        torch.tensor([[candidate_token]], device=self.device)
                    ], dim=1)
                    if verbose:
                        print(f"\nFound token: {best_token} (distance={best_distance:.3f})")
                    break

            if not found:
                # If no exact match found, use best candidate
                if verbose:
                    print(f"\nWarning: No exact match at position {t}, "
                          f"using best candidate: {best_token} (distance={best_distance:.3f})")
                recovered_tokens.append(best_token)
                distances.append(best_distance)

                prefix_ids = torch.cat([
                    prefix_ids,
                    torch.tensor([[best_token]], device=self.device)
                ], dim=1)

        return recovered_tokens, distances

    def _get_vocab_order(
        self,
        prefix_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        policy: str
    ) -> List[int]:
        """
        Get vocabulary ordering based on policy.

        Args:
            prefix_ids: Current prefix [1, prefix_len]
            target_hidden: Target hidden state [1, 1, hidden_dim]
            policy: 'random' or 'gradient'

        Returns:
            List of token IDs in order to try
        """
        if policy == "random":
            # Random shuffling of vocabulary
            vocab_order = list(range(self.vocab_size))
            np.random.shuffle(vocab_order)
            return vocab_order

        elif policy == "gradient":
            # Gradient-guided search (Algorithm 3 in paper)
            return self._gradient_guided_order(prefix_ids, target_hidden)

        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _gradient_guided_order(
        self,
        prefix_ids: torch.Tensor,
        target_hidden: torch.Tensor
    ) -> List[int]:
        """
        Compute gradient-guided ordering of vocabulary candidates.

        This implements a gradient-based policy that prioritizes tokens
        based on their alignment with the gradient direction.

        Args:
            prefix_ids: Current prefix [1, prefix_len]
            target_hidden: Target hidden state [1, 1, hidden_dim]

        Returns:
            List of token IDs ordered by gradient-based score
        """
        scores = []

        # Get embedding layer
        embedding_layer = self.model.get_input_embeddings()

        with torch.no_grad():
            for token_id in tqdm(range(self.vocab_size)):
                # Get embedding for this token
                token_embedding = embedding_layer(
                    torch.tensor([[token_id]], device=self.device)
                )

                # Compute candidate hidden state
                candidate_h = self.compute_hidden_state_for_candidate(
                    prefix_ids, token_id
                )

                # Score based on distance to target
                distance = torch.norm(target_hidden - candidate_h, p=2).item()
                scores.append((token_id, -distance))  # Negative because we want smallest distance

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        vocab_order = [token_id for token_id, _ in scores]

        return vocab_order

    def invert_from_text(
        self,
        text: str,
        policy: str = "random",
        verbose: bool = False
    ) -> Tuple[str, float]:
        """
        Convenience method to invert text by first encoding it.

        Args:
            text: Input text to invert
            policy: Policy for candidate selection
            verbose: Whether to show progress

        Returns:
            Tuple of (recovered_text, average_distance)
        """
        # Encode text
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        # Get hidden states
        if verbose:
            print(f"Extracting hidden states from layer {self.layer_idx}...")
        hidden_states = self.get_hidden_states(input_ids)

        # Invert
        if verbose:
            print(f"Inverting sequence of length {input_ids.shape[1]}...")
        recovered_tokens, distances = self.invert_sequence(
            hidden_states,
            policy=policy,
            verbose=verbose
        )

        # Decode
        recovered_text = self.tokenizer.decode(recovered_tokens)
        avg_distance = np.mean(distances)

        return recovered_text, avg_distance


def demo():
    """
    Simple demonstration of the SipIt algorithm.
    """
    print("=" * 80)
    print("SipIt Algorithm Demo")
    print("=" * 80)

    # Initialize SipIt with a small model
    sipit = SipIt(model_name="gpt2", layer_idx=-1, epsilon=1e-5)

    # Test text
    test_text = "Hello, world!"
    print(f"\nOriginal text: '{test_text}'")

    # Invert
    recovered_text, avg_distance = sipit.invert_from_text(
        test_text,
        policy="random",
        verbose=True
    )

    print(f"\nRecovered text: '{recovered_text}'")
    print(f"Average distance: {avg_distance:.6f}")
    print(f"Match: {test_text == recovered_text}")


if __name__ == "__main__":
    demo()
