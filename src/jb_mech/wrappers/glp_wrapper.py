"""
Wrapper for the Generative Latent Prior (GLP) library.

Provides a unified interface for loading GLP models, extracting meta-neurons,
and running 1-D scalar probing for safety-related classification tasks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from ..config import (
    GLP_MODEL,
    GLP_CHECKPOINT,
    GLP_U_TIMESTEP,
    TARGET_LAYER,
    GLP_DIR,
    add_third_party_to_path,
)


class GLPWrapper:
    """
    Unified interface for GLP operations on Llama 3.1 8B.

    This wrapper handles:
    - GLP model loading from HuggingFace
    - Raw activation extraction using baukit
    - Meta-neuron extraction via diffusion model
    - 1-D scalar probing for binary classification
    """

    def __init__(
        self,
        device: str = "cuda:0",
        weights_folder: str = GLP_MODEL,
        checkpoint: str = GLP_CHECKPOINT,
        load_model: bool = True,
    ):
        """
        Initialize the GLP wrapper.

        Args:
            device: Device to load model on
            weights_folder: HuggingFace model ID or local path
            checkpoint: Checkpoint name (e.g., "final")
            load_model: Whether to load the model immediately
        """
        # Add GLP to path
        add_third_party_to_path()

        self.device = device
        self.weights_folder = weights_folder
        self.checkpoint = checkpoint
        self.model = None
        self.layers = None

        if load_model:
            self._load_model()

    def _load_model(self):
        """Load the GLP model."""
        from glp.denoiser import load_glp
        from glp.script_probe import get_meta_neurons_locations

        print(f"Loading GLP model: {self.weights_folder}")
        self.model = load_glp(self.weights_folder, device=self.device, checkpoint=self.checkpoint)
        self.layers = get_meta_neurons_locations(self.model)
        print(f"GLP loaded with {len(self.layers)} meta-neuron layers")

    def extract_activations_baukit(
        self,
        hf_model,
        hf_tokenizer,
        texts: List[str],
        layer: int = TARGET_LAYER,
        token_idx: str = "last",
    ) -> torch.Tensor:
        """
        Extract raw activations from a HuggingFace model using baukit.

        Args:
            hf_model: HuggingFace model
            hf_tokenizer: HuggingFace tokenizer
            texts: List of text prompts
            layer: Layer index to extract
            token_idx: Token position ("last", "first", or int)

        Returns:
            Tensor of activations (batch, 1, hidden_dim)
        """
        from baukit import TraceDict

        layer_name = f"model.layers.{layer}"
        activations = []

        for text in tqdm(texts, desc="Extracting activations"):
            inputs = hf_tokenizer(text, return_tensors="pt", padding=True).to(hf_model.device)

            with torch.no_grad():
                with TraceDict(hf_model, layers=[layer_name], retain_output=True) as ret:
                    _ = hf_model(**inputs)

                # Get output activation
                act = ret[layer_name].output
                if isinstance(act, tuple):
                    act = act[0]

                # Select token position
                if token_idx == "last":
                    # Get last non-padding token
                    seq_len = inputs.attention_mask.sum(dim=1) - 1
                    act = act[0, seq_len[0].item(), :]
                elif token_idx == "first":
                    act = act[0, 0, :]
                else:
                    act = act[0, token_idx, :]

                activations.append(act.cpu())

        return torch.stack(activations)[:, None, :]

    def get_meta_neurons(
        self,
        activations: torch.Tensor,
        u: float = GLP_U_TIMESTEP,
        seed: int = 42,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Extract meta-neurons from raw activations via GLP diffusion.

        Args:
            activations: Raw activations (batch, 1, hidden_dim) or (batch, hidden_dim)
            u: Diffusion timestep (0-1, higher = more abstract)
            seed: Random seed for reproducibility
            batch_size: Batch size for processing

        Returns:
            Meta-neurons tensor (n_layers * n_timesteps, batch, dim)
        """
        if self.model is None:
            raise RuntimeError("GLP model not loaded. Initialize with load_model=True.")

        from glp.script_probe import get_meta_neurons_layer_time

        # Ensure correct shape
        if activations.ndim == 3:
            X = activations[:, 0, :].float()
        else:
            X = activations.float()

        u_tensor = torch.tensor([u])[:, None]

        meta_neurons, (layer_size, u_size) = get_meta_neurons_layer_time(
            self.model,
            self.device,
            X,
            u_tensor,
            self.layers,
            seed=seed,
            batch_size=batch_size,
        )

        return meta_neurons

    def run_probing(
        self,
        X_train_meta: torch.Tensor,
        y_train: torch.Tensor,
        X_test_meta: torch.Tensor,
        y_test: torch.Tensor,
        topk: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Run 1-D scalar probing on meta-neurons.

        Args:
            X_train_meta: Training meta-neurons (n_layers, batch, dim)
            y_train: Training labels (batch,)
            X_test_meta: Test meta-neurons (n_layers, batch, dim)
            y_test: Test labels (batch,)
            topk: Number of top features to select

        Returns:
            Tuple of (val_aucs, test_aucs, top_indices)
        """
        from glp.script_probe import prefilter_and_reshape_to_oned, run_sklearn_logreg_batched

        # Prefilter and reshape for 1-D probing
        X_train_1d, X_test_1d, top_indices = prefilter_and_reshape_to_oned(
            X_train_meta, X_test_meta, y_train, self.device, topk=topk
        )

        # Run logistic regression
        val_aucs, test_aucs = run_sklearn_logreg_batched(
            X_train_1d, y_train, X_test_1d, y_test, device=self.device
        )

        return val_aucs, test_aucs, top_indices

    def find_best_neuron(
        self,
        X_train_meta: torch.Tensor,
        y_train: torch.Tensor,
        X_test_meta: torch.Tensor,
        y_test: torch.Tensor,
        topk: int = 512,
    ) -> Tuple[int, float, float]:
        """
        Find the best single neuron for a binary classification task.

        Args:
            X_train_meta: Training meta-neurons
            y_train: Training labels
            X_test_meta: Test meta-neurons
            y_test: Test labels
            topk: Number of top features to consider

        Returns:
            Tuple of (best_neuron_idx, val_auc, test_auc)
        """
        val_aucs, test_aucs, top_indices = self.run_probing(
            X_train_meta, y_train, X_test_meta, y_test, topk=topk
        )

        best_idx = np.argmax(val_aucs)
        best_neuron = top_indices[best_idx]

        return best_neuron, val_aucs[best_idx], test_aucs[best_idx]

    def load_persona_vector(self, persona_path: Union[str, Path]) -> torch.Tensor:
        """
        Load a pre-computed persona steering vector.

        Args:
            persona_path: Path to the .pt file

        Returns:
            Persona vector tensor
        """
        return torch.load(persona_path, map_location="cpu")

    def close(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self.layers = None

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SafetySpaceAnalyzer:
    """
    Build and analyze the (H, J, R) safety coordinate system using GLP meta-neurons.

    H = Harm detection (harmful vs benign prompts)
    J = Jailbreak success (successful vs failed jailbreaks)
    R = Refusal (model refuses vs complies)
    """

    def __init__(self, glp_wrapper: GLPWrapper):
        """
        Initialize the safety space analyzer.

        Args:
            glp_wrapper: Initialized GLPWrapper instance
        """
        self.glp = glp_wrapper

        # Best neuron indices for each dimension
        self.h_neuron_idx = None
        self.j_neuron_idx = None
        self.r_neuron_idx = None

        # Neuron directions (for steering)
        self.h_direction = None
        self.j_direction = None
        self.r_direction = None

        # AUC scores
        self.h_auc = None
        self.j_auc = None
        self.r_auc = None

        # Correlation matrix
        self.correlation_matrix = None

    def train_h_probe(
        self,
        harmful_meta: torch.Tensor,
        benign_meta: torch.Tensor,
        train_split: float = 0.8,
        topk: int = 512,
    ) -> Tuple[int, float]:
        """
        Train the H (harm detection) probe.

        Args:
            harmful_meta: Meta-neurons for harmful prompts
            benign_meta: Meta-neurons for benign prompts
            train_split: Fraction for training
            topk: Number of features to consider

        Returns:
            Tuple of (best_neuron_idx, test_auc)
        """
        X_meta, y, train_idx, test_idx = self._prepare_data(
            harmful_meta, benign_meta, train_split
        )

        self.h_neuron_idx, val_auc, self.h_auc = self.glp.find_best_neuron(
            X_meta[:, train_idx, :],
            y[train_idx],
            X_meta[:, test_idx, :],
            y[test_idx],
            topk=topk,
        )

        print(f"H (Harm) best neuron: {self.h_neuron_idx}, AUC: {self.h_auc:.4f}")
        return self.h_neuron_idx, self.h_auc

    def train_j_probe(
        self,
        success_meta: torch.Tensor,
        fail_meta: torch.Tensor,
        train_split: float = 0.8,
        topk: int = 512,
    ) -> Tuple[int, float]:
        """
        Train the J (jailbreak success) probe.

        Args:
            success_meta: Meta-neurons for successful jailbreaks
            fail_meta: Meta-neurons for failed jailbreaks
            train_split: Fraction for training
            topk: Number of features to consider

        Returns:
            Tuple of (best_neuron_idx, test_auc)
        """
        X_meta, y, train_idx, test_idx = self._prepare_data(
            success_meta, fail_meta, train_split
        )

        self.j_neuron_idx, val_auc, self.j_auc = self.glp.find_best_neuron(
            X_meta[:, train_idx, :],
            y[train_idx],
            X_meta[:, test_idx, :],
            y[test_idx],
            topk=topk,
        )

        print(f"J (Jailbreak) best neuron: {self.j_neuron_idx}, AUC: {self.j_auc:.4f}")
        return self.j_neuron_idx, self.j_auc

    def train_r_probe(
        self,
        refusal_meta: torch.Tensor,
        compliance_meta: torch.Tensor,
        train_split: float = 0.8,
        topk: int = 512,
    ) -> Tuple[int, float]:
        """
        Train the R (refusal) probe.

        Args:
            refusal_meta: Meta-neurons for refusal responses
            compliance_meta: Meta-neurons for compliance responses
            train_split: Fraction for training
            topk: Number of features to consider

        Returns:
            Tuple of (best_neuron_idx, test_auc)
        """
        X_meta, y, train_idx, test_idx = self._prepare_data(
            refusal_meta, compliance_meta, train_split
        )

        self.r_neuron_idx, val_auc, self.r_auc = self.glp.find_best_neuron(
            X_meta[:, train_idx, :],
            y[train_idx],
            X_meta[:, test_idx, :],
            y[test_idx],
            topk=topk,
        )

        print(f"R (Refusal) best neuron: {self.r_neuron_idx}, AUC: {self.r_auc:.4f}")
        return self.r_neuron_idx, self.r_auc

    def _prepare_data(
        self,
        pos_meta: torch.Tensor,
        neg_meta: torch.Tensor,
        train_split: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Prepare data for probing."""
        n_pos = pos_meta.shape[1]
        n_neg = neg_meta.shape[1]

        X_meta = torch.cat([pos_meta, neg_meta], dim=1)
        y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])

        # Shuffle and split
        n_total = n_pos + n_neg
        perm = torch.randperm(n_total).numpy()
        split = int(n_total * train_split)

        train_idx = perm[:split]
        test_idx = perm[split:]

        return X_meta, y, train_idx, test_idx

    def project_to_hjr(self, meta_neurons: torch.Tensor) -> torch.Tensor:
        """
        Project meta-neurons to (H, J, R) space.

        Args:
            meta_neurons: Meta-neurons tensor (n_layers, batch, dim)

        Returns:
            Tensor of shape (batch, 3) with [H, J, R] coordinates
        """
        if any(idx is None for idx in [self.h_neuron_idx, self.j_neuron_idx, self.r_neuron_idx]):
            raise ValueError("Not all probes trained. Call train_h_probe, train_j_probe, train_r_probe first.")

        # Get max activation across layers for each neuron
        H = meta_neurons[:, :, self.h_neuron_idx].max(dim=0).values
        J = meta_neurons[:, :, self.j_neuron_idx].max(dim=0).values
        R = meta_neurons[:, :, self.r_neuron_idx].max(dim=0).values

        return torch.stack([H, J, R], dim=1)

    def check_orthogonality(self, all_meta: torch.Tensor) -> torch.Tensor:
        """
        Check orthogonality of H, J, R axes.

        Args:
            all_meta: All meta-neurons for computing correlations

        Returns:
            Correlation matrix (3, 3)
        """
        hjr = self.project_to_hjr(all_meta)
        corr = np.corrcoef(hjr.cpu().numpy().T)
        self.correlation_matrix = torch.tensor(corr)

        print("H, J, R Correlation Matrix:")
        print(f"      H       J       R")
        print(f"H  {corr[0, 0]:.3f}   {corr[0, 1]:.3f}   {corr[0, 2]:.3f}")
        print(f"J  {corr[1, 0]:.3f}   {corr[1, 1]:.3f}   {corr[1, 2]:.3f}")
        print(f"R  {corr[2, 0]:.3f}   {corr[2, 1]:.3f}   {corr[2, 2]:.3f}")

        return self.correlation_matrix

    def save_infrastructure(self, path: Union[str, Path]):
        """
        Save the trained safety space infrastructure.

        Args:
            path: Path to save the .pt file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "h_neuron_idx": self.h_neuron_idx,
            "j_neuron_idx": self.j_neuron_idx,
            "r_neuron_idx": self.r_neuron_idx,
            "h_auc": self.h_auc,
            "j_auc": self.j_auc,
            "r_auc": self.r_auc,
            "correlation_matrix": self.correlation_matrix,
        }, path)

        print(f"Saved safety space infrastructure to {path}")

    def load_infrastructure(self, path: Union[str, Path]):
        """
        Load a previously trained safety space infrastructure.

        Args:
            path: Path to the .pt file
        """
        path = Path(path)
        data = torch.load(path)

        self.h_neuron_idx = data["h_neuron_idx"]
        self.j_neuron_idx = data["j_neuron_idx"]
        self.r_neuron_idx = data["r_neuron_idx"]
        self.h_auc = data["h_auc"]
        self.j_auc = data["j_auc"]
        self.r_auc = data["r_auc"]
        self.correlation_matrix = data.get("correlation_matrix")

        print(f"Loaded safety space infrastructure from {path}")
        print(f"H neuron: {self.h_neuron_idx}, AUC: {self.h_auc:.4f}")
        print(f"J neuron: {self.j_neuron_idx}, AUC: {self.j_auc:.4f}")
        print(f"R neuron: {self.r_neuron_idx}, AUC: {self.r_auc:.4f}")
