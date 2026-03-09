"""
Embedding generation for LLM vulnerability descriptions.

Uses OpenAI text-embedding-3-small with efficient batching and caching.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import openai
from tqdm import tqdm


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 2048  # OpenAI max
    cache_dir: Path = Path("outputs/vuln_analysis/embeddings_cache")


class VulnerabilityEmbedder:
    """
    Generate and cache embeddings for vulnerability descriptions.

    Uses OpenAI's text-embedding-3-small model with efficient batching.
    Caches embeddings to avoid recomputation.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.client = openai.OpenAI()  # Uses OPENAI_API_KEY
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, texts: List[str]) -> Path:
        """Generate cache path based on hash of input texts."""
        content_hash = hashlib.md5(
            json.dumps(texts, sort_keys=True).encode()
        ).hexdigest()[:16]
        return self.config.cache_dir / f"embeddings_{self.config.model}_{content_hash}.npz"

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed texts with batching and caching.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            (N, D) array of embeddings
        """
        cache_path = self._get_cache_path(texts)
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)["embeddings"]

        embeddings = []
        batch_size = self.config.batch_size

        # Calculate number of batches
        n_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches", total=n_batches)

        for i in iterator:
            batch = texts[i:i + batch_size]

            # Filter out empty strings
            batch_clean = [t if t.strip() else "empty" for t in batch]

            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch_clean,
            )
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings, dtype=np.float32)
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"Cached embeddings to {cache_path}")

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        response = self.client.embeddings.create(
            model=self.config.model,
            input=[text if text.strip() else "empty"],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
