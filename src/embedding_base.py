"""
Base class for embedding generators.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingGeneratorBase(ABC):
    """Base class for embedding generators."""
    
    @abstractmethod
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def estimate_cost(self, texts: List[str]) -> float:
        """Estimate cost for generating embeddings."""
        pass
