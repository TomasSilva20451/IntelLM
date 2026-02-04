"""
Local Embeddings using sentence-transformers (FREE, no API calls).

Uses Hugging Face models that run locally on your machine.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import warnings

from .embedding_base import EmbeddingGeneratorBase

# Suppress warnings
warnings.filterwarnings('ignore')


class LocalEmbeddingGenerator(EmbeddingGeneratorBase):
    """Generate embeddings locally using sentence-transformers (FREE)."""
    
    # Popular free models (in order of quality/speed tradeoff)
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "name": "all-MiniLM-L6-v2",
            "description": "Fast, good quality (384 dimensions)",
            "size": "small"
        },
        "all-mpnet-base-v2": {
            "name": "all-mpnet-base-v2",
            "description": "Best quality, slower (768 dimensions)",
            "size": "medium"
        },
        "paraphrase-MiniLM-L6-v2": {
            "name": "paraphrase-MiniLM-L6-v2",
            "description": "Good for semantic similarity (384 dimensions)",
            "size": "small"
        }
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            # Load model (will download on first use)
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch
            show_progress: Whether to show progress
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if not self.model:
            self._load_model()
        
        # Generate embeddings (runs locally, no API calls)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def estimate_cost(self, texts: List[str]) -> float:
        """
        Estimate cost for generating embeddings.
        
        Args:
            texts: List of texts
        
        Returns:
            Always 0.0 (local embeddings are free!)
        """
        return 0.0
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about a model."""
        return cls.AVAILABLE_MODELS.get(model_name, {})
