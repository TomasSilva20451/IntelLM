"""
OpenAI Embeddings integration with rate limiting and batch processing.

Generates embeddings for text chunks using OpenAI's embedding API.
"""

import os
import time
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .rate_limiter import RateLimiter
from .chunker import TextChunker
from .embedding_base import EmbeddingGeneratorBase

# Load environment variables
load_dotenv()


class OpenAIEmbeddingGenerator(EmbeddingGeneratorBase):
    """Generate embeddings using OpenAI API with rate limiting."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Embedding model to use
            rate_limiter: Rate limiter instance (creates new one if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize rate limiter if not provided
        self.rate_limiter = rate_limiter or RateLimiter(
            embedding_model=model
        )
        
        # Initialize chunker for token estimation
        self.chunker = TextChunker()
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate total tokens for a list of texts."""
        total = 0
        for text in texts:
            total += self.chunker.estimate_tokens(text)
        return total
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with batching and rate limiting.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch
            show_progress: Whether to print progress
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            if show_progress:
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            
            # Estimate tokens for rate limiting
            tokens_needed = self._estimate_tokens(batch)
            
            # Acquire rate limit permission
            success, wait_time = self.rate_limiter.acquire(
                tokens_needed=tokens_needed,
                wait=True
            )
            
            if not success:
                if show_progress:
                    print(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                # Retry acquisition
                self.rate_limiter.acquire(tokens_needed=tokens_needed, wait=True)
            
            # Generate embeddings with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Failed to generate embeddings after {max_retries} retries: {str(e)}")
                    
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    if show_progress:
                        print(f"Error generating embeddings, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            # Small delay between batches to be respectful
            if batch_idx + batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    
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
            Estimated cost in USD
        """
        total_tokens = self._estimate_tokens(texts)
        return self.rate_limiter.estimate_cost(total_tokens)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return self.rate_limiter.get_stats()
