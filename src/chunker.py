"""
Text Chunker for splitting documents into manageable chunks.

Implements sliding window chunking with configurable size and overlap.
Uses tiktoken for accurate token counting.
"""

import tiktoken
from typing import List, Dict, Optional


class TextChunker:
    """Chunk text into smaller pieces with overlap for better context preservation."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base"  # Used by GPT-3.5/4
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
        
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text or not text.strip():
            return []
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        
        # If text fits in one chunk, return as-is
        if len(tokens) <= self.chunk_size:
            return [{
                'text': text,
                'metadata': metadata or {}
            }]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'chunk_size_tokens': len(chunk_tokens),
                'total_chunks': None  # Will be updated later
            })
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
            
            # Move start index forward, accounting for overlap
            if end_idx >= len(tokens):
                break
            
            start_idx = end_idx - self.chunk_overlap
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dictionaries with 'text' and 'metadata' keys
        
        Returns:
            List of chunked documents
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def estimate_chunks(self, text: str) -> int:
        """
        Estimate number of chunks needed for text.
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated number of chunks
        """
        tokens = self.estimate_tokens(text)
        if tokens <= self.chunk_size:
            return 1
        
        # Calculate chunks accounting for overlap
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        chunks = 1 + (tokens - self.chunk_size) // effective_chunk_size
        return max(1, chunks)
