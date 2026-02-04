"""
RAG Chain - Orchestrates the full RAG pipeline.

Combines retrieval and generation to answer questions based on documents.
"""

from typing import List, Dict, Optional
from .vector_store import VectorStore
from .embedding_base import EmbeddingGeneratorBase
from .llm_client import OllamaClient


class RAGChain:
    """Retrieval-Augmented Generation chain."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGeneratorBase,
        llm_client: OllamaClient,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            llm_client: LLM client instance
            top_k: Number of chunks to retrieve
            similarity_threshold: Maximum distance (0-1) for chunks to be considered relevant.
                                  Chunks with distance > threshold will trigger general knowledge fallback.
                                  For cosine similarity: 0 = identical, 1 = completely different.
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def _extract_document_name_from_query(self, question: str) -> Optional[str]:
        """
        Extract document name from query if mentioned.
        
        Args:
            question: User question
        
        Returns:
            Document name if found, None otherwise
        """
        # Get all available document sources
        try:
            available_sources = self.vector_store.get_unique_sources()
        except:
            return None
        
        if not available_sources:
            return None
        
        # Check if any document name is mentioned in the question
        question_lower = question.lower()
        
        for source in available_sources:
            # Check if the document name (or part of it) appears in the question
            source_lower = source.lower()
            source_name_only = source.split('/')[-1]  # Get just filename
            
            # Check for full filename match
            if source_name_only.lower() in question_lower or source_lower in question_lower:
                return source
            
            # Check for partial matches (at least 5 characters)
            if len(source_name_only) >= 5:
                # Try matching significant parts of the filename
                parts = source_name_only.replace('.pdf', '').replace('.docx', '').split('_')
                for part in parts:
                    if len(part) >= 5 and part.lower() in question_lower:
                        return source
        
        return None
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '')
            text = chunk.get('text', '')
            
            # Build citation
            citation = f"[{i}]"
            if page:
                citation += f" {source} (page {page})"
            else:
                citation += f" {source}"
            
            context_parts.append(f"{citation}\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str = None, max_context_length: int = 3000, use_general_knowledge: bool = False) -> str:
        """
        Build prompt for LLM with context and question.
        
        Args:
            question: User question
            context: Retrieved context chunks (None if using general knowledge)
            max_context_length: Maximum context length in characters (truncate if longer)
            use_general_knowledge: If True, use general knowledge prompt instead of document context
        
        Returns:
            Formatted prompt
        """
        if use_general_knowledge:
            # General knowledge mode - no document context
            system_prompt = """Answer the question using your general knowledge. Clearly state at the beginning that you're using general knowledge since no relevant information was found in the provided documents. Be concise and accurate."""
            
            prompt = f"""{system_prompt}

Q: {question}
A:"""
        else:
            # Document mode - use provided context
            if context:
                # Truncate context if too long to prevent slow generation
                if len(context) > max_context_length:
                    # Try to truncate at sentence boundaries
                    truncated = context[:max_context_length]
                    last_period = truncated.rfind('.')
                    if last_period > max_context_length * 0.8:  # Only truncate at period if not too early
                        context = truncated[:last_period + 1]
                    else:
                        context = truncated + "..."
            else:
                context = ""
            
            # More concise system prompt for faster processing
            system_prompt = """Answer based on the provided context from documents. Be concise and accurate. Cite sources with [1], [2], etc."""

            prompt = f"""{system_prompt}

Context:
{context}

Q: {question}
A:"""
        
        return prompt
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.5,  # Lower default temperature
        max_tokens: Optional[int] = 800,  # Limit response length
        max_context_length: int = 3000,  # Limit context length
        filter_sources: Optional[List[str]] = None  # Filter by specific sources
    ) -> Dict[str, any]:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve (overrides default)
            temperature: LLM temperature (lower = faster, more deterministic)
            max_tokens: Maximum tokens to generate
            max_context_length: Maximum context length in characters
            filter_sources: Optional list of source document names to filter by
        
        Returns:
            Dictionary with 'answer', 'sources', and 'chunks'
        """
        # Use provided top_k or default
        k = top_k or self.top_k
        
        # Step 0: Check if user is asking about a specific document (if no filter_sources provided)
        filter_metadata = None
        if not filter_sources:
            filter_document = self._extract_document_name_from_query(question)
            if filter_document:
                filter_metadata = {"source": filter_document}
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Step 2: Retrieve relevant chunks (with source filter if specified)
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=k,
            filter_metadata=filter_metadata,
            filter_sources=filter_sources
        )
        
        # Check if chunks are relevant based on similarity threshold
        # If best match has distance > threshold, treat as no relevant chunks
        relevant_chunks = []
        if retrieved_chunks:
            # Get the distance of the best match (first result, lowest distance)
            best_distance = retrieved_chunks[0].get('distance')
            if best_distance is not None and best_distance <= self.similarity_threshold:
                # Only use chunks that meet the similarity threshold
                relevant_chunks = [chunk for chunk in retrieved_chunks 
                                 if chunk.get('distance') is None or chunk.get('distance') <= self.similarity_threshold]
        
        if not retrieved_chunks or not relevant_chunks:
            # No relevant chunks found - fall back to general knowledge
            # Step 3: Build prompt for general knowledge
            prompt = self._build_prompt(question, context=None, use_general_knowledge=True)
            
            # Step 4: Generate answer using general knowledge
            answer = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                stream=False,
                num_predict=max_tokens
            )
            
            return {
                'answer': answer.strip(),
                'sources': [],
                'chunks': [],
                'knowledge_source': 'general_knowledge'
            }
        
        # Step 3: Format context (use only relevant chunks)
        context = self._format_context(relevant_chunks)
        
        # Step 4: Build prompt (with context truncation)
        prompt = self._build_prompt(question, context, max_context_length=max_context_length, use_general_knowledge=False)
        
        # Step 5: Generate answer with token limit
        answer = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            stream=False,
            num_predict=max_tokens
        )
        
        # Extract sources (from relevant chunks only)
        sources = []
        for chunk in relevant_chunks:
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        return {
            'answer': answer.strip(),
            'sources': sources,
            'chunks': relevant_chunks,
            'knowledge_source': 'documents'
        }
    
    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = 800,
        max_context_length: int = 3000,
        filter_sources: Optional[List[str]] = None  # Filter by specific sources
    ):
        """
        Answer a question with streaming response.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve (overrides default)
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            max_context_length: Maximum context length in characters
            filter_sources: Optional list of source document names to filter by
        
        Yields:
            Text chunks as they are generated
        """
        # Use provided top_k or default
        k = top_k or self.top_k
        
        # Step 0: Check if user is asking about a specific document (if no filter_sources provided)
        filter_metadata = None
        if not filter_sources:
            filter_document = self._extract_document_name_from_query(question)
            if filter_document:
                filter_metadata = {"source": filter_document}
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Step 2: Retrieve relevant chunks (with source filter if specified)
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=k,
            filter_metadata=filter_metadata,
            filter_sources=filter_sources
        )
        
        # Check if chunks are relevant based on similarity threshold
        # If best match has distance > threshold, treat as no relevant chunks
        relevant_chunks = []
        if retrieved_chunks:
            # Get the distance of the best match (first result, lowest distance)
            best_distance = retrieved_chunks[0].get('distance')
            if best_distance is not None and best_distance <= self.similarity_threshold:
                # Only use chunks that meet the similarity threshold
                relevant_chunks = [chunk for chunk in retrieved_chunks 
                                 if chunk.get('distance') is None or chunk.get('distance') <= self.similarity_threshold]
        
        if not retrieved_chunks or not relevant_chunks:
            # No relevant chunks found - fall back to general knowledge
            # Step 3: Build prompt for general knowledge
            prompt = self._build_prompt(question, context=None, use_general_knowledge=True)
            
            # Step 4: Generate answer using general knowledge with streaming
            for chunk in self.llm_client.generate_stream(
                prompt=prompt,
                temperature=temperature,
                num_predict=max_tokens
            ):
                yield chunk
            return
        
        # Step 3: Format context (use only relevant chunks)
        context = self._format_context(relevant_chunks)
        
        # Step 4: Build prompt (with context truncation)
        prompt = self._build_prompt(question, context, max_context_length=max_context_length, use_general_knowledge=False)
        
        # Step 5: Generate answer with streaming and token limit
        for chunk in self.llm_client.generate_stream(
            prompt=prompt,
            temperature=temperature,
            num_predict=max_tokens
        ):
            yield chunk
