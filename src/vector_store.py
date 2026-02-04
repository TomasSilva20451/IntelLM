"""
Vector Store using Chroma for persistent storage and similarity search.

Stores document embeddings with metadata for retrieval.
"""

import os
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Vector store for storing and retrieving document embeddings."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist Chroma database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (auto-generated if not provided)
        """
        if not texts or not embeddings:
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{hash(text[:50])}" for i, text in enumerate(texts)]
        
        # Ensure all lists have the same length
        min_length = min(len(texts), len(embeddings), len(metadatas), len(ids))
        texts = texts[:min_length]
        embeddings = embeddings[:min_length]
        metadatas = metadatas[:min_length]
        ids = ids[:min_length]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            List of dictionaries with 'text', 'metadata', 'distance', and 'id'
        """
        # Build where clause for filtering
        where = filter_metadata if filter_metadata else None
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def delete_documents_by_source(self, source: str) -> int:
        """
        Delete all documents with a specific source.
        
        Args:
            source: Source document name to delete
        
        Returns:
            Number of documents deleted
        """
        # Get all document IDs for this source
        all_docs = self.get_all_documents()
        ids_to_delete = []
        
        for doc in all_docs:
            metadata = doc.get('metadata', {})
            if metadata.get('source') == source:
                ids_to_delete.append(doc.get('id'))
        
        if ids_to_delete:
            # Delete documents by IDs
            self.collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        
        return 0
    
    def get_document_count_by_source(self, source: str) -> int:
        """
        Get the number of chunks for a specific source document.
        
        Args:
            source: Source document name
        
        Returns:
            Number of chunks for this source
        """
        all_docs = self.get_all_documents()
        count = 0
        for doc in all_docs:
            metadata = doc.get('metadata', {})
            if metadata.get('source') == source:
                count += 1
        return count
    
    def get_all_documents(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all documents from the collection.
        
        Args:
            limit: Optional limit on number of documents
        
        Returns:
            List of document dictionaries
        """
        count = self.collection.count()
        if limit:
            count = min(count, limit)
        
        if count == 0:
            return []
        
        results = self.collection.get(limit=count)
        
        documents = []
        if results['ids']:
            for i in range(len(results['ids'])):
                documents.append({
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                })
        
        return documents
    
    def get_unique_sources(self) -> List[str]:
        """Get list of unique source documents."""
        all_docs = self.get_all_documents()
        sources = set()
        
        for doc in all_docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source')
            if source:
                sources.add(source)
        
        return sorted(list(sources))
