"""
Streamlit UI for RAG Application.

Main application interface for uploading documents, processing them,
and asking questions based on the documents.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple

# Import modules
from src.document_processor import DocumentProcessor
from src.chunker import TextChunker
from src.vector_store import VectorStore
from src.rate_limiter import RateLimiter
from src.embeddings import OpenAIEmbeddingGenerator
from src.local_embeddings import LocalEmbeddingGenerator
from src.llm_client import OllamaClient
from src.rag_chain import RAGChain

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Application - IntelLM",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure file uploader settings (increase max upload size if needed)
# Note: Streamlit default is 200MB, but we can configure it via .streamlit/config.toml

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rate_limiter' not in st.session_state:
    st.session_state.rate_limiter = None
if 'embedding_generator' not in st.session_state:
    st.session_state.embedding_generator = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'embedding_type' not in st.session_state:
    st.session_state.embedding_type = "local"  # Default to free local embeddings
if 'local_embedding_model' not in st.session_state:
    st.session_state.local_embedding_model = "all-MiniLM-L6-v2"
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []  # List of selected document names


def initialize_components():
    """Initialize all components with error handling."""
    try:
        embedding_type = st.session_state.get('embedding_type', 'local')
        
        # Initialize embedding generator based on selected type
        if st.session_state.embedding_generator is None or \
           st.session_state.get('last_embedding_type') != embedding_type:
            
            if embedding_type == "openai":
                # Check OpenAI API key
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
                    return False
                
                # Initialize rate limiter for OpenAI
                if st.session_state.rate_limiter is None:
                    rpm = st.session_state.get('rpm_limit', 60)
                    tpm = st.session_state.get('tpm_limit', 1_000_000)
                    st.session_state.rate_limiter = RateLimiter(
                        rpm=rpm,
                        tpm=tpm,
                        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                    )
                
                try:
                    st.session_state.embedding_generator = OpenAIEmbeddingGenerator(
                        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                        rate_limiter=st.session_state.rate_limiter
                    )
                except ValueError as e:
                    st.error(f"⚠️ {str(e)}")
                    return False
            else:
                # Use local embeddings (FREE)
                local_model = st.session_state.get('local_embedding_model', 'all-MiniLM-L6-v2')
                try:
                    st.session_state.embedding_generator = LocalEmbeddingGenerator(
                        model_name=local_model
                    )
                except Exception as e:
                    st.error(f"⚠️ Failed to load local embedding model: {str(e)}")
                    return False
            
            st.session_state.last_embedding_type = embedding_type
        
        # Initialize vector store
        if st.session_state.vector_store is None:
            try:
                st.session_state.vector_store = VectorStore(
                    persist_directory="./chroma_db"
                )
            except Exception as e:
                st.error(f"⚠️ Error initializing vector store: {str(e)}")
                return False
        
        # Initialize LLM client
        if st.session_state.llm_client is None:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = st.session_state.get('ollama_model', os.getenv("OLLAMA_MODEL", "llama2"))
            st.session_state.llm_client = OllamaClient(
                base_url=ollama_url,
                model=ollama_model
            )
            
            # Check Ollama connection (non-blocking, just log warning)
            try:
                if not st.session_state.llm_client.check_connection():
                    # Don't show error in sidebar, just log - connection will be retried when needed
                    pass
            except Exception:
                # Connection check failed, but don't block initialization
                pass
        
        # Initialize RAG chain
        if st.session_state.rag_chain is None:
            top_k = st.session_state.get('top_k', 5)
            # Similarity threshold: chunks with cosine distance > 0.7 are considered not relevant
            # Lower threshold = stricter (only very similar chunks), Higher threshold = more lenient
            similarity_threshold = st.session_state.get('similarity_threshold', 0.7)
            st.session_state.rag_chain = RAGChain(
                vector_store=st.session_state.vector_store,
                embedding_generator=st.session_state.embedding_generator,
                llm_client=st.session_state.llm_client,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        
        return True
    except Exception as e:
        st.error(f"⚠️ Error initializing components: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def process_documents(files: List[Tuple[str, bytes]]):
    """Process uploaded documents."""
    if not files:
        return
    
    try:
        with st.spinner("Processing documents..."):
            # Initialize components
            if not initialize_components():
                return
            
            # Process documents
            processor = DocumentProcessor()
            documents = processor.process_multiple_files(files)
            
            if not documents:
                st.warning("No text could be extracted from the uploaded files.")
                return
            
            # Chunk documents
            chunk_size = st.session_state.get('chunk_size', 1000)
            chunk_overlap = st.session_state.get('chunk_overlap', 200)
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = chunker.chunk_documents(documents)
            
            if not chunks:
                st.warning("No chunks created from documents.")
                return
            
            # Estimate cost
            texts = [chunk['text'] for chunk in chunks]
            estimated_cost = st.session_state.embedding_generator.estimate_cost(texts)
            
            # Show cost estimate
            if estimated_cost > 0:
                st.info(f"📊 Will process {len(chunks)} chunks. Estimated cost: ${estimated_cost:.4f}")
            else:
                st.info(f"📊 Will process {len(chunks)} chunks. Cost: FREE (using local embeddings)")
            
            # Generate embeddings
            with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
                embeddings = st.session_state.embedding_generator.generate_embeddings(
                    texts=texts,
                    show_progress=True
                )
            
            # Store in vector database
            with st.spinner("Storing embeddings in vector database..."):
                texts_list = [chunk['text'] for chunk in chunks]
                metadatas_list = [chunk['metadata'] for chunk in chunks]
                
                st.session_state.vector_store.add_documents(
                    texts=texts_list,
                    embeddings=embeddings,
                    metadatas=metadatas_list
                )
            
            st.success(f"✅ Successfully processed {len(chunks)} chunks from {len(files)} files!")
            st.session_state.processing_status = "completed"
            # Clear uploaded files after successful processing to allow new uploads
            if 'file_uploader' in st.session_state:
                del st.session_state.file_uploader
            
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.session_state.processing_status = "error"


def main():
    """Main application."""
    st.title("📚 RAG Application - IntelLM")
    st.markdown("**Answer questions from your own documents using AI**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize components
        if not initialize_components():
            st.error("Failed to initialize components. Check your configuration.")
            st.stop()
        
        # File upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or images (screenshots supported)",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            key="file_uploader",  # Add key to prevent state issues
            help="Supported formats: PDF, DOCX, PNG, JPG, GIF, BMP, TIFF, WEBP. Screenshots (PNG) are fully supported!"
        )
        
        # Show currently uploaded files
        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} file(s) ready to process")
            for f in uploaded_files:
                st.caption(f"  • {f.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            process_clicked = st.button("Process Documents", type="primary", use_container_width=True)
            if process_clicked:
                if uploaded_files:
                    files = [(f.name, f.read()) for f in uploaded_files]
                    process_documents(files)
                else:
                    st.warning("Please upload files first.")
        
        with col2:
            if st.button("Clear Files", use_container_width=True):
                # Clear the file uploader by removing its key from session state
                if 'file_uploader' in st.session_state:
                    del st.session_state.file_uploader
                st.rerun()
        
        st.divider()
        
        # Document selection for filtering
        if st.session_state.vector_store and st.session_state.vector_store.get_collection_size() > 0:
            try:
                available_sources = st.session_state.vector_store.get_unique_sources()
                if available_sources:
                    st.subheader("📋 Select Documents for Context")
                    st.caption("Choose which documents to use when answering questions. Leave unchecked to use all documents.")
                    
                    # Initialize selected_documents if not set or if sources changed
                    if 'selected_documents' not in st.session_state:
                        st.session_state.selected_documents = []
                    
                    # Remove any selected documents that no longer exist
                    st.session_state.selected_documents = [
                        doc for doc in st.session_state.selected_documents 
                        if doc in available_sources
                    ]
                    
                    # Create checkboxes for each document
                    selected_docs = []
                    
                    for source in available_sources:
                        # Check if this document is currently selected
                        is_selected = source in st.session_state.selected_documents
                        
                        # Get chunk count for this source
                        all_docs = st.session_state.vector_store.get_all_documents()
                        chunk_count = len([
                            doc for doc in all_docs
                            if doc.get('metadata', {}).get('source') == source
                        ])
                        
                        # Create checkbox with document name and chunk count
                        checkbox_key = f"doc_checkbox_{source}"
                        checked = st.checkbox(
                            f"📄 {source}",
                            value=is_selected,
                            key=checkbox_key,
                            help=f"{chunk_count} chunk(s)"
                        )
                        
                        if checked:
                            selected_docs.append(source)
                    
                    # Update session state only if selection actually changed
                    # This prevents unnecessary reruns that might affect chat history
                    previous_selection = st.session_state.get('selected_documents', [])
                    if set(selected_docs) != set(previous_selection):
                        st.session_state.selected_documents = selected_docs
                        # Note: We don't clear chat_history here - conversation continues!
                    
                    # Show filter status
                    if st.session_state.selected_documents:
                        st.info(f"🔍 **Filtering active:** {len(st.session_state.selected_documents)} document(s) selected")
                    else:
                        st.info("ℹ️ **No filter:** Using all documents")
                    
                    st.divider()
            except Exception as e:
                st.warning(f"Could not load document list: {str(e)}")
        
        # Rate limit settings
        st.subheader("💰 Rate Limiting")
        rpm_limit = st.slider(
            "Requests per minute (RPM)",
            min_value=1,
            max_value=300,
            value=st.session_state.get('rpm_limit', 60),
            help="Limit API requests per minute"
        )
        tpm_limit = st.slider(
            "Tokens per minute (TPM)",
            min_value=10000,
            max_value=10000000,
            value=st.session_state.get('tpm_limit', 1_000_000),
            step=10000,
            help="Limit tokens per minute"
        )
        
        if rpm_limit != st.session_state.get('rpm_limit', 60) or \
           tpm_limit != st.session_state.get('tpm_limit', 1_000_000):
            st.session_state.rpm_limit = rpm_limit
            st.session_state.tpm_limit = tpm_limit
            if st.session_state.rate_limiter:
                st.session_state.rate_limiter.update_limits(rpm=rpm_limit, tpm=tpm_limit)
        
        # Show rate limit stats (only for OpenAI)
        if st.session_state.get('embedding_type') == 'openai' and st.session_state.rate_limiter:
            stats = st.session_state.rate_limiter.get_stats()
            st.metric("Current RPM", f"{stats['current_rpm']}/{stats['rpm_limit']}")
            st.metric("Current TPM", f"{stats['current_tpm']:,}/{stats['tpm_limit']:,}")
            st.metric("Total Cost", f"${stats['estimated_cost']:.4f}")
        elif st.session_state.get('embedding_type') == 'local':
            st.info("💰 Using FREE local embeddings - No API costs!")
        
        st.divider()
        
        # Chunking settings
        st.subheader("📝 Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=2000,
            value=st.session_state.get('chunk_size', 1000),
            step=100
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=500,
            value=st.session_state.get('chunk_overlap', 200),
            step=50
        )
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        
        st.divider()
        
        # Embedding settings
        st.subheader("🔤 Embedding Settings")
        embedding_type = st.radio(
            "Embedding Provider",
            options=["local", "openai"],
            index=0 if st.session_state.get('embedding_type', 'local') == 'local' else 1,
            format_func=lambda x: "Local (FREE)" if x == "local" else "OpenAI (Paid)",
            help="Choose between free local embeddings or OpenAI API"
        )
        st.session_state.embedding_type = embedding_type
        
        if embedding_type == "local":
            local_models = LocalEmbeddingGenerator.get_available_models()
            local_model = st.selectbox(
                "Local Model",
                options=local_models,
                index=local_models.index(st.session_state.get('local_embedding_model', 'all-MiniLM-L6-v2')) if st.session_state.get('local_embedding_model', 'all-MiniLM-L6-v2') in local_models else 0,
                help="Free local embedding model (runs on your machine)"
            )
            st.session_state.local_embedding_model = local_model
            
            # Show model info
            model_info = LocalEmbeddingGenerator.get_model_info(local_model)
            if model_info:
                st.caption(f"📊 {model_info.get('description', '')}")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ OpenAI API key not set. Using local embeddings instead.")
                st.session_state.embedding_type = "local"
        
        # Reset embedding generator if type changed
        if embedding_type != st.session_state.get('last_embedding_type'):
            st.session_state.embedding_generator = None
        
        st.divider()
        
        # RAG settings
        st.subheader("🔍 RAG Settings")
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=20,
            value=st.session_state.get('top_k', 5),
            help="Number of chunks to retrieve (fewer = faster)"
        )
        st.session_state.top_k = top_k
        
        # Performance settings
        st.subheader("⚡ Performance Settings")
        max_tokens = st.slider(
            "Max Response Tokens",
            min_value=100,
            max_value=2000,
            value=st.session_state.get('max_tokens', 800),
            step=100,
            help="Maximum tokens in response (lower = faster)"
        )
        st.session_state.max_tokens = max_tokens
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('temperature', 0.5),
            step=0.1,
            help="Lower = faster, more deterministic. Higher = more creative"
        )
        st.session_state.temperature = temperature
        
        use_streaming = st.checkbox(
            "Use Streaming",
            value=st.session_state.get('use_streaming', True),
            help="Stream responses for better UX (recommended)"
        )
        st.session_state.use_streaming = use_streaming
        
        # LLM model selection
        # Fetch available models from Ollama
        default_model = st.session_state.get('ollama_model', os.getenv("OLLAMA_MODEL", "llama2"))
        available_models = [default_model]  # Start with default
        
        # Try to fetch models from Ollama
        # First ensure components are initialized
        if not st.session_state.llm_client:
            initialize_components()
        
        if st.session_state.llm_client:
            try:
                fetched_models = st.session_state.llm_client.list_models()
                if fetched_models:
                    available_models = fetched_models
                    # Ensure default model is in the list
                    if default_model not in available_models:
                        available_models.insert(0, default_model)
            except Exception:
                # If fetching fails, keep default model
                pass
        
        # Find the index of the current model
        try:
            model_index = available_models.index(default_model)
        except ValueError:
            model_index = 0
        
        ollama_model = st.selectbox(
            "Ollama Model",
            options=available_models,
            index=model_index,
            help="Select a model installed in Ollama"
        )
        
        if ollama_model != st.session_state.get('ollama_model'):
            st.session_state.ollama_model = ollama_model
            if st.session_state.llm_client:
                st.session_state.llm_client.set_model(ollama_model)
            # Reinitialize RAG chain
            st.session_state.rag_chain = None
        
        st.divider()
        
        # Vector store status
        st.subheader("💾 Vector Store")
        if st.session_state.vector_store:
            collection_size = st.session_state.vector_store.get_collection_size()
            st.metric("Documents", collection_size)
            
            if collection_size > 0:
                sources = st.session_state.vector_store.get_unique_sources()
                st.write(f"**Sources:** {len(sources)}")
                if st.button("Clear Database", type="secondary"):
                    st.session_state.vector_store.delete_collection()
                    st.session_state.chat_history = []
                    st.rerun()
        
        st.divider()
        
        # Educational info
        with st.expander("ℹ️ How RAG Works"):
            st.markdown("""
            **Retrieval-Augmented Generation (RAG)** combines:
            
            1. **Document Processing**: Extract text from your files
            2. **Chunking**: Split into manageable pieces
            3. **Embedding**: Convert to vectors using AI
            4. **Storage**: Store in vector database
            5. **Retrieval**: Find relevant chunks for your question
            6. **Generation**: Use LLM to answer with context
            
            **Why this reduces hallucinations:**
            - LLMs are grounded in YOUR documents
            - Answers cite specific sources
            - No reliance on training data alone
            """)
    
    # Main area
    if st.session_state.vector_store and st.session_state.vector_store.get_collection_size() == 0:
        st.info("👆 Upload and process documents in the sidebar to get started!")
    else:
        # Show document context
        if st.session_state.vector_store:
            collection_size = st.session_state.vector_store.get_collection_size()
            if collection_size > 0:
                with st.expander("📄 Document Context", expanded=True):
                    try:
                        sources = st.session_state.vector_store.get_unique_sources()
                        total_chunks = collection_size
                        
                        st.markdown(f"""
**📊 Document Summary:**
- **Total Documents:** {len(sources)}
- **Total Chunks:** {total_chunks}
- **Sources:** {', '.join(sources[:5])}{'...' if len(sources) > 5 else ''}
                        """)
                        
                        # Show document list with delete buttons
                        if sources:
                            st.markdown("**📚 Processed Documents:**")
                            # Get all documents once (more efficient)
                            all_docs = st.session_state.vector_store.get_all_documents()
                            
                            for i, source in enumerate(sources, 1):
                                # Get chunks for this source
                                source_chunks = [
                                    doc for doc in all_docs
                                    if doc.get('metadata', {}).get('source') == source
                                ]
                                chunk_count = len(source_chunks)
                                
                                # Create columns for document info and delete button
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"{i}. **{source}** ({chunk_count} chunk{'s' if chunk_count != 1 else ''})")
                                    
                                    # Show preview of first chunk
                                    if source_chunks:
                                        preview_text = source_chunks[0].get('text', '')
                                        if preview_text:
                                            preview = preview_text[:200]
                                            if len(preview_text) > 200:
                                                preview += "..."
                                            st.caption(f"   Preview: {preview}")
                                
                                with col2:
                                    # Delete button for this document
                                    delete_key = f"delete_{source}_{i}"
                                    if st.button("🗑️", key=delete_key, help=f"Delete {source}"):
                                        try:
                                            deleted_count = st.session_state.vector_store.delete_documents_by_source(source)
                                            if deleted_count > 0:
                                                st.success(f"✅ Deleted {deleted_count} chunk(s) from {source}")
                                                # Reset RAG chain to reflect changes
                                                st.session_state.rag_chain = None
                                                st.session_state.chat_history = []
                                                st.rerun()
                                            else:
                                                st.warning(f"No chunks found for {source}")
                                        except Exception as e:
                                            st.error(f"Error deleting document: {str(e)}")
                                
                                st.divider()
                    except Exception as e:
                        import traceback
                        st.error(f"Could not load document context: {str(e)}")
                        st.code(traceback.format_exc())
    
    # Chat interface
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for i, entry in enumerate(st.session_state.chat_history):
        # Handle different formats: 3 items (old), 4 items (previous), 5 items (new with knowledge_source)
        if len(entry) == 3:
            question, answer, sources = entry
            used_documents = None
            knowledge_source = 'documents'  # Default assumption for old entries
        elif len(entry) == 4:
            question, answer, sources, used_documents = entry
            knowledge_source = 'documents'  # Default assumption for entries without knowledge_source
        else:
            question, answer, sources, used_documents, knowledge_source = entry
        
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            st.write(answer)
            
            # Show knowledge source indicator
            if knowledge_source == 'general_knowledge':
                st.caption("🌍 **General knowledge** (no relevant documents found)")
            else:
                st.caption("📄 **Answer based on documents**")
            
            # Show which documents were used for filtering (if documents mode)
            if knowledge_source == 'documents' and used_documents:
                st.caption(f"📋 Used documents: {', '.join(used_documents[:3])}{'...' if len(used_documents) > 3 else ''}")
            
            # Show sources (only if documents were used)
            if knowledge_source == 'documents' and sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(f"- {source}")
    
    # Question input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        if not st.session_state.rag_chain:
            st.error("Please process documents first!")
        else:
            # Add question to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Generate answer
            with st.chat_message("assistant"):
                try:
                    use_streaming = st.session_state.get('use_streaming', True)
                    max_tokens = st.session_state.get('max_tokens', 800)
                    temperature = st.session_state.get('temperature', 0.5)
                    top_k = st.session_state.get('top_k', 5)
                    
                    # Get selected documents for filtering
                    # IMPORTANT: Only use filter if documents are actually selected
                    # Empty list means "use all documents" - don't filter
                    selected_docs = st.session_state.get('selected_documents', [])
                    filter_sources = selected_docs if selected_docs else None  # None = use all documents
                    
                    if filter_sources:
                        st.info(f"🔍 **Using selected documents:** {', '.join(filter_sources[:3])}{'...' if len(filter_sources) > 3 else ''}")
                    else:
                        st.info("ℹ️ **No filter:** Searching across all documents")
                    
                    # Check if user is asking about a specific document (only if no manual selection)
                    filter_document = None
                    if not filter_sources:
                        filter_document = st.session_state.rag_chain._extract_document_name_from_query(question)
                        if filter_document:
                            st.info(f"🔍 Auto-detected document: **{filter_document}**")
                    
                    # Check if chunks will be retrieved (to determine knowledge_source for streaming)
                    query_embedding = st.session_state.rag_chain.embedding_generator.generate_embedding(question)
                    filter_metadata = None
                    if not filter_sources:
                        filter_document = st.session_state.rag_chain._extract_document_name_from_query(question)
                        if filter_document:
                            filter_metadata = {"source": filter_document}
                    
                    preview_chunks = st.session_state.rag_chain.vector_store.search(
                        query_embedding=query_embedding,
                        n_results=top_k,
                        filter_metadata=filter_metadata,
                        filter_sources=filter_sources
                    )
                    # Check similarity threshold to determine if chunks are relevant
                    similarity_threshold = st.session_state.rag_chain.similarity_threshold
                    relevant_preview_chunks = []
                    if preview_chunks:
                        best_distance = preview_chunks[0].get('distance')
                        if best_distance is not None and best_distance <= similarity_threshold:
                            relevant_preview_chunks = [chunk for chunk in preview_chunks 
                                                      if chunk.get('distance') is None or chunk.get('distance') <= similarity_threshold]
                    knowledge_source = 'general_knowledge' if not preview_chunks or not relevant_preview_chunks else 'documents'
                    
                    if use_streaming:
                        # Streaming response
                        answer_placeholder = st.empty()
                        full_answer = ""
                        
                        # Show knowledge source indicator
                        if knowledge_source == 'documents':
                            st.info("📄 **Answer based on documents**")
                        else:
                            st.info("🌍 **Answer using general knowledge** (no relevant documents found)")
                        
                        with st.spinner("Generating response..."):
                            try:
                                for chunk in st.session_state.rag_chain.query_stream(
                                    question=question,
                                    top_k=top_k,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    filter_sources=filter_sources  # Already None if no selection
                                ):
                                    full_answer += chunk
                                    answer_placeholder.write(full_answer + "▌")
                                
                                answer_placeholder.write(full_answer)
                                answer = full_answer
                                
                                # Use preview chunks for sources display
                                sources = list(set([chunk.get('metadata', {}).get('source', 'Unknown') for chunk in preview_chunks]))
                                chunks = preview_chunks
                                
                            except Exception as e:
                                st.error(f"Error during streaming: {str(e)}")
                                # Fallback to non-streaming
                                result = st.session_state.rag_chain.query(
                                    question=question,
                                    top_k=top_k,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    filter_sources=filter_sources  # Already None if no selection
                                )
                                answer = result['answer']
                                sources = result['sources']
                                chunks = result['chunks']
                                knowledge_source = result.get('knowledge_source', 'documents')
                                st.write(answer)
                    else:
                        # Non-streaming response
                        with st.spinner("Thinking..."):
                            result = st.session_state.rag_chain.query(
                                question=question,
                                top_k=top_k,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                filter_sources=filter_sources  # Already None if no selection
                            )
                            answer = result['answer']
                            sources = result['sources']
                            chunks = result['chunks']
                            knowledge_source = result.get('knowledge_source', 'documents')
                            
                            # Show knowledge source indicator
                            if knowledge_source == 'documents':
                                st.info("📄 **Answer based on documents**")
                            else:
                                st.info("🌍 **Answer using general knowledge** (no relevant documents found)")
                            
                            st.write(answer)
                    
                    # Show sources (only if documents were used)
                    if knowledge_source == 'documents' and sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.write(f"- {source}")
                    
                    # Show retrieved chunks (only if documents were used)
                    if knowledge_source == 'documents' and chunks:
                        with st.expander("🔍 Retrieved Chunks"):
                            for i, chunk in enumerate(chunks, 1):
                                metadata = chunk.get('metadata', {})
                                source = metadata.get('source', 'Unknown')
                                st.markdown(f"**Chunk {i}** - {source}")
                                st.text(chunk.get('text', '')[:200] + "...")
                    
                    # Add to chat history with document selection info and knowledge source
                    used_documents = filter_sources if filter_sources else ["All documents"]
                    st.session_state.chat_history.append((question, answer, sources, used_documents, knowledge_source))
                        
                except Exception as e:
                    error_msg = str(e)
                    if "Ollama" in error_msg or "connection" in error_msg.lower():
                        st.error(f"⚠️ Cannot connect to Ollama. Please ensure Ollama is running at {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
                        st.info("💡 If using Docker, make sure the 'ollama' service is running: `docker-compose ps`")
                    else:
                        st.error(f"⚠️ Error generating answer: {error_msg}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
