# RAG Application - IntelLM

An AI-powered RAG (Retrieval-Augmented Generation) application that answers questions from your own documents (PDFs, DOCX, images) instead of relying only on the LLM's training knowledge.

## Features

- **Multi-format Support**: Process PDFs, DOCX files, and images (via OCR)
- **Vector Database**: Uses Chroma for semantic search and retrieval
- **Local LLM**: Uses Ollama for local LLM inference (no cloud costs for generation)
- **OpenAI Embeddings**: Uses OpenAI for high-quality embeddings (with rate limiting)
- **Cost Control**: Built-in rate limiting and cost tracking to prevent unexpected API costs
- **Docker Support**: Easy deployment with Docker Compose

## Architecture

The application follows a standard RAG pipeline:

1. **Document Processing**: Extract text from PDFs, DOCX files, and images
2. **Chunking**: Split documents into manageable chunks with overlap
3. **Embedding**: Generate embeddings using OpenAI API
4. **Vector Storage**: Store embeddings in Chroma vector database
5. **Retrieval**: Find relevant chunks using semantic search
6. **Generation**: Use local LLM (Ollama) to generate answers with context

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- Ollama installed (or use Docker Compose which includes it)

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
cd IntelLM
```

2. Create `.env` file:
```bash
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. Start the application:
```bash
docker-compose up --build
```

4. Access the application:
- Streamlit UI: http://localhost:8501
- Ollama API: http://localhost:11434

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR (for image processing):
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# Set OLLAMA_BASE_URL=http://localhost:11434
```

4. Start Ollama (if not using Docker):
```bash
# Install Ollama from https://ollama.ai
ollama serve
# Pull a model: ollama pull llama2
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or image files
2. **Process Documents**: Click "Process Documents" to extract text and generate embeddings
3. **Ask Questions**: Type your question in the chat interface
4. **View Answers**: The AI will answer based on your documents with source citations

## Configuration

### Rate Limiting

Configure rate limits in the Streamlit UI sidebar to control API costs:
- **Requests per minute (RPM)**: Default 60
- **Tokens per minute (TPM)**: Default 1,000,000

### Chunking

Adjust chunk size and overlap in settings:
- **Chunk Size**: Default 1000 tokens
- **Overlap**: Default 200 tokens

### LLM Model

Change the Ollama model in settings. Popular options:
- `llama2`
- `mistral`
- `codellama`

## Project Structure

```
IntelLM/
├── requirements.txt
├── README.md
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── chunker.py
│   ├── rate_limiter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   └── llm_client.py
├── app.py
├── data/
└── chroma_db/
```

## Learning Outcomes

This implementation demonstrates:

- **How LLMs work**: Tokenization, context windows, prompt engineering
- **Why hallucinations happen**: LLMs generate from training data without grounding
- **How vector DBs reduce wrong answers**: Semantic search retrieves relevant context, grounding the LLM in your documents

## Cost Management

The application includes built-in cost controls:

- Rate limiting prevents excessive API calls
- Cost estimation before processing large documents
- Usage tracking and display
- Configurable limits via UI

OpenAI embedding costs (approximate):
- `text-embedding-3-small`: $0.02 per 1M tokens
- `text-embedding-ada-002`: $0.10 per 1M tokens

## Troubleshooting

### Ollama Connection Issues

If using Docker, ensure the `ollama` service is running:
```bash
docker-compose ps
docker-compose logs ollama
```

### Tesseract OCR Not Found

Ensure Tesseract is installed and in your PATH. For Docker, it's included in the image.

### Chroma Database Issues

If you encounter database errors, you can reset the database:
- In the UI: Click "Clear Database" in the sidebar
- Or delete the `chroma_db/` directory

## License

MIT License
