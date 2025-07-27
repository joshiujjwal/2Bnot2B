# 2Bnot2B
Local Retrieval-Augmented Generation (RAG) on The Complete Works of William Shakespeare 

# Overview
- Ingest
- Chunk
- Embed
- Store
- Query
- Retrieve
- Generate

# Core Components
- Vector Database
- Embedding Model
- LLM

# Tech Stack
- Docker ( Easy deployment and distribution )
- Ollama ( Running LLM models locally )
- Python with Flask framework for RAG and Web Interface

# Prerequisites 
- [Docker](https://docs.docker.com/get-started/get-docker/)
- [docker-compose-plugin](https://docs.docker.com/compose/install/)

# Running application
1. Clone 
```bash
git clone https://github.com/joshiujjwal/2Bnot2B.git
cd 2Bnot2B
```
2. Build and run
```bash
docker compose up -d
```

# Key Design TradeOff
## Document Processing 
1. Chunk Size and Overlap for storing data
    - Current 1000 Larger Size more context, less precision
    - Overlap 200 

## Vector Database and Embeddings
1. Using Sentence Transformers to convert text to fixed size vectors
    - Using pretrained model all-MiniLM-L6-v2
    - Lightweight for Local apps not accurate as larger models or APIs
2. Vector storage for data Chroma

## Retrieval
1. For a given query, retrieve documents from vector data based on similarity search
2. With using metadata can show source for response