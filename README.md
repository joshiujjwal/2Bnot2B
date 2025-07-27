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
