#!/bin/bash

# Start Ollama in background
ollama serve &

# Wait for Ollama to be ready
sleep 10

# Pull the model for LLMs
ollama pull gemma3n:e4b

# Start the web application
python app/main.py