FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

COPY . .
RUN pip install --upgrade pip
# Copy requirements and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Expose ports
EXPOSE 8000 11434

ENTRYPOINT ["/entrypoint.sh"]
