FROM python:3.11-slim

WORKDIR /app

# Ensure consistent HOME and cache paths for build and runtime
ENV HOME=/root
ENV CHROMA_CACHE_DIR=/root/.cache/chroma

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache ChromaDB's ONNX embedding model during build
# (avoids 79MB download on every cold start)
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ef = ONNXMiniLM_L6_V2(); ef(['test'])" \
    && ls -la /root/.cache/chroma/onnx_models/ \
    && echo "ONNX model pre-cached successfully"

# Copy application code
COPY bridge.py .
COPY memory.py .
COPY vector_memory.py .
COPY google_services.py .
COPY knowledge_base.py .
COPY persona.txt .
COPY agent_memory.json .

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Run the bridge
CMD ["python", "bridge.py"]
