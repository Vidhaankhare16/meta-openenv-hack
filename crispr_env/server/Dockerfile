FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package source
COPY . .

# Expose port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV WORKERS=4
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MAX_CONCURRENT_ENVS=100

CMD ["sh", "-c", "uvicorn crispr_env.server.app:app --host $HOST --port $PORT --workers $WORKERS"]
