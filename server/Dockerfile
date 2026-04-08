FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package source
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

ENV WORKERS=1
ENV PORT=7860
ENV HOST=0.0.0.0
ENV MAX_CONCURRENT_ENVS=100

CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
