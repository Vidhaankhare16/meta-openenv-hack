FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package source
COPY . .

# OpenEnv LocalDockerProvider maps -p {local}:8000, so default to 8000.
# HF Spaces routes to this port via app_port: 8000 in README frontmatter.
EXPOSE 8000

ENV WORKERS=1
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MAX_CONCURRENT_ENVS=100

CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
