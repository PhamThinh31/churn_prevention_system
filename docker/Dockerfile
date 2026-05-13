FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt requirements-cpu.txt /app/
RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY src /app/src
COPY configs /app/configs

# Artifacts are mounted at /app/data at runtime (see docker-compose.yml).
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
