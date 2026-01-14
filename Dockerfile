FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for matplotlib + build wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

# Copy app code
COPY src /app/src
COPY data /app/data
COPY README.md /app/README.md

# Make src importable
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "poos_backtest.main"]
