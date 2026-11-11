FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DFV_DATA_MODE=sample \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 dfvuser

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && grep -v '^-e \.' requirements.txt > /tmp/reqs.lock \
    && pip install --no-cache-dir -r /tmp/reqs.lock \
    && rm -f /tmp/reqs.lock

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/sample ./data/sample
COPY data/reference ./data/reference

RUN pip install --no-cache-dir --no-deps . \
    && mkdir -p /app/data/raw /app/data/interim /app/data/processed /app/data/cache /app/reports \
    && chown -R dfvuser:dfvuser /app

USER dfvuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "dfv.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
