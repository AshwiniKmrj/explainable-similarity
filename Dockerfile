FROM python:3.12-slim
WORKDIR /app
# CPU-only torch: the default PyPI wheel bundles CUDA libraries (gigabytes) that are never used here.
# Installed first so sentence-transformers finds torch already satisfied. 2.15.0 is the version the tests ran with.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py embedder.py explain.py ./

# Run as an unprivileged user. The model's code is downloaded into HF_HOME on first start; mount a volume there
# to keep it between runs.
RUN useradd --create-home appuser && mkdir /hf-cache && chown appuser /hf-cache
ENV HF_HOME=/hf-cache
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# --no-server-header: responses do not announce which server software is running.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
