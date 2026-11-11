# Deployment Guide (`DubaiFairValue`)

`DubaiFairValue` is packaged to run cleanly on **Streamlit Community Cloud** (or Hugging Face Spaces) **without any private raw DLD files or API keys**, and includes a slim non-root Docker image for the FastAPI backend.

---

## 1. Streamlit Community Cloud (Dashboard)

The repository already ships the three deployment artifacts required by Streamlit Community Cloud:
- [`data/sample/`](../data/sample/) (`2.18 MB` total committed sample dataset, `ResidualBoostedLGBM` CI model bundle, and precomputed analysis JSONs).
- [`requirements.txt`](../requirements.txt) (exported runtime dependencies + `-e .` editable local install).
- [`packages.txt`](../packages.txt) (`libgomp1` for Linux OpenMP support required by `lightgbm`).
- [`.streamlit/config.toml`](../.streamlit/config.toml) (consistent theme + headless production flags).

### Exact Deployment Steps
1. Push your GitHub repository to `main` (never commit `data/raw/`, `data/processed/`, or `.env`).
2. Visit <https://share.streamlit.io/>, sign in with GitHub, and click **New app**.
3. Configure:
   - **Repository:** `<your-github-username>/dubai-fair-value`
   - **Branch:** `main`
   - **Main file path:** `src/dfv/app/Home.py`
   - **Python version (Advanced settings):** `3.12`
4. **Secrets (Optional):** Leave blank to run completely free with the deterministic rule-based NL search fallback. To enable LLM query parsing on Page 9 (*Ask the Data*), paste in **App settings -> Secrets**:
   ```toml
   LLM_API_KEY = "sk-ant-..."
   LLM_PROVIDER = "anthropic"
   LLM_MODEL = "claude-3-5-sonnet-latest"
   ```
5. Click **Deploy!** The app boots automatically in clearly labeled **Sample Data Mode**.

---

## 2. Containerized FastAPI Service (`Dockerfile`)

The [`Dockerfile`](../Dockerfile) builds a slim, non-root `python:3.12-slim` image (`dfvuser`, UID `10001`) with `libgomp1` and `data/sample/` baked in:

```bash
# Build API image
docker build -t dubaifairvalue-api .

# Run locally on port 8000 (Sample mode, zero secrets required)
docker run --rm -p 8000:8000 dubaifairvalue-api
```

Verify live endpoints:
- Health check: `curl http://127.0.0.1:8000/health`
- Interactive Swagger / OpenAPI docs: <http://127.0.0.1:8000/docs>

### Optional: Serve Full Processed DLD Model via Docker Volume Mount
Once you have generated `data/processed/` locally via `make data features train analyze`:
```bash
docker run --rm -p 8000:8000 \
  -e DFV_DATA_MODE=processed \
  -v "$(pwd)/data/processed:/app/data/processed:ro" \
  dubaifairvalue-api
```

### Optional: Pass LLM Key & Custom CORS Origins
```bash
docker run --rm -p 8000:8000 \
  -e LLM_API_KEY="sk-ant-..." \
  -e LLM_PROVIDER="anthropic" \
  -e DFV_CORS_ORIGINS="https://your-dashboard.streamlit.app" \
  dubaifairvalue-api
```
