# Streamlit Deployment Guide

This guide outlines the steps to deploy the **NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform** to Streamlit Community Cloud.

---

## 🚀 Deployment Instructions

### 1. Repository Setup
1. Push the final recruiter-ready branch (`nf-lrd-v2-recruiter-ready` or `main`) to your GitHub repository:
   ```bash
   git push origin nf-lrd-v2-recruiter-ready
   ```

### 2. Streamlit Community Cloud Setup
1. Log in to the [Streamlit Community Cloud console](https://share.streamlit.io/).
2. Click **New app** in the upper-right corner.
3. Configure the deployment settings:
   - **Repository**: Choose your cloned GitHub repository (e.g., `username/NF-LRD`).
   - **Branch**: Select the active branch `nf-lrd-v2-recruiter-ready`.
   - **Main file path**: Set this to `app/streamlit_app.py` (do not use default `streamlit_app.py` directly).
   - **App URL**: Choose a custom subdomain (e.g., `nifty-regimes-terminal`).
4. Click **Deploy!**

---

## 🔑 Secrets & Credentials

* **API Keys & Databases**: The platform operates entirely on public Yahoo Finance API connections and pre-calculated local files. **No secrets or credentials are required.**
* **Secrets configuration**: During the Streamlit Cloud setup, you can leave the "Secrets" text area completely blank. If needed for future expansions, consult the [.streamlit/secrets.toml.example](file:///.streamlit/secrets.toml.example) file.

---

## 📦 How Demo Data Works (Fail-Safe Data Ingestion)

Streamlit Cloud instances can occasionally encounter network timeouts or rate-limiting bans from Yahoo Finance (`yfinance`). To prevent application startup crashes:
1. **Fallback Loader**: If a `yfinance` API fetch fails, the ingestion engine automatically falls back to reading the pre-stored CSV files in [demo_data/](file:///c:/Desktop/NF_LRD/demo_data/) containing baseline NIFTY 50 history.
2. **Pre-computed Model Parquets**: Discovered regimes are loaded from the parquet binaries located in [models/reports/](file:///c:/Desktop/NF_LRD/models/reports/) generated during the offline walk-forward pipeline. This guarantees that recruitment managers can view the charts and interact with the terminal immediately upon page load without waiting for online training.

---

## 🛠️ Troubleshooting Common Errors

### 1. `ModuleNotFoundError` or Missing Dependencies
* **Error**: The app crashes showing `ModuleNotFoundError: No module named 'scikit-learn'`.
* **Fix**: Ensure that the `requirements.txt` file is located at the root of the repository, as Streamlit Community Cloud automatically searches for and installs packages declared in the root `requirements.txt`.

### 2. Slow Initial Page Load or Untouchable Grey Overlay
* **Error**: The browser displays a greyed-out overlay for several seconds when navigation tabs are clicked.
* **Fix**: The Plotly chart rendering has been optimized using `st.cache_resource` at the top of the entrypoint file. If the app is slow on the first load, wait a few seconds for Streamlit to build the cache; subsequent navigation will load in under 1 millisecond.

### 3. Pyrefly or Pyright Import Errors
* **Error**: Linter or language servers cannot resolve modules starting with `src.`.
* **Fix**: The workspace uses `pyproject.toml` and `pyrightconfig.json` with `search-path = ["."]` to ensure that the python path correctly resolves root-relative imports. No changes to code files are needed.
