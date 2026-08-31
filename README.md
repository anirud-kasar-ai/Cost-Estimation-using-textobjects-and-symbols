# Cost Estimation — Drawing Split & Symbol Count

This repo contains **two standalone tools** plus sample bid drawing PDFs:

| Path | Role | Setup |
|------|------|--------|
| [`dual-pathway-drawing-split/`](dual-pathway-drawing-split/) | PDF → wing crops + 588px zoom tiles (no symbol count) | [SETUP](dual-pathway-drawing-split/SETUP.md) |
| [`symbol-count-vision/`](symbol-count-vision/) | Count legend symbols on tiles / zoom folders | [SETUP](symbol-count-vision/SETUP.md) |
| [`Real data/`](Real%20data/) | Sample bid drawing PDFs (Git LFS) | — |

## Quick start

See **[SETUP.md](SETUP.md)** for full steps.

```bash
# 1) Export zooms from a bid PDF
cd dual-pathway-drawing-split
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # set GROQ_API_KEY
python scripts/run_batch.py --input "path\to\pdfs"

# 2) Count symbols on a zoom folder or single tile
cd ../symbol-count-vision
python -m venv .venv
pip install -r requirements.txt
copy .env.example .env   # set GEMINI_API_KEY
uvicorn app:app --host 127.0.0.1 --port 8767
```

- Drawing split UI: http://127.0.0.1:8766  
- Symbol count UI: http://127.0.0.1:8767  

**Never commit `.env`.** Use each folder’s `.env.example`. API keys and job storage are gitignored.

## Layout

```
├── README.md
├── SETUP.md
├── .gitignore
├── Real data/              # Sample bid PDFs (Git LFS)
├── dual-pathway-drawing-split/  # PDF → wings + zoom tiles (Stage 1; no counting)
└── symbol-count-vision/    # Legend-gated symbol counting (CV localize, taxonomy classify)
```

Sample drawings are in [`Real data/`](Real%20data/) (Git LFS). After clone:

```bash
git lfs install
git lfs pull
```
