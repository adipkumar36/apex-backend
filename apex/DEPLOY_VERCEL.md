# Vercel deployment

This folder (`apex/`) is the Vercel project root. It contains the FastAPI app in
`main.py`, `requirements.txt`, and the bundled UI in `static/index.html`.

## Deploy

1. Put this folder in a GitHub repository (or upload the project to GitHub).
2. In Vercel, choose **Add New → Project** and import that repository.
3. Set **Root Directory** to `apex` if the repository contains a parent folder.
   If the repository itself contains these files at its top level, leave Root
   Directory as `.`.
4. Deploy. Vercel detects the FastAPI `app` in `main.py`; `vercel.json` sets the
   function timeout to 300 seconds.
5. In **Settings → Environment Variables**, optionally add `ANTHROPIC_API_KEY`
   for AI features. `APEX_MAILTO` and `APEX_CORE_API_KEY` are also optional.
   Redeploy after changing environment variables.
6. Open the deployed URL and check `/api/health`.

## Deployment limits to know

- Reviewer accounts, saved reports, and indexed papers use SQLite. On Vercel,
  the app stores this database under `/tmp/apex.db`; Vercel's function storage
  is temporary, so these records and indexes can disappear between instances
  or deployments. Do not use reviewer accounts for important or sensitive data
  until the app is changed to use a persistent hosted database.
- The multilingual Sentence-Transformers model is downloaded while the app
  starts. The model plus PyTorch and the other dependencies may exceed Vercel's
  500 MB uncompressed Python Function bundle limit, and model loading can make
  cold starts slow. If the deployment build reports a bundle-size error, or
  `/api/health` reports `hashing-fallback`, this app needs a hosted embedding
  service or a different backend host for full-quality semantic matching.
- OCR needs the Tesseract system program, which is not installed by
  `requirements.txt`; image and scanned-PDF OCR may not work on Vercel as-is.
- The app's accounts are documented in its README as prototype accounts. Keep
  this deployment for demos until authentication and persistent storage are
  production-ready.

## Local preview

From this folder, install `requirements.txt`, then run `vercel dev` with the
Vercel CLI. `python main.py` is not the start command; for a plain local run use
`uvicorn main:app --reload --port 8000`.
