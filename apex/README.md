# ApexResearch AI – backend (complete)

## Chalane ka tareeka
Windows: `run.bat` double-click (venv + install + ingest + server, sab ek saath).
Mac/Linux: `./run.sh`
Phir http://localhost:8000 kholo.  Pehli baar model download hota hai (~470 MB), thoda wait karo.
Text, PDF, DOCX/ODT, HTML, aur OCR se nikle image/scanned PDF text ki languages analysis report mein detect hoti hain. Multi-format page par har extracted file ki detected language bhi dikhai jaati hai. Images/scanned PDFs ke OCR ke liye Windows Tesseract install karna hota hai; Hindi OCR model missing ho to first Hindi scan usse official Tesseract repository se download karta hai. Steps `OCR_SETUP.md` mein hain. Default `APEX_OCR_LANG=auto` installed aur app-managed OCR languages use karta hai. Check `http://localhost:8000/api/health` par `ocr.available` aur `ocr.selected_language`.
17. Profile dashboard – top-right profile icon se account details (name, email, institution, role, research area), submission metrics, recent activity, and sign-out access. Dashboard panels use a dark background.
19. Chat voice assistant – chat mic se Hindi (`HI`) ya Indian English (`EN`) mein question dictate karein, assistant replies ko `Listen` se sun sakte hain. It remembers recent turns for follow-up questions and keeps suggested next questions visible. Chrome/Edge microphone permission is required.
20. Password visibility – reviewer sign-in/create-account form has a show/hide eye button.

Check: http://localhost:8000/api/health
- `model` = sentence-transformers/... hona chahiye (hashing-fallback matlab sentence-transformers install nahi hua)
- `index` = faiss-hnsw (faiss install nahi hua to numpy chalega, app phir bhi kaam karega)

## Features
1. Repository search (FAISS HNSW approximate nearest-neighbour) – apne papers `papers/` mein daalo, `python ingest.py papers`
2. Live arXiv + OpenAlex search – asli papers ke links ke saath (test: /api/live?q=your topic)
3. Sentence-level evidence – highlighted sentences + unke closest match ka text
4. Hindi / multilingual – Hindi proposal English papers se match hota hai.
   Live search ke liye Hindi keywords translate hote hain -> `ANTHROPIC_API_KEY` chahiye (bina key ke sirf repository check hota hai)
5. Novelty advisor – key ho to Claude likhta hai, warna rule-based advice
6. PDF report download – report ke neeche "Download PDF report" button
   (PDF mein Hindi text ke liye `APEX_PDF_FONT` = koi Devanagari .ttf, warna wo hissa "[non-Latin text]" dikhega)
7. Explainable analysis – transparency panel shows search sources, model, index and scoring method; idea decomposition gives passage scores; citation-aware flags suggest missing citations on overlapping passages; related-work suggestions list the closest five papers.
8. Open data discovery – live results include Semantic Scholar, Europe PMC and Crossref alongside arXiv and OpenAlex; CORE can be enabled with `APEX_CORE_API_KEY`. The top three arXiv matches are checked for extractable PDF full text before sentence evidence is produced. In Upload, use “Index arXiv abstracts” to add up to 1,000 results for a topic to the local vector database per request.
9. Batch review – Upload accepts a ZIP with up to 25 supported proposal files for per-file scoring. Keep the ZIP small; each file is limited to 20 MB.
10. Reviewer accounts – Reports supports account creation and sign-in; reports and review decisions sync through the backend database. Passwords are salted and PBKDF2-hashed; repeated failed attempts are rate-limited; sign-out revokes the server session. The session token is held in browser session storage. Local Windows runs keep the database in `%LOCALAPPDATA%\ApexResearch\apex.db` so extracted project versions share the same account database. On the first update, keep the old `apex.db` beside the updated files so the app can safely migrate it once. Back up this database before replacing or reinstalling the app. Accounts remain a prototype feature; do not expose this server to the public internet without production authentication, HTTPS, and a persistent secured database.
18. Research & app assistant – can explain app screens and similarity evidence, help shape research questions, proposal outlines, methods, and next steps. For open-ended AI responses, add `ANTHROPIC_API_KEY` to `.env`; without a key it gives built-in guidance. It must not invent citations or treat a similarity score as proof of plagiarism.
11. Multilingual identification – analysis detects dominant and mixed languages in a proposal (including English, Hindi and other supported languages) and shows the detected language names in the report. `langdetect` is installed automatically from `requirements.txt`.
12. Side-by-side evidence and novelty map – reports show each flagged passage beside the matched source and extract proposal question, method, dataset, contribution and a gap to investigate. With `ANTHROPIC_API_KEY`, the map is AI-assisted; without it, fields are extractive and labelled as such.
13. Quote/reference filters – optional controls exclude quoted passages and a trailing References/Bibliography/Works Cited section from scoring. They do not edit the user's submitted text; the report states the counts excluded.
14. Multilingual demo – Upload includes a Hindi–English sample and per-passage language labels. Cross-language matching still depends on the configured multilingual embedding model and having related work in the repository or live search.
15. Evaluation dashboard – Upload accepts a UTF-8 CSV with `text_a,text_b,label` columns (`1` similar, `0` different), and computes accuracy, precision, recall, F1, confusion counts and runtime from those labelled pairs. It does not estimate scholarly source coverage.
16. Domain classification – TF-IDF + Random Forest classifier trained from `domain_training.csv` with 144 prototype examples across 12 research domains. See `DOMAIN_CLASSIFIER.md` for the training data notes.

## Settings (.env)
`.env.example` ko `.env` naam se copy karo. Sab optional hai.

## Notes
- Local Windows database: `%LOCALAPPDATA%\ApexResearch\apex.db`. Back it up before a fresh start; removing a project-folder `apex.db` no longer resets the shared account database.
- Similarity % ke thresholds (`Embedder.low/high` in main.py) andaze se set hain; apne known papers pe test karke tune karna.
- Demo ke liye texts: DEMO.md

## Latest fixes
- Hindi PDF reports: Devanagari is shaped correctly (conjuncts/matras) using a bundled font in `fonts/` (FreeSans). Needs Pillow with Raqm; check `python -c "from PIL import features; print(features.check('raqm'))"`. If it prints False, Hindi parts show "[non-Latin text]" instead. Override the font with `APEX_PDF_FONT`.
- Legacy Kruti Dev / DevLys Hindi PDFs are converted to Unicode directly (no OCR needed).
- PDF/OCR extraction runs in worker threads, so a slow OCR no longer freezes the server.
- The UI shows the backend's real error message; the "check backend/CORS" hint appears only when the backend is truly unreachable.
- If Pillow has no Raqm (common on Windows), a Hindi PDF report opens as a print-ready page in the browser instead; choose "Save as PDF" in the print window. English-only reports still download as normal PDFs.
- Chatbot: answers stream in as they are written; it now knows your current analysis (score, risk, top matches, flagged passages) and can explain it; chat history is sent as real multi-turn messages; the Claude client is reused; chat is rate-limited per IP (20/min). Without `ANTHROPIC_API_KEY` it still answers result questions with your actual numbers.
