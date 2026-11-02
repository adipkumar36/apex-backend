"""Load your reference papers into the repository.
Usage:  python ingest.py ./papers      (txt, md, pdf, docx, html ...)
Safe to re-run: duplicates are skipped. Running server picks new docs up automatically."""
import sys
from pathlib import Path
from main import extract_text, store

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "papers")
added = 0
for p in sorted(folder.rglob("*")):
    if not p.is_file():
        continue
    try:
        text = extract_text(p.name, p.read_bytes()).strip()
    except Exception as e:
        print("skip", p.name, "-", getattr(e, "detail", e)); continue
    if len(text.split()) < 20:
        print("skip", p.name, "- too short"); continue
    ok = store.add_doc(p.stem.replace("_", " "), text, "Repository", "source")
    print(("added " if ok else "dup   ") + p.name); added += bool(ok)
print(f"done, {added} new document(s)")
