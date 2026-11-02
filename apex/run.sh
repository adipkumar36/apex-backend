#!/bin/sh
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python ingest.py papers
uvicorn main:app --port 8000
