@echo off
where tesseract >nul 2>&1
if not errorlevel 1 goto OCR_OK
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" goto OCR_OK
if exist "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" goto OCR_OK
echo.
echo WARNING: Tesseract OCR engine was not detected.
echo Images and scanned PDF pages will not extract text until Tesseract is installed.
echo Install instructions: see OCR_SETUP.md
:OCR_OK
if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python ingest.py papers
start http://localhost:8000
uvicorn main:app --port 8000
