# Image and scanned PDF OCR (Windows)

Image OCR uses two pieces: the Python wrapper (`pytesseract`, installed from `requirements.txt`) and the separate Tesseract OCR program. The app ZIP installs the Python wrapper, but it cannot install the Windows OCR program automatically.

## Install the OCR program

1. Download the Windows installer from the Tesseract project's Windows downloads page: <https://github.com/UB-Mannheim/tesseract/wiki>.
2. In the installer, select the language data you need, such as English (`eng`), Hindi (`hin`), Bengali (`ben`), Tamil (`tam`), Urdu (`urd`) or Arabic (`ara`).
3. Finish installation, close the current PowerShell/Command Prompt, and run `run.bat` again.
4. Open <http://localhost:8000/api/health>. Under `ocr`, check `available` is `true`. `languages` shows installed packs and `selected_language` shows what OCR will use. The default `auto` setting uses all installed OCR language models together (excluding orientation-only `osd`); installing `hin` alongside `eng` should show `eng+hin` selected.

The backend checks PATH and common Windows install locations. If Tesseract is installed elsewhere, put its full executable path in `.env` as:

```text
APEX_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

By default, `.env.example` sets `APEX_OCR_LANG=auto`. To choose specific OCR languages explicitly, set a `+`-joined list (every pack must be installed):

```text
APEX_OCR_LANG=eng+hin
```

Other examples are `eng+fra+deu` (English, French and German) or `eng+ben+tam` (English, Bengali and Tamil). Restart `run.bat` after changing `.env`. Tesseract supports using several installed language models together; processing may take longer as more packs are enabled.

The app preprocesses images (orientation, grayscale, contrast and upscaling), OCRs image files, and OCRs low-text pages in mixed or scanned PDFs. Clear, upright, high-resolution printed text works best; handwriting and blurry photographs can still produce poor results.

## Hindi PDFs that extract as Latin-looking gibberish

Some older Hindi PDFs use legacy fonts such as Kruti Dev. Their embedded text can extract as strings like `Hkkjrh;` instead of Devanagari. The app detects common cases and OCRs those pages. If the Tesseract `hin` model is missing, the first Hindi scan downloads the official fast Hindi model from the Tesseract `tessdata_fast` repository into `%LOCALAPPDATA%\ApexResearch\tessdata`; no PDF text is sent to download it. Internet access is needed once. If that download is unavailable, install the Hindi (`hin`) pack manually using the Windows installer above. The model is then used alongside English (`eng`) for Hindi OCR.

## Update: legacy Hindi PDFs now work without OCR

Pages that use Kruti Dev / DevLys fonts are detected (by font name and text pattern) and converted straight to Unicode Hindi, so the Tesseract `hin` pack is no longer required for them. Other legacy fonts (Chanakya, Shusha, etc.) and scanned pages still use OCR, which needs the `hin` pack. If the automatic model download is blocked, the app tries two mirrors; otherwise install `hin` manually as described above.
