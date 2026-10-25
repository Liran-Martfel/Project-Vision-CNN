# VISION · Facial-emotion recognition app

Take or upload a photo of a face, and a CNN trained on FER-2013 recognizes the emotion
(angry, disgust, fear, happy, neutral, sad, surprise). Hebrew and English, phone and desktop.

```
Streamlit (app.py)  ──HTTP──▶  FastAPI (api.py)  ──▶  vision_core.py  ──▶  models/training_model.keras
     UI only                    validation +           the inference
     requests                   call chain             functions
```

The frontend never loads the model: it only talks to the API over HTTP.

## Files

| File | What it does |
|---|---|
| `vision_core.py` | The inference functions from the Colab notebook, copied verbatim, plus model and face-detector loading |
| `api.py` | FastAPI backend: `GET /health`, `POST /predict` |
| `app.py` | Streamlit frontend |
| `i18n.py` | All UI text in Hebrew and English |
| `styles.py` | The app's CSS (navy & beige glass) and SVG icons |
| `models/training_model.keras` | The trained model |
| `.streamlit/config.toml` | Theme and upload limit |

## Install

TensorFlow needs **Python 3.12** (it has no stable build for 3.14 yet).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Two terminals, both inside `vision_app/` with the virtual environment active.

```bash
# 1. the API (loads the model once, takes ~10 s)
uvicorn api:app --reload --port 8000

# 2. the website
streamlit run app.py
```

Open http://localhost:8501.

The app looks for the API at `http://localhost:8000`. If port 8000 is taken, run the API on
another port and point the app at it:

```bash
uvicorn api:app --port 8765
set API_URL=http://127.0.0.1:8765        # PowerShell: $env:API_URL="http://127.0.0.1:8765"
streamlit run app.py
```

## API

`GET /health` → `{"status": "ok"}`

`POST /predict` with one image as `multipart/form-data`, field `file` (JPG, JPEG, PNG or WEBP, up to 10 MB):

```json
{
  "emotion": "happy",
  "confidence": 0.997,
  "is_unknown": false,
  "answer": "I'm 99.70% sure that this emotion is: happy",
  "face_detected": true
}
```

- `is_unknown` is true when `create_answer` reports under 60% confidence.
- `face_detected` is false when no face was found and the whole image was analyzed.
- Phone photos stored sideways are rotated upright (EXIF orientation) before analysis.

Errors return JSON `{"error": "<code>", "detail": "..."}`:

| Status | `error` | When |
|---|---|---|
| 400 | `missing_file`, `empty_file`, `corrupt_image` | No file, empty file, damaged image |
| 413 | `file_too_large` | Over 10 MB |
| 415 | `not_an_image`, `unsupported_format` | Not an image, or not JPG/PNG/WEBP |
| 500 | `prediction_failed` | Unexpected failure during analysis |

## Notes

- **Phone camera needs HTTPS.** Browsers only allow the camera on `localhost` or HTTPS. On a phone
  over your Wi-Fi address, uploading from the gallery works but the camera is blocked; serving the
  app through HTTPS (for example ngrok) enables it.
- **Bonuses still to connect:** look for `# TODO (bonus):` in `api.py` and `app.py` for where the
  per-emotion probabilities (`emotions_probability`) plug in.
