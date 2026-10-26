# Project Vision · CNN

Facial-emotion recognition with a convolutional neural network, end to end: the model is trained in
Google Colab, and a web app (FastAPI + Streamlit) lets anyone take or upload a photo and see the emotion.

מערכת לזיהוי הבעות פנים באמצעות רשת נוירונים קונבולוציונית (CNN): אימון המודל ב-Google Colab, ואפליקציית רשת שמסווגת תמונות בזמן אמת.

| | |
|---|---|
| **Topic** | Facial expressions (7 emotions: angry, disgust, fear, happy, neutral, sad, surprise) |
| **Dataset** | FER+ (FER-2013 faces with Microsoft's improved labels), `contempt` removed → 7 classes |
| **Model** | CNN: 3 × (Conv2D + MaxPooling) → Dense 128 → Dropout 0.5 → Dense 7 softmax, 355,847 parameters |
| **Final accuracy** | **74.5%** on the FER+ test set (7,042 images) |
| **App** | FastAPI REST backend + Streamlit frontend, Hebrew/English, phone and desktop |

## Results

All three models share the same architecture; only the training data and settings changed.

| Model | Trained on | Test accuracy | |
|---|---|---|---|
| v1 `training_model.keras` | FER-2013 | ~59% | first version |
| **v2 `training_model_v2.keras`** | **FER+ (7 classes)** | **74.5%** | **used by the app** |
| v3 `training_model_v3.keras` | FER+ with `class_weight='balanced'` | 61% | experiment |

**Why v2:** FER+'s cleaner labels raised accuracy by about 15 points over v1. v3 tried to help the rare
emotions with class weights, but lost 13.5 points overall, so v2 was kept. v2's weak spots are the rare
classes: disgust (57 test images) is never predicted and fear is recognized 22% of the time
(per-class report in `PROJECT_CNN.ipynb`). Below 60% confidence the app answers "Unknown".

## Repository

```
.
├── PROJECT_CNN.ipynb        Colab notebook: data, training, evaluation (confusion matrix, report)
├── JOURNAL.md               learning log and experiments
├── README.md
└── vision_app/              the web app
    ├── api.py               FastAPI: GET /health, POST /predict
    ├── vision_core.py       inference functions from the notebook + model loading
    ├── app.py               Streamlit UI (talks to the API over HTTP only)
    ├── config.py            MODEL_FILENAME, MODEL_DATASET
    ├── i18n.py, styles.py   Hebrew/English text, design
    ├── models/training_model_v2.keras
    ├── requirements.txt
    └── README.md            full API reference and details
```

## Run it

Requires **Python 3.12** (TensorFlow has no stable build for 3.14 yet).

```bash
cd vision_app
py -3.12 -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

uvicorn api:app --port 8000            # terminal 1: the API (prints the loaded model and its SHA-256)
streamlit run app.py                   # terminal 2: the website → http://localhost:8501
```

If port 8000 is taken, run the API on another port and set `API_URL` before starting Streamlit
(see `vision_app/README.md`).

## Open it on a phone with ngrok

ngrok gives the app a public **HTTPS** address, which the phone camera needs.

1. Create a free account at ngrok.com and install ngrok (Windows: `winget install ngrok.ngrok`).
2. Connect your account once (the token stays on your computer, never in this repo):
   `ngrok config add-authtoken <YOUR_TOKEN>`
3. With both servers running: `ngrok http 8501`
4. Open the `https://….ngrok-free.dev` address it prints on your phone, and tap "Visit Site" on ngrok's notice page.

Only the Streamlit port is exposed; the API stays private on your computer.

## Credits

- **FER-2013:** Goodfellow et al., "Challenges in Representation Learning" (ICML 2013),
  available on Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- **FER+ labels:** Microsoft, Barsoum et al., "Training Deep Networks for Facial Expression Recognition
  with Crowd-Sourced Label Distribution" (2016): https://github.com/microsoft/FERPlus
- Face detection: OpenCV Haar cascade (`haarcascade_frontalface_default.xml`).

---

## רשימת משימות

**Progress: 0%** · 0 of 23 done · נותרו כ-10.3h

> העדיפויות והערכות הזמן הן הערכה, לא חלק מה-PDF. הבונוסים לא נכללים באחוז.

### שלב 1 · בחירת נושא ומערך נתונים — 0% · 0/2 · (45m)
- [ ] לבחור נושא זיהוי אחד: הבעות פנים / עטיית מסכה / פירות וירקות `[גבוה]` (15m)
- [ ] להוריד את מערך הנתונים המתאים מ-Kaggle `[גבוה]` (30m) [ממתין ל: בחירת נושא]
  - [הבעות פנים](https://www.kaggle.com/datasets/msambare/fer2013) · [מסכה](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset) · [פירות](https://www.kaggle.com/datasets/moltean/fruits)

### שלב 2 · בניית המודל ב-Colab — 0% · 0/8 · (5.3h)
- [ ] לפתוח סביבת עבודה ב-Google Colab ולחבר את Google Drive `[גבוה]` (15m)
- [ ] להעלות את מערך התמונות ולחלץ את הקבצים `[גבוה]` (30m) [ממתין ל: הורדת הנתונים, סביבת Colab]
- [ ] להגדיר את מספר הקטגוריות לפי הנושא (בפירות: 4–5 פירות) `[גבוה]` (15m) [ממתין ל: בחירת נושא]
- [ ] לטעון את התמונות ולהכין אותן לאימון `[גבוה]` (1h) [ממתין ל: העלאת התמונות, הגדרת הקטגוריות]
- [ ] לבנות מודל CNN `[גבוה]` (1h) [ממתין ל: הכנת התמונות]
- [ ] לאמן את המודל ולהעריך את הביצועים `[גבוה]` (2h) [ממתין ל: בניית המודל]
- [ ] לשמור את המודל לקובץ `[גבוה]` (10m) [ממתין ל: אימון המודל]
- [ ] להוריד את קובץ המודל למחשב `[גבוה]` (5m) [ממתין ל: שמירת המודל]

### שלב 3 · אפליקציית Streamlit — 0% · 0/10 · (3.5h)
- [ ] לטעון את קובץ המודל מהדיסק בעליית האפליקציה `[גבוה]` (30m) [ממתין ל: הורדת המודל]
- [ ] העלאת קובץ תמונה מהמחשב `[בינוני]` (15m)
- [ ] העלאת תמונה בצילום ישיר מהמצלמה `[בינוני]` (30m)
- [ ] העלאת תמונה מהגלריה `[בינוני]` (15m)
- [ ] כפתור "שמור תמונה" שמוריד את התמונה למחשב `[בינוני]` (30m) [ממתין ל: העלאה מהמחשב]
- [ ] עיבוד מקדים: שינוי גודל לגודל הקלט של המודל `[גבוה]` (15m) [ממתין ל: הכנת התמונות]
- [ ] עיבוד מקדים: נרמול ערכי הפיקסלים `[גבוה]` (10m) [ממתין ל: הכנת התמונות]
- [ ] עיבוד מקדים: התאמת צורת הנתונים לפורמט שהמודל מצפה לו `[גבוה]` (15m) [ממתין ל: הכנת התמונות]
- [ ] כפתור "בצע סיווג" ששולח את התמונה למודל `[גבוה]` (20m) [ממתין ל: טעינת המודל, העלאה מהמחשב, שינוי גודל, נרמול, התאמת הצורה]
- [ ] להציג את תוצאת הסיווג בצורה ברורה `[גבוה]` (30m) [ממתין ל: כפתור סיווג]

### שלב 4 · הגשה — 0% · 0/3 · (50m)
- [ ] לצרף את קובץ ה-Google Colab `[בינוני]` (10m) [ממתין ל: אימון המודל]
- [ ] להעלות את קבצי הקוד ואת קובץ המודל ל-GitHub (ריפו Project-Vision-CNN: קודם branch, ל-main אחרי בדיקה) `[גבוה]` (30m) [ממתין ל: הורדת המודל, הצגת התוצאה]
- [ ] לשלוח ל-pythonai211225+project3vision@gmail.com `[גבוה]` (10m) [ממתין ל: העלאה ל-GitHub, קובץ ה-Colab]

### בונוסים · רשות — 0% · 0/4 · (1.8h)
- [ ] להציג את הסבירות של כל אפשרות באחוזים `[נמוך]` (20m) [ממתין ל: הצגת התוצאה]
- [ ] להציג "unknown" כשרמת הביטחון נמוכה מ-60% `[נמוך]` (15m) [ממתין ל: סבירות באחוזים]
- [ ] לפתוח גישה לאפליקציה דרך האינטרנט עם ngrok `[נמוך]` (1h) [ממתין ל: הצגת התוצאה]
- [ ] לצרף צילום מסך מהטלפון של ההתחברות ושל התחזית `[נמוך]` (10m) [ממתין ל: גישה עם ngrok]
