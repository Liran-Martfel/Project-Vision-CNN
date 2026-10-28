# Project Vision — Facial Emotion Recognition

**[Visit the website →](https://spoof-nylon-unclip.ngrok-free.dev)**

An end-to-end computer vision project that classifies facial expressions with a convolutional neural network (CNN). The model is trained in Google Colab on FER+, and a bilingual web app lets users take or upload a photo to receive a prediction.

## At a glance

| Item | Details |
|---|---|
| **Task** | Facial-expression classification |
| **Emotions** | Angry, disgust, fear, happy, neutral, sad, surprise |
| **Dataset** | FER+ labels on FER-2013 images; contempt excluded, leaving 7 classes |
| **Model** | CNN with 3 convolution/pooling blocks, a 128-unit dense layer, and a 7-class softmax output |
| **Test accuracy** | **74.5%** on 7,042 FER+ test images (raw model output) |
| **Application** | FastAPI backend and Streamlit frontend |
| **Languages** | Hebrew and English |
| **Highlights** | Per-emotion probabilities, confidence threshold, and mobile-friendly interface |

## How it works

The app detects a face, converts it to grayscale, resizes it to 48 × 48 pixels, scales pixel values to the range 0–1, and passes it to the CNN. The app applies a prior correction to the model probabilities, then displays the predicted emotion, confidence, and probabilities for all seven classes. Predictions below 60% confidence are marked as **Unknown**.

The application separates the interface from inference: Streamlit sends the image to FastAPI over HTTP, and the API loads the model once at startup.

## Model and evaluation

The project compares three versions using the same CNN architecture:

| Version | Training data / setting | Test accuracy | Outcome |
|---|---|---:|---|
| v1 | FER-2013 | ~59% | Initial model |
| **v2** | **FER+ (7 classes)** | **74.5%** | **Selected for the app** |
| v3 | FER+ with balanced class weights | 61.2% | Experiment; not selected |

FER+ improved the v2 result over v1. The app then applies a prior correction to v2's output to reduce the effect of class imbalance (`balance_prediction` in the notebook and in `vision_core.py`). On the test set, this changed accuracy from **74.5% to 73.8%** and macro recall from **52.4% to 60.2%**. The correction improves the balance of predictions for less common emotions; it does not retrain the model.

## Using the website

1. Take a photo or upload an image.
2. Select **Identify the emotion** to run the prediction.
3. Review the predicted emotion, confidence, and probabilities for all seven emotions.

If face detection does not find a face, the app analyzes the full image and displays a notice. Results are model predictions and should not be treated as a definitive assessment of a person's emotional state.

## Running the server

**Requirements:** Python 3.12 (TensorFlow does not support newer Python versions yet).

Install the dependencies once:

```powershell
cd vision_app
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then start the API, the website, and its public address, each in its own terminal inside `vision_app`:

```powershell
uvicorn api:app --port 8000
```

```powershell
streamlit run app.py --server.port 8502
```

```powershell
ngrok http --url=spoof-nylon-unclip.ngrok-free.dev 8502
```

## Repository layout

| Path | What it is |
|---|---|
| `PROJECT_CNN.ipynb` | The Colab notebook: data, training, evaluation, and the inference functions |
| `JOURNAL.md` | The project journal: experiments, results, and what was learned |
| `vision_app/api.py` | FastAPI endpoints (`/health`, `/predict`) |
| `vision_app/app.py` | Streamlit interface |
| `vision_app/vision_core.py` | Model loading and the inference functions from the notebook |
| `vision_app/config.py` | Which model file the app loads |
| `vision_app/i18n.py` | Hebrew and English UI text |
| `vision_app/styles.py` | Interface styling |
| `vision_app/models/` | The model used by the app (`training_model_v2.keras`) |
| `vision_app/requirements.txt` | Python dependencies |

See [vision_app/README.md](vision_app/README.md) for the API reference and app-specific configuration details.

## Limitations

- The model uses 48 × 48 grayscale face images and may perform best on clear, frontal, well-lit faces.
- Fear and disgust are more difficult to recognize, in part because they have fewer training examples.
- Haar-cascade detection may miss small, tilted, or back-lit faces.
- Emotion classification is an estimate based on visual patterns; it does not establish how someone actually feels.

## Data and acknowledgments

- **FER-2013:** Goodfellow et al., “Challenges in Representation Learning,” ICML 2013. [Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
- **FER+:** Barsoum et al., “Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution,” 2016. [Microsoft FER+ repository](https://github.com/microsoft/FERPlus).
- **Face detection:** OpenCV Haar cascade, `haarcascade_frontalface_default.xml`.

## If the website doesn't open

| What you see | What it means | What to do |
|---|---|---|
| A **"You are about to visit"** page with a **Visit Site** button | ngrok's notice on the free plan, shown once per browser | Click **Visit Site**. The app opens, and that browser won't see the notice again |
| `ERR_NGROK_3200` (endpoint offline) | The site isn't running on the host computer | Start the three commands in [Running the server](#running-the-server), in three terminals. Closing a terminal stops that part |
| `ERR_NGROK_334` (endpoint already online) | Another ngrok window is already using the address | Close the other ngrok window (or run `taskkill /IM ngrok.exe /F`), then start ngrok again |
| `ERR_NGROK_121` (agent version too old) | The ngrok program is older than 3.20.0 | Install a newer version from [ngrok.com/download](https://ngrok.com/download). The `winget` package (3.3.1) is too old |
| `ngrok` is "not recognized", or `ngrok.exe` disappeared after `ngrok update` | Windows Defender quarantined the newest version as `Trojan:Win32/Kepavll!rfn` (a false positive) | Use version **3.39.11**, which Defender doesn't flag, and don't run `ngrok update` |
| The app opens but says **"The service isn't available right now"** | The API isn't running | Start `uvicorn api:app --port 8000` in `vision_app` and press **Try again** |

**Author:** Liran Martfel
