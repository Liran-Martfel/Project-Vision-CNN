# Project Vision — Facial Emotion Recognition

> **Try the app:** [Open Project Vision](http://192.168.1.95:8502)
> The link uses a private network address and is available only when the app is running on the host computer and your device can reach it on the same network.

An end-to-end computer vision project that classifies facial expressions with a convolutional neural network (CNN). The model is trained in Google Colab on FER+, and a bilingual web app lets users take or upload a photo to receive a prediction.

**Hebrew:** מערכת לזיהוי הבעות פנים באמצעות רשת נוירונים קונבולוציונית (CNN). המודל אומן על FER+, ואפליקציית רשת בעברית ובאנגלית מאפשרת לצלם או להעלות תמונה ולקבל חיזוי.

## At a glance

| | |
|---|---|
| **Task** | Facial-expression classification |
| **Emotions** | Angry, disgust, fear, happy, neutral, sad, surprise |
| **Dataset** | FER+ labels on FER-2013 images; contempt excluded, leaving 7 classes |
| **Model** | CNN with 3 convolution/pooling blocks, a 128-unit dense layer, and a 7-class softmax output |
| **Test accuracy** | **74.5%** on 7,042 FER+ test images (raw model output) |
| **Application** | FastAPI backend and Streamlit frontend |
| **Languages** | Hebrew and English |
| **Highlights** | Per-emotion probabilities, confidence threshold, and mobile-friendly interface |

## See it in action

<p align="center">
  <img src="screenshots/3_prediction_he.png" alt="Emotion prediction displayed in Hebrew" width="260">
  &nbsp;&nbsp;
  <img src="screenshots/5_prediction_en.png" alt="Emotion prediction displayed in English" width="260">
</p>

## How it works

The app detects a face, converts it to grayscale, resizes it to 48 × 48 pixels, scales pixel values to the range 0–1, and passes it to the CNN. The app applies a prior correction to the model probabilities, then displays the predicted emotion, confidence, and probabilities for all seven classes. Predictions below 60% confidence are marked as **Unknown**.

```text
Photo → Face detection → Preprocessing → CNN → Prior correction → Prediction
       OpenCV Haar       grayscale,      7 classes   rebalance      emotion,
       cascade          48 × 48, /255                probabilities confidence
```

The application separates the interface from inference: Streamlit sends the image to FastAPI over HTTP, and the API loads the model once at startup.

```text
Streamlit (app.py) ──HTTP──▶ FastAPI (api.py) ──▶ vision_core.py ──▶ CNN model
    interface                 validation and       preprocessing     inference
                              request handling      and prediction
```

## Model and evaluation

The project compares three versions using the same CNN architecture:

| Version | Training data / setting | Test accuracy | Outcome |
|---|---|---:|---|
| v1 | FER-2013 | ~59% | Initial model |
| **v2** | **FER+ (7 classes)** | **74.5%** | **Selected for the app** |
| v3 | FER+ with balanced class weights | 54% | Experiment; not selected |

FER+ improved the v2 result over v1. A separate experiment applies prior correction to v2's output to reduce the effect of class imbalance. On the test set, this changed accuracy from **74.5% to 73.8%** and macro recall from **52.4% to 60.2%**. The correction improves the balance of predictions for less common emotions; it does not retrain the model.

## Run locally

**Requirements:** Python 3.12. TensorFlow compatibility can depend on the Python version; use Python 3.12 for this project.

From the project directory, create and activate a virtual environment, then install the app dependencies:

```powershell
cd vision_app
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API and frontend in separate terminals, both in `vision_app` with the virtual environment activated:

```powershell
uvicorn api:app --port 8000
```

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The frontend connects to the API at `http://localhost:8000`.

### Using the app

1. Take a photo or upload an image.
2. Select **Identify the emotion** to run the prediction.
3. Review the predicted emotion, confidence, and probabilities for all seven emotions.

If face detection does not find a face, the app analyzes the full image and displays a notice. Results are model predictions and should not be treated as a definitive assessment of a person's emotional state.

## Open the app on a phone

The camera is generally available to a mobile browser only on `localhost` or over HTTPS. For testing on a phone, you can expose the Streamlit frontend with ngrok:

1. Install ngrok and create an account at [ngrok.com](https://ngrok.com/).
2. Configure your ngrok authentication token locally:
   ```powershell
   ngrok config add-authtoken <YOUR_TOKEN>
   ```
3. With the API and Streamlit running, start a tunnel to the frontend:
   ```powershell
   ngrok http 8501
   ```
4. Open the HTTPS address printed by ngrok on your phone.

Only the Streamlit port is exposed by this setup; the API remains local. Keep the token private, and note the app is available only while the computer and required services are running.

## Repository layout

```text
.
├── docs/                         Project notes, audit, and summary
├── model_versions/               Earlier model versions
├── notebook/                     Notebook workspace
├── screenshots/                  App screenshots
├── test_images/                  Sample images for testing
├── vision_app/
│   ├── api.py                     FastAPI endpoints
│   ├── app.py                     Streamlit interface
│   ├── config.py                  Model configuration
│   ├── i18n.py                    Hebrew and English UI text
│   ├── styles.py                  Interface styling
│   ├── vision_core.py             Model loading and inference
│   ├── models/                    Model used by the app
│   └── requirements.txt           Python dependencies
└── split_dataset.py               Dataset preparation utility
```

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

**Author:** Liran Martfel
