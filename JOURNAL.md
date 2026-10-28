# VISION · Project Journal

Facial-expression recognition with a CNN: what I built, what I tested, and what I learned along the way.

| Item | Details |
|---|---|
| **Topic** | Facial-expression recognition |
| **Dataset** | FER-2013 (Kaggle) at first, then **FER+** (the same images with Microsoft's corrected labels); 7 classes, 48×48 grayscale |
| **Classes** | angry, disgust, fear, happy, neutral, sad, surprise |
| **Environment** | Google Colab + Google Drive, Keras / TensorFlow; the app in Python 3.12 with FastAPI and Streamlit |
| **Final model** | `training_model_v2.keras` (v2, FER+): **74.5% test accuracy** on 7,042 FER+ test images |
| **In the app** | v2 + prior correction (`balance_prediction`, α = 0.5): 73.8% accuracy, macro recall 52.4% → 60.2% |
| **Website** | FastAPI backend + Streamlit frontend, Hebrew and English, public over ngrok |
| **Status** | All three phases done (last updated 2026-10-28). Left: submission |

---

## Contents

1. [Timeline](#1-timeline)
2. [Learning log](#2-learning-log)
3. [Experiments](#3-experiments)
4. [The web app](#4-the-web-app)
5. [Mistakes and lessons](#5-mistakes-and-lessons)
6. [Reference](#6-reference)
7. [Open issues and next steps](#7-open-issues-and-next-steps)

---

## 1. Timeline

| Date | What happened |
|---|---|
| 2026-10-14 | Project started from the course brief (`project_vision_tasks.pdf`). Task checklist created; GitHub repo set up with the checklist in the README. |
| 2026-10-20 | Chose facial expressions. Downloaded FER-2013 and extracted it locally. Re-split the local copy 70/30 (`split_dataset.py`). Locked `main` on GitHub so changes go through pull requests. |
| 2026-10-20 → 10-23 | Built the data pipeline and the CNN in Colab. Ran experiments 1–4; experiment 5 started. |
| 2026-10-23 | This journal written from the project summary. Experiment 5 finished: **58.0%**, the best so far. Chosen as the final model. |
| 2026-10-24 | Saved the final model as `training_model.keras` and downloaded it to the PC. Phase 2 complete. Wrote the single-image inference pipeline and split it into functions (`organize_img`, `predict`, `create_answer`). |
| 2026-10-25 | Added automatic face cropping with OpenCV (Haar cascade). Evaluated v1 with a confusion matrix: rare emotions were weak. First version of the web app (FastAPI + Streamlit), built around my functions. |
| 2026-10-25 → 10-26 | Switched to **FER+**: same model, **74.5%** (v2). Tried class weights (v3, 61.2%) and kept v2. |
| 2026-10-26 | Fixed iPhone gallery photos (Pillow reads them as MPO). Moved the app to v2, ran a full audit, added the all-emotion probabilities and connected the app to the internet with ngrok. |
| 2026-10-27 | Tested the site on my own phone over ngrok: camera, gallery upload, a confident result (happy, 100%) and an Unknown result (51.5%). |
| 2026-10-28 | Layout review and fixes (the inputs fold away, the result scrolls into view, two columns on desktop). Added the **prior correction** (`balance_prediction`, α = 0.5) to the notebook and the app. README rewritten. ngrok stopped working: the agent needed an update, and Windows Defender quarantined the newest version (3.39.13). Went back to 3.39.11, the version that had worked; the public link is up again. |

---

## 2. Learning log

What I did at each step, and what I understood by doing it.

### 2.1 Colab and the data

**Did:** mounted Google Drive, extracted the dataset ZIP to Colab's local disk, and stored the train/test folder paths in variables.

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile(zipfile_path, 'r') as face_exp_data:
    face_exp_data.extractall('/content/data')

train_data = '/content/data/train'
test_data  = '/content/data/test'
```

**Learned:**
- Colab runs on Google's server, not my PC. It can't see `C:`, and it only sees Drive after mounting. The mount doesn't survive a new runtime.
- Extract to `/content`, not Drive: reading thousands of images straight from Drive is very slow.
- `zipfile` is a **module**; `ZipFile` is a **class** inside it. Python is case-sensitive, so `zipfile.zipfile` fails.
- A **function** does something and returns a result; a **class** creates an object you keep working with (`z.namelist()`, `z.extractall()`).
- `with` is a **context manager**: it closes the file automatically, even after an error. `as` names the object.
- A path is just a **string with an address**. A variable can't "hold a folder".
- Two different splits: **train/test** is by role (learn vs. check); **happy/sad/...** is by label (the folder name *is* the correct answer).
- Security: `extractall` on an untrusted ZIP can write outside the target folder ("Zip Slip").

### 2.2 Preparing the images

**Did:** created two generators (training with augmentation, testing without) and connected them to the folders.

```python
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)
test_datagen  = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_data, target_size=(48, 48), batch_size=32,
    class_mode='categorical', color_mode='grayscale')
```

Output: `Found 28709 images belonging to 7 classes.` / `Found 7178 images belonging to 7 classes.`

**Learned:**
- A generator is a **recipe**: it defines *how* to process images. `flow_from_directory` is the **action** that scans the folders and produces a stream of images. `fit` gets the stream (`training_set`), never the path.
- **Normalization** (÷255) brings pixels to 0–1 for faster, more stable training.
- **Augmentation** creates random variations so the model learns "a smile is a smile" instead of memorizing pictures. Only on the training set, and only realistic changes (no upside-down faces).
- `target_size=(48, 48)` is the original size: bigger adds only blur and compute, smaller throws away eyebrows and mouth corners.
- `color_mode='grayscale'`: the default `'rgb'` copies the same channel 3 times with no new information.
- `class_mode` must match the loss and the output layer:

  | `class_mode` | Label | Loss | Output layer |
  |---|---|---|---|
  | `'binary'` | 0 / 1 | `binary_crossentropy` | `Dense(1, sigmoid)` |
  | `'categorical'` | one-hot | `categorical_crossentropy` | `Dense(7, softmax)` |
  | `'sparse'` | `3` | `sparse_categorical_crossentropy` | `Dense(7, softmax)` |

- `ImageDataGenerator` is deprecated. The modern route is `image_dataset_from_directory` plus augmentation layers inside the model.

### 2.3 Building the model

**Final model (experiment 5):**

```python
cnn = Sequential()
cnn.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(128, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=7, activation='softmax'))
```

**Learned, layer by layer:**
- **Sequential** is an empty, general container. It *becomes* a CNN when Conv2D layers go in. Each layer connects only to its neighbours.
- **Conv2D:** a 3×3 filter slides over the image, multiplying and summing at each position to build a **feature map**. 32 filters give 32 maps. ReLU then zeroes the negatives. `input_shape` goes on the first layer only and must match `target_size` + `color_mode`.
- **The code defines the structure; training defines the weights.** Weights start random and `fit` updates them.
- **More filters in deeper blocks (32 → 64 → 128):** there are few kinds of edges but many combinations of them (eyes, brows, lips). After pooling the maps are 4× smaller, so doubling filters costs little. *Lose "where", gain "what".*
- **MaxPooling2D** keeps the highest value in each 2×2 square (it picks, it doesn't sum). It halves height and width and has no weights.
- `(48,48,1)` and `(2,2)` are both **tuples**. Shape has 3 numbers (height, width, channels); a window has 2.
- **Flatten** turns the 4×4×128 maps into one vector of 2,048 numbers.
- **Dense:** each neuron takes every input, weights and sums them, applies an activation, and outputs one number. The first argument *is* `units`. Dense layers **shrink** toward the output (2,048 → 128 → 7), the opposite of the filters.
- **Output = `Dense(7, softmax)`:** one score per emotion. The model doesn't output a name; we take the highest score.
- **Activations:** ReLU in hidden layers (no vanishing gradient on the positive side, cheap). Softmax in the output: it looks at all 7 numbers together and splits 100% between them.
- **Dropout(0.5)** randomly switches off half of the *previous* layer's outputs on each training batch, so the model can't lean on a few neurons that memorize. Training only; never after the output layer.
- **Always rebuild from `cnn = Sequential()`.** Re-running only the `add` lines stacks duplicate layers, and `compile` doesn't reset weights.

**Model summary:**

| Layer | Output shape | Params |
|---|---|---|
| Conv2D | (None, 46, 46, 32) | 320 |
| MaxPooling2D | (None, 23, 23, 32) | 0 |
| Conv2D | (None, 21, 21, 64) | 18,496 |
| MaxPooling2D | (None, 10, 10, 64) | 0 |
| Conv2D | (None, 8, 8, 128) | 73,856 |
| MaxPooling2D | (None, 4, 4, 128) | 0 |
| Flatten | (None, 2048) | 0 |
| Dense | (None, 128) | 262,272 |
| Dropout | (None, 128) | 0 |
| Dense | (None, 7) | 903 |
| **Total** | | **355,847** |

Three blocks is the maximum here (48 → 23 → 10 → 4); a fourth would reach about 1×1. The deeper model is *lighter*, because a 4×4 map shrinks the Dense layer.

### 2.4 Compile

```python
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Learned:**
- **Optimizer** = how weights are updated. Adam (RMSprop + momentum) is the fast, stable default.
- **He is not an optimizer.** It's a way to set *initial* weights. Initialization decides where you start; the optimizer decides how you move.
- **Loss** = how wrong the model is, and the only thing training tries to reduce. Crossentropy punishes confident wrong answers hardest.
- **Metrics** = what *I* watch. They don't affect training. It's a list, so `['accuracy']`.

### 2.5 Training

```python
stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cnn.fit(x=training_set, validation_data=testing_set, epochs=50, callbacks=[stopper])
```

**Learned:**
- **Epoch** = one full pass over all 28,709 training images. **Batch** = 32 images, after which weights update. 28,709 ÷ 32 → 898 batches per epoch.
- `val_accuracy` is the number that matters: correct answers on images the model hasn't seen. `val_loss` also reflects confidence, and **shows overfitting first**.
- `val_accuracy` can beat training `accuracy`, because training runs with augmentation and Dropout and averages the whole epoch.
- **EarlyStopping** watches `val_loss`, stops after `patience` epochs in a row without improvement, and restores the best weights. `patience` must be far smaller than `epochs`, or it never triggers. `restore_best_weights` only applies if it actually stopped.
- **Colab autosave saves the notebook, not the model.** The model lives in server memory and disappears on disconnect. Save it to Drive right after training (`cnn.save('....keras')`), or during training with `ModelCheckpoint(save_best_only=True)`.

### 2.6 The saved model

**Did:** saved the experiment 5 model from Colab as `training_model.keras` (Keras 3.13.2, 2026-10-24) and downloaded it to the PC.

**What's inside the file:** a `.keras` file is a ZIP archive with three parts:

| Part | Size | What it holds |
|---|---|---|
| `config.json` | 7 KB | The architecture: every layer and its settings, plus the compile settings |
| `model.weights.h5` | 4.3 MB | The trained weights (355,847 numbers) and the Adam optimizer's state |
| `metadata.json` | 64 B | Keras version and save date |

**Checked against experiment 5:** input `(48, 48, 1)`; Conv2D 32 / 64 / 128 (3×3, ReLU), each followed by MaxPooling 2×2; Flatten; Dense 128 ReLU; Dropout 0.5; Dense 7 softmax. Compiled with Adam, categorical crossentropy and accuracy. All of it matches.

**Learned:**
- The weights alone would take ~1.4 MB (355,847 × 4 bytes). The file is ~3× that because it also stores Adam's two running averages for every weight, so training could continue from where it stopped.
- **The file doesn't store the class names.** The output is just 7 numbers in the order `flow_from_directory` assigned: alphabetical folder names, `angry, disgust, fear, happy, neutral, sad, surprise` (`training_set.class_indices`). The app has to use this exact order.
- 4.3 MB is well under GitHub's 100 MB limit, so the model can go in the repo without Git LFS.

**Correction (2026-10-25):** the saved `training_model.keras` is from a **retrain** of experiment 5 (5b, 58.9%), because Colab reset overnight and the first trained model was lost before `cnn.save` ran. This became **v1**. The final model is v2 (see experiment 6).

### 2.7 Inference: one photo through the model

**Did:** wrote the steps for a single photo, then split them into functions with clear inputs and outputs:

| Function | Job |
|---|---|
| `organize_img` | Grayscale (`convert('L')`), resize to 48×48, `/255`, reshape to `(1, 48, 48, 1)` |
| `predict` | Runs the model; returns the probabilities, the winning position and its emotion name |
| `create_answer` | Builds the sentence; below 60% confidence it says **Unknown** |
| `emotions_probability` | One line per emotion with its percentage |

```python
img = organize_img(croped_img)
prediction, prediction_output, full_prediction = predict(img, cnn_model)
answer = create_answer(prediction, prediction_output, full_prediction)
```

**Learned:**
- `def` only defines; a function runs when it is called. Variables inside it exist only inside it: `return` is the way out, and the `=` on the calling line catches it.
- What changes between runs (the path) is a parameter; what is fixed (48, `'L'`) stays inside.
- Functions **return** text instead of printing, so the same function works in Colab (`print`) and on the website.
- The 60% check has to compare the **probability** (`prediction[0][prediction_output]`), not the **position** (`prediction_output`). Comparing the position gave no error, just wrong answers.

### 2.8 Face detection (OpenCV)

**The problem:** FER images are tight crops of a face. A normal photo has a small face and a lot of background.

| Same photo, same model | Confidence in happy |
|---|---|
| Original (face + hair + wall) | 41%, Unknown |
| Cropped by hand | 99.97% |
| Cropped automatically (OpenCV) | 98.52% |

**Did:** added OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`). It answers **where** the face is (rectangles `[x, y, w, h]`); my model answers **what** emotion it shows. `face_locating` makes a grayscale array for the detector, and `face_ditector` crops the first face with `PIL_img.crop((x, y, x + w, y + h))`.

**Decisions:**
- **No face found:** continue with the whole image instead of stopping. The detector didn't find a face in the hand-cropped photo that scored 99.97%, so stopping would reject good images. The 60% check handles the bad ones.
- **Several faces:** take `faces[0]`.
- **OpenCV 5 removed the cascade detector.** Pin `opencv-python-headless<5` in Colab and in `requirements.txt`.

### 2.9 Evaluation: confusion matrix

**Did:** a second test generator with `shuffle=False`, so the predictions line up with the true labels, then `confusion_matrix` and `classification_report`.

```python
evaluation_set = test_datagen.flow_from_directory(test_data, target_size=(48, 48), batch_size=32,
                     class_mode='categorical', color_mode='grayscale', shuffle=False)
y_true = evaluation_set.classes
y_pred = np.argmax(cnn_model.predict(evaluation_set), axis=1)
```

**Learned:**

| Metric | Question it answers |
|---|---|
| recall | Of all the real X, how many did the model **find**? |
| precision | Of all the times the model **said** X, how often was it right? |
| macro avg | Every emotion counts equally, so rare emotions show up |
| weighted avg | Weighted by count, so the common emotions dominate |

**v1 on FER-2013:** happy 86% recall, but disgust 7% and fear 31%. Fear and surprise get confused in both directions (open eyes, raised eyebrows), and sad / neutral / fear form a triangle of subtle expressions. Disgust is a problem of **quantity**.

### 2.10 Prior correction (`balance_prediction`)

**The problem:** v2 learned "when unsure, say neutral". Neutral is 37% of the FER+ training images and disgust is 0.7%, so the model leans toward the common emotions.

**Did:** after `predict`, divide each probability by how common that emotion was in training (to the power α = 0.5), then scale back to 100%. The model isn't retrained; only its output is rebalanced.

```python
train_counts = np.array([2466, 191, 652, 7528, 10308, 3514, 3562])   # FER+ training images per emotion
train_prior = train_counts / train_counts.sum()
balanced = prediction / train_prior ** alpha
balanced = balanced / balanced.sum()
```

**Result on the FER+ test set (7,042 images):**

| Emotion | Recall without | Recall with |
|---|---|---|
| angry | 55% | **64%** |
| disgust | 0% | **21%** |
| fear | 22% | **44%** |
| happy | **87%** | 85% |
| neutral | **86%** | 77% |
| sad | 37% | **50%** |
| surprise | 79% | **80%** |
| **accuracy** | **74.5%** | 73.8% |
| **macro avg recall** | 52.4% | **60.2%** |

**Learned:** α decides how strong the correction is: 0 changes nothing, 1 divides by the full prior. α = 0.5 gives most of the rare emotions' gain (disgust 0% → 21%, fear 22% → 44%) for less than 1% of overall accuracy, a better trade than v3's class weights (−13 points of accuracy). The counts must be in the same alphabetical order as `prediction_class_name`; they add up to 28,221, the number `flow_from_directory` reports.

---

## 3. Experiments

Rule: **one change per experiment**, results written down before the next run, compared on `val_accuracy` and `val_loss`.

### Summary

| # | Change | Filters | Dropout | Epochs | Params | Best `val_acc` | Lowest `val_loss` | Diagnosis |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | 32 / 32 | – | 15 | ~420K | 55.6% (ep 11) | 1.193 (ep 11) | Overfitting from epoch 11 |
| 2 | 64 filters in block 2 | 32 / 64 | – | 15 | ~839K | 55.5% (ep 13) | 1.183 (ep 11) | Same overfitting, slower |
| 3 | + Dropout 0.5 | 32 / 64 | 0.5 | 15 | ~839K | 52.8% (ep 14) | 1.238 (ep 14) | No overfitting, still learning |
| 4 | + EarlyStopping, 35 epochs | 32 / 64 | 0.5 | 35 | ~839K | **56.8% (ep 34)** | **1.143 (ep 34)** | Still improving slowly |
| 5 | + Third block | 32 / 64 / 128 | 0.5 | 34 (stopped) | 355,847 | **58.0% (ep 29, restored)** | **1.104 (ep 29)** | Best on FER-2013: no overfitting, learns faster |
| 5b | Retrain of 5 (after a Colab reset) | 32 / 64 / 128 | 0.5 | 41 (stopped) | 355,847 | 58.9% (ep 36) | 1.095 (ep 36) | **v1**: `training_model.keras` |
| 6 | **FER+** instead of FER-2013 | 32 / 64 / 128 | 0.5 | 35 (stopped) | 355,847 | **74.5% (ep 30)** | **0.673 (ep 30)** | **v2: the final model** |
| 7 | FER+ + `class_weight` | 32 / 64 / 128 | 0.5 | 38 (stopped) | 355,847 | 61.2% (ep 33) | 1.042 (ep 33) | v3: rare emotions up, overall accuracy down |

Runs 1–5b: 28,709 training / 7,178 test images (the original FER-2013 80/20 split). Runs 6–7: FER+ with contempt removed, 28,221 training / 7,042 test images, so they can't be compared one-to-one with 1–5b. All runs: batch 32, Adam, categorical crossentropy.

### Experiment 1: Baseline

**Setup:** 2 blocks (32/32 filters), no Dropout, 15 epochs, ~420K params, ~79 s per epoch. Probably ran with `input_shape=(48,48,3)` (RGB) before the channel fix.

| | Epoch 1 | Epoch 11 | Epoch 14 |
|---|---|---|---|
| `accuracy` | 35% | 56% | 59%, still rising |
| `val_accuracy` | 44% | **55.6%** | 55%, stuck |
| `val_loss` | 1.45 | **1.19** (lowest) | 1.21, rising |

**Diagnosis:** my first read was "serious underfitting". **That was wrong.** Training kept improving while validation stalled and `val_loss` rose: that's **overfitting**. Underfitting would show both metrics low and both still rising.

**Decision:** more epochs alone would make it worse.

### Experiment 2: 64 filters in block 2

**Setup:** same as 1, block 2 → 64 filters, ~839K params, ~93 s per epoch.

**Result:** 55.5% vs. 55.6%, within noise. Same pattern: `val_loss` bottoms out at epoch 11, then rises.

**Diagnosis:** the number of filters isn't the limit; overfitting is.

**Decision:** keep 64. With Dropout, the extra capacity may pay off.

### Experiment 3: Add Dropout

**Setup:** `Dropout(0.5)` after the hidden Dense layer, 15 epochs. From here on everything is grayscale (`color_mode='grayscale'`, `(48,48,1)`).

| | Without Dropout | With Dropout |
|---|---|---|
| Best `val_accuracy` | 55.5% | 52.8% |
| `val_loss` at the end | rising | **still falling** |
| `accuracy` vs. `val_accuracy` | 60% above 55% | 49% **below** 52% |
| State | overfitting | **still learning** |

**Diagnosis:** looks worse at first glance, but the overfitting is gone and the model simply hasn't finished learning. Training `accuracy` can't be compared, because Dropout measures it with half the neurons off.

**Decision:** now more epochs make sense, protected by EarlyStopping.

### Experiment 4: EarlyStopping + 35 epochs

**Setup:** as 3, with `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`, 35 epochs. Model rebuilt from scratch (epoch 1 started at 31%).

**Result:** **56.8% `val_accuracy` and 1.143 `val_loss` at epoch 34, the best so far.** EarlyStopping never triggered: the longest run without improvement was 3 epochs (31–33), then epoch 34 improved. Training `accuracy` (56%) stayed below `val_accuracy` (57%), so no overfitting. Gains are slow: ~1.5% over the last 10 epochs.

**Decision:** the model can still grow. Next, add depth.

### Experiment 5: Third block

**Setup:** 32 / 64 / 128 filters. **One change only:** the added block. (I considered also raising 64 → 128, but that would be two changes at once.) Up to 50 epochs since experiment 4 was still improving at 35; `patience` kept at 5 for a fair comparison. 355,847 params, lighter than experiment 4.

**Compare against:** 56.8% / 1.143 from experiment 4.

**Result:**
- **EarlyStopping triggered for the first time.** The lowest `val_loss` was at epoch 29 (1.104). Epochs 30–34 brought no improvement (5 in a row), so training stopped at 34 and **epoch 29's weights were restored**.
- **Restored model: 58.0% `val_accuracy`**, up from 56.8% in experiment 4.
- The highest `val_accuracy` seen was 58.6% at epoch 31. But EarlyStopping watches `val_loss`, so it restored epoch 29. Lesson: the monitored metric decides which epoch you keep.
- **Learns faster:** reached 56% by epoch 15; experiment 4 needed until epoch 26.
- **No overfitting:** training `accuracy` (~58%) stays close to `val_accuracy`.

| | Exp. 4 (2 blocks) | Exp. 5 (3 blocks) |
|---|---|---|
| Best `val_accuracy` | 56.8% | **58.0%** |
| Lowest `val_loss` | 1.143 | **1.104** |
| Epoch reaching 56% | 26 | **15** |
| Params | ~839K | **355,847** |
| EarlyStopping | never triggered | stopped at 34, restored 29 |

**Diagnosis:** depth helped, and made the model lighter at the same time.

**Decision:** this is the final model. Save it, evaluate it, and move to Streamlit.

### Experiment 6: FER+ (v2)

**What FER+ is:** the **same** FER-2013 images, relabeled by Microsoft: 10 people tagged every image, which fixes FER-2013's known wrong labels. Same format (48×48 grayscale), so nothing in the pipeline changes.

**Setup:** the experiment 5 model, unchanged. FER+ has 8 emotions, so I deleted `contempt` (train and test) right after extracting, before `flow_from_directory`. The folder names differ (`anger`, `happiness`, `sadness`…) but sort in the same alphabetical order, so `prediction_class_name` and `Dense(7)` stay the same.

**Result: 74.5% on the FER+ test set**, with training and test accuracy close (~75%), so no overfitting.

| Emotion | Recall v1 | Recall v2 |
|---|---|---|
| happy | 86% | 87% |
| neutral | 61% | **86%** |
| surprise | 69% | **79%** |
| angry | 52% | 55% |
| sad | 44% | 37% |
| fear | 31% | 22% |
| disgust | 7% | **0%** |
| **macro avg recall** | 0.50 | 0.52 |

**Diagnosis, honestly:** FER+ relabeled many images as **neutral** (37% of the test set, up from 17%), and neutral is an emotion the model is good at. So accuracy jumped while macro recall barely moved, and the rare emotions stayed weak. Disgust got zero predictions, which is why sklearn warns that its precision is undefined.

**The surprise photo from my phone:** v1 55.8% (Unknown), **v2 94.6%**.

### Experiment 7: Class weights (v3)

**Setup:** the same as v2, plus `class_weight='balanced'`, so a mistake on a rare emotion costs more. Disgust got weight **21.1**, neutral **0.39**.

```python
weights = compute_class_weight(class_weight='balanced', classes=emotion_array, y=training_set.classes)
class_weights = dict(enumerate(weights))   # Keras needs {class_index: weight}, not an array
```

**Result:** 61.2% (epoch 33, restored). The weights **worked**: disgust 0% → 40%, fear 22% → 56%, sad 37% → 54% recall, macro recall 0.52 → **0.58**. But the model **overcorrects**: neutral fell from 86% to 56%, happy from 87% to 68%, 551 neutral images were labeled sad, and fear's precision dropped to 0.18. Training was also less stable (`val_accuracy` jumped between 52% and 61%). The surprise photo: 78%.

### Choosing the final model

| | v1 | **v2** | v3 |
|---|---|---|---|
| Data | FER-2013 | **FER+** | FER+ + class weights |
| Accuracy | 59% | **74.5%** | 61% |
| Macro avg recall | 0.50 | 0.52 | **0.58** |
| Surprise photo | 55.8% | **94.6%** | 78% |
| File | `training_model.keras` | **`training_model_v2.keras`** | `training_model_v3.keras` |

**Chose v2:** most photos people upload are neutral or happy, where v2 is best. Someone who smiles and gets "sad" will think the site is broken, while a missed disgust is noticed less. It's also the most confident on real phone photos. The rare emotions were improved later without retraining, with the prior correction (2.10).

### Rules I took from the experiments

1. One change at a time, or you can't tell what caused the result.
2. Write the result down before starting the next run.
3. Compare `val_accuracy` and `val_loss`, not training `accuracy`.
4. Look at the direction: are the metrics still improving at the end?
5. Rebuild from `Sequential()` before every experiment.
6. Choose what EarlyStopping monitors on purpose: it decides which epoch's weights you keep.
7. Save every experiment under a new name (`_v2`, `_v3`) so a working model is never overwritten, and save during training (`ModelCheckpoint`) because Colab can reset overnight.
8. Look past overall accuracy: macro recall and the confusion matrix show what happens to the rare emotions.

---

## 4. The web app

### 4.1 Architecture

```text
Browser / phone → Streamlit (app.py) → HTTP → FastAPI (api.py) → vision_core.py (my functions + the model)
```

- **Streamlit** is the interface only. It never imports TensorFlow or my functions.
- **FastAPI** loads the model and the face detector **once**, at startup, and runs my functions on every `POST /predict`. `GET /health` reports which model is loaded.
- **My functions are copied as they are** into `vision_core.py`. Only the Colab-specific parts (`drive.mount`, Drive paths) were replaced with a path built from the file's location. The app checks that the model file exists and stops with a clear error instead of silently loading another model.
- **Unknown** comes from what my `create_answer` returns, not from a second threshold. **"Was a face found?"**: if `face_ditector` returns the same image it received, no face was found. Neither needed a change to my functions.

### 4.2 What the site does

- Take a photo with the camera, or upload one from the computer or the phone's gallery (JPG, PNG, WEBP, up to 10 MB).
- **Save image** downloads it; **Identify the emotion** runs the model, and only after the click.
- The result shows the emotion, the confidence and **every emotion's probability** (bonus), or **Unknown** with the best guess below 60% (bonus), plus a notice when no face was found.
- Hebrew and English, with the whole layout flipping direction. Works on phones and desktops.
- Friendly messages for a wrong file type, a damaged image, a file over 10 MB, or the server being down.

### 4.3 Testing and fixes

- **iPhone gallery photos were rejected (415).** Pillow reads them as **MPO** (JPEG with extra frames), which wasn't on the allowed list. Found with ngrok's request inspector; fixed by allowing MPO.
- **Final audit (2026-10-26):** a fresh install from `requirements.txt`, 36 automated UI checks at phone and desktop sizes, the server-down case and the camera flow. All passed. The v2 model file was verified by its SHA-256 fingerprint and by `evaluate` (74.52% on FER+, the same as in Colab).
- **Layout review (2026-10-28):** after a photo is chosen the input cards fold away, the result scrolls into view after **Identify**, the image and result sit side by side on desktop, and the Identify button can't send the same request twice.
- **Matching the notebook:** the website's output for `vision-photo.jpg` is identical to the notebook's, to two decimals (surprise 94.19% with the prior correction).

### 4.4 ngrok: a tunnel, not a server

```text
Phone → https://spoof-nylon-unclip.ngrok-free.dev → ngrok on my PC → Streamlit (port 8502)
```

- The domain is just a reserved address. The site is reachable only while ngrok, `uvicorn` and `streamlit` all run on my computer: three terminals.
- Free account: one domain, one tunnel at a time, and a "Visit Site" warning page that each visitor sees once.
- **The token never goes into the code or GitHub.**
- Tested on my own phone on 2026-10-27: the ngrok page, the app, the camera permission, a happy result at 100%, and an Unknown result at 51.5%.

---

## 5. Mistakes and lessons

### Code errors

| Error | Cause | Lesson |
|---|---|---|
| `SyntaxError` on a path | URL without quotes | Text always goes in quotes |
| `zipfile.ZipFile('zipfile_path')` | Quotes around a variable name | Quoted = text, unquoted = variable |
| `FileNotFoundError` with a Drive link | `ZipFile` needs a file path, not a URL | Mount Drive and use `/content/drive/...` |
| `unicodeescape ... \UXXXXXXXX` | Windows path: `\U` read as a Unicode escape | Use `r'...'`, `/` or `\\`. And Colab can't see `C:` anyway |
| `IndentationError` | Lines after `with` not indented | 4 spaces for every line in the block |
| `'ZipFile' object is not subscriptable` | `face_exp_data['happy']` | A ZipFile is not a dict; extract with `extractall` |
| Error persisted after `extractall` | The `['happy']` lines were still there | Python runs top to bottom; extracting doesn't "fix" the object |
| `NameError: train_dategen` | Typo | One letter off is a different variable; use Tab completion |
| `expected axis -1 ... value 1, but received ... 3` | No `color_mode`, so it loaded RGB | `input_shape` must match loading: `color_mode='grayscale'` |
| `Dense(7, units=128)` | `units` given twice | The first argument *is* `units` |
| `'ReLU'`, `Relu`, `'categorical'` as activation | Wrong name or no quotes | Lowercase strings: `'relu'`, `'softmax'` |
| `optimizer='HE'` | He is initialization, not an optimizer | `'adam'` |
| `fit(x=train_data, ...)` | Passed the path, not the image stream | `fit` needs `training_set` |
| `EarlyStoppoing`, `restore_best_wights` | Typos | Keras rejects unknown names |
| `callbacks=stopper` | Not a list | `callbacks=[stopper]` |
| `patience=25` or `35` with 35–40 epochs | `patience` close to `epochs` | Keep `patience` much smaller, e.g. 5 |
| `img.resize(48,48)` | Two numbers instead of a tuple | `img.resize((48,48))` |
| `img.color_mode = 'grayscale'` on a PIL image | A Keras setting, not PIL. **Silent**: no error, the image stays RGB | `img.convert('L')`, and print `img.mode` to check |
| `np.array` before `resize` | NumPy's `resize` is different: it cuts numbers | Resize the PIL image first, then convert |
| `if prediction_output < 0.6` | Compared the **position** (e.g. 3), not the probability. **Silent** | `prediction[0][prediction_output] < 0.6` |
| My own `def load_model(...)` | Overwrote Keras's `load_model` and called itself | Never name a function after an existing tool |
| `module 'cv2' has no attribute 'CascadeClassifier'` | Colab installed OpenCV 5, which removed the detector | `pip install "opencv-python-headless<5"`, then restart the session |
| `face_img` overwritten by an array before `crop` | One variable used for two things | Two names: `PIL_img` for cropping, an array for the detector |
| Trained on FER-2013 instead of FER+ | The new ZIP extracted into a subfolder; the paths still pointed to the old one | Check with `os.listdir`; confirm `Found ...` and the batch count changed (882, not 898) |
| `flow_from_directory` found 8 classes | FER+ includes `contempt` | `rm -rf .../contempt` in train and test, before loading |
| The trained model was lost overnight | `cnn.save` hadn't run before Colab reset | `ModelCheckpoint` in `callbacks` saves to Drive during training |
| `'numpy.ndarray' object has no attribute 'get'` | `class_weight` got an array | `dict(enumerate(weights))`, under a separate name |
| iPhone gallery photos rejected (415) | Pillow reads them as MPO, which wasn't allowed | Allow MPO; found with ngrok's request inspector |
| `ERR_NGROK_18006` / `ERR_NGROK_334` | A domain not on my account / a domain already in use by another tunnel | Use the domain from the dashboard; stop the old tunnel first |
| `ERR_NGROK_3200` | The link's endpoint is offline: closing the terminals stopped ngrok | Start ngrok again (with the API and Streamlit running) |
| `ERR_NGROK_121` | The ngrok agent is too old (3.20.0 or newer required) | Any version from 3.20.0 works. On 2026-10-28 Windows Defender quarantined the brand-new 3.39.13 as `Trojan:Win32/Kepavll!rfn` (a reputation-based detection, very likely a false positive). Installed 3.39.11 instead, from ngrok's own server: checksum verified, signed by ngrok, Inc., and not flagged |

### Misunderstandings I corrected

| I thought | Actually |
|---|---|
| "Layers per neuron" | A layer **contains** neurons |
| Filter, multiply-and-sum and feature map are separate layers | They're what happens **inside** Conv2D |
| `Sequential` is "a CNN model" | A general container that becomes a CNN with Conv2D |
| "4 layers" | 4 (now 5) layer **types** |
| `1` means RGB | `1` is grayscale; RGB is 3 |
| MaxPooling sums | MaxPooling **picks** the maximum |
| The code line decides the weights | It defines **structure**; training decides weights |
| `input_shape` is a list and `pool_size` isn't | Both are **tuples**; they differ in length |
| Output layer with 1 or 3 neurons | **7**, one score per emotion |
| "Grow every layer", Dense too | Dense **shrinks** toward the output |
| Dropout after the output layer | **Between** the hidden Dense and the output |
| Dropout switches off the layer **after** it | It switches off the layer **before** it |
| "Serious underfitting" (exp. 1) | **Overfitting**: `val_loss` rising while training improves |
| "64 filters may be too many" for 28K images | More weights than images is normal in CNNs; 64 is on the small side |
| "Dropout made it worse" (exp. 3) | `val_loss` still falling, no overfitting: it needed more epochs |
| "55% is like a coin flip" | A guess among 7 gives 14%, so the model is 4× better |
| "Colab saves automatically" | It saves the **notebook**, not the **model** |
| "Accuracy went from 59% to 74.5%, so FER+ fixed the rare emotions" | Most of the jump is neutral; macro recall barely moved (0.50 → 0.52) |
| "Class weights are the fix for imbalance" | They helped the rare emotions but cost 13 points of accuracy; the prior correction did most of it for under 1 point |
| "`summary()` shows which model I loaded" | Models with the same architecture look identical; check with `evaluate` or the file's fingerprint |
| "ngrok hosts my site" | It's a tunnel to my computer; the site is up only while my PC runs the servers |

---

## 6. Reference

### Layer sizes
- Conv2D 3×3, no padding: `out = in − 2`
- MaxPooling 2×2: `out = floor(in / 2)`
- This model: 48 → 46 → 23 → 21 → 10 → 8 → 4
- Flatten: height × width × filters = 4 × 4 × 128 = 2,048

### Parameter counts
- Conv2D: `filters × (kh × kw × input_channels) + filters`
- Dense: `inputs × units + units`
- Conv layers are cheap because one filter is reused across the whole image. Most weights sit in Dense: in the 2-block/64 model the hidden Dense held 819,328 weights, ~97% of the model.

### Softmax
`e^score` for each class, then divide by the sum. Example: scores 2 / 1 / 0 → 7.4 / 2.7 / 1 → **67% / 24% / 9%**.

### Crossentropy loss
`loss = −ln(probability given to the correct answer)`

| P(correct) | Loss |
|---|---|
| 100% | 0 |
| 50% | 0.69 |
| 51% | **0.67** (≈ v2 on FER+) |
| 33% | **1.10** (≈ v1 on FER-2013) |
| 14% (random, 1/7) | 1.95 |
| 1% | 4.6 |

The loss is not a percentage; use it to compare epochs and experiments.

### Accuracy benchmarks on FER-2013

| Benchmark | Accuracy |
|---|---|
| Random guess among 7 | 14% |
| Always guess "happy" | ~25% |
| My model (exp. 4) | 56.8% |
| My model (exp. 5 / 5b, v1) | 58.0% / 58.9% |
| Humans | ~65% |
| Best research models | ~73–76% |

Realistic target for this project was **60–65%**. On **FER+** the same model reached **74.5%** (v2). That number isn't comparable with the FER-2013 table: FER+ has cleaner labels and far more neutral images.

### Training time
~80 s per epoch → 15 epochs ≈ 20 min, 35 ≈ 47 min, 50 ≈ 70 min.

### What limits accuracy on FER-2013
More depth (4–5+ blocks), BatchNormalization, longer training with a decaying learning rate, transfer learning and ensembles, and the data itself: noisy labels, ambiguous faces, and similar emotions (fear/surprise, sad/neutral).

---

## 7. Open issues and next steps

### Open issues
- **The test set doubles as validation.** EarlyStopping picks the best epoch by looking at `testing_set`, so 74.5% is slightly optimistic. Cleaner: split validation from training (`validation_split`) and use the test set once, at the end.
- **The notebook doesn't run top to bottom on a fresh runtime.** The v3 training cell uses `np` and `compute_class_weight` before the later cells import them, so "Run all" stops there with `NameError`.
- **The v3 number in the notebook.** The comment in the training cell says v3 reached 54%, but the same cell's log shows the restored epoch (33) at 61.2%, the figure in this journal.
- **Class imbalance.** Disgust has 54× fewer training images than neutral. The prior correction helps (disgust recall 0% → 21%), but more disgust and fear images would help more.
- **The website is only online while my computer is.** ngrok tunnels to my PC. Always-on would need real hosting (for example Streamlit Community Cloud plus a host for the API).
- **ngrok stays on 3.39.11.** Don't run `ngrok update` until Microsoft stops flagging the newest version; it would replace the working file with the one Defender quarantines.
- **Mixed imports.** `ImageDataGenerator` comes from `tensorflow.keras`, the layers from `keras`. Fine in Colab today; the first thing to check if odd errors appear elsewhere.

### Next steps
1. ~~Record experiment 5's result and compare it with 56.8%.~~ Done: 58.0%.
2. ~~Save the final model to Drive.~~ Done: `training_model.keras` (v1), then `training_model_v2.keras` (v2).
3. ~~Final evaluation with a confusion matrix per emotion.~~ Done: 2.9, experiment 6 and 2.10.
4. ~~Phase 3: build the app.~~ Done: FastAPI + Streamlit, with both bonuses (Unknown below 60%, ngrok) and every emotion's probability.
5. ~~Preprocess in the app exactly as in training.~~ Done: the app uses my own `organize_img`.
6. ~~Bring the public link back.~~ Done: ngrok 3.39.11, tested end to end over the public link on 2026-10-28.
7. Submit: the code, the model and the Colab notebook on GitHub, then the email.
