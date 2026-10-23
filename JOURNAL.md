# VISION · Project Journal

Facial-expression recognition with a CNN: what I built, what I tested, and what I learned along the way.

| | |
|---|---|
| **Topic** | Facial-expression recognition |
| **Dataset** | FER-2013 (Kaggle): 7 classes, 48×48 grayscale |
| **Classes** | angry, disgust, fear, happy, neutral, sad, surprise |
| **Environment** | Google Colab + Google Drive, Keras / TensorFlow |
| **Best result so far** | **56.8% validation accuracy** (experiment 4) |
| **Status** | Phase 2: training experiment 5 is running |

---

## Contents

1. [Timeline](#1-timeline)
2. [Learning log](#2-learning-log)
3. [Experiments](#3-experiments)
4. [Mistakes and lessons](#4-mistakes-and-lessons)
5. [Reference](#5-reference)
6. [Open issues and next steps](#6-open-issues-and-next-steps)

---

## 1. Timeline

| Date | What happened |
|---|---|
| 2026-10-14 | Project started from the course brief (`project_vision_tasks.pdf`). Task checklist created; GitHub repo set up with the checklist in the README. |
| 2026-10-20 | Chose facial expressions. Downloaded FER-2013 and extracted it locally. Re-split the local copy 70/30 (`split_dataset.py`). Locked `main` on GitHub so changes go through pull requests. |
| 2026-10-20 → 10-23 | Built the data pipeline and the CNN in Colab. Ran experiments 1–4; experiment 5 started. |
| 2026-10-23 | This journal written from the project summary. |

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

**Current model (experiment 5):**

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
| 5 | + Third block | 32 / 64 / 128 | 0.5 | ≤ 50 | 355,847 | *running* | *running* | *pending* |

All runs: 28,709 training / 7,178 test images (the original 80/20 split), batch 32, Adam, categorical crossentropy.

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

### Experiment 5: Third block (running)

**Setup:** 32 / 64 / 128 filters. **One change only:** the added block. (I considered also raising 64 → 128, but that would be two changes at once.) Up to 50 epochs since experiment 4 was still improving at 35; `patience` kept at 5 for a fair comparison. 355,847 params, lighter than experiment 4.

**Compare against:** 56.8% / 1.143 from experiment 4.

**Result:** *pending.*

### Rules I took from the experiments

1. One change at a time, or you can't tell what caused the result.
2. Write the result down before starting the next run.
3. Compare `val_accuracy` and `val_loss`, not training `accuracy`.
4. Look at the direction: are the metrics still improving at the end?
5. Rebuild from `Sequential()` before every experiment.

---

## 4. Mistakes and lessons

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

---

## 5. Reference

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
| 32% | **1.14** (≈ my model) |
| 14% (random, 1/7) | 1.95 |
| 1% | 4.6 |

The loss is not a percentage; use it to compare epochs and experiments.

### Accuracy benchmarks on FER-2013

| Benchmark | Accuracy |
|---|---|
| Random guess among 7 | 14% |
| Always guess "happy" | ~25% |
| **My model (exp. 4)** | **56.8%** |
| Humans | ~65% |
| Best research models | ~73–76% |

Realistic target for this project: **60–65%**.

### Training time
~80 s per epoch → 15 epochs ≈ 20 min, 35 ≈ 47 min, 50 ≈ 70 min.

### What limits accuracy on FER-2013
More depth (4–5+ blocks), BatchNormalization, longer training with a decaying learning rate, transfer learning and ensembles, and the data itself: noisy labels, ambiguous faces, and similar emotions (fear/surprise, sad/neutral).

---

## 6. Open issues and next steps

### Open issues
- **The test set doubles as validation.** EarlyStopping picks the best epoch by looking at `testing_set`, so 56.8% is slightly optimistic. Cleaner: split validation from training (`validation_split`) and use the test set once, at the end.
- **Two different splits.** Colab trains on the original 80/20 split. The local `data/` folder was re-split 70/30 with `split_dataset.py`. To train on 70/30, run the script in Colab after extracting. Results after a re-split aren't directly comparable with experiments 1–5.
- **Class imbalance.** `disgust` has ~16× fewer images than `happy`. Class weights or extra augmentation could help.
- **Mixed imports.** `ImageDataGenerator` comes from `tensorflow.keras`, the layers from `keras`. Fine in Colab today; the first thing to check if odd errors appear elsewhere.

### Next steps
1. Record experiment 5's result and compare it with 56.8%.
2. **Save the best model to Drive** (`.keras`) right after training.
3. Final evaluation, ideally with a confusion matrix per emotion.
4. Move to Phase 3 (Streamlit), and come back to improvements (BatchNormalization etc.) if time allows.
5. In the app, preprocess every image the same way as in training: resize to 48×48, grayscale, divide by 255, reshape to `(1, 48, 48, 1)`.
