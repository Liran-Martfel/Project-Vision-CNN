"""Liran's inference code from the Colab notebook.

The functions below are copied verbatim (body, signature, names, comments, typos).
Only the module-level setup changed: no google.colab, and the model path is built
from this file's location. The model and the face detector load once, at import.
"""

import hashlib
import os

from tensorflow.keras.models import load_model
from PIL import Image #Python Image Library
import numpy as np
import cv2 # openCV library which have face recegnition trained modelinside

from config import MODEL_FILENAME

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_FILENAME)

# Fail loudly: never fall back to another model file
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH} (set MODEL_FILENAME in config.py)")
with open(MODEL_PATH, "rb") as model_file:
    MODEL_SHA256 = hashlib.sha256(model_file.read()).hexdigest()
print(f"VISION model loaded: {MODEL_FILENAME} | {os.path.getsize(MODEL_PATH):,} bytes | sha256 {MODEL_SHA256}", flush=True)

cnn_model = load_model(MODEL_PATH)
prediction_class_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #creating the list of emotions so it can work on the website without the training data

def emotions_probability(prediction_class_name,prediction):
  probability = ""
  for i in range(len(prediction_class_name)):
      probability += f"{prediction_class_name[i]}: {prediction[0][i]* 100:.2f}%\n"
  return probability

def organize_img(img):
  img = img.convert('L') # 'L' = grayscale
  img = img.resize((48,48)) # resizing the img to fit the model (48,48)
  img = np.array(img) #transforming the img into an array in order to normalize it
  img = img / 255 #normaliting the img
  img = img.reshape(1,48,48,1)
  return img

def predict(img, cnn_model):
  prediction = cnn_model.predict(img) 
  prediction_output = np.argmax(prediction) #taking the place of the highest propability
  full_prediction = prediction_class_name[prediction_output] # position into emotion name
  return prediction, prediction_output, full_prediction

def create_answer(prediction, prediction_output, full_prediction):
  if prediction[0][prediction_output] < 0.6:
    answer = f"Unknown, I'm only {prediction[0][prediction_output] * 100:.2f}% sure that this emotion is: {full_prediction}"
  else:
    answer = f"I'm {prediction[0][prediction_output] * 100:.2f}% sure that this emotion is: {full_prediction}"
  return answer

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_locating (img_test):
  face_img = Image.open(img_test)
  face_img = face_img.convert('L')
  PIL_img = face_img # keep the grayscale PIL image before it becomes an array
  face_img = np.array(face_img)
  return face_img,PIL_img

def face_ditector(face_img,PIL_img):
  faces = face_detector.detectMultiScale(face_img)
  if len (faces) == 0:
    print ("I'm sorry, i couldn't ditect a face to analize")
    return PIL_img
  x = faces[0][0]
  y = faces[0][1]
  w = faces[0][2]
  h = faces[0][3]
  croped_img = PIL_img.crop((x, y, x + w, y + h))  # crop to the face box: (left, top, right, bottom); right = x+w, bottom = y+h
  return croped_img
