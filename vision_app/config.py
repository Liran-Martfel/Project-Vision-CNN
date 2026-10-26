"""Model settings shared by the API and the UI. Change the model here, in one place.

Kept free of heavy imports so the Streamlit frontend can read it without loading TensorFlow.
"""

MODEL_FILENAME = "training_model_v2.keras"  # inside models/ (v2: FER+, 74.5% accuracy)
MODEL_DATASET = "FER+"                      # shown in the footer and the API description
