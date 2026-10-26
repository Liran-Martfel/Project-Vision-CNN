"""Model settings shared by the API and the UI. Change the model here, in one place.

Kept free of heavy imports so the Streamlit frontend can read it without loading TensorFlow.
"""

MODEL_FILENAME = "training_model.keras"  # inside models/
MODEL_DATASET = "FER-2013"               # shown in the footer and the API description
