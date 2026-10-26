"""VISION REST API: exposes the emotion model over HTTP.

Run:  uvicorn api:app --reload --port 8000
"""

import io

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from config import MODEL_DATASET
from vision_core import (
    cnn_model,
    create_answer,
    face_ditector,
    face_locating,
    organize_img,
    predict,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# JPEG covers .jpg and .jpeg. MPO is also a JPEG: iPhone gallery photos embed a second image
# (HDR/depth data), so Pillow reports them as MPO; the first frame is the photo itself.
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
PNG_SAFE_MODES = {"1", "L", "LA", "I", "P", "RGB", "RGBA"}

app = FastAPI(title="VISION API", description=f"Facial-emotion recognition ({MODEL_DATASET}, 7 emotions)")


class BadImage(Exception):
    """An upload that can't be analyzed; becomes a 4xx JSON error."""

    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail


@app.exception_handler(BadImage)
def bad_image_handler(request: Request, exc: BadImage) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.error, "detail": exc.detail})


@app.exception_handler(RequestValidationError)
def missing_file_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "missing_file", "detail": "Send one image as multipart/form-data in the field 'file'."},
    )


@app.exception_handler(Exception)
def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "prediction_failed", "detail": "Something went wrong while analyzing the image."},
    )


def read_upload(file: UploadFile) -> bytes:
    """Read the upload, refusing empty files and files over the size limit."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise BadImage(400, "empty_file", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise BadImage(413, "file_too_large", "The image is larger than 10 MB.")
    return data


def upright_image_stream(data: bytes) -> io.BytesIO:
    """Check the bytes are a supported image, apply its EXIF orientation, and re-encode it.

    Phone photos are often stored sideways with an EXIF "rotate me" flag. Fixing that here,
    before face_locating, means the face detector sees the face upright.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise BadImage(415, "not_an_image", "The file is not an image that can be read.")

    if image_format not in ALLOWED_FORMATS:
        raise BadImage(415, "unsupported_format", f"{image_format} is not supported. Use JPG, PNG or WEBP.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = ImageOps.exif_transpose(image)
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise BadImage(400, "corrupt_image", "The image is damaged and could not be decoded.")

    if image.mode not in PNG_SAFE_MODES:  # e.g. CMYK JPEGs
        image = image.convert("RGB")

    stream = io.BytesIO()
    image.save(stream, format="PNG")  # lossless, so the model sees the same pixels
    stream.seek(0)
    return stream


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict_emotion(file: UploadFile = File(...)) -> dict:
    image_stream = upright_image_stream(read_upload(file))

    # Liran's call chain, in order
    face_img, PIL_img = face_locating(image_stream)
    croped_img = face_ditector(face_img, PIL_img)
    img = organize_img(croped_img)
    prediction, prediction_output, full_prediction = predict(img, cnn_model)
    answer = create_answer(prediction, prediction_output, full_prediction)

    return {
        "emotion": full_prediction,
        "confidence": float(prediction[0][prediction_output]),
        "is_unknown": answer.startswith("Unknown"),  # decided by create_answer, not here
        "answer": answer,
        "face_detected": croped_img is not PIL_img,  # face_ditector returns its input when no face is found
        # TODO (bonus): add the per-emotion probabilities here, e.g.
        #   "probabilities": emotions_probability(prediction_class_name, prediction)
    }
