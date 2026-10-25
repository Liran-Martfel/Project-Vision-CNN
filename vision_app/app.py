"""VISION web app (Streamlit frontend).

Talks to the FastAPI backend over HTTP only; it never loads the model itself.
Run:  streamlit run app.py   (with the API running, see README.md)
"""

import hashlib
import html
import os

import requests
import streamlit as st

from i18n import DEFAULT_LANGUAGE, DIRECTION, LANGUAGES, STRINGS, emotion_name, t
from styles import css, icon

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_MB = 10


# ---------- State ----------

def init_state() -> None:
    defaults = {
        "lang": DEFAULT_LANGUAGE,
        "widget_round": 0,     # bumping it gives the camera/uploader fresh keys, which clears them
        "camera_open": False,
        "source": None,        # "camera" or "upload": whichever the user used last
        "result": None,        # {"image_id": ..., "data": {...}} from the API
        "error": None,         # {"image_id": ..., "kind": "api_down" | error code}
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def use_source(source: str) -> None:
    st.session_state.source = source
    st.session_state.result = None
    st.session_state.error = None


def toggle_camera() -> None:
    st.session_state.camera_open = not st.session_state.camera_open


def start_over() -> None:
    st.session_state.widget_round += 1
    st.session_state.camera_open = False
    st.session_state.source = None
    st.session_state.result = None
    st.session_state.error = None


def current_image():
    """The image the user gave most recently, or the other one if that was cleared."""
    n = st.session_state.widget_round
    camera_shot = st.session_state.get(f"camera_{n}")
    uploaded = st.session_state.get(f"upload_{n}")
    if st.session_state.source == "camera" and camera_shot is not None:
        return camera_shot, "camera"
    if st.session_state.source == "upload" and uploaded is not None:
        return uploaded, "upload"
    if uploaded is not None:
        return uploaded, "upload"
    if camera_shot is not None:
        return camera_shot, "camera"
    return None, None


# ---------- API ----------

def api_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def request_prediction(name: str, data: bytes, mime: str) -> tuple[dict | None, str | None]:
    """Returns (result, None) on success, or (None, error_kind) on failure."""
    try:
        response = requests.post(f"{API_URL}/predict", files={"file": (name, data, mime)}, timeout=60)
    except requests.RequestException:
        return None, "api_down"
    if response.ok:
        return response.json(), None
    try:
        return None, response.json().get("error", "generic")
    except ValueError:
        return None, "generic"


# ---------- Pieces of UI ----------

def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_header(lang: str) -> None:
    st.markdown(
        f'<header class="v-header"><h1 class="v-title">{t(lang, "title")}</h1>'
        f'<p class="v-subtitle">{t(lang, "subtitle")}</p></header>',
        unsafe_allow_html=True,
    )
    with st.container(key="lang_row"):
        st.segmented_control(
            t(lang, "language_label"),
            options=list(LANGUAGES),
            format_func=LANGUAGES.get,
            key="lang",
            required=True,
            label_visibility="collapsed",
        )


def error_message(title: str, body: str) -> str:
    return (
        f'<div class="v-error" role="alert"><div class="v-error-icon">{icon("alert")}</div>'
        f'<div><p class="v-error-title">{title}</p><p class="v-error-body">{body}</p></div></div>'
    )


def error_card(title: str, body: str) -> str:
    return f'<div class="v-card">{error_message(title, body)}</div>'


def render_api_down(lang: str) -> None:
    with st.container(key="api_down_card"):
        st.markdown(error_message(t(lang, "api_down_title"), t(lang, "api_down_body")), unsafe_allow_html=True)
        st.button(t(lang, "retry"), key="retry_health", type="secondary", width="stretch")


def tile_head(name: str, title: str, hint: str) -> str:
    return (
        f'<div class="v-tile-head"><div class="v-tile-icon">{icon(name)}</div>'
        f'<div><p class="v-tile-title">{title}</p><p class="v-tile-hint">{hint}</p></div></div>'
    )


def render_inputs(lang: str) -> None:
    n = st.session_state.widget_round
    with st.container(key="input_card"):
        camera_col, upload_col = st.columns(2, gap="medium")
        with camera_col, st.container(key="camera_tile"):
            st.markdown(tile_head("camera", t(lang, "camera_title"), t(lang, "camera_hint")), unsafe_allow_html=True)
            open_now = st.session_state.camera_open
            st.button(
                t(lang, "camera_close" if open_now else "camera_open"),
                key="camera_toggle",
                on_click=toggle_camera,
                type="secondary",
                width="stretch",
            )
            if open_now:  # ask for the camera only when the user chooses to
                st.camera_input(
                    t(lang, "camera_label"),
                    key=f"camera_{n}",
                    on_change=use_source,
                    args=("camera",),
                    label_visibility="collapsed",
                )
        with upload_col, st.container(key="upload_tile"):
            st.markdown(tile_head("upload", t(lang, "upload_title"), t(lang, "upload_hint")), unsafe_allow_html=True)
            st.file_uploader(
                t(lang, "upload_label"),
                type=UPLOAD_TYPES,
                key=f"upload_{n}",
                on_change=use_source,
                args=("upload",),
                max_upload_size=MAX_UPLOAD_MB,
                label_visibility="collapsed",
            )
            st.markdown(f'<p class="v-types">{t(lang, "upload_types")}</p>', unsafe_allow_html=True)


def skeleton(lang: str) -> str:
    return (
        '<div class="v-card"><div class="v-skeleton" aria-busy="true">'
        '<div class="v-skel circle"></div><div class="v-skel line-lg"></div><div class="v-skel line-sm"></div>'
        f'<p class="v-skel-label" role="status">{t(lang, "analyzing")}</p></div></div>'
    )


def result_card(lang: str, data: dict) -> str:
    emotion = data["emotion"]
    name = html.escape(emotion_name(lang, emotion))
    confidence = f'<span class="v-num">{percent(data["confidence"])}</span>'
    if data["is_unknown"]:
        guess = t(lang, "best_guess", emotion=name, confidence=confidence)
        body = (
            f'<div class="v-result is-unknown" role="status" aria-live="polite">'
            f'<div class="v-result-icon">{icon("unknown", 40)}</div>'
            f'<p class="v-result-label">{t(lang, "result_label")}</p>'
            f'<p class="v-result-emotion">{t(lang, "unknown")}</p>'
            f'<p class="v-result-hint">{t(lang, "unknown_hint")}</p>'
            f'<p class="v-guess">{guess}</p>'
        )
    else:
        body = (
            f'<div class="v-result" role="status" aria-live="polite">'
            f'<div class="v-result-icon">{icon(emotion, 40)}</div>'
            f'<p class="v-result-label">{t(lang, "result_label")}</p>'
            f'<p class="v-result-emotion">{name}</p>'
            f'<span class="v-pill">{t(lang, "confidence")} {confidence}</span>'
        )
    # TODO (bonus): show the other emotions' percentages here (from the API's per-emotion probabilities)
    if not data["face_detected"]:
        body += f'<div class="v-notice">{icon("scan", 20)}<span>{t(lang, "no_face")}</span></div>'
    return f'<div class="v-card v-result-card">{body}</div></div>'


def render_preview(lang: str, image, source: str) -> None:
    data = image.getvalue()
    image_id = hashlib.sha1(data).hexdigest()
    mime = image.type or "image/jpeg"
    name = image.name if source == "upload" else "vision-photo.jpg"

    with st.container(key="preview_card"):
        st.markdown(f'<p class="v-section-title">{t(lang, "preview_title")}</p>', unsafe_allow_html=True)
        try:
            st.image(data, width="stretch", alt=t(lang, "preview_title"))
        except Exception:  # the file has an image extension but isn't a readable image
            st.markdown(error_message(t(lang, "error_title"), t(lang, "error_not_an_image")), unsafe_allow_html=True)
            st.button(t(lang, "start_over"), key="start_over", on_click=start_over, type="secondary", width="stretch")
            return
        identify_col, save_col = st.columns(2, gap="small")
        with identify_col:
            identify = st.button(t(lang, "identify"), key="identify", type="primary", width="stretch")
        with save_col:
            st.download_button(
                t(lang, "save"), data=data, file_name=name, mime=mime,
                key="save", on_click="ignore", width="stretch",
            )
        with st.container(key="start_over_row"):
            st.button(t(lang, "start_over"), key="start_over", on_click=start_over, type="tertiary")

    slot = st.empty()
    if identify:  # the model runs only after this click
        slot.markdown(skeleton(lang), unsafe_allow_html=True)
        result, error = request_prediction(name, data, mime)
        st.session_state.result = {"image_id": image_id, "data": result} if result else None
        st.session_state.error = {"image_id": image_id, "kind": error} if error else None

    result = st.session_state.result
    error = st.session_state.error
    if result and result["image_id"] == image_id:
        slot.markdown(result_card(lang, result["data"]), unsafe_allow_html=True)
    elif error and error["image_id"] == image_id:
        if error["kind"] == "api_down":
            slot.markdown(error_card(t(lang, "api_down_title"), t(lang, "api_down_body")), unsafe_allow_html=True)
        else:
            key = f"error_{error['kind']}"
            message = t(lang, key) if key in STRINGS[lang] else t(lang, "error_generic")
            slot.markdown(error_card(t(lang, "error_title"), message), unsafe_allow_html=True)
    else:
        slot.empty()


# ---------- Page ----------

def main() -> None:
    init_state()
    lang = st.session_state.lang
    st.set_page_config(page_title=t(lang, "page_title"), page_icon="◉", layout="centered")
    st.markdown(
        css(DIRECTION[lang], t(lang, "upload_label"), t(lang, "camera_take"), t(lang, "camera_clear")),
        unsafe_allow_html=True,
    )

    render_header(lang)
    status_slot = st.container()  # filled last, so the page never waits on the health check

    render_inputs(lang)
    image, source = current_image()
    if image is not None:
        render_preview(lang, image, source)

    st.markdown(f'<p class="v-footer">{t(lang, "footer")}</p>', unsafe_allow_html=True)

    if not api_is_up():
        with status_slot:
            render_api_down(lang)


main()
