"""Every piece of UI text, in Hebrew and English. The layout never hard-codes a string."""

LANGUAGES = {"he": "עברית", "en": "English"}
DEFAULT_LANGUAGE = "he"
DIRECTION = {"he": "rtl", "en": "ltr"}

STRINGS = {
    "he": {
        "page_title": "VISION · זיהוי רגשות",
        "title": "VISION",
        "subtitle": "צלמו או העלו תמונת פנים, והמודל יזהה את הרגש",
        "language_label": "שפה",
        "camera_title": "צילום במצלמה",
        "camera_hint": "צלמו את עצמכם עכשיו",
        "camera_open": "פתח מצלמה",
        "camera_close": "סגור מצלמה",
        "camera_label": "צלמו תמונה",
        "camera_take": "צלם",
        "camera_clear": "צלם שוב",
        "upload_title": "העלאת תמונה",
        "upload_hint": "מהמחשב או מגלריית הטלפון",
        "upload_label": "בחרו תמונה",
        "upload_types": "קבצי <bdi>JPG</bdi>, <bdi>PNG</bdi> או <bdi>WEBP</bdi>, עד <bdi>10MB</bdi>",
        "preview_title": "התמונה שלכם",
        "identify": "זהה את הרגש",
        "save": "שמור תמונה",
        "start_over": "התחל מחדש",
        "change_photo": "החלף תמונה",
        "result_placeholder": "לחצו על „זהה את הרגש” והתוצאה תופיע כאן",
        "analyzing": "מזהה את הרגש…",
        "result_label": "הרגש שזוהה",
        "confidence": "ביטחון",
        "unknown": "לא ודאי",
        "unknown_hint": "המודל לא בטוח מספיק כדי לקבוע",
        "best_guess": "הניחוש הטוב ביותר: {emotion} · {confidence}",
        "all_emotions": "כל הרגשות",
        "no_face": "לא נמצאו פנים בתמונה, ולכן נותחה התמונה כולה.",
        "api_down_title": "השירות לא זמין כרגע",
        "api_down_body": "לא הצלחנו להתחבר לשרת הזיהוי. ודאו שהוא פועל ונסו שוב.",
        "retry": "נסה שוב",
        "error_title": "לא הצלחנו לנתח את התמונה",
        "error_not_an_image": "הקובץ הזה אינו תמונה. נסו קובץ JPG, PNG או WEBP.",
        "error_unsupported_format": "סוג התמונה לא נתמך. נסו JPG, PNG או WEBP.",
        "error_file_too_large": "התמונה גדולה מ-10MB. נסו תמונה קטנה יותר.",
        "error_empty_file": "הקובץ ריק. נסו לבחור תמונה אחרת.",
        "error_corrupt_image": "התמונה פגומה ולא ניתן לפתוח אותה. נסו תמונה אחרת.",
        "error_generic": "משהו השתבש בזמן הניתוח. נסו שוב.",
        "footer": "מודל <bdi>CNN</bdi> שאומן על <bdi>{dataset}</bdi> · 7 רגשות",
    },
    "en": {
        "page_title": "VISION · Emotion recognition",
        "title": "VISION",
        "subtitle": "Take or upload a photo of a face, and the model will recognize the emotion",
        "language_label": "Language",
        "camera_title": "Take a photo",
        "camera_hint": "Use your camera right now",
        "camera_open": "Open camera",
        "camera_close": "Close camera",
        "camera_label": "Take a photo",
        "camera_take": "Take photo",
        "camera_clear": "Retake",
        "upload_title": "Upload a photo",
        "upload_hint": "From your computer or phone gallery",
        "upload_label": "Choose a photo",
        "upload_types": "JPG, PNG or WEBP, up to 10 MB",
        "preview_title": "Your photo",
        "identify": "Identify the emotion",
        "save": "Save image",
        "start_over": "Start over",
        "change_photo": "Change photo",
        "result_placeholder": "Press “Identify the emotion” and the result will appear here",
        "analyzing": "Identifying the emotion…",
        "result_label": "Detected emotion",
        "confidence": "Confidence",
        "unknown": "Unknown",
        "unknown_hint": "The model isn't sure enough to decide",
        "best_guess": "Best guess: {emotion} · {confidence}",
        "all_emotions": "All emotions",
        "no_face": "No face was found, so the whole image was analyzed.",
        "api_down_title": "The service isn't available right now",
        "api_down_body": "We couldn't reach the recognition server. Make sure it's running and try again.",
        "retry": "Try again",
        "error_title": "We couldn't analyze this image",
        "error_not_an_image": "This file isn't an image. Try a JPG, PNG or WEBP file.",
        "error_unsupported_format": "This image type isn't supported. Try JPG, PNG or WEBP.",
        "error_file_too_large": "The image is larger than 10 MB. Try a smaller one.",
        "error_empty_file": "The file is empty. Try choosing another image.",
        "error_corrupt_image": "The image is damaged and can't be opened. Try another one.",
        "error_generic": "Something went wrong during the analysis. Please try again.",
        "footer": "CNN model trained on {dataset} · 7 emotions",
    },
}

EMOTIONS = {
    "he": {
        "angry": "כועס",
        "disgust": "גועל",
        "fear": "פחד",
        "happy": "שמח",
        "neutral": "ניטרלי",
        "sad": "עצוב",
        "surprise": "מופתע",
    },
    "en": {
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
        "surprise": "Surprise",
    },
}


def t(lang: str, key: str, **values) -> str:
    """Look up a UI string in the given language, filling in any {placeholders}."""
    text = STRINGS[lang][key]
    return text.format(**values) if values else text


def emotion_name(lang: str, emotion: str) -> str:
    return EMOTIONS[lang].get(emotion, emotion)
