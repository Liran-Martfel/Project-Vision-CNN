"""The app's single CSS block: navy & beige liquid glass (Liran's house style) with Apple-style motion.

Streamlit renders its own DOM, so the selectors target Streamlit's stable data-testid
attributes and the `st-key-<key>` classes that keyed containers get.
"""

ICONS = {
    # 24x24, 2px stroke, round caps: one consistent line style for every glyph
    "angry": '<circle cx="12" cy="12" r="10"/><path d="M16 16.5s-1.5-2-4-2-4 2-4 2"/>'
             '<path d="M7.5 8 10 9.2"/><path d="M16.5 8 14 9.2"/><path d="M9 11h.01"/><path d="M15 11h.01"/>',
    "disgust": '<circle cx="12" cy="12" r="10"/><path d="M8 10h2.5"/><path d="M13.5 10H16"/>'
               '<path d="M8 15.5c1.2-1.2 2.3.8 3.5-.2s2.3-1 4.5.7"/>',
    "fear": '<circle cx="12" cy="12" r="10"/><path d="M7.5 8.2 10 7"/><path d="M16.5 8.2 14 7"/>'
            '<circle cx="9" cy="10.5" r="1.2"/><circle cx="15" cy="10.5" r="1.2"/>'
            '<path d="M8 16.5c1-1 2-1 2.7 0s1.6 1 2.6 0 1.7-1 2.7 0"/>',
    "happy": '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/>'
             '<path d="M9 9h.01"/><path d="M15 9h.01"/>',
    "neutral": '<circle cx="12" cy="12" r="10"/><path d="M8 15h8"/><path d="M9 9h.01"/><path d="M15 9h.01"/>',
    "sad": '<circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/>'
           '<path d="M9 9h.01"/><path d="M15 9h.01"/>',
    "surprise": '<circle cx="12" cy="12" r="10"/><path d="M7.5 6.8h3"/><path d="M13.5 6.8h3"/>'
                '<circle cx="9" cy="9.8" r="1"/><circle cx="15" cy="9.8" r="1"/><circle cx="12" cy="15.5" r="2"/>',
    "unknown": '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z"/>'
              '<circle cx="12" cy="13" r="3"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
            '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
            '<circle cx="12" cy="12" r="3.5"/>',
}


def icon(name: str, size: int = 24) -> str:
    """An inline SVG icon; decorative, so hidden from screen readers (the text next to it carries the meaning)."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">'
        f"{ICONS[name]}</svg>"
    )


def css(direction: str, browse_label: str, take_label: str, clear_label: str, has_image: bool = False) -> str:
    """The whole stylesheet. Direction and the relabelled Streamlit buttons depend on the language;
    once there is an image the inputs fold away and the page widens to two columns on desktop."""
    text_align = "right" if direction == "rtl" else "left"
    image_state = """
/* ---------- There is an image: the inputs fold away, image and result side by side ---------- */
.st-key-input_card { display: none !important; }
@media (min-width: 641px) {
  [data-testid="stMainBlockContainer"], .block-container { max-width: 1040px; }
}
""" if has_image else ""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Assistant:wght@400;500;600;700&display=swap');

/* ---------- Tokens: beige paper, navy ink, gold highlight ---------- */
:root {{
  --bg: #f4efe6;
  --bg-2: #fbf8f2;
  --sand: #e6d8bf;
  --navy: #1b2a4a;
  --navy-deep: #111c33;
  --accent: #24407a;
  --accent-soft: rgba(36, 64, 122, 0.10);
  --gold: #b88a3e;
  --gold-soft: rgba(184, 138, 62, 0.14);
  --text-muted: #5f6678;
  --success: #1f7a4a;
  --danger: #b5434c;
  --glass-bg: rgba(255, 252, 246, 0.55);
  --glass-strong: rgba(255, 252, 246, 0.78);
  --glass-border: rgba(255, 255, 255, 0.8);
  --glass-shadow: 0 12px 40px rgba(27, 42, 74, 0.14);
  --glass-blur: blur(26px) saturate(160%);
  --radius-lg: 22px;
  --radius-md: 14px;
  --ease-out: cubic-bezier(0.22, 1, 0.36, 1);  /* critically damped feel: fast start, no overshoot */
  --font: 'Poppins', 'Assistant', system-ui, sans-serif;
  --font-he: 'Assistant', 'Poppins', system-ui, sans-serif;
}}

/* ---------- Page ---------- */
html, body, .stApp {{ background: var(--bg); color: var(--navy); }}
.stApp {{ font-family: {"var(--font-he)" if direction == "rtl" else "var(--font)"}; }}
.stApp::before {{
  content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(55% 50% at 18% 20%, rgba(36, 64, 122, 0.30), transparent 72%),
    radial-gradient(58% 52% at 82% 15%, rgba(216, 174, 94, 0.38), transparent 72%),
    radial-gradient(60% 55% at 25% 88%, rgba(230, 216, 191, 0.75), transparent 72%),
    radial-gradient(55% 50% at 80% 85%, rgba(70, 110, 170, 0.22), transparent 72%),
    linear-gradient(160deg, var(--bg-2), var(--bg) 60%);
  filter: blur(50px);
  animation: drift 22s ease-in-out infinite alternate;
}}
@keyframes drift {{ from {{ transform: scale(1); }} to {{ transform: scale(1.08) translate(-2%, 2%); }} }}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent; position: relative; z-index: 1; }}
[data-testid="stMain"] {{ direction: {direction}; text-align: {text_align}; }}
/* Text only: Streamlit's icons are ligatures in their own font, so spans are left alone */
.stApp :is(p, h1, h2, h3, label, li, input, small, [data-testid="stMarkdownContainer"]) {{ font-family: inherit; }}
/* st.markdown aligns its text left by default, even in Hebrew; start follows the page direction */
.stApp [data-testid="stMarkdownContainer"] {{ text-align: start; }}

/* Product feel: no Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 600px; padding: 1.75rem 16px 3rem; position: static;
}}
/* Scripts that only scroll the page take no room */
[data-testid="stElementContainer"]:has(.v-js) {{ display: none; }}

/* ---------- Header ---------- */
.v-header {{ text-align: center; margin: 0.5rem 0 0.25rem; }}
.stApp .v-title {{
  margin: 0; font-family: var(--font); font-weight: 700;
  font-size: clamp(2.4rem, 9vw, 3.2rem); line-height: 1.05; letter-spacing: 0.04em;
  background: linear-gradient(90deg, var(--navy-deep), var(--accent), var(--gold));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.stApp .v-subtitle {{
  margin: 0.6rem auto 0; max-width: 60ch; color: var(--text-muted); text-wrap: balance; text-align: center;
  font-size: 1rem; line-height: 1.55; letter-spacing: 0;
}}

/* Language switch: a small pill in the page's top corner (left in Hebrew, right in English), out of the main column */
[data-testid="stLayoutWrapper"]:has(> .st-key-lang_switch) {{
  position: absolute; top: 12px; inset-inline-end: 16px; z-index: 5; width: max-content;
}}
.st-key-lang_switch, .st-key-lang_switch [data-testid="stElementContainer"] {{ width: max-content; }}
.st-key-lang_switch button {{
  min-height: 44px; padding: 0 16px; border-radius: 999px;
  border: 1px solid var(--glass-border); background: var(--glass-bg); color: var(--navy);
  backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur);
  transition: transform 100ms ease-out, background-color 150ms ease-out;
}}
.stApp .st-key-lang_switch button p {{ font-size: 0.9rem; font-weight: 600; white-space: nowrap; }}

/* ---------- Glass cards ---------- */
.st-key-input_card, .st-key-preview_card, .st-key-api_down_card, .v-card {{
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--glass-shadow), inset 0 1px 1px rgba(255, 255, 255, 0.6);
  padding: 22px 22px;
  animation: rise 320ms var(--ease-out) both;
}}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}

/* The two input tiles: equal weight, lighter inner items (never glass on glass) */
.st-key-input_card [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
.st-key-input_card [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
.st-key-input_card [data-testid="stLayoutWrapper"]:has(> .st-key-camera_tile, > .st-key-upload_tile) {{ flex: 1 1 auto; }}
.st-key-camera_tile, .st-key-upload_tile {{
  background: var(--glass-strong); border-radius: var(--radius-md);
  border: 1px solid rgba(27, 42, 74, 0.08); padding: 16px; flex: 1 1 auto;
}}
.v-tile-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }}
.v-tile-icon {{
  flex: none; width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent);
}}
.stApp .v-tile-title {{ margin: 0; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; color: var(--navy); }}
.stApp .v-tile-hint {{ margin: 2px 0 0; font-size: 0.875rem; color: var(--text-muted); line-height: 1.4; }}
.stApp .v-types {{ margin: 2px 0 0; font-size: 0.8rem; color: var(--text-muted); line-height: 1.4; }}

/* ---------- Buttons ---------- */
.stApp .stButton button p, .stApp .stDownloadButton button p {{ font-size: 1rem; font-weight: 600; }}
.stApp .stButton button[kind="tertiary"] p {{ font-size: 0.95rem; font-weight: 500; }}
.stButton button, .stDownloadButton button {{
  min-height: 48px; border-radius: var(--radius-md); font-weight: 600; font-size: 1rem;
  touch-action: manipulation;
  transition: transform 100ms ease-out, box-shadow 150ms ease-out, background-color 150ms ease-out;
}}
.stButton button:active, .stDownloadButton button:active {{
  transform: scale(0.97);  /* feedback on press, not on release */
}}
.stButton button[kind="primary"] {{
  border: none; color: #fff;
  background: linear-gradient(90deg, var(--accent), var(--navy-deep));
  box-shadow: 0 8px 20px rgba(27, 42, 74, 0.28);
}}
.stButton button[kind="primary"]:hover {{ box-shadow: 0 10px 24px rgba(27, 42, 74, 0.34); }}
.stButton button[kind="secondary"], .stDownloadButton button {{
  background: var(--accent-soft); color: var(--accent); border: 1px solid rgba(36, 64, 122, 0.35);
}}
.stButton button[kind="tertiary"] {{ color: var(--text-muted); min-height: 44px; }}
.stButton button[kind="tertiary"]:hover {{ color: var(--accent); }}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {{
  outline: 2px solid var(--accent); outline-offset: 2px;
}}

/* ---------- Streamlit's file uploader and camera, restyled and relabelled ---------- */
[data-testid="stFileUploaderDropzone"] {{
  background: var(--bg-2); border: 1.5px dashed rgba(36, 64, 122, 0.35);
  border-radius: var(--radius-md); padding: 14px; min-height: 64px;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{ display: none; }}
[data-testid="stFileUploaderDropzone"] > span {{ width: 100%; }}
[data-testid="stFileUploaderDropzone"] > span > button {{
  width: 100%; min-height: 48px; border-radius: var(--radius-md);
  background: var(--accent-soft); color: var(--accent); border: 1px solid rgba(36, 64, 122, 0.35);
}}
/* Streamlit's own English label is swapped for ours */
[data-testid="stFileUploaderDropzone"] > span > button [data-testid="stMarkdownContainer"] {{ display: none; }}
[data-testid="stFileUploaderDropzone"] > span > button span[data-has-shortcut]::after {{
  content: "{browse_label}"; font-size: 1rem; font-weight: 600; font-family: inherit; margin-inline-start: 6px;
}}
[data-testid="stFileChip"] {{ max-width: 100%; }}
/* Streamlit's small icon buttons (remove file, add file) get a 44px touch target */
[data-testid="stFileChipDeleteBtn"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"] {{
  min-width: 44px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center;
}}
/* The preview's hover toolbar (fullscreen) isn't part of the flow and its button is too small to tap */
.st-key-preview_card [data-testid="stElementToolbar"] {{ display: none; }}
[data-testid="stFileChipName"] {{ overflow: hidden; text-overflow: ellipsis; }}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img {{ border-radius: var(--radius-md) var(--radius-md) 0 0; }}
[data-testid="stCameraInputButton"] {{
  min-height: 48px; font-size: 0 !important; border-radius: 0 0 var(--radius-md) var(--radius-md);
  background: var(--navy); color: var(--bg-2); border: none;
  transition: transform 100ms ease-out;
}}
[data-testid="stCameraInputButton"]:active {{ transform: scale(0.97); }}
[data-testid="stCameraInputButton"] svg {{ display: none; }}
[data-testid="stCameraInputButton"]::after {{
  content: "{take_label}"; font-size: 1rem; font-weight: 600; color: var(--bg-2);
}}
[data-testid="stCameraInput"]:has(img) [data-testid="stCameraInputButton"]::after {{ content: "{clear_label}"; }}

/* ---------- Preview: title and "change photo" on one line, then the image and its actions ---------- */
.stApp .v-section-title {{ margin: 0; font-weight: 600; font-size: 1.05rem; color: var(--navy); }}
.st-key-preview_head {{ flex-wrap: nowrap; }}
.st-key-preview_head [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}  /* Streamlit's -1rem would push the title off center */
.st-key-preview_head .stButton button {{ min-height: 44px; padding-inline: 4px; }}
.st-key-preview_card [data-testid="stImage"] img {{
  border-radius: var(--radius-md); max-height: 380px; object-fit: contain; background: var(--bg-2);
}}
@media (min-width: 641px) {{
  .st-key-preview_card [data-testid="stImage"] img {{ max-height: 320px; }}
}}
.stButton button:disabled[kind="primary"] {{ opacity: 0.6; cursor: progress; }}

/* Result column before the first prediction (desktop only: on a phone it would just add length) */
.v-placeholder {{
  min-height: 248px; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; text-align: center;
}}
.v-placeholder .v-result-icon {{ width: 64px; height: 64px; background: var(--accent-soft); color: var(--text-muted); animation: none; }}
.stApp .v-placeholder .v-result-hint {{ max-width: 26ch; }}
@media (max-width: 640px) {{
  [data-testid="stElementContainer"]:has(.v-placeholder) {{ display: none; }}
}}

/* ---------- Result ---------- */
.v-result-card {{ animation: none; }}  /* the inner result materializes instead */
.v-result {{
  min-height: 248px; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; gap: 6px;
  animation: materialize 380ms var(--ease-out) both;  /* blur + scale + fade: the surface arrives */
}}
@keyframes materialize {{
  from {{ opacity: 0; transform: scale(0.96); filter: blur(6px); }}
  to {{ opacity: 1; transform: none; filter: none; }}
}}
.v-result-icon {{
  width: 76px; height: 76px; border-radius: 50%; display: grid; place-items: center; margin-bottom: 6px;
  background: var(--gold-soft); color: var(--accent);
  animation: pop 420ms 80ms var(--ease-out) both;
}}
@keyframes pop {{ from {{ opacity: 0; transform: scale(0.8); }} to {{ opacity: 1; transform: none; }} }}
.stApp .v-result-label {{ margin: 0; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); letter-spacing: 0.02em; }}
.stApp .v-result-emotion {{
  margin: 0; font-size: clamp(2.2rem, 9vw, 2.8rem); font-weight: 700; line-height: 1.1;
  letter-spacing: -0.02em; color: var(--navy);
}}
.stApp .v-pill {{
  display: inline-flex; align-items: center; gap: 6px; margin-top: 6px; padding: 6px 14px;
  border-radius: 999px; background: var(--accent-soft); color: var(--accent);
  font-weight: 600; font-size: 0.95rem;
}}
.v-num {{ direction: ltr; unicode-bidi: isolate; font-variant-numeric: tabular-nums; }}
.v-result.is-unknown .v-result-icon {{ background: var(--accent-soft); color: var(--text-muted); }}
.stApp .v-result.is-unknown .v-result-emotion {{ color: var(--text-muted); }}
.stApp .v-result-hint {{ margin: 2px 0 0; color: var(--text-muted); font-size: 0.95rem; }}
.stApp .v-guess {{ margin: 8px 0 0; color: var(--navy); font-weight: 500; }}
.stApp .v-notice {{
  display: flex; align-items: flex-start; gap: 10px; margin-top: 14px; padding: 12px 14px;
  border-radius: var(--radius-md); background: var(--gold-soft); color: var(--navy);
  font-size: 0.9rem; line-height: 1.45; text-align: start; max-width: 100%;
}}
.v-notice svg {{ flex: none; color: var(--gold); margin-top: 1px; }}

/* Bonus: every emotion's probability, quiet and secondary to the main result */
.v-probs {{ width: 100%; margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(27, 42, 74, 0.10); text-align: start; }}
.stApp .v-probs-title {{ margin: 0 0 8px; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); letter-spacing: 0.02em; }}
.v-prob-list {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }}
.stApp .v-prob {{
  display: grid; grid-template-columns: minmax(6.5rem, auto) 1fr 3.4rem; align-items: center; gap: 10px;
  margin: 0; padding: 0; font-size: 0.9rem; color: var(--text-muted);
}}
.v-prob-name {{ display: inline-flex; align-items: center; gap: 8px; }}
.v-prob-name svg {{ flex: none; color: var(--accent); opacity: 0.7; }}
.v-prob-bar {{ height: 6px; border-radius: 999px; background: rgba(27, 42, 74, 0.08); overflow: hidden; }}
.v-prob-fill {{
  display: block; height: 100%; border-radius: 999px; background: rgba(36, 64, 122, 0.45);
  transform-origin: {"right" if direction == "rtl" else "left"};
  animation: grow 520ms 120ms var(--ease-out) both;
}}
@keyframes grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
.v-prob-value {{ text-align: end; }}
.stApp .v-prob.is-top {{ color: var(--navy); font-weight: 600; }}
.v-prob.is-top .v-prob-fill {{ background: linear-gradient(90deg, var(--accent), var(--gold)); }}
.v-prob.is-top .v-prob-name svg {{ opacity: 1; }}
.v-result.is-unknown .v-prob.is-top .v-prob-fill {{ background: var(--text-muted); }}

/* Skeleton while the model runs: same size as the result, so nothing jumps */
.v-skeleton {{ min-height: 248px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }}
.v-skel {{
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(27,42,74,0.07) 25%, rgba(27,42,74,0.14) 50%, rgba(27,42,74,0.07) 75%);
  background-size: 200% 100%; animation: shimmer 1.2s linear infinite;
}}
.v-skel.circle {{ width: 76px; height: 76px; border-radius: 50%; }}
.v-skel.line-lg {{ width: 55%; height: 34px; }}
.v-skel.line-sm {{ width: 30%; height: 22px; }}
.v-skel-probs {{ width: 100%; margin-top: 18px; padding-top: 14px; display: grid; gap: 14px; border-top: 1px solid rgba(27, 42, 74, 0.10); }}
.v-skel.line-row {{ width: 100%; height: 10px; }}
.stApp .v-skel-label {{ margin: 4px 0 0; color: var(--text-muted); font-size: 0.9rem; }}
@keyframes shimmer {{ from {{ background-position: 200% 0; }} to {{ background-position: -200% 0; }} }}

/* Error / API-down card */
.v-error {{ display: flex; gap: 14px; align-items: flex-start; text-align: start; }}
.v-error-icon {{
  flex: none; width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
  background: rgba(181, 67, 76, 0.12); color: var(--danger);
}}
.stApp .v-error-title {{ margin: 0; font-weight: 600; font-size: 1.05rem; color: var(--navy); }}
.v-error-alt {{
  margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(27, 42, 74, 0.10); text-align: start;
}}
.v-error-alt[lang="he"] {{ font-family: var(--font-he); }}
.v-error-alt[lang="en"] {{ font-family: var(--font); }}
.stApp .v-error-alt .v-error-title {{ font-size: 0.95rem; }}
.stApp .v-error-alt .v-error-body {{ font-size: 0.875rem; }}
.stApp .v-error-body {{ margin: 4px 0 0; color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; }}

.stApp .v-footer {{ margin: 1.5rem 0 0; text-align: center; font-size: 0.8rem; color: var(--text-muted); }}

/* ---------- Mobile ---------- */
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding-top: 1.25rem; }}
  .st-key-preview_card, .v-card {{ padding: 18px 16px; }}
  /* One card level on a phone: the two input tiles are the cards, so they get the full width */
  .st-key-input_card {{
    padding: 0; background: none; border: none; box-shadow: none;
    backdrop-filter: none; -webkit-backdrop-filter: none;
  }}
  .st-key-camera_tile, .st-key-upload_tile {{ box-shadow: var(--glass-shadow); }}
}}

/* ---------- Accessibility preferences ---------- */
@media (prefers-reduced-motion: reduce) {{
  .stApp::before {{ animation: none; }}
  .st-key-input_card, .st-key-preview_card, .v-card, .v-result, .v-result-icon {{
    animation: fade 200ms ease-out both !important;
  }}
  .v-skel, .v-prob-fill {{ animation: none; }}
  .stButton button:active, .stDownloadButton button:active {{ transform: none; }}
}}
@keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@media (prefers-reduced-transparency: reduce) {{
  .st-key-input_card, .st-key-preview_card, .v-card, .st-key-lang_switch button {{
    background: var(--bg-2); backdrop-filter: none; -webkit-backdrop-filter: none;
  }}
}}
@media (prefers-contrast: more) {{
  .st-key-input_card, .st-key-preview_card, .v-card {{ background: var(--bg-2); border: 1px solid var(--navy); }}
  .v-tile-hint, .v-subtitle, .v-result-label {{ color: var(--navy); }}
}}
{image_state}</style>
"""
