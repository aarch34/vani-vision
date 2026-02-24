"""
app.py - Vani-Vision: Offline AI Socratic Tutor
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""

import time
import streamlit as st
from PIL import Image

import config
import ocr_engine
import llm_engine
import socratic_prompt
import multilingual
from understanding_meter import UnderstandingMeter

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

def inject_css():
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

    :root {{
        --primary:   {config.THEME_PRIMARY};
        --secondary: {config.THEME_SECONDARY};
        --bg:        {config.THEME_BG};
        --surface:   {config.THEME_SURFACE};
        --text:      {config.THEME_TEXT};
        --success:   {config.THEME_SUCCESS};
        --warning:   {config.THEME_WARNING};
        --danger:    {config.THEME_DANGER};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }}

    /* Header */
    .app-header {{
        background: linear-gradient(135deg, #1a1040 0%, #0f2027 100%);
        border-bottom: 1px solid rgba(108,99,255,0.4);
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 0 0 16px 16px;
    }}
    .app-title {{
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #F5A623);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .app-tagline {{
        font-size: 0.85rem;
        color: rgba(232,236,244,0.6);
        margin: 0;
    }}

    /* Cards */
    .card {{
        background: var(--surface);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: border-color 0.2s;
    }}
    .card:hover {{ border-color: rgba(108,99,255,0.5); }}

    /* Chat messages */
    .msg-ai {{
        background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(108,99,255,0.05));
        border-left: 3px solid var(--primary);
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.97rem;
        line-height: 1.6;
    }}
    .msg-student {{
        background: rgba(245,166,35,0.1);
        border-right: 3px solid var(--secondary);
        border-radius: 12px 0 0 12px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        text-align: right;
        font-size: 0.97rem;
    }}

    /* Understanding Meter */
    .meter-container {{
        background: rgba(0,0,0,0.3);
        border-radius: 100px;
        height: 22px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    .meter-fill {{
        height: 100%;
        border-radius: 100px;
        transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
        background: linear-gradient(90deg, #E74C3C, #F39C12, #2ECC71);
    }}
    .meter-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: rgba(232,236,244,0.7);
        margin-top: 0.25rem;
    }}

    /* Badge */
    .badge {{
        display: inline-block;
        background: var(--primary);
        color: white;
        border-radius: 100px;
        padding: 0.15rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }}

    /* OCR Box */
    .ocr-box {{
        background: rgba(0,0,0,0.4);
        border: 1px dashed rgba(108,99,255,0.5);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
        color: #A8B4CF;
        min-height: 60px;
        white-space: pre-wrap;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: var(--surface);
        border-right: 1px solid rgba(108,99,255,0.15);
    }}

    /* Streamlit tweaks */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, var(--primary), #9B59B6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s, transform 0.1s;
    }}
    .stButton > button:hover {{
        opacity: 0.88;
        transform: translateY(-1px);
    }}
    .stTextArea textarea {{
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 10px;
        color: var(--text);
        font-size: 0.95rem;
    }}
    hr {{ border-color: rgba(108,99,255,0.15); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ─── Session State Initialisation ────────────────────────────────────────────

def init_session():
    defaults = {
        "chat_history":    [],   # [{"role": "user"|"assistant", "content": "..."}]
        "ocr_text":        "",
        "subject":         "general",
        "language":        config.DEFAULT_LANGUAGE,
        "meter":           UnderstandingMeter(),
        "wrong_streak":    0,
        "session_active":  False,
        "captured_image":  None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ─── App Header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <span style="font-size:2.5rem">🎓</span>
        <div>
            <p class="app-title">Vāṇī-Vision</p>
            <p class="app-tagline">Offline AI Socratic Tutor · Empowering every learner in their own language</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_lang = st.selectbox(
        "🌐 Language / भाषा",
        options=list(config.SUPPORTED_LANGUAGES.keys()),
        index=list(config.SUPPORTED_LANGUAGES.keys()).index(st.session_state.language),
    )
    st.session_state.language = selected_lang
    lang_code = multilingual.get_lang_code(selected_lang)

    st.markdown("---")

    # Understanding Meter display
    meter = st.session_state.meter
    st.markdown(
        f"""
        <div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:0.85rem;font-weight:600;color:rgba(232,236,244,0.8)">
                    {multilingual.get_phrase("understanding", lang_code)}
                </span>
                <span class="badge">{meter.emoji} {meter._badge(meter.score)}</span>
            </div>
            <div class="meter-container">
                <div class="meter-fill" style="width:{meter.score}%"></div>
            </div>
            <div class="meter-label">
                <span>0%</span>
                <span style="color:{meter.color};font-weight:700">{meter.score}%</span>
                <span>100%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button(f"🔄 {multilingual.get_phrase('new_session', lang_code)}"):
        for key in ["chat_history", "ocr_text", "session_active", "captured_image"]:
            st.session_state[key] = [] if key == "chat_history" else (
                "" if key == "ocr_text" else (False if key == "session_active" else None)
            )
        st.session_state.meter = UnderstandingMeter()
        llm_engine.reset_demo()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.metric("Turns", meter.turn_count)
    st.metric("Score", f"{meter.score}%")
    st.metric("Subject", st.session_state.subject.capitalize())

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem;color:rgba(232,236,244,0.4);line-height:1.6">
            🔒 <b>Fully Offline</b> · No internet required<br>
            ⚡ CPU-only · Runs on Intel i3 + 8GB RAM<br>
            🤖 Powered by Phi-3 Mini (quantized)
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Main Layout: Two Columns ──────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.4], gap="large")

# ── LEFT: Camera / Upload ─────────────────────────────────────────────────────

with col_left:
    st.markdown("### 📷 Capture Question")

    input_mode = st.radio(
        "Input Mode",
        ["📤 Upload Image", "📸 Webcam Snapshot"],
        horizontal=True,
        label_visibility="collapsed",
    )

    captured_img = None

    if input_mode == "📤 Upload Image":
        uploaded = st.file_uploader(
            multilingual.get_phrase("upload_image", lang_code),
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            captured_img = Image.open(uploaded).convert("RGB")
            st.session_state.captured_image = captured_img

    else:
        if st.button(f"📸 {multilingual.get_phrase('capture_btn', lang_code)}"):
            with st.spinner("Opening webcam…"):
                frame = ocr_engine.capture_frame(config.WEBCAM_INDEX)
                if frame is not None:
                    captured_img = Image.fromarray(frame)
                    st.session_state.captured_image = captured_img
                    st.success("Frame captured!")
                else:
                    st.error("Could not open webcam. Please check connection.")

    # Show captured image + Run OCR
    if st.session_state.captured_image is not None:
        img = st.session_state.captured_image
        img_np = ocr_engine.pil_to_cv2(img)
        annotated = ocr_engine.draw_bounding_boxes(img_np)
        st.image(annotated, caption="Detected text regions highlighted", use_container_width=True)

        if st.button("🔍 Extract & Start Tutoring"):
            with st.spinner("Running OCR…"):
                text = ocr_engine.extract_text(img, preprocess=True)

            if text.strip():
                st.session_state.ocr_text   = text
                st.session_state.subject    = socratic_prompt.detect_subject(text)
                st.session_state.session_active = True
                st.session_state.chat_history   = []
                st.session_state.meter          = UnderstandingMeter()
                llm_engine.reset_demo()

                # Generate first Socratic question
                meter = st.session_state.meter
                sys_prompt = socratic_prompt.build_system_prompt(
                    language=selected_lang,
                    subject=st.session_state.subject,
                    understanding_score=meter.score,
                    wrong_answer_count=meter.wrong_streak,
                )
                intro = socratic_prompt.build_intro_message(text, selected_lang)

                with st.spinner("Vāṇī is thinking…"):
                    ai_response = llm_engine.generate(
                        system_message=sys_prompt,
                        history=[],
                        user_message=intro,
                    )

                st.session_state.chat_history.append(
                    {"role": "user", "content": intro}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ai_response}
                )
                st.rerun()
            else:
                st.warning("No text detected. Try a clearer image or better lighting.")

    # OCR Result Box
    if st.session_state.ocr_text:
        st.markdown("#### 📝 " + multilingual.get_phrase("extracted_text", lang_code))
        st.markdown(
            f'<div class="ocr-box">{st.session_state.ocr_text}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Subject detected: **{st.session_state.subject.capitalize()}**")

# ── RIGHT: Chat Interface ─────────────────────────────────────────────────────

with col_right:
    st.markdown("### 💬 Tutoring Session")

    chat_placeholder = st.container()

    with chat_placeholder:
        if not st.session_state.session_active:
            st.markdown(
                """
                <div class="card" style="text-align:center;padding:3rem 2rem">
                    <div style="font-size:3rem">📖</div>
                    <h3 style="color:rgba(232,236,244,0.7);font-weight:500;margin:0.5rem 0">
                        Ready to learn?
                    </h3>
                    <p style="color:rgba(232,236,244,0.4);font-size:0.9rem">
                        Upload an image or capture from webcam to start your Socratic tutoring session.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Render chat history
            for msg in st.session_state.chat_history:
                if msg["role"] == "assistant":
                    st.markdown(
                        f"""
                        <div class="msg-ai">
                            <div style="font-size:0.75rem;font-weight:600;color:var(--primary);margin-bottom:4px">
                                🤖 {multilingual.get_phrase("tutor_says", lang_code)}
                            </div>
                            {msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif msg["role"] == "user" and msg.get("display", True):
                    st.markdown(
                        f"""
                        <div class="msg-student">
                            <div style="font-size:0.75rem;font-weight:600;color:var(--secondary);margin-bottom:4px">
                                🎒 You
                            </div>
                            {msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Student Answer Input
    if st.session_state.session_active:
        st.markdown("---")
        student_input = st.text_area(
            multilingual.get_phrase("your_answer", lang_code),
            placeholder="Type your answer or thinking here…",
            height=100,
            key=f"student_input_{meter.turn_count}",
        )

        if st.button(
            f"✅ {multilingual.get_phrase('submit_btn', lang_code)}",
            disabled=not student_input.strip(),
        ):
            user_text = student_input.strip()
            meter     = st.session_state.meter

            # Score the response
            result = meter.update(
                student_reply=user_text,
                concept=st.session_state.subject,
                use_llm=False,
            )

            # Build next Socratic prompt
            sys_prompt = socratic_prompt.build_system_prompt(
                language=selected_lang,
                subject=st.session_state.subject,
                understanding_score=meter.score,
                wrong_answer_count=meter.wrong_streak,
            )

            with st.spinner("Vāṇī is thinking…"):
                ai_response = llm_engine.generate(
                    system_message=sys_prompt,
                    history=st.session_state.chat_history,
                    user_message=user_text,
                )

            # Display scoring feedback briefly
            verdict_emoji = {"correct": "✅", "partial": "🔶", "incorrect": "❌"}
            st.toast(
                f"{verdict_emoji.get(result['verdict'], '💬')} "
                f"{result['verdict'].capitalize()} | Score: {result['score']}%",
                icon=result["verdict"][0].upper(),
            )

            # Update history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_text, "display": True}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ai_response}
            )
            st.rerun()

        # Turn limit reached
        if meter.turn_count >= config.MAX_SOCRATIC_TURNS:
            st.info(
                "🎯 Great session! You have completed the Socratic dialogue. "
                "Start a new session to explore another problem.",
                icon="🎓",
            )

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;font-size:0.75rem;color:rgba(232,236,244,0.3);padding:0.5rem">
        Vāṇī-Vision v1.0 · Offline AI Socratic Tutor · Built for Bharat 🇮🇳 ·
        Runs 100% on-device · No internet · No data shared
    </div>
    """,
    unsafe_allow_html=True,
)
