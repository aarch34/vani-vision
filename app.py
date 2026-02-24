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
from emotion_engine import engine as emotion_bg_engine
from tts_engine import engine as tts_engine

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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

    :root {{
        --primary:   #8A2BE2; /* BlueViolet */
        --secondary: #FF2A6D; /* Neon Pink */
        --tertiary:  #05D9E8; /* Cyan */
        --bg-color:  #0B0F19; /* Deep Space */
        --surface:   rgba(26, 31, 46, 0.6); /* Glassmorphism surface */
        --surface-border: rgba(255, 255, 255, 0.08);
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
    }}

    /* Global Typography & Background */
    html, body, [class*="css"] {{
        font-family: 'Outfit', 'Noto Sans Devanagari', sans-serif !important;
        background-color: var(--bg-color) !important;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(138, 43, 226, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(5, 217, 232, 0.08) 0%, transparent 50%);
        background-attachment: fixed;
        color: var(--text-main) !important;
    }}

    /* Hide standard Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent !important;}}
    .stApp > header {{background-color: transparent !important;}}

    /* Elegant Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.25); }}

    /* Layout Constraints */
    .stApp > main > div.block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1300px;
    }}

    /* Custom App Header */
    .app-header {{
        background: var(--surface);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--surface-border);
        padding: 1.5rem 2.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }}
    .app-header::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, var(--primary), var(--tertiary), transparent);
        opacity: 0.6;
    }}
    .app-title {{
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFF 0%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .app-tagline {{
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--tertiary);
        margin: 0.2rem 0 0 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    /* Glass Cards */
    .card {{
        background: var(--surface);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--surface-border);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }}
    .card:hover {{
        border-color: rgba(138, 43, 226, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.4), 0 0 20px rgba(138, 43, 226, 0.15);
    }}

    /* Chat Messages */
    .msg-ai, .msg-student {{
        padding: 1.25rem 1.5rem;
        margin: 1.2rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        border-radius: 20px;
        position: relative;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        animation: fadeIn 0.4s ease-out;
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

    .msg-ai {{
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(138, 43, 226, 0.25);
        border-bottom-left-radius: 4px;
        margin-right: 10%;
    }}
    .msg-ai::before {{
        content: ''; position: absolute; top: -1px; left: -1px; bottom: -1px; width: 4px;
        background: linear-gradient(to bottom, var(--primary), var(--tertiary));
        border-radius: 4px 0 0 4px;
    }}
    
    .msg-student {{
        background: linear-gradient(145deg, rgba(138, 43, 226, 0.15), rgba(138, 43, 226, 0.05));
        border: 1px solid rgba(138, 43, 226, 0.4);
        border-bottom-right-radius: 4px;
        margin-left: 10%;
        text-align: right;
    }}

    .msg-header {{
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
        opacity: 0.9;
    }}
    .msg-ai .msg-header {{ color: var(--tertiary); }}
    .msg-student .msg-header {{ color: #FFF; }}

    /* Animated Understanding Meter */
    .meter-container {{
        background: rgba(0,0,0,0.5);
        border-radius: 100px;
        height: 14px;
        overflow: hidden;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }}
    .meter-fill {{
        height: 100%;
        border-radius: 100px;
        transition: width 1s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        background: linear-gradient(90deg, var(--secondary), var(--primary), var(--tertiary));
        background-size: 200% 100%;
        animation: gradientMove 3s linear infinite;
        box-shadow: 0 0 10px rgba(5, 217, 232, 0.5);
    }}
    @keyframes gradientMove {{ 0% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
    
    .meter-label {{
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Glow Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(138, 43, 226, 0.2);
        border: 1px solid rgba(138, 43, 226, 0.6);
        color: #FFF;
        border-radius: 100px;
        padding: 0.25rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.3);
    }}

    /* OCR Code Box */
    .ocr-box {{
        background: rgba(11, 15, 25, 0.7);
        border: 1px dashed rgba(5, 217, 232, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 0.95rem;
        color: var(--tertiary);
        min-height: 80px;
        white-space: pre-wrap;
        box-shadow: inset 0 4px 15px rgba(0,0,0,0.5);
    }}

    /* Streamlit Input Overrides */
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(6, 9, 20, 0.85) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid var(--surface-border);
    }}

    /* Primary Buttons */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, #4F46E5 100%) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.4) !important;
        text-transform: uppercase;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(138, 43, 226, 0.6) !important;
        border-color: rgba(255,255,255,0.4) !important;
    }}
    .stButton > button:active {{ transform: translateY(1px) !important; }}

    /* File Uploader area */
    [data-testid="stFileUploader"] {{
        background: var(--surface) !important;
        border: 2px dashed rgba(138, 43, 226, 0.5) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: border-color 0.3s ease !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--tertiary) !important;
    }}

    /* Text Areas & Select Boxes */
    .stTextArea textarea, 
    .stSelectbox div[data-baseweb="select"] > div {{
        background: rgba(11, 15, 25, 0.8) !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }}
    .stTextArea textarea:focus, 
    .stSelectbox div[data-baseweb="select"] > div:focus-within {{
        border-color: var(--tertiary) !important;
        box-shadow: 0 0 0 3px rgba(5, 217, 232, 0.2) !important;
    }}

    /* Radio Group */
    [data-testid="stRadio"] > div {{
        background: var(--surface);
        padding: 0.8rem;
        border-radius: 14px;
        border: 1px solid var(--surface-border);
    }}

    hr {{ border-color: rgba(255, 255, 255, 0.08); margin: 2rem 0; }}
    
    /* Headers & Text Formatting */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-main) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }}
    h3 {{ font-size: 1.6rem !important; margin-bottom: 1.2rem !important; color: #F8FAFC !important; }}
    p, li, div {{ color: var(--text-main); }}
    
    /* Metrics panel styling */
    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--tertiary) 0%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
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
        "current_emotion": "neutral",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ─── App Header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <span style="font-size:3.5rem; filter: drop-shadow(0 0 15px rgba(138,43,226,0.8));">✨</span>
        <div>
            <p class="app-title">Vāṇī-Vision</p>
            <p class="app-tagline">Offline AI Socratic Tutor · Empowering every learner</p>
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

    enable_audio = st.toggle("🔊 Enable Vāṇī Voice", value=True)
    tts_engine.set_muted(not enable_audio)

    st.markdown("---")

    if st.session_state.session_active:
        st.markdown("**Live Tracker**")
        st.markdown(
            '<img src="http://localhost:5050/video_feed" width="100%" style="border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">', 
            unsafe_allow_html=True
        )
        
        current_emotion = emotion_bg_engine.get_latest_emotion()
        st.session_state.current_emotion = current_emotion
        
        EMOTION_EMOJIS = {
            "happy": "😊", "sad": "😢", "angry": "😠", 
            "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐",
            "no_face": "🚫"
        }
        emotion_emoji = EMOTION_EMOJIS.get(current_emotion, "😐")
        st.markdown(f"**Emotion at Last Turn**: {current_emotion.upper()} {emotion_emoji}")

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
        st.session_state.current_emotion = "neutral"
        llm_engine.reset_demo()
        emotion_bg_engine.stop()
        tts_engine.stop()
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
                
                # Start backend emotion tracker
                emotion_bg_engine.start()

                # Generate first Socratic question
                meter = st.session_state.meter
                sys_prompt = socratic_prompt.build_system_prompt(
                    language=selected_lang,
                    subject=st.session_state.subject,
                    understanding_score=meter.score,
                    wrong_answer_count=meter.wrong_streak,
                    emotion=st.session_state.current_emotion,
                )
                intro = socratic_prompt.build_intro_message(text, selected_lang)

                with st.spinner("Vāṇī is thinking…"):
                    ai_response = llm_engine.generate(
                        system_message=sys_prompt,
                        history=[],
                        user_message=intro,
                    )
                
                # Start TTS and queue the first message
                tts_engine.start()
                tts_engine.say(ai_response)

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
                            <div class="msg-header">🤖 {multilingual.get_phrase("tutor_says", lang_code)}</div>
                            {msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif msg["role"] == "user" and msg.get("display", True):
                    st.markdown(
                        f"""
                        <div class="msg-student">
                            <div class="msg-header">🎒 You</div>
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
                emotion=st.session_state.current_emotion,
            )

            # Build next Socratic prompt
            sys_prompt = socratic_prompt.build_system_prompt(
                language=selected_lang,
                subject=st.session_state.subject,
                understanding_score=meter.score,
                wrong_answer_count=meter.wrong_streak,
                emotion=st.session_state.current_emotion,
            )

            with st.spinner("Vāṇī is thinking…"):
                ai_response = llm_engine.generate(
                    system_message=sys_prompt,
                    history=st.session_state.chat_history,
                    user_message=user_text,
                )
                
            # Queue TTS voice
            tts_engine.start()
            tts_engine.say(ai_response)

            # Display scoring feedback briefly
            verdict_emoji = {"correct": "✅", "partial": "🔶", "incorrect": "❌"}
            st.toast(
                f"{result['verdict'].capitalize()} | Score: {result['score']}%",
                icon=verdict_emoji.get(result["verdict"], "💬"),
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
